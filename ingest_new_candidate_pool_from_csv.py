#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, re, hashlib, datetime, sys
from typing import List, Dict, Any, Set
from pymilvus import connections, db, Collection, utility
from sentence_transformers import SentenceTransformer

encoder = None
# ---------- Config ----------
MILVUS_URI = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_DB  = os.getenv("MILVUS_DB", "default")
COLL_NAME  = os.getenv("NEW_COLLECTION", "new_candidate_pool")
CSV_PATH   = os.getenv("CSV_PATH", "candidate_pool_1000.csv")

# Embedding device: 'cpu' (safe), 'mps' (Apple GPU), or 'cuda'
EMBED_DEVICE   = os.getenv("EMBED_DEVICE", "cpu").lower()   # default to CPU to avoid MPS OOMs
EMBED_MODEL    = os.getenv("EMBED_MODEL_NAME", "intfloat/e5-base-v2")
EMBED_MAX_LEN  = int(os.getenv("EMBED_MAX_LEN", "256"))     # shorter = less memory
BATCH          = int(os.getenv("BATCH", "256"))             # ingest batch to Milvus
ENC_BSZ        = int(os.getenv("ENC_BATCH", "32"))          # encoder batch size (we’ll backoff if OOM)

BASE_PARTS = ["backend","frontend","devops","security","data","mlops","cloud","systems","mobile","blockchain"]
REMAP = {"skills":"skills_extracted","titles":"top_titles_mentioned","summary":"semantic_summary"}

# ---------- Utilities ----------
def _norm(n: str) -> str:
    n = (n or "").strip()
    n = re.sub(r"[^a-zA-Z0-9_]+","_", n)
    n = re.sub(r"_+","_", n).strip("_")
    return n.lower() or "col"

def role_from(title: str, skills_and_tools: str) -> str:
    t = (title or "").lower() + " " + (skills_and_tools or "").lower()
    if any(k in t for k in ["react","angular","vue","frontend","front end"]): return "frontend"
    if any(k in t for k in ["django","spring","fastapi","flask","node","express","nest","backend","back end",".net","asp.net","rust"]): return "backend"
    if any(k in t for k in ["sre","devops","platform","terraform","ansible","kubernetes","k8s","helm","docker"]): return "devops"
    if any(k in t for k in ["security","soc","siem","iam","pentest","zero trust","cyber"]): return "security"
    if any(k in t for k in ["data engineer","etl","dbt","spark","hadoop","snowflake","bigquery","data analyst"]): return "data"
    if any(k in t for k in ["mlops","ml ops","model serving","feature store"]): return "mlops"
    if any(k in t for k in ["aws","gcp","azure","cloud engineer","cloud architect"]): return "cloud"
    if any(k in t for k in ["vmware","nutanix","olvm","sccm","dns","dhcp","ad","rhel","linux admin","systems"]): return "systems"
    if any(k in t for k in ["android","ios","mobile"]): return "mobile"
    if any(k in t for k in ["blockchain","solidity","evm","web3"]): return "blockchain"
    return "backend"

def years_band_from(v: str) -> str:
    try:
        f = float(v) if v not in (None,"") else None
    except:
        f = None
    if f is None: return ""
    if f < 3: return "junior"
    if f < 6: return "mid"
    return "senior"

def clouds_from(skills: str, tools: str) -> List[str]:
    t = (skills or "").lower() + " " + (tools or "").lower()
    out, seen = [], set()
    for flag, tag in (("aws","AWS"),("gcp","GCP"),("azure","AZURE")):
        if flag in t and tag not in seen:
            out.append(tag); seen.add(tag)
    return out[:3]

def det_uuid(name: str, email: str, city: str, state: str, country: str) -> str:
    base = "|".join([name or "", email or "", city or "", state or "", country or ""])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]

def zero_vec(d: int=768): return [0.0]*d

def milvus_in_list(field: str, values) -> str:
    # Safely build: field in ["v1","v2",...]
    quoted = []
    for v in values:
        s = str(v).replace('"', r'\"')
        quoted.append(f'"{s}"')
    return f'{field} in [{",".join(quoted)}]'

# ---------- Encoder with OOM-safe fallback ----------
# keep make_encoder() as-is
def make_encoder():
    enc = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    try:
        enc.max_seq_length = EMBED_MAX_LEN
    except Exception:
        pass
    return enc


encoder = make_encoder()

# replace your existing encode_batch() with this version
def encode_batch(texts: List[str], bsz: int) -> List[List[float]]:
    global encoder  # <--- declare first thing
    if not texts:
        return []
    cur_bsz = max(1, bsz)
    device = EMBED_DEVICE
    while True:
        try:
            emb = encoder.encode(
                texts,
                normalize_embeddings=True,
                batch_size=cur_bsz,
                show_progress_bar=False
            )
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "mps" in msg:
                if cur_bsz > 2:
                    cur_bsz = max(2, cur_bsz // 2)
                    print(f"[warn] OOM at batch={cur_bsz*2}. Retrying with batch={cur_bsz} on {device}…")
                    continue
                if device != "cpu":
                    print(f"[warn] OOM persists on {device}. Switching to CPU and retrying…")
                    os.environ["EMBED_DEVICE"] = "cpu"
                    # Rebuild encoder on CPU
                    encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
                    try:
                        encoder.max_seq_length = EMBED_MAX_LEN
                    except Exception:
                        pass
                    device = "cpu"
                    cur_bsz = 8
                    continue
            raise


# ---------- Main ----------
def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

from pymilvus import DataType

def _to_float(x):
    if x in (None, "", "null", "None"): return None
    try: return float(x)
    except: return None

def _to_int(x):
    if x in (None, "", "null", "None"): return None
    try: return int(float(x))  # handles "3.0" safely
    except: return None

def _to_str(x):
    if x is None: return ""
    return str(x)

def _to_str_list(x):
    # Accept already-a-list OR comma/semicolon separated strings
    if x in (None, "", "null", "None"): return []
    if isinstance(x, list): return [str(v) for v in x if v not in (None, "")]
    if isinstance(x, str):
        # split on commas/semicolons
        parts = re.split(r"[;,]", x)
        return [p.strip() for p in parts if p.strip()]
    return [str(x)]

def _to_vec(x, dim=768):
    # Ensure a 768-d float list (zero-pad or truncate)
    if isinstance(x, list) and len(x) == dim:
        try: return [float(v) for v in x]
        except: return [0.0]*dim
    if isinstance(x, list):
        try:
            arr = [float(v) for v in x][:dim]
            if len(arr) < dim: arr += [0.0]*(dim - len(arr))
            return arr
        except:
            return [0.0]*dim
    return [0.0]*dim

def build_field_type_map(col: Collection):
    """Return {field_name: (dtype, extra)} so we can coerce per schema."""
    m = {}
    for f in col.schema.fields:
        extra = {}
        if f.dtype == DataType.FLOAT_VECTOR:
            extra["dim"] = f.params.get("dim", 768)
        if f.dtype == DataType.ARRAY:
            extra["elem_type"] = f.element_type
        if f.dtype == DataType.VARCHAR:
            extra["max_length"] = getattr(f, "max_length", None)
        m[f.name] = (f.dtype, extra)
    return m

def coerce_row_to_schema(row: dict, ftypes: dict) -> dict:
    """Return a new row dict with values coerced to the Milvus schema types."""
    out = {}
    for name, (dt, extra) in ftypes.items():
        v = row.get(name, None)
        if dt == DataType.FLOAT:
            out[name] = _to_float(v)
        elif dt == DataType.INT64:
            out[name] = _to_int(v)
        elif dt == DataType.VARCHAR:
            out[name] = _to_str(v)
        elif dt == DataType.ARRAY:
            # we only use ARRAY<VARCHAR> in this schema
            out[name] = _to_str_list(v)
        elif dt == DataType.FLOAT_VECTOR:
            out[name] = _to_vec(v, dim=extra.get("dim", 768))
        else:
            # pass-through for types we don't use here
            out[name] = v
    return out


def main():
    connections.connect("default", uri=MILVUS_URI)
    try: db.using_database(MILVUS_DB)
    except: pass

    if not utility.has_collection(COLL_NAME):
        print(f"Collection {COLL_NAME} not found. Run create_new_candidate_pool.py first.", file=sys.stderr)
        return 2

    col = Collection(COLL_NAME)
    have = {f.name for f in col.schema.fields}
    has_clouds_text = "clouds_text" in have
    has_clouds_arr  = "clouds" in have   # ARRAY field from create script

    # ---- read + normalize + derive ----
    rows: List[Dict[str,Any]] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for raw in rdr:
            r = {}
            for k, v in raw.items():
                n = REMAP.get(_norm(k), _norm(k))
                if n == "clouds":
                    # keep CSV clouds as text only
                    if has_clouds_text:
                        r["clouds_text"] = v
                else:
                    r[n] = v

            name  = r.get("name","")
            email = r.get("email","")
            city  = r.get("location_city","")
            state = r.get("location_state","")
            country = r.get("location_country","")
            r["candidate_id"] = r.get("candidate_id") or det_uuid(name, email, city, state, country)

            titles = r.get("top_titles_mentioned","") or r.get("title","")
            skills = r.get("skills_extracted","") or r.get("skills","")
            tools  = r.get("tools_and_technologies","") or r.get("tools","")
            exp    = r.get("total_experience_years","")

            role = role_from(titles, skills + " " + tools)
            r["role_family"] = r.get("role_family") or role
            r["years_band"]  = r.get("years_band")  or years_band_from(exp)
            if has_clouds_arr:
                r["clouds"] = clouds_from(skills, tools)

            if not r.get("last_updated"):
                r["last_updated"] = datetime.datetime.utcnow().isoformat(timespec="seconds")

            if "assigned_recruiter" not in r or not r.get("assigned_recruiter"):
                r["assigned_recruiter"] = "unknown"
            if "offer_status" not in r or not r.get("offer_status"):
                r["offer_status"] = "unknown"

            rows.append(r)

    # existence check
    pks = [r["candidate_id"] for r in rows]
    exists: Set[str] = set()
    for chunk in batched(pks, 1000):
        expr = milvus_in_list("candidate_id", chunk)
        try:
            res = col.query(expr=expr, output_fields=["candidate_id"], limit=len(chunk))
            exists.update([d["candidate_id"] for d in res])
        except Exception:
            pass

    # encode with OOM-safe helper
    sum_texts = [
        " ".join([r.get("semantic_summary","") or "", r.get("employment_history","") or "", r.get("skills_extracted","") or ""]).strip()
        for r in rows
    ]
    skl_texts = [
        " ".join([r.get("skills_extracted","") or "", r.get("tools_and_technologies","") or ""]).strip()
        for r in rows
    ]
    print(f"[info] encoding summary texts on {os.getenv('EMBED_DEVICE', EMBED_DEVICE)} (max_len={EMBED_MAX_LEN}) …")
    sum_emb = encode_batch(sum_texts, bsz=max(2, min(ENC_BSZ, 64)))
    print(f"[info] encoding skills texts on {os.getenv('EMBED_DEVICE', EMBED_DEVICE)} …")
    skl_emb = encode_batch(skl_texts, bsz=max(2, min(ENC_BSZ, 64)))

    for i, r in enumerate(rows):
        r["summary_embedding"] = sum_emb[i] if sum_texts[i] else zero_vec()
        r["skills_embedding"]  = skl_emb[i] if skl_texts[i] else zero_vec()

    new_rows = [r for r in rows if r["candidate_id"] not in exists]
    upd_rows = [r for r in rows if r["candidate_id"] in exists]

    # updates
    def safe_update(row: Dict[str,Any]):
        pk = row["candidate_id"]
        scalar_fields = ["role_family","years_band","last_updated","assigned_recruiter","offer_status",
                         "skills_extracted","tools_and_technologies","semantic_summary","employment_history",
                         "top_titles_mentioned","total_experience_years","location_city","location_state","location_country"]
        if has_clouds_text: scalar_fields.append("clouds_text")
        if has_clouds_arr:  scalar_fields.append("clouds")
        for k in scalar_fields:
            if k in have and k in row and row[k] is not None:
                try: col.update(expr=f'candidate_id == "{pk}"', field_name=k, value=row[k])
                except Exception: pass
        for vf in ["summary_embedding","skills_embedding"]:
            if vf in have and vf in row and row[vf]:
                try:
                    col.update(expr=f'candidate_id == "{pk}"', field_name=vf, value=row[vf])
                except Exception:
                    # fallback: delete+insert
                    part = row.get("role_family","backend")
                    try:
                        if not col.has_partition(part): col.create_partition(part)
                    except Exception:
                        part = "backend"
                    try:
                        col.delete(expr=f'candidate_id == "{pk}"')
                        row_obj = {fn.name: row.get(fn.name, None) for fn in col.schema.fields}
                        col.insert([row_obj], partition_name=part)

                    except Exception:
                        pass

    for r in upd_rows:
        safe_update(r)

    # inserts by partition
    have_names = [f.name for f in col.schema.fields]
    part_buf: Dict[str, Dict[str, List[Any]]] = {}

    def ensure_part(pname: str):
        if not col.has_partition(pname):
            try: col.create_partition(pname)
            except Exception: pass

    def ensure_buf(pname: str):
        if pname not in part_buf:
            part_buf[pname] = {fn: [] for fn in have_names}

    def flush_part(pname: str):
    buf = part_buf.get(pname)
    if not buf:
        return

    field_names = list(buf.keys())
    if not field_names:
        return

    # Determine number of rows we’re about to send.
    # If any field has a list, use its length; if scalars sneak in, treat them as length-1.
    def _len(v):
        return len(v) if isinstance(v, list) else (1 if v is not None else 0)

    n = 0
    for v in buf.values():
        lv = _len(v)
        if lv > n:
            n = lv
    if n == 0:
        return

    # Build list-of-dicts (row-wise)
    rows = []
    for i in range(n):
        row = {}
        for fn in field_names:
            v = buf[fn]
            if isinstance(v, list):
                row[fn] = v[i] if i < len(v) else None
            else:
                # scalar → place it in the first row only, None for the rest
                row[fn] = v if i == 0 else None
        rows.append(row)

    ensure_part(pname)
    col.insert(rows, partition_name=pname)   # list[dict] OK

    # clear buffers
    for k in field_names:
        buf[k].clear()


    # Build list-of-dicts (row-wise)
    rows = []
    for i in range(n):
         row = {}
         for fn in field_names:
             v = buf[fn]
             if isinstance(v, list):
                 row[fn] = v[i] if i < len(v) else None
             else:
                # scalar → place it in the first row only, None for the rest
                 row[fn] = v if i == 0 else None
         rows.append(row)



    ensure_part(pname)
    col.insert(rows, partition_name=pname)   # <-- list[dict] OK
    # clear buffers
    for k in field_names:
        buf[k].clear()


    for r in new_rows:
        part = r.get("role_family","backend")
        ensure_buf(part)
        for fn in have_names:
            part_buf[part][fn].append(r.get(fn, None))
        if len(next(iter(part_buf[part].values()))) >= BATCH:
            flush_part(part)

    for p in list(part_buf.keys()):
        flush_part(p)

    col.flush()
    print(f"[done] upserted: new={len(new_rows)} updated={len(upd_rows)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
