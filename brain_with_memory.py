#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recruiter Brain (terminal)
- IntentRouter: greeting / search / memory
- JDExtractor: role_family, years_band, clouds, location, must_skills, urgent
- Eligibility + Rank + Presenter
- Milvus search on new_candidate_pool (scalar-first + optional vector, efSearch=32)
- Client memory:
   * On session start: upsert/find client in `client_memory`, recall last summary
   * On each successful search: append parsed JD + knobs to last_queries; bump searches_count
   * On session end ("bye"/Ctrl+C): persist a brief conversation summary + serialized embedding to preferences

Env:
  MILVUS_URI=http://34.135.232.156:19530
  MILVUS_DB=default
  CANDIDATE_COLL=new_candidate_pool
  CLIENT_MEMORY_COLL=client_memory
  TOP_K=8
  EF_SEARCH=32
  USE_VECTOR=1        (0 = scalar only)
  EMBED_MODEL=intfloat/e5-base-v2
  EMBED_DEVICE=cpu    (cpu|mps|cuda)
  EMBED_MAXLEN=256
"""

import os, re, sys, json, textwrap, hashlib, datetime
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import connections, db, Collection, utility, DataType

# -------------------- Config --------------------
MILVUS_URI   = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_DB    = os.getenv("MILVUS_DB", "default")
CAND_COLL    = os.getenv("CANDIDATE_COLL", "new_candidate_pool")
MEM_COLL     = os.getenv("CLIENT_MEMORY_COLL", "client_memory")

TOP_K        = int(os.getenv("TOP_K", "8"))
EF_SEARCH    = int(os.getenv("EF_SEARCH", "32"))
USE_VECTOR   = int(os.getenv("USE_VECTOR", "1"))
EMBED_MODEL  = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu").lower()
EMBED_MAXLEN = int(os.getenv("EMBED_MAXLEN", "256"))

ROLE_FAMILIES = ["backend","frontend","devops","security","data","mlops","cloud","systems","mobile","blockchain"]
ROLE_HINTS = {
    "frontend": ["react","angular","vue","nextjs","frontend","front end","typescript","ui"],
    "backend":  ["django","drf","spring","springboot","fastapi","flask","node","express","nest",".net","asp.net","golang","rust","grpc","rest","microservice","server"],
    "devops":   ["sre","devops","platform","terraform","ansible","kubernetes","k8s","helm","docker","ci/cd","argo","tekton"],
    "security": ["security","soc","siem","iam","pentest","zero trust","threat","mitre"],
    "data":     ["data engineer","etl","dbt","spark","hadoop","snowflake","bigquery","redshift","airflow","kafka","databricks","data pipeline","data warehouse"],
    "mlops":    ["mlops","model serving","feature store","inference","vector db","rag","embedding","llmops"],
    "cloud":    ["cloud engineer","cloud architect","aws","gcp","azure"],
    "systems":  ["vmware","nutanix","olvm","sccm","dns","dhcp","ad","rhel","linux admin","windows server","backup","veeam","commvault"],
    "mobile":   ["android","ios","mobile","swift","kotlin","react native","flutter"],
    "blockchain":["blockchain","solidity","evm","web3","smart contract"],
}
CLOUD_EQUIV = {"aws":"AWS","gcp":"GCP","azure":"AZURE"}

FIELDS = [
    "candidate_id","name","top_titles_mentioned",
    "location_city","location_state","location_country",
    "total_experience_years","skills_extracted","tools_and_technologies",
    "employment_history","semantic_summary","clouds","role_family","years_band",
    "last_updated","source_channel","assigned_recruiter"
]

# -------------------- Embedding (lazy) --------------------
_encoder = None
def get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
        try: _encoder.max_seq_length = EMBED_MAXLEN
        except: pass
    except Exception as e:
        print(f"[warn] encoder init failed ({e}); continuing w/o vectorization", file=sys.stderr)
        _encoder = None
    return _encoder

def embed_text_once(text: str) -> Optional[List[float]]:
    enc = get_encoder()
    if not enc or not text:
        return None
    try:
        arr = enc.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return arr[0].tolist()
    except Exception as e:
        print(f"[warn] embed failed: {e}", file=sys.stderr)
        return None

def serialize_vec(vec: Optional[List[float]], max_len_chars=7000) -> str:
    if not vec:
        return ""
    s = ",".join(f"{x:.6f}" for x in vec)
    # keep under VARCHAR(8192) headroom
    return s[:max_len_chars]

# -------------------- Intent Router --------------------
GREETING_SET = {
    "hi","hello","hey","how are you","how’s it going","good morning","good afternoon","good evening",
    "thanks","thank you","cool","ok","okay","yo","sup"
}
MEMORY_HINTS = ["recap","summary","last slate","preferences","client memory","what did we send","remind me"]

def is_greeting(txt: str) -> bool:
    t = txt.strip().lower()
    return (len(t.split()) <= 4 and any(p in t for p in GREETING_SET))

def is_memory(txt: str) -> bool:
    t = txt.lower()
    return any(h in t for h in MEMORY_HINTS)

def has_jd_signal(txt: str) -> bool:
    t = txt.lower()
    if re.search(r"\b(\d+)\+?\s*(yrs|years|y)\b", t): return True
    if any(k in t for k in ["need","looking for","require","must have","jd","role","position"]): return True
    if any(k in t for k in ["aws","gcp","azure","django","spring","kubernetes","datadog","prometheus","grafana","grpc","rest","sql","postgres","mysql","sql server"]): return True
    if any(k in t for k in ROLE_FAMILIES): return True
    return False

def route_intent(user_text: str) -> str:
    if is_greeting(user_text): return "greeting"
    if is_memory(user_text):   return "memory"
    if has_jd_signal(user_text): return "search"
    return "unknown"

# -------------------- JD extraction --------------------
def infer_role_family(text: str) -> Optional[str]:
    t = text.lower()
    for fam in ROLE_FAMILIES:
        if re.search(rf"\b{fam}\b", t):
            return fam
    best, score = None, 0
    for fam, hints in ROLE_HINTS.items():
        hit = sum(1 for h in hints if h in t)
        if hit > score:
            best, score = fam, hit
    return best

def years_band_from_text(text: str) -> Optional[str]:
    t = text.lower()
    if "junior" in t: return "junior"
    if "mid" in t or "mid-level" in t or "middle" in t: return "mid"
    if "senior" in t or "lead" in t or "staff" in t or "principal" in t: return "senior"
    m = re.search(r"(\d+)\+?\s*(?:yrs|years|y)", t)
    if m:
        y = int(m.group(1))
        if y < 3: return "junior"
        if y < 6: return "mid"
        return "senior"
    return None

def extract_clouds(text: str) -> List[str]:
    t = text.lower()
    out = []
    for k, v in CLOUD_EQUIV.items():
        if k in t: out.append(v)
    return out[:3]

def extract_location(text: str) -> Dict[str,str]:
    t = text.lower()
    loc = {}
    if "remote" in t: loc["mode"] = "remote"
    if "onsite" in t or "on-site" in t: loc["mode"] = "onsite"
    return loc

def extract_must_skills(text: str) -> List[str]:
    t = text.lower()
    TOKS = [
        "django","drf","rest","grpc","aws","gcp","azure","kubernetes","eks","gke","aks","docker",
        "datadog","prometheus","grafana","postgres","mysql","sql server","kafka","spark","airflow",
        "spring","springboot","fastapi","flask",".net","golang","rust","terraform","ansible","jenkins","gh actions","github actions",
    ]
    out = []
    for tok in TOKS:
        if tok in t: out.append(tok)
    if "gh actions" in out and "github actions" not in out:
        out.append("github actions")
    return list(dict.fromkeys(out))

def parse_jd(user_text: str) -> Dict[str, Any]:
    return {
        "raw": user_text,
        "role_family": infer_role_family(user_text),
        "years_band": years_band_from_text(user_text),
        "clouds": extract_clouds(user_text),
        "location": extract_location(user_text),
        "must_skills": extract_must_skills(user_text),
        "urgent": ("urgent" in user_text.lower()),
    }

# -------------------- Milvus connect --------------------
def connect_milvus():
    connections.connect("default", uri=MILVUS_URI)
    try: db.using_database(MILVUS_DB)
    except Exception: pass
    if not utility.has_collection(CAND_COLL):
        raise RuntimeError(f"Collection {CAND_COLL} not found.")
    if not utility.has_collection(MEM_COLL):
        raise RuntimeError(f"Collection {MEM_COLL} not found.")
    return Collection(CAND_COLL), Collection(MEM_COLL)

# -------------------- Search helpers --------------------
def arr_contains_expr(field: str, values: List[str]) -> Optional[str]:
    if not values: return None
    clauses = [f'contains({field}, "{v}")' for v in values]
    return "(" + " or ".join(clauses) + ")"

def build_scalar_expr(jd: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    terms = []
    part = None
    rf = jd.get("role_family")
    if rf:
        terms.append(f'role_family == "{rf}"')
        part = rf if rf in ROLE_FAMILIES else None
    yb = jd.get("years_band")
    if yb:
        terms.append(f'years_band == "{yb}"')
    cloud_expr = arr_contains_expr("clouds", jd.get("clouds", []))
    if cloud_expr:
        terms.append(cloud_expr)
    return (" and ".join(terms)) if terms else "", part

def normalize_tokens(s: str) -> List[str]:
    if not s: return []
    s = s.lower()
    s = re.sub(r"[^a-z0-9\+\.\#\- ]+"," ", s)
    return [t for t in s.split() if t]

EQUIV = {
    "github actions": ["gh actions","github-actions"],
    "rest": ["restful","rest api","restapis"],
    "grpc": ["g rpc"],
    "postgres": ["postgresql","postgre","psql"],
    "kubernetes": ["k8s","eks","gke","aks"],
    "datadog": ["data dog"],
    "prometheus": ["prom"],
    "grafana": ["gra f a na"],
    "aws": ["amazon web services"],
}

def expand_equivalents(req: List[str]) -> List[str]:
    out = set()
    for r in req:
        r0 = r.lower(); out.add(r0)
        for k, alts in EQUIV.items():
            if r0 == k or r0 in alts:
                out.add(k); out.update(alts)
    return list(out)

def years_meets_band(years: Optional[float], band: Optional[str]) -> bool:
    if band is None: return True
    if years is None: return False
    if band == "junior": return years < 3
    if band == "mid":    return 3 <= years < 6
    if band == "senior": return years >= 6
    return True

def evidence_lines(c: Dict[str,Any], jd: Dict[str,Any]) -> List[str]:
    lines = []
    years = c.get("total_experience_years")
    try: years = float(years) if years not in (None,"") else None
    except: years = None
    if years is not None:
        lines.append(f"{years:.1f}y total exp")
    # clouds
    if c.get("clouds"):
        try:
            if isinstance(c["clouds"], str):
                # serialized list from some pipelines → show raw
                lines.append(f"Cloud: {c['clouds']}")
            else:
                lines.append(f"Cloud: {'/'.join(c['clouds'])}")
        except Exception:
            pass
    # skill hits
    must = expand_equivalents(jd.get("must_skills", []))
    toks = set(normalize_tokens((c.get("skills_extracted") or "") + " " + (c.get("tools_and_technologies") or "")))
    hits = [m for m in must if m in toks][:2]
    if hits: lines.append("Has: " + ", ".join(hits))
    # quick summary
    summ = (c.get("semantic_summary") or c.get("employment_history") or "")[:140].strip()
    if summ: lines.append(summ + ("…" if len(summ)==140 else ""))
    return lines[:4] if lines else ["Good skill/role alignment"]

def milvus_search(cand_col: Collection, jd: Dict[str, Any]) -> List[Dict[str, Any]]:
    expr, partition = build_scalar_expr(jd)
    results: List[Dict[str,Any]] = []

    # Optional vector
    use_vec = USE_VECTOR and len(jd.get("raw","")) >= 20
    if use_vec:
        vec = embed_text_once(jd["raw"])
        if vec:
            params = {"metric_type":"COSINE","params":{"ef": int(os.getenv("EF_SEARCH", EF_SEARCH))}}
            try:
                res = cand_col.search(
                    data=[vec],
                    anns_field="summary_embedding",
                    param=params,
                    limit=max(TOP_K*6, 60),
                    expr=expr if expr else None,
                    output_fields=FIELDS,
                    partition_names=[partition] if partition else None,
                    consistency_level="Bounded"
                )
                for h in res[0]:
                    d = h.entity
                    row = {f: d.get(f, None) for f in FIELDS}
                    row["_distance"] = h.distance
                    results.append(row)
            except Exception as e:
                print(f"[warn] vector search failed: {e}", file=sys.stderr)

    # Scalar fallback / supplement
    if not results:
        try:
            rows = cand_col.query(
                expr=expr if expr else None,
                output_fields=FIELDS,
                limit=max(TOP_K*20, 200),
                consistency_level="Bounded"
            )
            for r in rows:
                r["_distance"] = None
                results.append(r)
        except Exception as e:
            print(f"[warn] scalar query failed: {e}", file=sys.stderr)

    # Dedup keep freshest
    uniq: Dict[str, Dict[str,Any]] = {}
    for r in results:
        cid = r.get("candidate_id") or ""
        lu  = r.get("last_updated") or ""
        prev = uniq.get(cid)
        if not prev or str(lu) > str(prev.get("last_updated","")):
            uniq[cid] = r
    return list(uniq.values())

def score_candidate(c: Dict[str,Any], jd: Dict[str,Any]) -> Tuple[float, Dict[str,Any]]:
    musts = 0; satisfied = 0; gaps = []

    # role family
    musts += 1
    if jd.get("role_family") and c.get("role_family") == jd["role_family"]:
        satisfied += 1
    else:
        gaps.append("role mismatch")

    # years band
    yb = jd.get("years_band")
    years = c.get("total_experience_years")
    try: years = float(years) if years not in (None,"") else None
    except: years = None
    musts += 1
    if years_meets_band(years, yb):
        satisfied += 1
    else:
        gaps.append(f"years band mismatch (has {years}, needs {yb})")

    # clouds
    req_clouds = jd.get("clouds", [])
    if req_clouds:
        musts += 1
        cand_clouds = c.get("clouds") or []
        ok = False
        try:
            # support both list and string-serialized
            if isinstance(cand_clouds, list):
                ok = any(cc in cand_clouds for cc in req_clouds)
            else:
                ok = any(cc in str(cand_clouds) for cc in req_clouds)
        except Exception:
            ok = False
        if ok: satisfied += 1
        else: gaps.append(f"clouds missing ({'/'.join(req_clouds)})")

    # must skills
    must_list = expand_equivalents(jd.get("must_skills", []))
    if must_list:
        musts += 1
        toks = set(normalize_tokens((c.get("skills_extracted") or "") + " " + (c.get("tools_and_technologies") or "")))
        if any(m in toks for m in must_list):
            satisfied += 1
        else:
            gaps.append("key skills missing")

    score = (satisfied / max(1, musts))

    # slight bump for more overlaps
    if must_list:
        toks = set(normalize_tokens((c.get("skills_extracted") or "") + " " + (c.get("tools_and_technologies") or "")))
        overlap = sum(1 for m in must_list if m in toks)
        score = min(1.0, score + 0.02*overlap)

    return score, {"gaps": gaps}

def bucketize(cands: List[Dict[str,Any]], jd: Dict[str,Any]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    urgent = jd.get("urgent", False)
    perfect_thr = 0.80
    near_thr    = 0.70 if urgent else 0.75

    scored = []
    for c in cands:
        s, meta = score_candidate(c, jd)
        c["_score"] = s
        c["_meta"]  = meta
        scored.append(c)

    scored.sort(key=lambda x: (x["_score"], str(x.get("last_updated",""))), reverse=True)
    perfect = [x for x in scored if x["_score"] >= perfect_thr]
    nearfit = [x for x in scored if near_thr <= x["_score"] < perfect_thr]

    out_p = perfect[:TOP_K]
    out_n = nearfit[: max(0, TOP_K - len(out_p))]
    return out_p, out_n

def nice_loc(c: Dict[str,Any]) -> str:
    parts = [c.get("location_city") or "", c.get("location_state") or "", c.get("location_country") or ""]
    parts = [p for p in parts if p]
    return ", ".join(parts) if parts else "—"

def format_slate(perfect: List[Dict[str,Any]], nearfit: List[Dict[str,Any]], jd: Dict[str,Any]) -> str:
    def rowline(c):
        titles = c.get('top_titles_mentioned','[]')
        if isinstance(titles, list): titles = str(titles)
        return f"- {c.get('name','(no name)')} | {titles} | {nice_loc(c)}"

    out = []
    out.append("Here’s a slate (Perfect vs Near-fit)  [style: details]\n")

    out.append("Perfect matches:")
    if not perfect:
        out.append("- (none yet)")
    else:
        for c in perfect:
            out.append(rowline(c))
            out.append("  why: " + "; ".join(evidence_lines(c, jd)))

    if nearfit:
        out.append("\nNear-fit:")
        for c in nearfit:
            gaps = c.get("_meta",{}).get("gaps", [])
            gap_str = (f" | gaps: {', '.join(gaps)}" if gaps else "")
            out.append(rowline(c))
            out.append("  why: " + "; ".join(evidence_lines(c, jd)) + gap_str)

    # brief narrative
    role = jd.get("role_family") or "the role"
    clouds = "/".join(jd.get("clouds", [])) if jd.get("clouds") else "n/a"
    band = jd.get("years_band") or "unspecified band"
    out.append("\nSummary narrative:")
    out.append(
        f"I found {len(perfect)+len(nearfit)} candidates aligned to **{role}** "
        f"(band={band}, clouds={clouds}). Showing top {min(TOP_K, len(perfect)+len(nearfit))}."
    )
    return "\n".join(out)

# -------------------- Client memory helpers --------------------
def upsert_client(mem_col: Collection, company: str, phone_or_contact: str) -> str:
    # Try fetch
    expr = f'client_company == "{company}" and phone == "{phone_or_contact}"'
    try:
        res = mem_col.query(expr=expr, output_fields=["client_id"], limit=1)
        if res: return res[0]["client_id"]
    except Exception:
        pass
    # Create
    import uuid
    cid = str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat(timespec="seconds")
    row = {
        "client_id": cid,
        "client_company": company,
        "phone": phone_or_contact,
        "email": "",
        "contact_name": "",
        "notes": "",
        "preferences": "",
        "last_queries": "",
        "created_at": now,
        "updated_at": now,
        "searches_count": 0,
        "desired_headcount": 0,
        "status": "Active",
        "cm_dummy_vec": [0.0, 0.0],
    }
    mem_col.insert([row])
    mem_col.flush()
    return cid

def read_memory(mem_col: Collection, cid: str) -> Dict[str,Any]:
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"',
                            output_fields=["preferences","last_queries","searches_count","updated_at"],
                            limit=1)
        return res[0] if res else {}
    except Exception:
        return {}

def append_memory(mem_col: Collection, cid: str, field: str, payload: str, max_len=7900):
    # Append with delimiter; trim from the left if too long
    prev = ""
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=[field], limit=1)
        prev = res[0].get(field, "") if res else ""
    except Exception:
        prev = ""
    blob = (prev + ("\n---\n" if prev else "") + payload)
    if len(blob) > max_len:
        blob = blob[-max_len:]
    try:
        mem_col.update(expr=f'client_id == "{cid}"', field_name=field, value=blob)
        mem_col.update(expr=f'client_id == "{cid}"', field_name="updated_at",
                       value=datetime.datetime.utcnow().isoformat(timespec="seconds"))
    except Exception:
        pass

def bump_search_count(mem_col: Collection, cid: str):
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=["searches_count"], limit=1)
        cur = res[0]["searches_count"] if res else 0
        mem_col.update(expr=f'client_id == "{cid}"', field_name="searches_count", value=int(cur)+1)
    except Exception:
        pass

# -------------------- Conversation orchestration --------------------
def greeting():
    return ("Doing well — thanks for asking!\n"
            "Kick off a search by sharing: role (backend/frontend/…), experience band (junior/mid/senior), "
            "cloud (AWS/GCP/Azure), and MUST skills (REST/gRPC/Kubernetes/Datadog, etc.).\n")

def memory_card(mem_col: Collection, cid: str) -> str:
    m = read_memory(mem_col, cid)
    if not m: return "(No previous memory yet.)"
    last = (m.get("last_queries") or "").split("\n---\n")[-1][:260] if m.get("last_queries") else ""
    pref = (m.get("preferences") or "").split("\n---\n")[-1][:260] if m.get("preferences") else ""
    bits = []
    if last: bits.append(f"Last query memo: {last}")
    if pref: bits.append(f"Last preference memo: {pref}")
    return "\n".join(bits) if bits else "(Memory exists but no recent notes.)"

def handle_memory_intent(mem_col: Collection, cid: str) -> str:
    return "Memory recap:\n" + memory_card(mem_col, cid)

def handle_unknown():
    return ("I can help you find candidates. Tell me the role, band, cloud(s), and MUST skills. "
            "Example: “Need a mid backend engineer on AWS with REST/gRPC and Kubernetes; Datadog preferred.”\n")

def handle_search(cand_col: Collection, mem_col: Collection, cid: str, user_text: str) -> str:
    jd = parse_jd(user_text)
    # Eligibility: need a role family to proceed
    if not jd.get("role_family"):
        return ("Got it. Which role family should I target (backend, frontend, devops, security, data, mlops, cloud, systems, mobile, blockchain)?\n")

    # Log choices (knobs + parsed JD) to memory
    knobs = {
        "TOP_K": TOP_K, "EF_SEARCH": EF_SEARCH, "USE_VECTOR": USE_VECTOR,
        "thresholds": {"perfect": 0.80, "near_fit": 0.70 if jd.get("urgent") else 0.75}
    }
    memo = json.dumps({"jd": jd, "knobs": knobs}, ensure_ascii=False)
    append_memory(mem_col, cid, "last_queries", memo)
    bump_search_count(mem_col, cid)

    # Search
    rows = milvus_search(cand_col, jd)
    if not rows:
        return ("Sorry — I didn’t find candidates with those constraints. Want me to relax something or draft a JD to source externally?\n")

    perfect, nearfit = bucketize(rows, jd)
    slate = format_slate(perfect, nearfit, jd)
    return slate + "\n\nNeed a short email summary or a CSV export?"

# -------------------- Main loop with session memory --------------------
def main():
    # Connect & collections
    try:
        connections.connect("default", uri=MILVUS_URI)
        try: db.using_database(MILVUS_DB)
        except Exception: pass
        if not utility.has_collection(CAND_COLL):
            print(f"[fatal] Milvus collection {CAND_COLL} not found.", file=sys.stderr); return 2
        if not utility.has_collection(MEM_COLL):
            print(f"[fatal] Milvus collection {MEM_COLL} not found.", file=sys.stderr); return 2
    except Exception as e:
        print(f"[fatal] Milvus connect failed: {e}", file=sys.stderr); return 2

    cand_col = Collection(CAND_COLL)
    mem_col  = Collection(MEM_COLL)

    # Session: identify client (company + phone/contact)
    print("RecruiterBrain ready.\n")
    client_company = input("Client Company? ").strip() or "UnknownCo"
    client_contact = input("Phone (or Contact Key)? ").strip() or "unknown"

    client_id = upsert_client(mem_col, client_company, client_contact)
    print("\nHello! (memory key established)\n")
    # Recall last memory snippet
    recap = memory_card(mem_col, client_id)
    if recap and "No previous" not in recap:
        print("Last time:\n" + recap + "\n")

    transcript: List[str] = []
    print("You can type your JD or ask for recap. Say 'bye' to end.\n")

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            user = "bye"
        if not user:
            continue
        transcript.append(f"USER: {user}")

        if user.lower() in {"bye","goodbye","exit","quit"}:
            # Persist conversation summary + embedding
            summary_text = "\n".join(transcript[-30:])  # last 30 turns max
            conv_embed = serialize_vec(embed_text_once(summary_text))
            summary_blob = json.dumps({
                "ended_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                "company": client_company,
                "contact_key": client_contact,
                "summary_excerpt": summary_text[-900:],  # keep small
                "embedding": conv_embed
            }, ensure_ascii=False)
            append_memory(mem_col, client_id, "preferences", summary_blob)
            print("bye!\n")
            break

        # Intent routing
        if is_greeting(user):
            resp = greeting()
        elif is_memory(user):
            resp = handle_memory_intent(mem_col, client_id)
        elif has_jd_signal(user):
            resp = handle_search(cand_col, mem_col, client_id, user)
        else:
            resp = handle_unknown()

        transcript.append(f"BOT: {resp}")
        print(resp)

    return 0

if __name__ == "__main__":
    sys.exit(main())
