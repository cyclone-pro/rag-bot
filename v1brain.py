#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recruiter Brain (terminal) — Milvus-safe + client history + 30 query recipes

What’s included
- IntentRouter: greeting / search / memory / recipe
- JDExtractor: role_family, years_band, clouds, location, must_skills, urgent
- Eligibility + Rank + Presenter
- Milvus search on new_candidate_pool (hybrid; efSearch configurable)
- Client memory:
   * On session start: upsert/find client in `client_memory` (by client_company + phone), show last recap
   * After EVERY turn: append a <300 char, 2–3 sentence human recap; keep only last 3
   * On session end: compact snapshot saved too
- QueryEngine: Milvus-safe implementations of your 30 “SQL-like” questions

Env:
  MILVUS_URI=http://34.135.232.156:19530
  MILVUS_DB=default
  CANDIDATE_COLL=new_candidate_pool
  RESUME_CHUNKS_COLL=resume_chunks
  JOB_DESC_COLL=job_descriptions
  CLIENT_MEMORY_COLL=client_memory
  TOP_K=8
  EF_SEARCH=128
  USE_VECTOR=1
  EMBED_MODEL=intfloat/e5-base-v2
  EMBED_DEVICE=cpu    (cpu|mps|cuda)
  EMBED_MAXLEN=256
"""

import os, re, sys, json, datetime, time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict

from pymilvus import connections, db, Collection, utility

# -------------------- Config --------------------
MILVUS_URI   = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_DB    = os.getenv("MILVUS_DB", "default")
CAND_COLL    = os.getenv("CANDIDATE_COLL", "new_candidate_pool")
RESUME_COLL  = os.getenv("RESUME_CHUNKS_COLL", "resume_chunks")
JD_COLL      = os.getenv("JOB_DESC_COLL", "job_descriptions")
MEM_COLL     = os.getenv("CLIENT_MEMORY_COLL", "client_memory")

TOP_K        = int(os.getenv("TOP_K", "8"))
EF_SEARCH    = int(os.getenv("EF_SEARCH", "128"))
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
    "candidate_id","name","email","phone","linkedin_url","top_titles_mentioned",
    "location_city","location_state","location_country",
    "total_experience_years","skills_extracted","tools_and_technologies","languages_spoken",
    "employment_history","semantic_summary","clouds","domains_of_expertise","certifications",
    "role_family","years_band","offer_status","availability_status","remote_preference",
    "source_channel","assigned_recruiter","hiring_manager_notes","interview_feedback",
    "genai_relevance_score","nlp_relevance_score","mlops_relevance_score",
    "computer_vision_relevance_score","medical_domain_score","cad_relevance_score","construction_domain_score",
    "last_updated_ts"  # INT64 epoch recommended
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
    return s[:max_len_chars]

# -------------------- Intent Router --------------------
GREETING_SET = {
    "hi","hello","hey","how are you","how’s it going","good morning","good afternoon","good evening",
    "thanks","thank you","cool","ok","okay","yo","sup"
}
MEMORY_HINTS = ["recap","summary","last slate","preferences","client memory","what did we send","remind me"]
RECIPE_HINTS = ["top 10","common skills","gaps","genai","nlp","chicago","backend","remote","medical","clinical",
                "cv","mlops","cad","resume_chunks","k8s","docker","pytorch","lead","certifications","source_channel",
                "language diversity","10 yrs","senior","domain overlap","assigned_recruiter","pass-through",
                "high interview scores","funnel","passive fits","security","phi","contract-ready","stale",
                "reasons","heatmap","vertex ai","bim","conversion","duplicates","near-duplicate","shortlist","avatar"]

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

def looks_like_recipe(txt: str) -> bool:
    return any(k in txt.lower() for k in RECIPE_HINTS)

def route_intent(user_text: str) -> str:
    if is_greeting(user_text): return "greeting"
    if is_memory(user_text):   return "memory"
    if looks_like_recipe(user_text): return "recipe"
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
    for c in [CAND_COLL, MEM_COLL]:
        if not utility.has_collection(c):
            raise RuntimeError(f"Collection {c} not found.")
    cand_col = Collection(CAND_COLL)
    mem_col  = Collection(MEM_COLL)
    resume_col = Collection(RESUME_COLL) if utility.has_collection(RESUME_COLL) else None
    jd_col = Collection(JD_COLL) if utility.has_collection(JD_COLL) else None
    return cand_col, mem_col, resume_col, jd_col

# -------------------- Expr helpers (Milvus-safe) --------------------
def field_is_array(col: Collection, name: str) -> bool:
    try:
        for f in col.schema.fields:
            if f.name == name:
                return getattr(f, "element_type", None) is not None
    except Exception:
        pass
    return False

def ts_days_ago(days: int) -> int:
    return int(time.time()) - days*86400

def expr_in(field: str, values: List[str]) -> str:
    vals = ",".join([f'"{v}"' for v in values])
    return f"{field} in [{vals}]"

def expr_and(*parts: str) -> str:
    parts = [p for p in parts if p]
    return " and ".join(parts) if parts else ""

# -------------------- Token utils --------------------
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

# -------------------- Client memory (last 3) --------------------
MAX_INTERACTIONS = 3
SEP = "\n---\n"

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds")

def clamp_300_chars(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:300]

def summarize_turn(user_text: str, bot_text: str) -> str:
    jd = parse_jd(user_text)
    parts = []
    if jd.get("role_family") or jd.get("years_band") or jd.get("clouds"):
        role = jd.get("role_family") or "a role"
        band = jd.get("years_band") or "any band"
        clouds = "/".join(jd.get("clouds", [])) or "any cloud"
        parts.append(f"You asked for {role} ({band}, {clouds}).")
    else:
        t = user_text.strip()
        if len(t) > 120: t = t[:120] + "…"
        parts.append(f'You said: "{t}".')
    resp = bot_text.strip().replace("\n", " ")
    resp = re.sub(r"\s+", " ", resp)
    if len(resp) > 160: resp = resp[:160] + "…"
    parts.append(f"I shared results and next steps: {resp}")
    parts.append("We’ll refine as needed.")
    return clamp_300_chars(" ".join(parts))

def parse_pref_blocks(pref_blob: str) -> List[Dict[str,Any]]:
    out = []
    if not pref_blob: return out
    for chunk in pref_blob.strip().split(SEP):
        chunk = chunk.strip()
        if not chunk: continue
        try:
            out.append(json.loads(chunk))
        except Exception:
            continue
    return out

def serialize_pref_blocks(blocks: List[Dict[str,Any]]) -> str:
    raw = SEP.join(json.dumps(b, ensure_ascii=False) for b in blocks)
    if len(raw) <= 8192:
        return raw
    i = 0
    while i < len(blocks) and len(raw) > 8192:
        i += 1
        raw = SEP.join(json.dumps(b, ensure_ascii=False) for b in blocks[i:])
    return raw[-8192:]

def append_interaction(mem_col: Collection, cid: str, user_text: str, bot_text: str):
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=["preferences"], limit=1)
        blob = (res[0].get("preferences") or "") if res else ""
    except Exception:
        blob = ""
    blocks = parse_pref_blocks(blob)
    entry = {
        "type": "turn",
        "ts_iso": now_iso(),
        "user": user_text,
        "bot": bot_text,
        "summary": summarize_turn(user_text, bot_text)
    }
    blocks.append(entry)
    turns = [b for b in blocks if b.get("type") == "turn"][-MAX_INTERACTIONS:]
    others = [b for b in blocks if b.get("type") != "turn"]
    blocks = others + turns
    new_blob = serialize_pref_blocks(blocks)
    try:
        mem_col.update(expr=f'client_id == "{cid}"', field_name="preferences", value=new_blob)
        mem_col.update(expr=f'client_id == "{cid}"', field_name="updated_at", value=now_iso())
    except Exception:
        pass

def append_session_end(mem_col: Collection, cid: str, transcript: List[str], company: str, contact: str):
    text = "\n".join(transcript[-30:])
    entry = {
        "type": "session_end",
        "ts_iso": now_iso(),
        "company": company,
        "contact_key": contact,
        "summary": clamp_300_chars(text[-900:])
    }
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=["preferences"], limit=1)
        blob = (res[0].get("preferences") or "") if res else ""
    except Exception:
        blob = ""
    blocks = parse_pref_blocks(blob)
    blocks.append(entry)
    new_blob = serialize_pref_blocks(blocks)
    try:
        mem_col.update(expr=f'client_id == "{cid}"', field_name="preferences", value=new_blob)
        mem_col.update(expr=f'client_id == "{cid}"', field_name="updated_at", value=now_iso())
    except Exception:
        pass

def last_interaction_summary(mem_col: Collection, client_company: str, phone: str) -> Optional[str]:
    try:
        res = mem_col.query(
            expr=f'client_company == "{client_company}" and phone == "{phone}"',
            output_fields=["preferences"],
            limit=1
        )
        if not res: return None
        blob = res[0].get("preferences") or ""
        blocks = parse_pref_blocks(blob)
        for b in reversed(blocks):
            if b.get("type") == "turn" and b.get("summary"):
                when = b.get("ts_iso","")
                return f"Last time ({when}): {b['summary']}"
    except Exception:
        return None
    return None

# -------------------- Search core --------------------
CLOUDS_IS_ARRAY = False  # set at runtime

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

    # clouds (only if clouds is ARRAY<VARCHAR>)
    clouds = jd.get("clouds", [])
    if clouds and CLOUDS_IS_ARRAY:
        clauses = [f'contains(clouds, "{c}")' for c in clouds]
        terms.append("(" + " or ".join(clauses) + ")")

    return (" and ".join(terms)) if terms else "", part

def milvus_hybrid_search(cand_col: Collection, jd_text: str, expr: Optional[str], limit: int, output_fields: List[str]) -> List[Dict[str,Any]]:
    results: List[Dict[str,Any]] = []
    vec = embed_text_once(jd_text) if (USE_VECTOR and len(jd_text or "") >= 20) else None
    if vec:
        params = {"metric_type":"COSINE","params":{"ef": int(os.getenv("EF_SEARCH", EF_SEARCH))}}
        try:
            res = cand_col.search(
                data=[vec], anns_field="summary_embedding", param=params,
                limit=limit, expr=expr if expr else None,
                output_fields=output_fields, consistency_level="Bounded"
            )
            for h in res[0]:
                d = h.entity
                row = {f: d.get(f, None) for f in output_fields}
                row["_cosine"] = h.distance
                results.append(row)
        except Exception as e:
            print(f"[warn] vector search failed: {e}", file=sys.stderr)

    if not results:
        try:
            rows = cand_col.query(
                expr=expr if expr else None, output_fields=output_fields,
                limit=max(limit*3, 200), consistency_level="Bounded"
            )
            for r in rows:
                r["_cosine"] = None
                results.append(r)
        except Exception as e:
            print(f"[warn] scalar query failed: {e}", file=sys.stderr)

    return results

def app_side_cloud_filter(rows: List[Dict[str,Any]], clouds: List[str]) -> List[Dict[str,Any]]:
    if not clouds or CLOUDS_IS_ARRAY:
        return rows
    keep = []
    for r in rows:
        raw = r.get("clouds")
        s = ",".join(raw) if isinstance(raw, list) else str(raw or "")
        if any(c in s for c in clouds):
            keep.append(r)
    return keep

# -------------------- Ranking / evidence --------------------
def years_meets_band(years: Optional[float], band: Optional[str]) -> bool:
    if band is None: return True
    if years is None: return False
    if band == "junior": return years < 3
    if band == "mid":    return 3 <= years < 6
    if band == "senior": return years >= 6
    if band in {"5-7","7-9"}:
        lo, hi = band.split("-"); y = float(years); return float(lo) <= y <= float(hi)
    return True

def evidence_lines(c: Dict[str,Any], jd: Dict[str,Any]) -> List[str]:
    lines = []
    yrs = c.get("total_experience_years")
    try: yrs = float(yrs) if yrs not in (None,"") else None
    except: yrs = None
    if yrs is not None: lines.append(f"{yrs:.1f}y total exp")
    if c.get("clouds"):
        try:
            if isinstance(c["clouds"], list): lines.append("Cloud: " + "/".join(c["clouds"]))
            else: lines.append(f"Cloud: {c['clouds']}")
        except: pass
    must = expand_equivalents(jd.get("must_skills", []))
    toks = set(normalize_tokens((c.get("skills_extracted") or "") + " " + (c.get("tools_and_technologies") or "")))
    hits = [m for m in must if m in toks][:2]
    if hits: lines.append("Has: " + ", ".join(hits))
    summ = (c.get("semantic_summary") or c.get("employment_history") or "")[:140].strip()
    if summ: lines.append(summ + ("…" if len(summ)==140 else ""))
    return lines[:4] if lines else ["Good skill/role alignment"]

def score_candidate(c: Dict[str,Any], jd: Dict[str,Any]) -> Tuple[float, Dict[str,Any]]:
    musts = 0; satisfied = 0; gaps = []

    musts += 1
    if jd.get("role_family") and c.get("role_family") == jd["role_family"]: satisfied += 1
    else: gaps.append("role mismatch")

    yb = jd.get("years_band")
    years = c.get("total_experience_years")
    try: years = float(years) if years not in (None,"") else None
    except: years = None
    musts += 1
    if years_meets_band(years, yb): satisfied += 1
    else: gaps.append(f"years band mismatch")

    req_clouds = jd.get("clouds", [])
    if req_clouds:
        musts += 1
        cand_clouds = c.get("clouds") or []
        ok = False
        try:
            if isinstance(cand_clouds, list): ok = any(cc in cand_clouds for cc in req_clouds)
            else: ok = any(cc in str(cand_clouds) for cc in req_clouds)
        except: ok = False
        if ok: satisfied += 1
        else: gaps.append(f"clouds missing")

    must_list = expand_equivalents(jd.get("must_skills", []))
    if must_list:
        musts += 1
        toks = set(normalize_tokens((c.get("skills_extracted") or "") + " " + (c.get("tools_and_technologies") or "")))
        if any(m in toks for m in must_list): satisfied += 1
        else: gaps.append("key skills missing")

    score = (satisfied / max(1, musts))
    if must_list:
        toks = set(normalize_tokens((c.get("skills_extracted") or "") + " " + (c.get("tools_and_technologies") or "")))
        overlap = sum(1 for m in must_list if m in toks)
        score = min(1.0, score + 0.02*overlap)

    meta = {"gaps": gaps}
    return score, meta

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

    scored.sort(key=lambda x: (x["_score"], x.get("_cosine") or 0.0, str(x.get("last_updated_ts",""))), reverse=True)
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
    if not perfect: out.append("- (none yet)")
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
    role = jd.get("role_family") or "the role"
    clouds = "/".join(jd.get("clouds", [])) if jd.get("clouds") else "n/a"
    band = jd.get("years_band") or "unspecified band"
    out.append("\nSummary narrative:")
    out.append(
        f"I found {len(perfect)+len(nearfit)} candidates aligned to **{role}** "
        f"(band={band}, clouds={clouds}). Showing top {min(TOP_K, len(perfect)+len(nearfit))}."
    )
    return "\n".join(out)

# -------------------- Query Engine (30 recipes) --------------------
class QueryEngine:
    def __init__(self, cand: Collection, resume: Optional[Collection]):
        self.cand = cand
        self.resume = resume

    # 1. Top 10 for a req + why
    def top_k_for_req(self, jd_text: str, jd_filters: Dict[str,Any], K: int = 10) -> List[Dict[str,Any]]:
        expr = []
        if jd_filters.get("role_family"):
            expr.append(f'role_family == "{jd_filters["role_family"]}"')
        if jd_filters.get("min_exp") is not None:
            expr.append(f'total_experience_years >= {float(jd_filters["min_exp"])}')
        # optional location filters can be exact string matches if present
        ex = " and ".join(expr) if expr else None
        rows = milvus_hybrid_search(self.cand, jd_text, ex, max(K*6, 60), FIELDS)
        rows = app_side_cloud_filter(rows, jd_filters.get("clouds", []))
        # rank: cosine + simple overlap heuristic (optional)
        for r in rows:
            r["_final"] = (0.5*(r.get("_cosine") or 0.0))
        rows.sort(key=lambda x: x.get("_final", 0.0), reverse=True)
        return rows[:K]

    # 2. 3 common skills + 3 gaps vs JD
    def common_and_gaps(self, rows: List[Dict[str,Any]], jd_required: List[str]) -> Dict[str,Any]:
        bag = Counter()
        for r in rows[:20]:
            toks = []
            if isinstance(r.get("tools_and_technologies"), list):
                toks += [t.lower() for t in r["tools_and_technologies"]]
            toks += normalize_tokens(r.get("skills_extracted",""))
            bag.update(toks)
        common = [w for w,_ in bag.most_common(10)]
        jd_req = [s.lower() for s in jd_required]
        gaps = [s for s in jd_req if s not in common][:3]
        return {"common": common[:3], "gaps": gaps[:3]}

    # 3. GenAI + NLP in Chicago or relocatable
    def genai_nlp_chicago(self) -> List[Dict[str,Any]]:
        ex = expr_and("genai_relevance_score >= 0.75",
                      "nlp_relevance_score >= 0.6",
                      '(location_city == "Chicago" or relocation_willingness == "yes")')
        return self.cand.query(expr=ex, output_fields=FIELDS, limit=200)

    # 4. Backend, 5–8 yrs, remote, start ≤ 2 weeks
    def backend_5to8_remote_quick(self) -> List[Dict[str,Any]]:
        ex = expr_and('role_family == "backend"',
                      'years_band in ["5-7","7-9"]',
                      'remote_preference == "remote"',
                      'availability_status in ["immediate","2 weeks"]')
        return self.cand.query(expr=ex, output_fields=FIELDS, limit=200)

    # 5. Medical + clinical NLP
    def medical_clinical_nlp(self) -> List[Dict[str,Any]]:
        ex = expr_and("medical_domain_score >= 0.6", "nlp_relevance_score >= 0.6")
        rows = self.cand.query(expr=ex, output_fields=FIELDS, limit=200)
        rows.sort(key=lambda r: (float(r.get("medical_domain_score") or 0) +
                                 float(r.get("nlp_relevance_score") or 0)), reverse=True)
        return rows

    # 6. CV + MLOps + cloud (AWS/GCP)
    def cv_mlops_cloud(self) -> List[Dict[str,Any]]:
        ex = expr_and("computer_vision_relevance_score >= 0.6",
                      "mlops_relevance_score >= 0.6")
        rows = self.cand.query(expr=ex, output_fields=FIELDS, limit=300)
        # app-side cloud filter if needed
        rows = app_side_cloud_filter(rows, ["AWS","GCP"])
        rows.sort(key=lambda r: (float(r.get("computer_vision_relevance_score") or 0) +
                                 float(r.get("mlops_relevance_score") or 0)), reverse=True)
        return rows

    # 7. Top CAD for construction; include portfolio_url
    def cad_construction(self) -> List[Dict[str,Any]]:
        ex = expr_and("cad_relevance_score >= 0.5", "construction_domain_score >= 0.5")
        rows = self.cand.query(expr=ex, output_fields=FIELDS + ["portfolio_url"], limit=200)
        rows.sort(key=lambda r: (0.6*float(r.get("cad_relevance_score") or 0) +
                                 0.4*float(r.get("construction_domain_score") or 0)), reverse=True)
        return rows

    # 8. Semantic matches from resume_chunks to JD
    def resume_chunks_to_jd(self, jd_text: str, K: int = 10) -> List[str]:
        if not self.resume:
            return []
        vec = embed_text_once(jd_text)
        if not vec:
            return []
        params = {"metric_type":"COSINE","params":{"ef": EF_SEARCH}}
        try:
            res = self.resume.search(
                data=[vec], anns_field="chunk_embedding", param=params,
                limit=1000, output_fields=["candidate_id"], consistency_level="Bounded"
            )
        except Exception as e:
            print(f"[warn] resume_chunks search failed: {e}", file=sys.stderr)
            return []
        cand_ids = []
        seen = set()
        for h in res[0]:
            cid = h.entity.get("candidate_id")
            if cid and cid not in seen:
                seen.add(cid); cand_ids.append(cid)
            if len(cand_ids) >= K: break
        return cand_ids

    # 9. K8s + Docker + PyTorch + leadership
    def k8s_docker_pytorch_lead(self) -> List[Dict[str,Any]]:
        rows = self.cand.query(expr=None, output_fields=FIELDS, limit=500)
        keep = []
        for r in rows:
            tools = r.get("tools_and_technologies", [])
            titles = r.get("top_titles_mentioned", [])
            tset = set([t.lower() for t in (tools or [])])
            if {"kubernetes","docker","pytorch"}.issubset(tset) and any(t in (titles or []) for t in ["Lead","Manager","Staff","Principal"]):
                keep.append(r)
        return keep

    # 10. Active certifications relevant to JD; flag expired/missing
    def certs_relevant(self, jd_certs: List[str]) -> List[Dict[str,Any]]:
        rows = self.cand.query(expr=None, output_fields=FIELDS, limit=500)
        jset = set([c.lower() for c in jd_certs])
        out = []
        for r in rows:
            certs = r.get("certifications", [])
            cset = set([str(c).lower() for c in (certs or [])])
            r["_certs_missing"] = list(jset - cset)
            out.append(r)
        return out

    # 11. Best source_channel this week
    def best_source_channel_week(self) -> Dict[str,Any]:
        ex = f"last_updated_ts >= {ts_days_ago(7)}"
        rows = self.cand.query(expr=ex, output_fields=FIELDS + ["summary_embedding"], limit=2000)
        agg = defaultdict(list)
        for r in rows:
            sc = r.get("source_channel") or "unknown"
            agg[sc].append(float(r.get("genai_relevance_score") or 0.0))
        summary = {k: {"count": len(v), "avg_genai_score": (sum(v)/len(v) if v else 0.0)} for k,v in agg.items()}
        return summary

    # 12. New candidates today/this week by role_family
    def new_by_rolefamily(self, days: int = 7) -> Dict[str,int]:
        ex = f"last_updated_ts >= {ts_days_ago(days)}"
        rows = self.cand.query(expr=ex, output_fields=["role_family"], limit=5000)
        cnt = Counter([r.get("role_family","unknown") for r in rows])
        return dict(cnt)

    # 13. Language diversity for top 50 (reuse list)
    def language_histogram(self, rows: List[Dict[str,Any]]) -> Dict[str,int]:
        langs = []
        for r in rows[:50]:
            ls = r.get("languages_spoken") or []
            langs += ls
        return dict(Counter(langs))

    # 14. Senior titles + ≥10 yrs
    def senior_10yrs(self) -> List[Dict[str,Any]]:
        ex = "total_experience_years >= 10"
        rows = self.cand.query(expr=ex, output_fields=FIELDS, limit=1000)
        keep = []
        for r in rows:
            titles = r.get("top_titles_mentioned", [])
            if any(t in (titles or []) for t in ["Staff","Lead","Principal","Architect","Manager"]):
                keep.append(r)
        return keep

    # 15. Domain overlap + evidence (app-side token match)
    def domain_overlap(self, jd_domain: str) -> List[Dict[str,Any]]:
        rows = self.cand.query(expr=None, output_fields=FIELDS, limit=1000)
        token = jd_domain.lower()
        keep = []
        for r in rows:
            doms = r.get("domains_of_expertise") or []
            if any(token in str(d).lower() for d in doms):
                keep.append(r)
        return keep

    # 16. Best assigned_recruiter by pass-through (requires external numeric data; stub grouping)
    def recruiter_pass_through(self) -> Dict[str,Any]:
        rows = self.cand.query(expr=None, output_fields=["assigned_recruiter","offer_status"], limit=5000)
        stages = defaultdict(lambda: {"phone":0, "manager":0})
        for r in rows:
            ar = r.get("assigned_recruiter") or "unknown"
            st = r.get("offer_status") or ""
            if st in {"phone_round","video_round"}: stages[ar]["phone"] += 1
            if st in {"manager_round","offered","accepted"}: stages[ar]["manager"] += 1
        out = {}
        for k,v in stages.items():
            denom = max(1, v["phone"])
            out[k] = {"phone": v["phone"], "manager_or_beyond": v["manager"], "pass_through": v["manager"]/denom}
        return out

    # 17. High interview scores not advanced + why (simple heuristic)
    def high_score_not_advanced(self) -> List[Dict[str,Any]]:
        rows = self.cand.query(expr='offer_status in ["applied","phone_round","video_round"]',
                               output_fields=FIELDS + ["interview_feedback_score"], limit=2000)
        keep = []
        for r in rows:
            sc = float(r.get("interview_feedback_score") or 0.0)
            if sc >= 0.7:
                keep.append(r)
        return keep

    # 18. Offer_status funnel for active JDs (approx)
    def funnel(self) -> Dict[str,int]:
        rows = self.cand.query(expr='offer_status in ["applied","phone_round","video_round","manager_round","offered","accepted","rejected"]',
                               output_fields=["offer_status"], limit=10000)
        return dict(Counter([r.get("offer_status","unknown") for r in rows]))

    # 19. Top 10 passive fits for new JD
    def passive_fits(self, jd_text: str, K: int = 10) -> List[Dict[str,Any]]:
        ex = 'offer_status not in ["applied"]'
        rows = milvus_hybrid_search(self.cand, jd_text, ex, max(K*8, 80), FIELDS)
        return rows[:K]

    # 20. Security/PHI compliance signals
    def phi_signals(self) -> List[Dict[str,Any]]:
        rows = self.cand.query(expr=None, output_fields=FIELDS, limit=2000)
        KEYS = {"hipaa","phi","soc2","hitrust","pii","gxp"}
        out = []
        for r in rows:
            blob = " ".join([str(r.get("hiring_manager_notes","")), str(r.get("domains_of_expertise","")),
                             str(r.get("tools_and_technologies",""))]).lower()
            if any(k in blob for k in KEYS): out.append(r)
        return out

    # 21. Contract-ready, availability & notes
    def contract_ready(self) -> List[Dict[str,Any]]:
        ex = expr_and('availability_status in ["immediate","2 weeks"]',
                      '(relocation_willingness == "yes" or remote_preference == "remote")')
        return self.cand.query(expr=ex, output_fields=FIELDS, limit=500)

    # 22. Stale but high match → outreach list
    def stale_high_match(self, jd_text: str, days: int = 180, K: int = 50) -> List[Dict[str,Any]]:
        ex = f"last_updated_ts < {ts_days_ago(days)}"
        rows = milvus_hybrid_search(self.cand, jd_text, ex, max(K*8, 200), FIELDS)
        rows.sort(key=lambda r: r.get("_cosine") or 0.0, reverse=True)
        return rows[:K]

    # 23. Top 5 reasons high/low in video interviews (light topic sketch)
    def reasons_video_interviews(self) -> Dict[str,List[str]]:
        rows = self.cand.query(expr='offer_status == "video_round"',
                               output_fields=["interview_feedback"], limit=1000)
        texts = [str(r.get("interview_feedback","")) for r in rows]
        # naive keyphrase tally
        KEYS = ["communication","system design","coding","culture","leadership","ownership",
                "requirements","mlops","cloud","kubernetes","python","java"]
        cnt = Counter()
        for t in texts:
            low = t.lower()
            for k in KEYS:
                if k in low: cnt[k]+=1
        common = [k for k,_ in cnt.most_common(10)]
        return {"top_reasons": common[:5], "other_reasons": common[5:10]}

    # 24. Role_family → skills heatmap
    def heatmap(self) -> Dict[str,Dict[str,int]]:
        rows = self.cand.query(expr=None, output_fields=["role_family","tools_and_technologies","genai_relevance_score","mlops_relevance_score"], limit=5000)
        grid: Dict[str,Counter] = defaultdict(Counter)
        for r in rows:
            rf = r.get("role_family","unknown") or "unknown"
            for t in (r.get("tools_and_technologies") or []):
                grid[rf][t] += 1
        # convert to plain dicts
        return {rf: dict(grid[rf].most_common(10)) for rf in grid}

    # 25. GenAI Platform Eng on GCP + MLOps + Python + Vertex AI
    def genai_platform_gcp(self) -> List[Dict[str,Any]]:
        ex = 'mlops_relevance_score >= 0.6'
        rows = self.cand.query(expr=ex, output_fields=FIELDS, limit=2000)
        rows = app_side_cloud_filter(rows, ["GCP"])
        keep = []
        for r in rows:
            tools = set([t.lower() for t in (r.get("tools_and_technologies") or [])])
            if "python" in tools and ("vertex ai" in tools or "vertexai" in tools):
                keep.append(r)
        return keep

    # 26. CAD + BIM, construction, onsite TX
    def cad_bim_tx(self) -> List[Dict[str,Any]]:
        ex = expr_and("cad_relevance_score >= 0.5",
                      "construction_domain_score >= 0.6",
                      '(location_state == "TX" or relocation_willingness == "yes")',
                      'remote_preference != "remote"')
        rows = self.cand.query(expr=ex, output_fields=FIELDS, limit=1000)
        keep = []
        for r in rows:
            if "BIM" in (r.get("tools_and_technologies") or []):
                keep.append(r)
        return keep

    # 27. Segments with best conversion (90d) — requires segment tags
    def segments_conversion(self) -> Dict[str,Any]:
        ex = f"last_updated_ts >= {ts_days_ago(90)}"
        rows = self.cand.query(expr=ex, output_fields=["segment_tag","offer_status"], limit=10000)
        agg = defaultdict(lambda: {"phone":0,"manager":0})
        for r in rows:
            seg = r.get("segment_tag") or "unknown"
            st  = r.get("offer_status") or ""
            if st in {"phone_round","video_round"}: agg[seg]["phone"] += 1
            if st in {"manager_round","offered","accepted"}: agg[seg]["manager"] += 1
        out = {}
        for k,v in agg.items():
            denom = max(1, v["phone"])
            out[k] = {"phone": v["phone"], "manager": v["manager"], "conversion": v["manager"]/denom}
        return out

    # 28. Duplicate / near-duplicate profiles
    def duplicates(self) -> Dict[str,List[str]]:
        rows = self.cand.query(expr=None, output_fields=["candidate_id","email","phone","linkedin_url"], limit=20000)
        by_key = defaultdict(list)
        for r in rows:
            for k in ["email","phone","linkedin_url"]:
                val = (r.get(k) or "").strip().lower()
                if val: by_key[(k,val)].append(r.get("candidate_id"))
        return {f"{k}:{v}": ids for (k,v),ids in by_key.items() if len(ids)>1}

    # 29. Shortlist packet (top 8) for REQ-123 — pass in rows
    def shortlist_packet(self, rows: List[Dict[str,Any]], K: int = 8) -> List[Dict[str,Any]]:
        pkt = []
        for r in rows[:K]:
            pkt.append({
                "name": r.get("name"), "email": r.get("email"), "phone": r.get("phone"),
                "linkedin_url": r.get("linkedin_url"),
                "total_experience_years": r.get("total_experience_years"),
                "skills_extracted": r.get("skills_extracted"),
                "tools_and_technologies": r.get("tools_and_technologies"),
                "domains_of_expertise": r.get("domains_of_expertise"),
                "certifications": r.get("certifications"),
                "cosine_score": r.get("_cosine"),
                "gaps_vs_JD": r.get("_meta",{}).get("gaps", [])
            })
        return pkt

    # 30. Avatar question script to hit top 5 gaps (returns prompt text)
    def avatar_questions(self, gaps: List[str]) -> str:
        gaps = [g for g in gaps if g][:5]
        if not gaps: return "No gaps detected."
        return ("Generate behavior-based interview questions to probe: " +
                ", ".join(gaps) +
                ". Start easy and progress to hard. Include at least one follow-up per gap.")

# -------------------- Client memory helpers (upsert) --------------------
def upsert_client(mem_col: Collection, company: str, phone_or_contact: str) -> str:
    expr = f'client_company == "{company}" and phone == "{phone_or_contact}"'
    try:
        res = mem_col.query(expr=expr, output_fields=["client_id"], limit=1)
        if res: return res[0]["client_id"]
    except Exception:
        pass
    import uuid
    cid = str(uuid.uuid4())
    now = now_iso()
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
    mem_col.insert([row]); mem_col.flush()
    return cid

def append_memory(mem_col: Collection, cid: str, field: str, payload: str, max_len=7900):
    prev = ""
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=[field], limit=1)
        prev = res[0].get(field, "") if res else ""
    except Exception:
        prev = ""
    blob = (prev + (SEP if prev else "") + payload)
    if len(blob) > max_len:
        blob = blob[-max_len:]
    try:
        mem_col.update(expr=f'client_id == "{cid}"', field_name=field, value=blob)
        mem_col.update(expr=f'client_id == "{cid}"', field_name="updated_at", value=now_iso())
    except Exception:
        pass

def bump_search_count(mem_col: Collection, cid: str):
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=["searches_count"], limit=1)
        cur = res[0]["searches_count"] if res else 0
        mem_col.update(expr=f'client_id == "{cid}"', field_name="searches_count", value=int(cur)+1)
    except Exception:
        pass

# -------------------- Orchestration --------------------
def greeting():
    return ("Doing well — thanks for asking!\n"
            "Kick off a search by sharing: role (backend/frontend/…), experience band (junior/mid/senior), "
            "cloud (AWS/GCP/Azure), and MUST skills (REST/gRPC/Kubernetes/Datadog, etc.).\n")

def handle_unknown():
    return ("I can help you find candidates. Tell me the role, band, cloud(s), and MUST skills. "
            "Example: “Need a mid backend engineer on AWS with REST/gRPC and Kubernetes; Datadog preferred.”\n")

def handle_search(cand_col: Collection, mem_col: Collection, cid: str, user_text: str) -> str:
    jd = parse_jd(user_text)
    if not jd.get("role_family"):
        return ("Got it. Which role family should I target (backend, frontend, devops, security, data, mlops, cloud, systems, mobile, blockchain)?\n")
    knobs = {
        "TOP_K": TOP_K, "EF_SEARCH": EF_SEARCH, "USE_VECTOR": USE_VECTOR,
        "thresholds": {"perfect": 0.80, "near_fit": 0.70 if jd.get("urgent") else 0.75}
    }
    memo = json.dumps({"jd": jd, "knobs": knobs}, ensure_ascii=False)
    append_memory(mem_col, cid, "last_queries", memo)
    bump_search_count(mem_col, cid)

    expr, partition = build_scalar_expr(jd)
    rows = milvus_hybrid_search(cand_col, jd["raw"], expr, max(TOP_K*8, 80), FIELDS)
    rows = app_side_cloud_filter(rows, jd.get("clouds", []))
    if not rows:
        return ("Sorry — I didn’t find candidates with those constraints. Want me to relax something or draft a JD to source externally?\n")
    perfect, nearfit = bucketize(rows, jd)
    slate = format_slate(perfect, nearfit, jd)
    return slate + "\n\nNeed a short email summary or a CSV export?"

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

    cand_col, mem_col, resume_col, jd_col = connect_milvus()

    # detect array schema once
    global CLOUDS_IS_ARRAY
    CLOUDS_IS_ARRAY = field_is_array(cand_col, "clouds")

    engine = QueryEngine(cand_col, resume_col)

    # Session: identify client (company + phone/contact)
    print("RecruiterBrain ready.\n")
    client_company = input("Client Company? ").strip() or "UnknownCo"
    client_contact = input("Phone (or Contact Key)? ").strip() or "unknown"

    client_id = upsert_client(mem_col, client_company, client_contact)
    print("\nHello! (memory key established)\n")

    recap = last_interaction_summary(mem_col, client_company, client_contact)
    if recap:
        print("Last time:\n" + recap + "\n")

    transcript: List[str] = []
    print("Type a JD, or say things like 'top 10 for this req', 'language diversity', 'backend 5-8 remote', etc. Say 'bye' to end.\n")

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            user = "bye"
        if not user:
            continue
        transcript.append(f"USER: {user}")

        if user.lower() in {"bye","goodbye","exit","quit"}:
            append_session_end(mem_col, client_id, transcript, client_company, client_contact)
            print("bye!\n"); break

        intent = route_intent(user)
        if intent == "greeting":
            resp = greeting()

        elif intent == "memory":
            recap2 = last_interaction_summary(mem_col, client_company, client_contact)
            resp = (recap2 or "No previous memory yet.") + "\n"

        elif intent == "recipe":
            u = user.lower()
            if "top 10" in u:
                jd_text = input("Paste JD text to match: ").strip()
                rows = engine.top_k_for_req(jd_text, {"role_family": infer_role_family(jd_text),
                                                      "clouds": extract_clouds(jd_text),
                                                      "min_exp": None}, K=10)
                pkt = engine.shortlist_packet(rows, K=10)
                resp = f"Prepared {len(pkt)} profiles for your req."
            elif "common skills" in u or "gaps" in u:
                jd_text = input("Paste JD text to match: ").strip()
                rows = engine.top_k_for_req(jd_text, {"role_family": infer_role_family(jd_text)}, K=20)
                ans = engine.common_and_gaps(rows, extract_must_skills(jd_text))
                resp = f"Common: {', '.join(ans['common'])}. Gaps: {', '.join(ans['gaps'])}."
            elif "genai" in u and "nlp" in u:
                resp = f"Found {len(engine.genai_nlp_chicago())} GenAI+NLP candidates in Chicago/relocatable."
            elif "backend" in u and "remote" in u:
                resp = f"Found {len(engine.backend_5to8_remote_quick())} backend candidates 5–8y, remote, quick start."
            elif "medical" in u or "clinical" in u:
                resp = f"Found {len(engine.medical_clinical_nlp())} medical+clinical NLP candidates."
            elif "cv" in u and "mlops" in u:
                resp = f"Found {len(engine.cv_mlops_cloud())} CV+MLOps candidates on AWS/GCP."
            elif "cad" in u and "construction" in u and "portfolio" in u:
                resp = f"Found {len(engine.cad_construction())} CAD+construction candidates with portfolio."
            elif "resume_chunks" in u:
                jd_text = input("Paste JD text to match chunks: ").strip()
                resp = f"Top chunk candidate_ids: {engine.resume_chunks_to_jd(jd_text)}"
            elif "k8s" in u and "docker" in u and "pytorch" in u:
                resp = f"Found {len(engine.k8s_docker_pytorch_lead())} with K8s+Docker+PyTorch and leadership."
            elif "certifications" in u:
                jd_certs = input("Required certs (comma): ").strip().split(",")
                resp = f"Checked certs for {len(engine.certs_relevant([c.strip() for c in jd_certs]))} candidates."
            elif "source_channel" in u and "week" in u:
                resp = json.dumps(engine.best_source_channel_week(), ensure_ascii=False)
            elif "new candidates" in u:
                resp = json.dumps(engine.new_by_rolefamily(7), ensure_ascii=False)
            elif "language" in u and "diversity" in u:
                jd_text = input("Paste JD text for top-50 pool: ").strip()
                rows = engine.top_k_for_req(jd_text, {"role_family": infer_role_family(jd_text)}, K=50)
                resp = json.dumps(engine.language_histogram(rows), ensure_ascii=False)
            elif "10 yrs" in u or "senior" in u:
                resp = f"Found {len(engine.senior_10yrs())} senior candidates (>=10y) with senior titles."
            elif "domain overlap" in u:
                dom = input("JD domain string: ").strip()
                resp = f"Found {len(engine.domain_overlap(dom))} candidates overlapping domain tokens."
            elif "assigned_recruiter" in u or "pass-through" in u:
                resp = json.dumps(engine.recruiter_pass_through(), ensure_ascii=False)
            elif "high interview" in u and "not advanced" in u:
                resp = f"Found {len(engine.high_score_not_advanced())} high-score candidates not advanced."
            elif "funnel" in u:
                resp = json.dumps(engine.funnel(), ensure_ascii=False)
            elif "passive fits" in u:
                jd_text = input("Paste NEW JD text: ").strip()
                resp = f"Top passive fits: {len(engine.passive_fits(jd_text, 10))}."
            elif "security" in u or "phi" in u:
                resp = f"Found {len(engine.phi_signals())} candidates with PHI/security keywords."
            elif "contract-ready" in u:
                resp = f"Found {len(engine.contract_ready())} contract-ready candidates."
            elif "stale" in u and "outreach" in u:
                jd_text = input("Paste JD text: ").strip()
                resp = f"Outreach list prepared: {len(engine.stale_high_match(jd_text))} candidates."
            elif "reasons" in u and "video" in u:
                resp = json.dumps(engine.reasons_video_interviews(), ensure_ascii=False)
            elif "heatmap" in u:
                resp = json.dumps(engine.heatmap(), ensure_ascii=False)
            elif "vertex" in u and "gcp" in u:
                resp = f"Found {len(engine.genai_platform_gcp())} GenAI Platform Eng profiles on GCP with Vertex AI."
            elif "bim" in u and "tx" in u:
                resp = f"Found {len(engine.cad_bim_tx())} CAD+BIM in TX (onsite/hybrid)."
            elif "segments" in u and "conversion" in u:
                resp = json.dumps(engine.segments_conversion(), ensure_ascii=False)
            elif "duplicate" in u:
                resp = json.dumps(engine.duplicates(), ensure_ascii=False)
            elif "shortlist" in u:
                jd_text = input("Paste JD text to shortlist: ").strip()
                rows = engine.top_k_for_req(jd_text, {"role_family": infer_role_family(jd_text)}, K=8)
                pkt = engine.shortlist_packet(rows, K=8)
                resp = json.dumps(pkt, ensure_ascii=False)
            elif "avatar" in u and "gaps" in u:
                gaps = input("Provide comma-separated gaps: ").strip().split(",")
                resp = engine.avatar_questions([g.strip() for g in gaps])
            else:
                resp = "Recipe not recognized. Try: top 10, common skills, backend 5-8 remote, medical clinical, cv mlops cloud, language diversity, funnel, duplicates, shortlist, avatar gaps."

        elif intent == "search":
            resp = handle_search(cand_col, mem_col, client_id, user)

        else:
            resp = handle_unknown()

        # store per turn
        append_interaction(mem_col, client_id, user, resp)
        transcript.append(f"BOT: {resp}")
        print(resp)

    return 0

if __name__ == "__main__":
    sys.exit(main())
