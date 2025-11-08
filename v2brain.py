#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, json, datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import connections, db, Collection, utility

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
CLOUD_EQUIV   = {"aws":"AWS","gcp":"GCP","azure":"AZURE"}

FIELDS = [
    "candidate_id","name","top_titles_mentioned",
    "location_city","location_state","location_country",
    "total_experience_years","skills_extracted","tools_and_technologies",
    "employment_history","semantic_summary","clouds","role_family","years_band",
    "last_updated","source_channel","assigned_recruiter",
    # optional scores if you have them; keep harmless if missing
    "genai_relevance_score","mlops_relevance_score","nlp_relevance_score",
    "medical_domain_score","computer_vision_relevance_score",
    "availability_status","remote_preference","relocation_willingness",
    "offer_status","languages_spoken","domains_of_expertise","certifications"
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

# -------------------- Intent helpers --------------------
GREETING_SET = {"hi","hello","hey","how are you","how’s it going","good morning","good afternoon","good evening","thanks","thank you","cool","ok","okay","yo","sup"}
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
    if any(k in t for k in ["aws","gcp","azure","django","spring","springboot","kubernetes","datadog","prometheus","grafana","grpc","rest","sql","postgres","mysql","sql server"]): return True
    if any(k in t for k in ROLE_FAMILIES): return True
    return False

# -------------------- JD parse (lightweight) --------------------
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
def infer_role_families(text: str) -> List[str]:
    """Return all role families hinted by the JD text, ranked by signal strength."""
    t = text.lower()
    scores: Dict[str, int] = {}

    # direct word hits get a small baseline score so they outrank weak hints
    for fam in ROLE_FAMILIES:
        if re.search(rf"\b{fam}\b", t):
            scores[fam] = max(scores.get(fam, 0), len(fam))

    # hint overlaps add to the scoreboard; keep any family with >=1 hit
    for fam, hints in ROLE_HINTS.items():
        hit = sum(1 for h in hints if h in t)
        if hit >= 1:
            scores[fam] = max(scores.get(fam, 0), hit)

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda item: (-item[1], ROLE_FAMILIES.index(item[0])))
    return [fam for fam, _ in ranked]
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
    t = text.lower(); out=[]
    for k, v in CLOUD_EQUIV.items():
        if k in t: out.append(v)
    return out[:3]
def extract_must_skills(text: str) -> List[str]:
    t = text.lower()
    TOKS = ["django","drf","rest","grpc","aws","gcp","azure","kubernetes","eks","gke","aks","docker",
            "datadog","prometheus","grafana","postgres","mysql","sql server","kafka","spark","airflow",
            "spring","springboot","fastapi","flask",".net","golang","rust","terraform","ansible",
            "jenkins","github actions","gh actions"]
    out=[tok for tok in TOKS if tok in t]
    if "gh actions" in out and "github actions" not in out: out.append("github actions")
    # normalize to tokens we’ll match in post-filter
    return sorted(set(out))

def parse_jd(user_text: str) -> Dict[str, Any]:
    families = infer_role_families(user_text)
    return {
        "raw": user_text,
        "role_families": families,
        "role_family": families[0] if families else None,
        "years_band": years_band_from_text(user_text),
        "clouds": extract_clouds(user_text),
        "must_skills": extract_must_skills(user_text),
        "urgent": ("urgent" in user_text.lower()),
    }
COUNT_PAT = re.compile(r"\b(how many|total)\b.*\b(candidates|profiles)\b|^\s*(count|total)\s*$", re.I)

def get_candidate_count(cand_col: Collection) -> int:
    # Try modern property, then fall back to stats JSON
    try:
        return int(getattr(cand_col, "num_entities"))
    except Exception:
        try:
            stats = utility.get_collection_stats(cand_col.name)
            # stats may be a JSON string or dict depending on client
            if isinstance(stats, str):
                stats = json.loads(stats)
            return int(stats.get("row_count") or stats.get("rowCount") or 0)
        except Exception:
            return 0

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

# -------------------- App-side filtering (generic) --------------------
def norm_tokens(val) -> set:
    if val is None: return set()
    if isinstance(val, list):
        s = " ".join(map(str, val))
    else:
        s = str(val)
    s = re.sub(r"[^a-z0-9\+\.\#\- ]+"," ", s.lower())
    return set(s.split())

def generic_postfilter(rows: List[Dict[str,Any]], need=None, any_of=None,
                       titles_any=None, domains_any=None, clouds_any=None) -> List[Dict[str,Any]]:
    need = [w.lower() for w in (need or [])]
    any_of = [w.lower() for w in (any_of or [])]
    titles_any = [w.lower() for w in (titles_any or [])]
    domains_any = [w.lower() for w in (domains_any or [])]
    clouds_any = [w.lower() for w in (clouds_any or [])]

    out=[]
    for r in rows:
        skills = norm_tokens(r.get("skills_extracted")) | norm_tokens(r.get("tools_and_technologies"))
        titles = norm_tokens(r.get("top_titles_mentioned"))
        domains= norm_tokens(r.get("domains_of_expertise"))
        clouds = norm_tokens(r.get("clouds"))
        if need and not all(w in skills for w in need): continue
        if any_of and not any(w in skills for w in any_of): continue
        if titles_any and not any(w in titles for w in titles_any): continue
        if domains_any and not any(w in domains for w in domains_any): continue
        if clouds_any and not any(w in clouds for w in clouds_any): continue
        out.append(r)
    return out

# -------------------- Unified search (Milvus-safe) --------------------
def fetch_from_milvus(cand_col: Collection, expr: Optional[str], limit=5000, fields=None) -> List[Dict[str,Any]]:
    try:
        rows = cand_col.query(expr=expr if expr else None, output_fields=fields or FIELDS, limit=limit, consistency_level="Bounded")
        return rows
    except Exception as e:
        print(f"[warn] Milvus query failed: {e}", file=sys.stderr)
        return []

def unified_search(cand_col: Collection, *, jd_text=None, expr=None,
                   need=None, any_of=None, titles_any=None, domains_any=None, clouds_any=None,
                   top_k=50) -> List[Dict[str,Any]]:
    # 1) SAFE expr (only scalars you know exist & are comparable)
    base = fetch_from_milvus(cand_col, expr, limit=5000, fields=FIELDS)
    if not base:
        return []

    # 2) generic post-filter for arrays/fuzzy text
    filt = generic_postfilter(base, need=need, any_of=any_of,
                              titles_any=titles_any, domains_any=domains_any, clouds_any=clouds_any)

    # 3) optional vector re-rank (client-side top_k cut; server vector already ran if you used cand_col.search)
    # Here we keep it simple; if you want server-side ANN rerank, do a separate cand_col.search without expr,
    # then intersect IDs. For now, just cap.
    return filt[:top_k]

# -------------------- Ranking & presentation --------------------
def years_meets_band(years: Optional[float], band: Optional[str]) -> bool:
    if band is None: return True
    if years is None: return False
    if band == "junior": return years < 3
    if band == "mid":    return 3 <= years < 6
    if band == "senior": return years >= 6
    return True

def evidence_lines(c: Dict[str,Any], jd: Dict[str,Any]) -> List[str]:
    lines=[]
    years=c.get("total_experience_years")
    try: years=float(years) if years not in (None,"") else None
    except: years=None
    if years is not None: lines.append(f"{years:.1f}y total exp")
    clouds=c.get("clouds")
    if clouds:
        try:
            if isinstance(clouds, list): lines.append(f"Cloud: {'/'.join(clouds)}")
            else: lines.append(f"Cloud: {clouds}")
        except: pass
    must = [m.lower() for m in (jd.get("must_skills") or [])]
    toks = norm_tokens((c.get("skills_extracted") or "")) | norm_tokens((c.get("tools_and_technologies") or ""))
    hits = [m for m in must if m in toks][:2]
    if hits: lines.append("Has: " + ", ".join(hits))
    summ = (c.get("semantic_summary") or c.get("employment_history") or "")[:140].strip()
    if summ: lines.append(summ + ("…" if len(summ)==140 else ""))
    return lines[:4] if lines else ["Good alignment"]

def nice_loc(c: Dict[str,Any]) -> str:
    parts=[c.get("location_city") or "", c.get("location_state") or "", c.get("location_country") or ""]
    parts=[p for p in parts if p]
    return ", ".join(parts) if parts else "—"

def format_slate(rows: List[Dict[str,Any]], jd: Dict[str,Any]) -> str:
    out=["Here’s a slate (top matches)\n"]
    if not rows:
        out.append("- (none yet)")
    else:
        for r in rows:
            titles = r.get('top_titles_mentioned','[]')
            if isinstance(titles, list): titles = str(titles)
            out.append(f"- {r.get('name','(no name)')} | {titles} | {nice_loc(r)}")
            out.append("  why: " + "; ".join(evidence_lines(r, jd)))
    families = jd.get("role_families") or []
    role = "/".join(families) if families else (jd.get("role_family") or "the role")
    clouds = "/".join(jd.get("clouds", [])) if jd.get("clouds") else "n/a"
    band = jd.get("years_band") or "unspecified band"
    out.append(f"\nSummary: {len(rows)} candidates aligned to {role} (band={band}, clouds={clouds}). Showing top {min(TOP_K, len(rows))}.")
    return "\n".join(out)

# -------------------- Client memory helpers (keep last 3) --------------------
def upsert_client(mem_col: Collection, company: str, phone_or_contact: str) -> str:
    expr = f'client_company == "{company}" and phone == "{phone_or_contact}"'
    try:
        res = mem_col.query(expr=expr, output_fields=["client_id"], limit=1)
        if res: return res[0]["client_id"]
    except Exception:
        pass
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
        "preferences": "",   # we’ll store last 3 convo recaps here as JSON lines
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

def _short_recap(user_text:str, bot_text:str) -> str:
    # 2–3 sentences, <300 chars, plain English with punctuation.
    ut = re.sub(r"\s+"," ", user_text).strip()
    bt = re.sub(r"\s+"," ", bot_text).strip()
    msg = f"You asked: {ut}. I responded with a brief slate and next steps. Outcome: continued search setup."
    if len(msg) > 290:
        msg = msg[:287] + "..."
    return msg

def append_interaction(mem_col: Collection, cid: str, user_text: str, bot_text: str):
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=["preferences"], limit=1)
        prev = res[0].get("preferences","") if res else ""
    except Exception:
        prev = ""
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    entry = {"ts": ts, "summary": _short_recap(user_text, bot_text)}
    # store as JSONL (Milvus VARCHAR-friendly)
    lines = [ln for ln in prev.split("\n") if ln.strip()]
    lines.append(json.dumps(entry, ensure_ascii=False))
    if len(lines) > 3:  # keep last 3
        lines = lines[-3:]
    blob = "\n".join(lines)
    try:
        mem_col.update(expr=f'client_id == "{cid}"', field_name="preferences", value=blob)
        mem_col.update(expr=f'client_id == "{cid}"', field_name="updated_at", value=datetime.datetime.utcnow().isoformat(timespec="seconds"))
    except Exception as e:
        print(f"[warn] memory update failed: {e}", file=sys.stderr)

def latest_memory_snippet(mem_col: Collection, cid: str) -> str:
    try:
        res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=["preferences"], limit=1)
        blob = res[0].get("preferences","") if res else ""
    except Exception:
        blob=""
    if not blob.strip():
        return "(No previous memory yet.)"
    try:
        last = json.loads(blob.strip().split("\n")[-1])
        return f'Last time ({last.get("ts","")}): {last.get("summary","")}'
    except Exception:
        return "(Memory exists but could not parse.)"

def _get_mem_row(mem_col: Collection, cid: str) -> Dict[str, Any]:
    res = mem_col.query(expr=f'client_id == "{cid}"', output_fields=["*"], limit=1) or [{}]
    return res[0] if res else {}

def _write_mem_row(mem_col: Collection, row: Dict[str, Any]):
    # Prefer upsert if your client supports it, else delete + insert
    try:
        upsert = getattr(mem_col, "upsert", None)
        if callable(upsert):
            mem_col.upsert([row])
            mem_col.flush()
            return
    except Exception:
        pass
    # Fallback: delete then insert
    try:
        mem_col.delete(expr=f'client_id == "{row["client_id"]}"')
    except Exception:
        pass
    mem_col.insert([row])
    mem_col.flush()

def append_interaction(mem_col: Collection, cid: str, user_text: str, bot_text: str):
    prev = _get_mem_row(mem_col, cid)
    blob = (prev.get("preferences") or "")
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    entry = {"ts": ts, "summary": _short_recap(user_text, bot_text)}

    lines = [ln for ln in blob.split("\n") if ln.strip()]
    lines.append(json.dumps(entry, ensure_ascii=False))
    if len(lines) > 3:
        lines = lines[-3:]
    new_blob = "\n".join(lines)

    # rebuild the full row, preserving all existing fields
    updated = dict(prev)
    updated["client_id"] = cid
    updated["preferences"] = new_blob
    updated["updated_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds")
    # Ensure required fields exist (Milvus schema strict about missing fields)
    for k in ["client_company","phone","email","contact_name","notes","last_queries",
              "created_at","searches_count","desired_headcount","status","cm_dummy_vec"]:
        updated.setdefault(k, "" if k not in ["searches_count","desired_headcount","cm_dummy_vec"] else (0 if k != "cm_dummy_vec" else [0.0,0.0]))
    _write_mem_row(mem_col, updated)

def latest_memory_snippet(mem_col: Collection, cid: str) -> str:
    blob = (_get_mem_row(mem_col, cid).get("preferences") or "").strip()
    if not blob:
        return "(No previous memory yet.)"
    try:
        last = json.loads(blob.split("\n")[-1])
        return f'Last time ({last.get("ts","")}): {last.get("summary","")}'
    except Exception:
        return "(Memory exists but could not parse.)"


# -------------------- Smalltalk reply --------------------
def smalltalk_reply(user: str) -> str:
    u = user.strip()
    if len(u) > 200: u = u[:200] + "…"
    return (f"You: {u}  Nice to hear from you. "
            "Tell me the role, experience band, clouds, and must-have skills, and I’ll pull a slate. "
            "You can also say things like 'top 10 for this req' or 'language diversity for top 50'.")

# -------------------- Main orchestration --------------------
def main():
    # Connect
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

    print("RecruiterBrain ready.\n")
    client_company = input("Client Company? ").strip() or "UnknownCo"
    client_contact = input("Phone (or Contact Key)? ").strip() or "unknown"
    client_id = upsert_client(mem_col, client_company, client_contact)

    print("\nHello! (memory key established)\n")
    recap = latest_memory_snippet(mem_col, client_id)
    print("Last time:\n" + recap + "\n")

    print("Type a JD, or ask things like 'backend 5-8 remote k8s grpc', 'genai vertex gcp python', 'language diversity', etc. Say 'bye' to end.\n")

    transcript: List[str] = []
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            user = "bye"
        if not user: 
            continue

        if user.lower() in {"bye","goodbye","exit","quit"}:
            # session end summary
            summary_text = "\n".join(transcript[-30:])
            conv_embed = serialize_vec(embed_text_once(summary_text))
            end_blob = json.dumps({
                "ended_at": datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z",
                "company": client_company,
                "contact_key": client_contact,
                "summary_excerpt": summary_text[-900:],
                "embedding": conv_embed
            }, ensure_ascii=False)
            # store as a final turn as well to ensure recap appears next boot
            append_interaction(mem_col, client_id, "Session end", "Saved session summary.")
            print("bye!\n")
            break
        
        if COUNT_PAT.search(user):
            n = get_candidate_count(cand_col)
            resp = f"We currently have {n} candidates indexed."
            append_interaction(mem_col, client_id, user, resp)
            transcript.append(f"USER: {user}"); transcript.append(f"BOT: {resp}")
            print(resp)
            continue
        jd = parse_jd(user)
        try:
            if is_greeting(user):
                resp = smalltalk_reply(user)

            elif is_memory(user):
                resp = latest_memory_snippet(mem_col, client_id)

            elif has_jd_signal(user):
                # SAFE expr: only scalars in Milvus (role_family / years_band if you stored as scalar string)
                expr_terms=[]
                families = jd.get("role_families") or []
                if families:
                    if len(families) == 1:
                        expr_terms.append(f'role_family == "{families[0]}"')
                    else:
                        fam_clause = ", ".join(f'"{fam}"' for fam in families)
                        expr_terms.append(f'role_family in [{fam_clause}]')
                if jd.get("years_band"):   expr_terms.append(f'years_band == "{jd["years_band"]}"')
                expr = " and ".join(expr_terms) if expr_terms else None

                rows = unified_search(
                    cand_col,
                    jd_text=jd.get("raw"),
                    expr=expr,
                    need=jd.get("must_skills"),                # skills/tools post-filter
                    clouds_any=[c.lower() for c in jd.get("clouds", [])],
                    top_k=TOP_K
                )
                if not rows:
                    resp = "I didn’t find candidates with those constraints. Want me to relax skills or clouds and try again?"
                else:
                    resp = format_slate(rows, jd)

            else:
                resp = smalltalk_reply(user)

        except Exception as e:
            resp = "I hit an error running that search. I can retry with fewer constraints or only vector."
            print(f"[warn] handler error: {e}", file=sys.stderr)

        # store one recap line for this turn (last 3 kept)
        append_interaction(mem_col, client_id, user, resp)
        transcript.append(f"USER: {user}")
        transcript.append(f"BOT: {resp}")
        print(resp)

    return 0

if __name__ == "__main__":
    sys.exit(main())
