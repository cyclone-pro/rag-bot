"""
Recruiter RAG Chatbot — v3.1 (Analytics + Open-Ended, Fault-Tolerant)
=====================================================================
- Milvus-backed (optional CSV mode) + optional OpenAI for formatting/embeddings
- Open-ended planner for candidate search
- NEW: Analytics intent routing for KPIs (rates, percentages, by-month, etc.)
- Strict matching for must-haves; explainable "why"
- Milvus-safe pagination & retries; graceful fallbacks

Environment:
------------
OPENAI_API_KEY=...                 # optional; enables nice formatting + embeddings
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
MILVUS_URI=http://my-milvus:19530
MILVUS_DB=default
MILVUS_COLLECTION=candidate_pool
USE_VECTOR=1                       # enable vector prefilter
STRICT_ALL=1                       # require all must-haves in candidate mode
MEDICAL_SCORE_MIN=0.5              # signals threshold for healthcare
LIMIT=10000                        # full-scan cap (<=16384 internally)
TZ_STR=America/Chicago
# Local CSV mode (no Milvus):
USE_LOCAL_CSV=0
LOCAL_CSV_PATH=./candidate_pool_1000.csv
"""
from __future__ import annotations

import os, re, json, math, time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter

# ------------------------- Configuration -------------------------
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED      = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

MILVUS_URI        = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_DB         = os.getenv("MILVUS_DB", "default")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "candidate_pool")

USE_VECTOR        = os.getenv("USE_VECTOR", "1") == "1"
STRICT_ALL        = os.getenv("STRICT_ALL", "1") == "1"
MEDICAL_SCORE_MIN = float(os.getenv("MEDICAL_SCORE_MIN", "0.5"))
SCAN_LIMIT        = min( max(1,int(os.getenv("LIMIT","10000"))) , 16384)   # Milvus window cap
PAGE_SIZE         = 4096                                                   # for paginated query()

TZ_STR = os.getenv("TZ_STR", "America/Chicago")
try:
    import zoneinfo
    TZ = zoneinfo.ZoneInfo(TZ_STR)
except Exception:
    TZ = timezone(timedelta(hours=-5))

def now_tz() -> datetime:
    return datetime.now(tz=TZ)

# -------------------------- Optional deps ------------------------
_openai_ok = True
try:
    from openai import OpenAI
except Exception:
    _openai_ok = False

_pymilvus_ok = True
try:
    from pymilvus import connections, db, Collection
except Exception:
    _pymilvus_ok = False

_pd = None
USE_LOCAL_CSV = os.getenv("USE_LOCAL_CSV", "0") == "1"
LOCAL_CSV_PATH = os.getenv("LOCAL_CSV_PATH", "./candidate_pool_1000.csv")
if USE_LOCAL_CSV:
    try:
        import pandas as pd
        _pd = pd
    except Exception:
        pass

# ---------------------------- Schema ----------------------------
ALL_FIELDS: List[str] = [
    "candidate_id","name","email","phone","linkedin_url","portfolio_url",
    "location_city","location_state","location_country","relocation_willingness","remote_preference","availability_status",
    "total_experience_years","education_level","degrees","institutions","languages_spoken",
    "primary_industry","sub_industries",
    "skills_extracted","tools_and_technologies","certifications",
    "top_titles_mentioned","domains_of_expertise","employment_history","semantic_summary","keywords_summary","career_stage","genai_relevance_score",
    "medical_domain_score","construction_domain_score","cad_relevance_score","nlp_relevance_score","computer_vision_relevance_score",
    "data_engineering_relevance_score","mlops_relevance_score",
    "evidence_skills","evidence_domains","evidence_certifications","evidence_tools",
    "source_channel","hiring_manager_notes","interview_feedback","offer_status","assigned_recruiter","resume_embedding_version","last_updated",
    "summary_embedding","skills_embedding"
]

ID_FIELD = "candidate_id"

# -------------------------- Utilities ---------------------------
def parse_dt(v: Any) -> Optional[datetime]:
    if v is None or (isinstance(v, float) and math.isnan(v)): return None
    if isinstance(v, datetime): return v.astimezone(TZ)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z","%Y-%m-%d %H:%M:%S%z","%Y-%m-%d %H:%M:%S","%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(v), fmt)
            if not dt.tzinfo: dt = dt.replace(tzinfo=TZ)
            return dt.astimezone(TZ)
        except Exception:
            continue
    try:
        from dateutil import parser as dparser
        dt = dparser.parse(str(v))
        if not dt.tzinfo: dt = dt.replace(tzinfo=TZ)
        return dt.astimezone(TZ)
    except Exception:
        return None

def norm_join(doc: dict, fields: List[str]) -> str:
    return " ".join([str(doc.get(f,"") or "") for f in fields])

def has_all_needles(hay: str, needles: List[str]) -> bool:
    return all(any(n.lower() in hay.lower() for n in [needle, needle.replace(" ", "")]) for needle in needles)

def any_of_terms(hay: str, terms: List[str]) -> bool:
    return any(t.lower() in hay.lower() for t in terms)

def dedupe_by_id(rows: List[dict], id_key: str = ID_FIELD) -> List[dict]:
    seen = set(); out = []
    for r in rows:
        rid = r.get(id_key)
        if rid in seen: continue
        seen.add(rid); out.append(r)
    return out

def percent(n:int, d:int) -> float:
    return 0.0 if d==0 else round(100.0*n/d, 2)

# ----------------------- Synonyms/Signals ------------------------
DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "healthcare": ["healthcare","medical","clinical","hospital","med-tech","healthvcare","ehr","emr","fhir","hl7"],
    "oncology":   ["oncology","cancer","tumor","oncological","radiology","pathology","chemo","imaging ai"],
    "hipaa":      ["hipaa","hippa","hipa","hipaa certified","hipaa compliance","hipaa-compliant"],
    "django":     ["django","python django","django rest","drf"],
    "pytorch":    ["pytorch","torch","torchvision","lightning"],
    "backend":    ["backend","server-side","api","microservices"],
    "python":     ["python","py"],
}

# --------------------------- LLM Wraps ---------------------------
class LLM:
    def __init__(self):
        self.enabled = _openai_ok and bool(OPENAI_API_KEY)
        self.client = OpenAI(api_key=OPENAI_API_KEY) if self.enabled else None

    def format(self, facts: Dict[str,Any]) -> str:
        if not facts.get("ok"):
            return "Sorry — we don't have any such info for that."
        if not self.enabled:
            out = []
            if facts.get("title"): out.append(f"*{facts['title']}*")
            if facts.get("summary"): out.append(facts["summary"])
            if facts.get("table"): out += facts["table"]
            return "\n".join(out) if out else "Done."
        sys = (
            "You are RecruiterBot. STRICT: Use ONLY the provided JSON facts. "
            "Never invent names or numbers. If ok=false, answer exactly with the apology. "
            "Keep responses concise and management-ready; bullets/tables welcome."
        )
        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                messages=[
                    {"role":"system","content":sys},
                    {"role":"user","content":json.dumps(facts, ensure_ascii=False)}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            out = []
            if facts.get("title"): out.append(f"*{facts['title']}*")
            if facts.get("summary"): out.append(facts["summary"])
            if facts.get("table"): out += facts["table"]
            return "\n".join(out) if out else "Done."

class Embedder:
    def __init__(self):
        self.enabled = _openai_ok and bool(OPENAI_API_KEY) and USE_VECTOR
        self.client = OpenAI(api_key=OPENAI_API_KEY) if self.enabled else None
        self.model  = OPENAI_EMBED
    def embed(self, text: str) -> Optional[List[float]]:
        if not self.enabled: return None
        try:
            res = self.client.embeddings.create(model=self.model, input=text)
            return res.data[0].embedding
        except Exception:
            return None

# ----------------------- Data Access Layer -----------------------
class Store:
    def __init__(self):
        self.df = None; self.col = None
        if USE_LOCAL_CSV and _pd is not None:
            self.df = _pd.read_csv(LOCAL_CSV_PATH)
        elif _pymilvus_ok:
            connections.connect("default", uri=MILVUS_URI)
            try: db.using_database(MILVUS_DB)
            except Exception: pass
            self.col = Collection(MILVUS_COLLECTION)
            try: self.col.load()
            except Exception: pass
        else:
            raise RuntimeError("No Milvus connection and no local CSV. Set USE_LOCAL_CSV=1 for CSV mode.")

    # Safe, paginated query that respects 16,384 window, with retries
    def fetch_all(self, fields: List[str]) -> List[Dict[str,Any]]:
        if self.df is not None:
            sub = self.df[[c for c in fields if c in self.df.columns]].copy()
            return sub.to_dict(orient="records")

        expr = f'{ID_FIELD} != ""'
        remaining = min(SCAN_LIMIT, 16384)
        offset = 0
        results: List[Dict[str,Any]] = []
        while remaining > 0:
            limit = min(PAGE_SIZE, remaining, 16384 - offset)
            if limit <= 0: break
            tries = 0
            while True:
                try:
                    batch = self.col.query(expr=expr, output_fields=fields, limit=limit, offset=offset)
                    break
                except Exception:
                    tries += 1
                    if tries >= 3: raise
                    time.sleep(0.2 * (2 ** (tries-1)))
            if not batch: break
            results.extend(batch)
            offset += limit
            remaining -= limit
            if offset >= 16384:   # Milvus window reached; stop scanning
                break
        return results

    def vector_search(self, text: str, anns_field: str, filter_expr: Optional[str], top_k: int = 400) -> List[Dict[str,Any]]:
        if self.col is None or not USE_VECTOR:
            return []
        emb = Embedder().embed(text)
        if not emb:
            return []
        params = {"metric_type":"COSINE","params":{"ef":128}}
        tries = 0
        while True:
            try:
                res = self.col.search(
                    data=[emb],
                    anns_field=anns_field,
                    param=params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=ALL_FIELDS
                )
                hits = res[0] if res else []
                return [h.entity for h in hits]
            except Exception:
                tries += 1
                if tries >= 3: return []
                time.sleep(0.2 * (2 ** (tries-1)))

# ----------------------------- Planner ---------------------------
ANALYTICS_HOOKS = [
    "rate","ratio","percentage","compare","by month","quarter",
    "conversion","acceptance","highest","which source channels","hiring rate","top skills","common among rejected"
]

def is_analytics(q: str) -> bool:
    ql = q.lower()
    return any(tok in ql for tok in ANALYTICS_HOOKS) or ql.strip().endswith("?")

class Planner:
    """
    Candidate-mode planner (analytics has its own routing).
    Converts manager query to a plan with:
    - vector_query, must_have {skills, certs, domains, titles}, time_window_days, limit
    """
    def __init__(self):
        self.llm = LLM()

    def _heuristic(self, q: str) -> Dict[str,Any]:
        ql = q.lower()
        plan = {"vector_query": q, "limit": 10, "must_have": {"skills":[], "certs":[], "domains":[], "titles":[]}, "time_window_days": None}
        m = re.search(r"(top|any|show)?\s*(\d{1,2})\b", ql)
        if m:
            try: plan["limit"] = max(1, min(50, int(m.group(2))))
            except: pass
        # map families
        if any(t in ql for t in DEFAULT_SYNONYMS["django"]):  plan["must_have"]["skills"].append("django")
        if any(t in ql for t in DEFAULT_SYNONYMS["python"]):  plan["must_have"]["skills"].append("python")
        if any(t in ql for t in DEFAULT_SYNONYMS["pytorch"]): plan["must_have"]["skills"].append("pytorch")
        if "backend" in ql:                                    plan["must_have"]["titles"].append("backend")
        if any(t in ql for t in DEFAULT_SYNONYMS["hipaa"]):   plan["must_have"]["certs"].append("hipaa")
        if any(t in ql for t in DEFAULT_SYNONYMS["healthcare"]): plan["must_have"]["domains"].append("healthcare")
        if any(t in ql for t in DEFAULT_SYNONYMS["oncology"]): plan["must_have"]["domains"].append("oncology")
        # time windows
        if "last 2 days" in ql: plan["time_window_days"] = 2
        elif "last week" in ql or "last 7 days" in ql: plan["time_window_days"] = 7
        elif "last 30 days" in ql or "last month" in ql: plan["time_window_days"] = 30
        return plan

    def make_plan(self, q: str) -> Dict[str,Any]:
        if not self.llm.enabled:
            return self._heuristic(q)
        sys = (
            "Return ONLY JSON for a search plan: "
            "{vector_query:str, limit:int, must_have:{skills:[], certs:[], domains:[], titles:[]}, time_window_days:int|null}"
        )
        try:
            client = self.llm.client
            resp = client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0,
                messages=[{"role":"system","content":sys},{"role":"user","content":q}]
            )
            txt = resp.choices[0].message.content.strip()
            plan = json.loads(txt)
            plan.setdefault("limit", 10)
            mh = plan.setdefault("must_have", {})
            for k in ("skills","certs","domains","titles"):
                mh.setdefault(k, [])
            return plan
        except Exception:
            return self._heuristic(q)

# --------------------------- Search Engine -----------------------
class Engine:
    def __init__(self, store: Store):
        self.store = store

    # -------- Candidate mode (hybrid retrieval) --------
    def _row_passes(self, r: Dict[str,Any], plan: Dict[str,Any]) -> bool:
        mh = plan.get("must_have", {})
        sskills  = norm_join(r, ["skills_extracted","tools_and_technologies","evidence_skills"])
        scerts   = norm_join(r, ["certifications","evidence_certifications"])
        sdomains = norm_join(r, ["domains_of_expertise","primary_industry","evidence_domains","semantic_summary"])
        stitles  = norm_join(r, ["top_titles_mentioned","keywords_summary"])

        if STRICT_ALL:
            if mh.get("skills")  and not has_all_needles(sskills,  mh["skills"]):  return False
            if mh.get("certs")   and not has_all_needles(scerts,   mh["certs"]):   return False
            if mh.get("domains") and not has_all_needles(sdomains, mh["domains"]): return False
            if mh.get("titles")  and not has_all_needles(stitles,  mh["titles"]):  return False
        else:
            ok = False
            for blob, keys in [(sskills,mh.get("skills",[])),(scerts,mh.get("certs",[])),(sdomains,mh.get("domains",[])),(stitles,mh.get("titles",[]))]:
                if keys and has_all_needles(blob, keys): ok = True
            if not ok: return False

        if any_of_terms(" ".join(mh.get("domains", [])), ["healthcare","medical","clinical","hospital"]):
            score = float(r.get("medical_domain_score", 0) or 0)
            if score < MEDICAL_SCORE_MIN and not any_of_terms(sdomains, ["fhir","hl7","ehr","emr","healthcare","medical"]):
                return False

        days = plan.get("time_window_days")
        if days:
            cutoff = now_tz() - timedelta(days=days)
            dt = parse_dt(r.get("last_updated"))
            if not dt or dt < cutoff: return False
        return True

    def _why(self, r: Dict[str,Any], plan: Dict[str,Any]) -> str:
        mh = plan.get("must_have", {})
        fields = {
            "skills":  norm_join(r, ["skills_extracted","tools_and_technologies","evidence_skills"]),
            "certs":   norm_join(r, ["certifications","evidence_certifications"]),
            "domains": norm_join(r, ["domains_of_expertise","primary_industry","evidence_domains","semantic_summary"]),
            "titles":  norm_join(r, ["top_titles_mentioned","keywords_summary"]),
            "history": norm_join(r, ["employment_history","semantic_summary","hiring_manager_notes","interview_feedback"]),
        }
        snippets: List[str] = []
        for cat, needles in mh.items():
            if not needles: continue
            blob = fields.get(cat,"")
            hits = [n for n in needles if n.lower() in blob.lower()]
            if hits:
                snippets.append(f"{cat}: " + ", ".join(hits))
        try:
            med_score = float(r.get("medical_domain_score", 0) or 0)
            if med_score >= MEDICAL_SCORE_MIN and any_of_terms(" ".join(mh.get("domains", [])), ["healthcare","medical","clinical","hospital"]):
                snippets.append(f"medical_domain_score={med_score}")
        except Exception:
            pass
        if not snippets:
            return ""  # avoid noisy fallback
        return " | ".join(snippets)[:220]

    def _as_lines(self, rows: List[Dict[str,Any]], plan: Dict[str,Any], limit: int) -> List[str]:
        out = []
        for r in rows[:limit]:
            loc = ", ".join([x for x in [r.get("location_city",""), r.get("location_state",""), r.get("location_country","")] if x])
            rec = r.get("assigned_recruiter") or "—"
            titles = r.get("top_titles_mentioned") or ""
            line = f"- {r.get('name','N/A')} | {titles} | {loc} | recruiter:{rec}"
            why = self._why(r, plan)
            if why: line += f"\n  why: {why}"
            out.append(line)
        return out

    def candidate_search(self, plan: Dict[str,Any]) -> List[Dict[str,Any]]:
        rows: List[Dict[str,Any]] = []
        if USE_VECTOR:
            rows = self.store.vector_search(plan.get("vector_query",""), "skills_embedding", filter_expr=None, top_k=400)
            if not rows:
                rows = self.store.vector_search(plan.get("vector_query",""), "summary_embedding", filter_expr=None, top_k=400)
        if not rows:
            rows = self.store.fetch_all(ALL_FIELDS)
        rows = dedupe_by_id(rows, ID_FIELD)
        filtered = [r for r in rows if self._row_passes(r, plan)]
        return filtered

    # -------- Analytics helpers (pure scalar, deterministic) --------
    def _families(self):
        hires = {"hired","accepted"}
        offers= {"offered","hired","accepted","declined"}
        rejects={"rejected","declined"}
        return hires, offers, rejects

    def _month_key(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m")

    # A1) Construction hiring rate by month (+ highlight last month)
    def analytics_construction_rate_by_month(self) -> dict:
        rows = self.store.fetch_all(["construction_domain_score","offer_status","last_updated"])
        if not rows: return {"ok": False}
        hires, _, _ = self._families()
        buckets: Dict[str, Dict[str,int]] = defaultdict(lambda: {"total":0,"hired":0})
        for r in rows:
            if float(r.get("construction_domain_score",0) or 0) < 0.5: continue
            dt = parse_dt(r.get("last_updated"))
            if not dt: continue
            key = self._month_key(dt)
            st  = (r.get("offer_status","") or "").lower()
            buckets[key]["total"] += 1
            if st in hires: buckets[key]["hired"] += 1
        if not buckets: return {"ok": False}
        months = sorted(buckets.keys())
        last = months[-1]
        rate_last = percent(buckets[last]["hired"], buckets[last]["total"])
        table = [f"- {m}: {buckets[m]['hired']}/{buckets[m]['total']} ({percent(buckets[m]['hired'], buckets[m]['total'])}%)" for m in months]
        return {"ok": True, "title":"Construction hiring rate by month", "summary": f"Last month ({last}) rate: {rate_last}%.", "table": table}

    # A2) Hiring success — Backend + Python (% hired among candidates matching)
    def analytics_backend_python_success_rate(self) -> dict:
        rows = self.store.fetch_all(["top_titles_mentioned","skills_extracted","offer_status"])
        if not rows: return {"ok": False}
        hires, _, _ = self._families()
        total=hired=0
        for r in rows:
            titles = (r.get("top_titles_mentioned") or "").lower()
            skills = (r.get("skills_extracted") or "").lower()
            if ("backend" in titles or "backend" in skills) and "python" in skills:
                total += 1
                if (r.get("offer_status","") or "").lower() in hires: hired += 1
        if total==0: return {"ok": False}
        return {"ok": True, "title":"Success rate — Backend (Python)", "summary": f"{hired}/{total} → {percent(hired,total)}%"}

    # A3) % of AWS/Azure-certified who got hired
    def analytics_aws_azure_hire_percent(self) -> dict:
        rows = self.store.fetch_all(["certifications","offer_status"])
        if not rows: return {"ok": False}
        hires, _, _ = self._families()
        total=hired=0
        for r in rows:
            certs = (r.get("certifications") or "").lower()
            if "aws" in certs or "azure" in certs:
                total += 1
                if (r.get("offer_status","") or "").lower() in hires: hired += 1
        if total==0: return {"ok": False}
        return {"ok": True, "title":"Hire % among AWS/Azure-certified", "summary": f"{hired}/{total} → {percent(hired,total)}%"}

    # A4) Source channels with highest hire rate (this quarter)
    def analytics_best_source_channels_this_quarter(self) -> dict:
        rows = self.store.fetch_all(["source_channel","offer_status","last_updated"])
        if not rows: return {"ok": False}
        hires, _, _ = self._families()
        now = now_tz()
        q = (now.month-1)//3
        start = datetime(now.year, q*3+1, 1, tzinfo=TZ)
        end   = datetime(now.year+1,1,1,tzinfo=TZ) if q==3 else datetime(now.year, q*3+4, 1, tzinfo=TZ)
        buckets = defaultdict(lambda: {"total":0,"hired":0})
        for r in rows:
            dt = parse_dt(r.get("last_updated"))
            if not dt or not (start <= dt < end): continue
            src = (r.get("source_channel") or "Unknown").strip() or "Unknown"
            buckets[src]["total"] += 1
            if (r.get("offer_status","") or "").lower() in hires:
                buckets[src]["hired"] += 1
        if not buckets: return {"ok": False}
        ordered = sorted(buckets.items(), key=lambda kv: percent(kv[1]['hired'], kv[1]['total']), reverse=True)
        table = [f"- {s}: {v['hired']}/{v['total']} ({percent(v['hired'],v['total'])}%)" for s,v in ordered]
        top = ordered[0][0] if ordered else "—"
        return {"ok": True, "title":"Best source channels (this quarter)", "summary": f"Top performer: {top}.", "table": table}

    # A5) Top in-demand skills among hires (this month)
    def analytics_top_skills_hires_this_month(self) -> dict:
        rows = self.store.fetch_all(["skills_extracted","offer_status","last_updated"])
        if not rows: return {"ok": False}
        hires, _, _ = self._families()
        month_start = now_tz().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        counter = Counter()
        for r in rows:
            if (r.get("offer_status","") or "").lower() not in hires: continue
            dt = parse_dt(r.get("last_updated"))
            if not dt or dt < month_start: continue
            skills = [s.strip().lower() for s in re.split(r"[;,/\|]| {2,}", r.get("skills_extracted") or "") if s.strip()]
            for s in skills: counter[s]+=1
        if not counter: return {"ok": False}
        table = [f"- {k}: {v}" for k,v in counter.most_common(20)]
        return {"ok": True, "title":"Top skills among hires (this month)", "table": table}

    # A6) Common skills/certs among rejected AI/ML candidates
    def analytics_common_rejected_ai_ml(self) -> dict:
        rows = self.store.fetch_all(["top_titles_mentioned","skills_extracted","certifications","offer_status"])
        if not rows: return {"ok": False}
        _, _, rejects = self._families()
        sks=Counter(); cts=Counter()
        for r in rows:
            if (r.get("offer_status","") or "").lower() not in rejects: continue
            titles=(r.get("top_titles_mentioned") or "").lower()
            if not any(t in titles for t in ["ai","ml","machine learning","deep learning"]): continue
            for s in [x.strip().lower() for x in re.split(r"[;,/\|]| {2,}", r.get("skills_extracted") or "") if x.strip()]:
                sks[s]+=1
            for c in [x.strip().lower() for x in re.split(r"[;,/\|]| {2,}", r.get("certifications") or "") if x.strip()]:
                cts[c]+=1
        if not sks and not cts: return {"ok": False}
        table=[]
        if sks: table.append("Skills: " + ", ".join([f"{k}({v})" for k,v in sks.most_common(15)]))
        if cts: table.append("Certs: " + ", ".join([f"{k}({v})" for k,v in cts.most_common(15)]))
        return {"ok": True, "title":"Common skills/certs — rejected AI/ML", "table": table}

    # A7) Medical/Healthcare AI hires in last 30 days (count)
    def analytics_medical_ai_recent_hires(self) -> dict:
        rows = self.store.fetch_all(["medical_domain_score","top_titles_mentioned","skills_extracted","offer_status","last_updated"])
        if not rows: return {"ok": False}
        hires, _, _ = self._families()
        cutoff = now_tz() - timedelta(days=30)
        out = []
        for r in rows:
            st = (r.get("offer_status","") or "").lower()
            if st not in hires: continue
            dt = parse_dt(r.get("last_updated"))
            if not dt or dt < cutoff: continue
            med = float(r.get("medical_domain_score", 0) or 0) >= MEDICAL_SCORE_MIN
            ai = any(t in ((r.get("top_titles_mentioned") or "") + " " + (r.get("skills_extracted") or "")).lower() for t in ["ai","ml","machine learning","deep learning","pytorch","tensorflow"])
            if med and ai: out.append(r)
        if not out: return {"ok": False}
        return {"ok": True, "title":"Recent healthcare AI hires (30d)", "summary": f"{len(out)} candidate(s)."}

# ------------------------------ Chatbot --------------------------
class Chatbot:
    def __init__(self):
        self.store = Store()
        self.engine = Engine(self.store)
        self.planner = Planner()
        self.llm = LLM()

    def answer(self, user_q: str) -> str:
        # Handle multiple questions: answer the first one cleanly
        for sep in ["?","；","।"]:
            if sep in user_q and user_q.count(sep) > 1:
                user_q = user_q.split(sep)[0] + "?"
                break

        if is_analytics(user_q):
            ql = user_q.lower()
            # route analytics intents
            if "construction" in ql and ("last month" in ql or "compare" in ql or "by month" in ql):
                facts = self.engine.analytics_construction_rate_by_month()
                return self.llm.format(facts)
            if ("aws" in ql or "azure" in ql) and ("percent" in ql or "percentage" in ql or "rate" in ql):
                facts = self.engine.analytics_aws_azure_hire_percent()
                return self.llm.format(facts)
            if "backend" in ql and "python" in ql and ("success rate" in ql or "hiring rate" in ql):
                facts = self.engine.analytics_backend_python_success_rate()
                return self.llm.format(facts)
            if "source channels" in ql or ("source" in ql and "rate" in ql and "quarter" in ql):
                facts = self.engine.analytics_best_source_channels_this_quarter()
                return self.llm.format(facts)
            if ("top" in ql or "most" in ql) and "skills" in ql and ("this month" in ql or "month" in ql):
                facts = self.engine.analytics_top_skills_hires_this_month()
                return self.llm.format(facts)
            if "common" in ql and ("rejected" in ql) and ("ai/ml" in ql or "ai" in ql or "ml" in ql):
                facts = self.engine.analytics_common_rejected_ai_ml()
                return self.llm.format(facts)
            if ("healthcare" in ql or "medical" in ql) and ("ai" in ql or "ml" in ql) and ("recent" in ql or "last 30 days" in ql):
                facts = self.engine.analytics_medical_ai_recent_hires()
                return self.llm.format(facts)
            # fallback for unknown analytics
            return "Sorry — we don't have any such info for that."

        # Candidate mode (open-ended search)
        plan = self.planner.make_plan(user_q)
        rows = self.engine.candidate_search(plan)
        if not rows:
            return "Sorry — we don't have any such info for that."
        limit = plan.get("limit", 10)
        rows = rows[:limit]
        facts = {
            "ok": True,
            "title": "Search results",
            "summary": f"Returning {len(rows)} of {len(rows)} matched candidates.",
            "table": self.engine._as_lines(rows, plan, limit)
        }
        return self.llm.format(facts)

def main():
    bot = Chatbot()
    print("Recruiter RAG Chatbot v3.1 — ask anything. Ctrl+C to exit.")
    while True:
        try:
            q = input("\nYou: ").strip()
            if not q: continue
            a = bot.answer(q)
            print("\nRecruiterBot:", a)
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

if __name__ == "__main__":
    main()

