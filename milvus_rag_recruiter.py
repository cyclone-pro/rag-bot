#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus + OpenAI RAG Recruiter Chatbot
====================================
- Connects to a Milvus collection "candidate_pool" (or any given via env).
- Answers 30 manager questions using scalar filters and (optionally) vector search
  over `summary_embedding` and `skills_embedding` (dimension 768, HNSW cosine).
- STRICT about accuracy: only summarizes facts fetched from Milvus.
- If required fields are missing (e.g., interview dates, stage transitions),
  responds: "Sorry — we don't have any such info for that."

Env Vars
--------
OPENAI_API_KEY=...              # for formatting + (optional) embeddings
OPENAI_MODEL=gpt-4o-mini        # chat model to prettify the facts (optional)
OPENAI_EMBED_MODEL=text-embedding-3-large  # for vector queries (optional)
MILVUS_URI=http://my-milvus:19530
MILVUS_TOKEN=...                # if auth enabled
MILVUS_DB=default               # db (Milvus 2.4+)
MILVUS_COLLECTION=candidate_pool
USE_VECTOR=1                    # 1 to enable vector search when helpful
LIMIT=20000                     # max rows to pull for Python-side filtering

Local testing (no Milvus)
-------------------------
Set USE_LOCAL_CSV=1 and LOCAL_CSV_PATH=./candidate_pool_1000.csv
"""
import os
import json
import math
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter

# ---------- Config ----------
def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file without extra deps."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in ("'", '"')):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()

# ---------- Config ----------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED     = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

MILVUS_URI       = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN     = os.getenv("MILVUS_TOKEN", "")
MILVUS_DB        = os.getenv("MILVUS_DB", "default")
MILVUS_COLLECTION= os.getenv("MILVUS_COLLECTION", "candidate_pool")
USE_VECTOR       = os.getenv("USE_VECTOR", "1") == "1"
LIMIT            = int(os.getenv("LIMIT", "20000"))

USE_LOCAL_CSV    = os.getenv("USE_LOCAL_CSV", "0") == "1"
LOCAL_CSV_PATH   = os.getenv("LOCAL_CSV_PATH", "./candidate_pool_1000.csv")

# Time zone: America/Chicago (manager base)
try:
    import zoneinfo
    TZ = zoneinfo.ZoneInfo("America/Chicago")
except Exception:
    TZ = timezone(timedelta(hours=-5))  # coarse CST fallback

# ---------- Optional deps ----------
_openai_ok = True
try:
    from openai import OpenAI
except Exception:
    _openai_ok = False

_pymilvus_ok = True
try:
    from pymilvus import connections, db, utility, Collection
except Exception:
    _pymilvus_ok = False

_pd = None
if USE_LOCAL_CSV:
    try:
        import pandas as pd
        _pd = pd
    except Exception:
        pass

# ---------- Schema (from your screenshots) ----------
ALL_FIELDS = [
    # identity & contact
    "candidate_id","name","email","phone","linkedin_url","portfolio_url",
    # location & prefs
    "location_city","location_state","location_country","relocation_willingness","remote_preference","availability_status",
    # background
    "total_experience_years","education_level","degrees","institutions","languages_spoken",
    "primary_industry","sub_industries",
    # skills/tools/certs
    "skills_extracted","tools_and_technologies","certifications",
    # resume/title summary
    "top_titles_mentioned","domains_of_expertise","employment_history","semantic_summary","keywords_summary","career_stage","genai_relevance_score",
    # domain/relevance scores
    "medical_domain_score","construction_domain_score","cad_relevance_score","nlp_relevance_score","computer_vision_relevance_score",
    "data_engineering_relevance_score","mlops_relevance_score",
    # evidence
    "evidence_skills","evidence_domains","evidence_certifications","evidence_tools",
    # process
    "source_channel","hiring_manager_notes","interview_feedback","offer_status","assigned_recruiter","resume_embedding_version","last_updated",
    # vectors
    "summary_embedding","skills_embedding"
]

FIELDS = {
    "id":"candidate_id",
    "name":"name",
    "titles":"top_titles_mentioned",
    "skills":"skills_extracted",
    "certs":"certifications",
    "domains":"domains_of_expertise",
    "medical":"medical_domain_score",
    "construction":"construction_domain_score",
    "industry":"primary_industry",
    "source":"source_channel",
    "recruiter":"assigned_recruiter",
    "offer":"offer_status",
    "updated":"last_updated",
    # vectors
    "summary_vec":"summary_embedding",
    "skills_vec":"skills_embedding",
}

# ---------- Helpers ----------
def now_cst() -> datetime: return datetime.now(tz=TZ)

def parse_dt(val: Any) -> Optional[datetime]:
    if val is None or (isinstance(val, float) and math.isnan(val)): return None
    if isinstance(val, datetime): return val.astimezone(TZ)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z","%Y-%m-%d %H:%M:%S%z","%Y-%m-%d %H:%M:%S","%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(val), fmt)
            if not dt.tzinfo: dt = dt.replace(tzinfo=TZ)
            return dt.astimezone(TZ)
        except Exception: pass
    try:
        from dateutil import parser as dparser
        dt = dparser.parse(str(val))
        if not dt.tzinfo: dt = dt.replace(tzinfo=TZ)
        return dt.astimezone(TZ)
    except Exception:
        return None

def contains(text: Optional[str], kw: str) -> bool:
    return bool(text and kw.lower() in text.lower())

def any_contains(text: Optional[str], kws: List[str]) -> bool:
    return any(contains(text, k) for k in kws)

def tokenize_list(s: Optional[str]) -> List[str]:
    if not s: return []
    toks = re.split(r"[;,/\|]| {2,}", s)
    return [t.strip() for t in toks if t and t.strip()]

def percent(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(100.0*n/d, 2)

# ---------- OpenAI wrappers ----------
class LLM:
    def __init__(self):
        self.enabled = _openai_ok and bool(OPENAI_API_KEY)
        self.client = OpenAI(api_key=OPENAI_API_KEY) if self.enabled else None

    def chat(self, facts: Dict[str, Any]) -> str:
        if not facts.get("ok"): return "Sorry — we don't have any such info for that."
        if not self.enabled:
            # Deterministic fallback
            out = []
            if facts.get("title"): out.append(f"*{facts['title']}*")
            if facts.get("summary"): out.append(facts["summary"])
            if facts.get("table"): out += facts["table"]
            return "\n".join(out) if out else "Done."
        sys = (
            "You are RecruiterBot, a precise recruiting analyst.\n"
            "STRICT RULE: Only use the JSON facts provided. Do NOT invent quantities.\n"
            "If ok=false or rows are empty, reply exactly: 'Sorry — we don't have any such info for that.'\n"
            "Be concise, friendly, and management-ready. Use bullets or short tables where helpful."
        )
        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                messages=[
                    {"role":"system","content":sys},
                    {"role":"user","content":json.dumps(facts, ensure_ascii=False)},
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            # Fallback
            out = []
            if facts.get("title"): out.append(f"*{facts['title']}*")
            if facts.get("summary"): out.append(facts["summary"])
            if facts.get("table"): out += facts["table"]
            return "\n".join(out) if out else "Done."

class Embedder:
    def __init__(self):
        self.enabled = _openai_ok and bool(OPENAI_API_KEY)
        self.client = OpenAI(api_key=OPENAI_API_KEY) if self.enabled else None
        self.model  = OPENAI_EMBED

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.enabled: return None
        try:
            res = self.client.embeddings.create(model=self.model, input=text)
            return res.data[0].embedding
        except Exception:
            return None

# ---------- Data Access (Milvus or CSV) ----------
class CandidateStore:
    def __init__(self):
        self.use_local = USE_LOCAL_CSV
        self.df = None
        self.local_records: Optional[List[Dict[str, Any]]] = None
        self.col = None
        if self.use_local:
            path = Path(LOCAL_CSV_PATH)
            if not path.exists():
                raise RuntimeError(f"USE_LOCAL_CSV=1 but CSV not found at {path}")
            if _pd is not None:
                self.df = _pd.read_csv(path)
            else:
                try:
                    with path.open(newline="", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        self.local_records = [dict(row) for row in reader]
                except Exception as exc:
                    raise RuntimeError(f"Failed to read CSV at {path}: {exc}") from exc
        elif _pymilvus_ok:
            # Connect & load collection
            connections.connect("default", uri=MILVUS_URI, token=MILVUS_TOKEN or None)
            try:
                db.using_database(MILVUS_DB)
            except Exception:
                pass
            self.col = Collection(MILVUS_COLLECTION)
            try: self.col.load()
            except Exception: pass
        else:
            raise RuntimeError("No Milvus client and no CSV. Install pymilvus or set USE_LOCAL_CSV=1.")

    def fetch_all(self, fields: List[str]) -> List[Dict[str, Any]]:
        if self.use_local:
            if self.df is not None:
                subset = self.df[ [c for c in fields if c in self.df.columns] ].copy()
                return subset.to_dict(orient="records")
            if self.local_records is not None:
                return [{f: row.get(f) for f in fields} for row in self.local_records]
            raise RuntimeError("USE_LOCAL_CSV=1 but pandas unavailable and CSV could not be loaded.")
        assert self.col is not None
        # Milvus doesn't have "*" query; do a permissive expr and large limit.
        expr = f'{FIELDS["id"]} != ""'
        return self.col.query(expr=expr, output_fields=fields, limit=LIMIT)

    def vector_search(self, text: str, vec_field: str, filter_expr: Optional[str], top_k: int=200) -> List[Dict[str, Any]]:
        # When local CSV: return empty (no vectors); Milvus: run search()
        if self.use_local or self.col is None: return []
        emb = Embedder().embed(text) if USE_VECTOR else None
        if not emb: return []
        params = {"metric_type":"COSINE","params":{"ef":128}}
        req = self.col.search(
            data=[emb],
            anns_field=vec_field,
            param=params,
            limit=top_k,
            expr=filter_expr,
            output_fields=ALL_FIELDS
        )
        hits = req[0] if req else []
        return [h.entity for h in hits]

# ---------- Manager Questions ----------
class ManagerLogic:
    def __init__(self, store: CandidateStore):
        self.store = store

    def _rows_today(self) -> List[Dict[str, Any]]:
        rows = self.store.fetch_all([FIELDS["offer"],FIELDS["updated"],FIELDS["recruiter"],FIELDS["titles"]])
        start = now_cst().replace(hour=0,minute=0,second=0,microsecond=0)
        end = start + timedelta(days=1)
        out = []
        for r in rows:
            dt = parse_dt(r.get(FIELDS["updated"]))
            if dt and start <= dt < end: out.append(r)
        return out

    def _display(self, rows: List[Dict[str, Any]], limit:int=10) -> List[str]:
        lines = []
        for r in rows[:limit]:
            name = r.get("name","N/A")
            title = r.get("top_titles_mentioned","")
            city  = r.get("location_city",""); state=r.get("location_state",""); country=r.get("location_country","")
            src   = r.get("source_channel",""); rec=r.get("assigned_recruiter","")
            parts = [name, title, ", ".join([x for x in [city,state,country] if x]), f"source:{src}" if src else "", f"recruiter:{rec}" if rec else ""]
            lines.append("- " + " | ".join([p for p in parts if p]))
        return lines

    # 1) Hired in last 2 days & medical
    def q1(self):
        rows = self.store.fetch_all(ALL_FIELDS)
        cutoff = now_cst() - timedelta(days=2)
        out = []
        for r in rows:
            st = (r.get("offer_status","") or "").lower()
            if st not in {"hired","accepted"}: continue
            dt = parse_dt(r.get("last_updated"))
            if not dt or dt < cutoff: continue
            med = (float(r.get("medical_domain_score",0) or 0) > 0.5) or any_contains(r.get("domains_of_expertise",""), ["medical","health","healthcare"])
            if med: out.append(r)
        return {"ok": bool(out), "title":"Hires in last 2 days (medical background)", "summary": f"Found {min(10,len(out))} candidate(s).", "table": self._display(out,10)}

    # 2) Interviews conducted today by recruiter + outcomes -> missing structured fields
    def q2(self):
        return {"ok": False}

    # 3) Interviews scheduled next 3 days -> missing schedule fields
    def q3(self):
        return {"ok": False}

    # 4) Most submissions today + shortlist→interview conversion -> missing submission/shortlist fields
    def q4(self):
        return {"ok": False}

    # 5) Moved Interviewed→Offered today + departments -> no transition history / department
    def q5(self):
        return {"ok": False}

    # 6) Rejected today + top 3 rejection reasons (heuristic from free-text)
    def q6(self):
        rows = self.store.fetch_all(["name","top_titles_mentioned","interview_feedback","hiring_manager_notes","offer_status","last_updated"])
        start = now_cst().replace(hour=0,minute=0,second=0,microsecond=0)
        end = start + timedelta(days=1)
        out = []
        counts = Counter()
        reason_kws = ["communication","experience","culture","salary","skills","visa","location","availability","timeline","overqualified","underqualified"]
        for r in rows:
            st = (r.get("offer_status","") or "").lower()
            if st != "rejected": continue
            dt = parse_dt(r.get("last_updated"))
            if not dt or not (start <= dt < end): continue
            note = (r.get("interview_feedback") or "") + " " + (r.get("hiring_manager_notes") or "")
            for kw in reason_kws:
                if contains(note, kw): counts[kw]+=1
            out.append(r)
        table = self._display(out,50)
        if not out: return {"ok": False}
        top = ", ".join([f"{k}({v})" for k,v in counts.most_common(3)]) if counts else "—"
        return {"ok": True, "title":"Rejected today", "summary": f"Top reasons: {top}", "table": table}

    # 7) Accepted offers this week
    def q7(self):
        rows = self.store.fetch_all(ALL_FIELDS)
        now = now_cst()
        week_start = (now - timedelta(days=now.weekday())).replace(hour=0,minute=0,second=0,microsecond=0)
        week_end = week_start + timedelta(days=7)
        out = []
        for r in rows:
            st = (r.get("offer_status","") or "").lower()
            if st not in {"hired","accepted"}: continue
            dt = parse_dt(r.get("last_updated"))
            if dt and week_start <= dt < week_end: out.append(r)
        return {"ok": bool(out), "title":"Offer acceptances this week", "summary": f"{len(out)} acceptance(s) this week.", "table": self._display(out,50)}

    # 8) Success rate Backend (Python) — optionally vector assisted
    def q8(self):
        # Vector prefilter if enabled
        vec_rows = []
        if USE_VECTOR:
            vec_rows = self.store.vector_search("backend python developer", "skills_embedding", None, top_k=500)
        rows = vec_rows if vec_rows else self.store.fetch_all([FIELDS["titles"],FIELDS["skills"],FIELDS["offer"]])
        total=hired=0
        for r in rows:
            titles = (r.get("top_titles_mentioned") or "") + " " + (r.get("semantic_summary") or "")
            skills = r.get("skills_extracted") or ""
            if any_contains(titles,["backend"]) or any_contains(skills,["backend"]):
                if any_contains(skills,["python"]):
                    total += 1
                    if (r.get("offer_status","") or "").lower() in {"hired","accepted"}: hired += 1
        if total==0: return {"ok": False}
        return {"ok": True, "title":"Hiring success — Backend (Python)", "summary": f"{hired}/{total} → {percent(hired,total)}%"}

    # 9) Construction hiring rate last month vs previous
    def q9(self):
        rows = self.store.fetch_all(["construction_domain_score","offer_status","last_updated"])
        buckets = defaultdict(lambda: {"total":0,"hired":0})
        for r in rows:
            if float(r.get("construction_domain_score",0) or 0) <= 0.5: continue
            dt = parse_dt(r.get("last_updated"));  st=(r.get("offer_status","") or "").lower()
            if not dt: continue
            key = dt.strftime("%Y-%m")
            buckets[key]["total"]+=1
            if st in {"hired","accepted"}: buckets[key]["hired"]+=1
        if not buckets: return {"ok": False}
        now = now_cst()
        last_month_key = (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
        table = [f"- {m}: {v['hired']}/{v['total']} ({percent(v['hired'],v['total'])}%)" for m,v in sorted(buckets.items())]
        lm = buckets.get(last_month_key)
        summary = f"Last month ({last_month_key}) rate: {percent(lm['hired'],lm['total'])}%." if lm else "No data for last month."
        return {"ok": True, "title":"Construction hiring rate by month", "summary": summary, "table": table}

    # 10) Dept with highest interview→hire ratio -> missing interview counts
    def q10(self): return {"ok": False}

    # 11) Medical domain offer acceptance rate (30d)
    def q11(self):
        rows = self.store.fetch_all(["medical_domain_score","offer_status","last_updated"])
        cutoff = now_cst() - timedelta(days=30)
        offered=accepted=0
        for r in rows:
            if float(r.get("medical_domain_score",0) or 0) <= 0.5: continue
            dt = parse_dt(r.get("last_updated"));  st=(r.get("offer_status","") or "").lower()
            if not dt or dt < cutoff: continue
            if st in {"offered","hired","accepted","declined"}:
                offered += 1
                if st in {"hired","accepted"}: accepted += 1
        if offered==0: return {"ok": False}
        return {"ok": True, "title":"Medical — Offer acceptance (last 30 days)", "summary": f"{accepted}/{offered} → {percent(accepted,offered)}%"}

    # 12) Offered but declined + reasons
    def q12(self):
        rows = self.store.fetch_all(["name","top_titles_mentioned","offer_status","interview_feedback","hiring_manager_notes","last_updated"])
        out=[]
        for r in rows:
            if (r.get("offer_status","") or "").lower()!="declined": continue
            note = (r.get("interview_feedback") or "") or (r.get("hiring_manager_notes") or "")
            out.append({"name":r.get("name"),"top_titles_mentioned":r.get("top_titles_mentioned"),"note":note})
        if not out: return {"ok": False}
        table = [f"- {x['name']} | {x['top_titles_mentioned']} | notes: {x['note'][:160]}" for x in out[:50]]
        return {"ok": True, "title":"Declined offers (recent)", "table": table}

    # 13) Conversion by role category
    def q13(self):
        rows = self.store.fetch_all(["top_titles_mentioned","offer_status"])
        cats = {
            "Frontend":["frontend","react","angular","vue"],
            "Backend":["backend","java","spring","django","node"],
            "Data Engineer":["data engineer","etl","spark","hadoop"],
            "AI Engineer":["ai engineer","ml engineer","machine learning","deep learning"],
            "MLOps":["mlops","kubeflow","sagemaker","mlflow"],
        }
        table=[]
        for cat, kws in cats.items():
            total=hired=0
            for r in rows:
                titles = r.get("top_titles_mentioned") or ""
                if any_contains(titles, kws):
                    total+=1
                    if (r.get("offer_status","") or "").lower() in {"hired","accepted"}: hired+=1
            if total>0: table.append(f"- {cat}: {hired}/{total} ({percent(hired,total)}%)")
        return {"ok": bool(table), "title":"Conversion by role category", "table": table}

    # 14) % AWS or Azure-certified who got hired
    def q14(self):
        rows = self.store.fetch_all(["certifications","offer_status"])
        total=hired=0
        for r in rows:
            certs = r.get("certifications") or ""
            if any_contains(certs,["aws","azure"]):
                total+=1
                if (r.get("offer_status","") or "").lower() in {"hired","accepted"}: hired+=1
        if total==0: return {"ok": False}
        return {"ok": True, "title":"Hire % among AWS/Azure-certified", "summary": f"{hired}/{total} → {percent(hired,total)}%"}

    # 15) Best source channels (this quarter)
    def q15(self):
        rows = self.store.fetch_all(["source_channel","offer_status","last_updated"])
        now = now_cst(); q=(now.month-1)//3
        start = datetime(now.year, q*3+1, 1, tzinfo=TZ)
        end   = datetime(now.year+1,1,1,tzinfo=TZ) if q==3 else datetime(now.year, q*3+4, 1, tzinfo=TZ)
        buckets=defaultdict(lambda: {"total":0,"hired":0})
        for r in rows:
            dt = parse_dt(r.get("last_updated"));  st=(r.get("offer_status","") or "").lower()
            if not dt or not (start <= dt < end): continue
            src = (r.get("source_channel") or "Unknown").strip() or "Unknown"
            buckets[src]["total"]+=1
            if st in {"hired","accepted"}: buckets[src]["hired"]+=1
        if not buckets: return {"ok": False}
        table=[f"- {s}: {v['hired']}/{v['total']} ({percent(v['hired'],v['total'])}%)" for s,v in sorted(buckets.items(), key=lambda kv: percent(kv[1]['hired'],kv[1]['total']), reverse=True)]
        return {"ok": True, "title":"Best source channels (this quarter)", "table": table}

    # 16) Rank recruiters by total hires this week
    def q16(self):
        rows = self.store.fetch_all(["assigned_recruiter","offer_status","last_updated"])
        now = now_cst(); week_start=(now - timedelta(days=now.weekday())).replace(hour=0,minute=0,second=0,microsecond=0); week_end=week_start+timedelta(days=7)
        counts=defaultdict(int)
        for r in rows:
            st=(r.get("offer_status","") or "").lower(); dt=parse_dt(r.get("last_updated"))
            if st in {"hired","accepted"} and dt and week_start<=dt<week_end:
                rec=(r.get("assigned_recruiter") or "Unassigned").strip() or "Unassigned"
                counts[rec]+=1
        if not counts: return {"ok": False}
        table=[f"- {rec}: {cnt} hire(s) this week" for rec,cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
        return {"ok": True, "title":"Recruiter hires this week", "table": table, "summary":"Avg submission→hire time unavailable (no submission timestamps)."}

    # 17) Lowest offer acceptance rates by recruiter
    def q17(self):
        rows = self.store.fetch_all(["assigned_recruiter","offer_status"])
        offered=defaultdict(int); accepted=defaultdict(int)
        for r in rows:
            rec=(r.get("assigned_recruiter") or "Unassigned").strip() or "Unassigned"
            st=(r.get("offer_status","") or "").lower()
            if st in {"offered","hired","accepted","declined"}: offered[rec]+=1
            if st in {"hired","accepted"}: accepted[rec]+=1
        data=[]
        for rec in offered:
            rate=percent(accepted[rec], offered[rec]) if offered[rec] else 0.0
            data.append((rec, offered[rec], accepted[rec], rate))
        if not data: return {"ok": False}
        data.sort(key=lambda x:x[3])
        table=[f"- {rec}: {acc}/{off} ({rate}%)" for rec,off,acc,rate in data[:10]]
        return {"ok": True, "title":"Lowest offer acceptance rates (by recruiter)", "table": table}

    # 18) Top 3 recruiters to final interview -> missing stage markers
    def q18(self): return {"ok": False}

    # 19) Recruiters with declining pipeline in last 14 days (hires w2→w1)
    def q19(self):
        rows = self.store.fetch_all(["assigned_recruiter","offer_status","last_updated"])
        now = now_cst(); w1_start=now - timedelta(days=7); w2_start=now - timedelta(days=14)
        hires_w1=defaultdict(int); hires_w2=defaultdict(int)
        for r in rows:
            st=(r.get("offer_status","") or "").lower(); dt=parse_dt(r.get("last_updated"))
            if st not in {"hired","accepted"} or not dt: continue
            rec=(r.get("assigned_recruiter") or "Unassigned").strip() or "Unassigned"
            if dt>=w1_start: hires_w1[rec]+=1
            elif w2_start<=dt<w1_start: hires_w2[rec]+=1
        diffs=[]
        for rec in set(list(hires_w1.keys())+list(hires_w2.keys())):
            diffs.append((rec, hires_w2.get(rec,0), hires_w1.get(rec,0), hires_w1.get(rec,0)-hires_w2.get(rec,0)))
        if not diffs: return {"ok": False}
        declines=[d for d in diffs if d[3] < 0]
        if not declines: return {"ok": True, "title":"Pipeline performance (14 days)", "summary":"No declines detected week-over-week."}
        table=[f"- {rec}: hires last week {w1}, prior week {w2} (Δ {delta})" for rec,w2,w1,delta in sorted(declines, key=lambda x:x[3])]
        return {"ok": True, "title":"Declining pipeline performance (14 days)", "table": table}

    # 20) Stage distribution by recruiter
    def q20(self):
        rows = self.store.fetch_all(["assigned_recruiter","offer_status"])
        dist=defaultdict(lambda: defaultdict(int))
        for r in rows:
            rec=(r.get("assigned_recruiter") or "Unassigned").strip() or "Unassigned"
            stage=(r.get("offer_status") or "Unknown").strip() or "Unknown"
            dist[rec][stage]+=1
        if not dist: return {"ok": False}
        table=[]
        for rec,stages in dist.items():
            cells=", ".join([f"{s}:{n}" for s,n in sorted(stages.items())])
            table.append(f"- {rec} → {cells}")
        return {"ok": True, "title":"Stage distribution by recruiter", "table": table}

    # 21) Most in-demand skills among hires this month
    def q21(self):
        rows = self.store.fetch_all(["skills_extracted","offer_status","last_updated"])
        month_start = now_cst().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        counter=Counter()
        for r in rows:
            st=(r.get("offer_status","") or "").lower()
            if st not in {"hired","accepted"}: continue
            dt=parse_dt(r.get("last_updated"))
            if not dt or dt < month_start: continue
            for sk in tokenize_list(r.get("skills_extracted")): counter[sk.lower()]+=1
        if not counter: return {"ok": False}
        table=[f"- {k}: {v}" for k,v in counter.most_common(15)]
        return {"ok": True, "title":"Top skills among hires (this month)", "table": table}

    # 22) Common skills/certs among rejected AI/ML roles
    def q22(self):
        rows = self.store.fetch_all(["top_titles_mentioned","skills_extracted","certifications","offer_status"])
        sks=Counter(); cts=Counter()
        for r in rows:
            if (r.get("offer_status","") or "").lower()!="rejected": continue
            titles=r.get("top_titles_mentioned") or ""
            if not any_contains(titles,["ai","ml","machine learning","deep learning"]): continue
            for s in tokenize_list(r.get("skills_extracted")): sks[s.lower()]+=1
            for c in tokenize_list(r.get("certifications")): cts[c.lower()]+=1
        if not sks and not cts: return {"ok": False}
        table=[]
        if sks: table.append("Skills: " + ", ".join([f"{k}({v})" for k,v in sks.most_common(10)]))
        if cts: table.append("Certs: " + ", ".join([f"{k}({v})" for k,v in cts.most_common(10)]))
        return {"ok": True, "title":"Common skills/certs among rejected AI/ML candidates", "table": table}

    # 23) Hired recently with strong healthcare/medical AI exp
    def q23(self):
        rows = self.store.fetch_all(ALL_FIELDS)
        cutoff = now_cst() - timedelta(days=30)
        out=[]
        for r in rows:
            st=(r.get("offer_status","") or "").lower()
            if st not in {"hired","accepted"}: continue
            dt=parse_dt(r.get("last_updated"))
            if not dt or dt < cutoff: continue
            med=(float(r.get("medical_domain_score",0) or 0)>0.5) or any_contains(r.get("domains_of_expertise",""),["medical","health"])
            ai = any_contains((r.get("top_titles_mentioned") or "") + " " + (r.get("skills_extracted") or ""), ["ai","ml","pytorch","tensorflow","machine learning","deep learning"])
            if med and ai: out.append(r)
        if not out: return {"ok": False}
        return {"ok": True, "title":"Recent hires — medical AI", "table": self._display(out,30)}

    # 24) Longest average time-to-hire by title -> missing submission/created timestamps
    def q24(self): return {"ok": False}

    # 25) Missing skills among pipeline Data Engineering roles
    def q25(self):
        rows = self.store.fetch_all(["top_titles_mentioned","skills_extracted","offer_status"])
        hired=Counter(); pipe=Counter()
        for r in rows:
            titles = r.get("top_titles_mentioned") or ""
            if not any_contains(titles,["data engineer"]): continue
            sks=[s.lower() for s in tokenize_list(r.get("skills_extracted"))]
            if (r.get("offer_status","") or "").lower() in {"hired","accepted"}:
                for s in sks: hired[s]+=1
            else:
                for s in sks: pipe[s]+=1
        if not hired or not pipe: return {"ok": False}
        missing=[]
        for sk,cnt in hired.most_common(50):
            if pipe.get(sk,0) < max(1, cnt//3): missing.append((sk,cnt,pipe.get(sk,0)))
        table=[f"- {sk}: hired {h}, pipeline {p}" for sk,h,p in missing[:20]]
        return {"ok": True, "title":"Missing skills in DE pipeline (vs hired)", "table": table}

    # 26) Pipeline >45 days — stagnating
    def q26(self):
        rows = self.store.fetch_all(["name","top_titles_mentioned","offer_status","last_updated","assigned_recruiter","location_city","location_state","location_country","source_channel"])
        cutoff = now_cst() - timedelta(days=45)
        out=[]
        for r in rows:
            st=(r.get("offer_status","") or "").lower()
            if st in {"hired","rejected","declined"}: continue
            dt=parse_dt(r.get("last_updated"))
            if dt and dt < cutoff: out.append(r)
        if not out: return {"ok": False}
        return {"ok": True, "title":"Stagnating pipeline >45 days", "table": self._display(out,50)}

    # 27) Dept highest rejection at first interview -> no first-interview markers
    def q27(self): return {"ok": False}

    # 28) Passed interviews but no feedback > 1 week -> no pass flags / timestamps
    def q28(self): return {"ok": False}

    # 29) Roles with >3 interviews and no hires -> interview counts missing
    def q29(self): return {"ok": False}

    # 30) Daily success summary
    def q30(self):
        rows = self._rows_today()
        hires=rejs=moves=0
        by_role=Counter(); by_rec=Counter()
        for r in rows:
            st=(r.get("offer_status","") or "").lower()
            if st in {"hired","accepted"}:
                hires+=1
                by_rec[(r.get("assigned_recruiter") or "Unassigned").strip() or "Unassigned"]+=1
                title=(r.get("top_titles_mentioned") or "Unknown").lower()
                by_role[title]+=1
            elif st in {"rejected","declined"}:
                rejs+=1
            else:
                moves+=1
        if hires==0 and rejs==0 and moves==0: return {"ok": False}
        top_rec = ", ".join([f"{k}:{v}" for k,v in by_rec.most_common(5)]) if by_rec else "—"
        top_roles = ", ".join([f"{k}:{v}" for k,v in by_role.most_common(5)]) if by_role else "—"
        summary=f"Hires: {hires} | Rejections: {rejs} | Other stage moves: {moves}. Top recruiters: {top_rec}. Roles with most movement: {top_roles}."
        return {"ok": True, "title":"Daily success summary", "summary": summary}

# ---------- Router ----------
ROUTES = [
    ("hired in the last 2 days", "q1"),
    ("interviews were conducted today", "q2"),
    ("interviews are scheduled in the next 3 days", "q3"),
    ("submitted the most profiles today", "q4"),
    ("moved from ‘Interviewed’ to ‘Offered’", "q5"),
    ("rejected today", "q6"),
    ("accepted offers this week", "q7"),
    ("success rate of hiring for Backend Engineer roles requiring Python", "q8"),
    ("hiring rate in the Construction department", "q9"),
    ("highest interview-to-hire ratio overall", "q10"),
    ("Medical domain candidates", "q11"),
    ("offered roles but declined", "q12"),
    ("conversion rates by role category", "q13"),
    ("AWS or Azure certifications", "q14"),
    ("source channels", "q15"),
    ("Rank recruiters by their total hires this week", "q16"),
    ("lowest offer acceptance rates", "q17"),
    ("final interview round", "q18"),
    ("declining pipeline performance", "q19"),
    ("distribution of candidates across hiring stages", "q20"),
    ("most in-demand skills among hired candidates this month", "q21"),
    ("common among rejected candidates for AI/ML roles", "q22"),
    ("hired recently with strong healthcare or medical AI experience", "q23"),
    ("longest average time-to-hire", "q24"),
    ("skills are missing among current active pipeline candidates for Data Engineering roles", "q25"),
    ("pipeline for more than 45 days", "q26"),
    ("highest rejection rate at the first interview stage", "q27"),
    ("passed interviews but haven’t received feedback", "q28"),
    ("more than 3 interviews were conducted without any hires", "q29"),
    ("Generate a success summary for the day", "q30"),
]

class Chatbot:
    def __init__(self):
        self.store = CandidateStore()
        self.logic = ManagerLogic(self.store)
        self.llm = LLM()

    def answer(self, q: str) -> str:
        low = q.lower()
        handler = None
        for key, fn in ROUTES:
            if key.lower() in low:
                handler = getattr(self.logic, fn); break
        if handler is None and "medical background" in low and "last 2 days" in low:
            handler = self.logic.q1
        if handler is None:
            return "Sorry — we don't have any such info for that."
        facts = handler()
        return self.llm.chat(facts)

def main():
    bot = Chatbot()
    print("Recruiter RAG Chatbot (Milvus) — ask your question. Ctrl+C to exit.")
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
