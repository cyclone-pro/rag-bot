#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recruiter Brain — terminal chat (points to `new_candidate_pool`)
- USE_VECTOR defaults to 1 (vector+scalar)
- JD short-text expansion before encoding
- efSearch=32, top_k=8
"""

import os, re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pymilvus import connections, db, Collection, utility

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB  = os.getenv("MILVUS_DB", "default")
CAND_COL   = os.getenv("CANDIDATES_COLLECTION", "new_candidate_pool")  # UPDATED
USE_VECTOR = os.getenv("USE_VECTOR", "1") == "1"                        # UPDATED default
EF_SEARCH  = int(os.getenv("EF_SEARCH", "32"))
TOP_K      = int(os.getenv("TOP_K", "8"))
HIDE_PII   = True

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/e5-base-v2")
_encoder = None
if USE_VECTOR:
    try:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"[info] encoder: {EMBED_MODEL_NAME}")
    except Exception as e:
        print(f"[warn] encoder unavailable; falling back to scalar-only: {e}")
        _encoder = None

ROLE_PARTITIONS = ["backend","frontend","devops","security","data","mlops","cloud","systems","mobile","blockchain"]

EQUIV = {
    "ci_cd": {"canonical":"github actions", "aliases":["github actions","jenkins","gitlab ci","circleci","azure devops"]},
    "monitoring": {"canonical":"datadog", "aliases":["datadog","prometheus","grafana","new relic","cloudwatch"]},
    "api": {"canonical":"rest+grpc", "aliases":["rest","grpc","graph ql","graphql"]},
    "orchestration": {"canonical":"kubernetes", "aliases":["kubernetes","k8s","eks","gke","aks"]},
    "cloud": {"canonical":"aws", "aliases":["aws","gcp","azure"]},
    "db": {"canonical":"rdbms", "aliases":["postgres","mysql","sql server","oracle","mariadb"]},
    "python_web": {"canonical":"django", "aliases":["django","fastapi","flask"]},
}

def conn():
    connections.connect("default", uri=MILVUS_URI)
    try: db.using_database(MILVUS_DB)
    except: pass
    return Collection(CAND_COL)

@dataclass
class Plan:
    role_family: Optional[str] = None
    min_years: Optional[float] = None
    musts: List[str] = field(default_factory=list)
    clouds: List[str] = field(default_factory=list)
    leadership: bool = False
    communication: bool = False
    strict: bool = False
    urgent: bool = False
    desired: int = 8

def expand_jd(jd: str) -> str:
    """Light expansion for very short JDs to stabilize recall."""
    base = jd.strip()
    if len(base.split()) >= 8:
        return base
    return f"{base}. Relevant tech terms: Django, REST, gRPC, Kubernetes, AWS, GCP, Postgres, MySQL, CI/CD, Datadog, Prometheus, Grafana, leadership, communication."

def parse_jd(jd: str) -> Plan:
    t = jd.lower()
    p = Plan()
    role_map = {
        "frontend":"frontend", "react":"frontend", "angular":"frontend", "vue":"frontend",
        "backend":"backend", "django":"backend", "spring":"backend", "spring boot":"backend",
        "fastapi":"backend", "flask":"backend", "node":"backend", "express":"backend", "nest":"backend",
        "asp.net":"backend", "rust":"backend",
        "devops":"devops", "sre":"devops", "platform":"devops",
        "security":"security", "cyber":"security",
        "data engineer":"data", "etl":"data", "dbt":"data", "data analyst":"data",
        "mlops":"mlops", "ml ops":"mlops",
        "cloud":"cloud", "aws":"cloud", "gcp":"cloud", "azure":"cloud",
        "systems":"systems", "vmware":"systems", "nutanix":"systems", "olvm":"systems",
        "mobile":"mobile", "ios":"mobile", "android":"mobile",
        "blockchain":"blockchain", "solidity":"blockchain", "evm":"blockchain",
    }
    for k,v in role_map.items():
        if k in t: p.role_family = v; break
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years|yrs)", t)
    if m: p.min_years = float(m.group(1))
    for c in ["aws","gcp","azure"]:
        if c in t: p.clouds.append(c.upper())
    for g in EQUIV.values():
        for alias in g["aliases"]:
            if alias in t: p.musts.append(g["canonical"]); break
    p.musts = list(dict.fromkeys(p.musts))
    if any(x in t for x in ["lead","team lead","mentor","architect"]): p.leadership=True
    if any(x in t for x in ["communication","client-facing","presented","stakeholder"]): p.communication=True
    return p

def years_band(y: Optional[float]) -> Optional[str]:
    if y is None: return None
    if y < 3: return "junior"
    if y < 6: return "mid"
    return "senior"

def build_expr(col: Collection, p: Plan) -> Tuple[str, List[str]]:
    have = {f.name for f in col.schema.fields}
    clauses, parts = [], []
    if p.role_family and col.has_partition(p.role_family):
        parts = [p.role_family]
    yb = years_band(p.min_years)
    if yb and "years_band" in have: clauses.append(f'years_band == "{yb}"')
    if p.clouds and "clouds" in have:
        ins = ",".join([f'"{c}"' for c in p.clouds])
        clauses.append(f'clouds in [{ins}]')
    clauses.append('candidate_id != ""')
    return (" and ".join(clauses) if clauses else ""), parts

TEXT_FIELDS = ["skills_extracted","tools_and_technologies","semantic_summary","employment_history","keywords_summary","top_titles_mentioned"]
def text_of(row: Dict[str,Any]) -> str:
    out = []
    for k in TEXT_FIELDS:
        v = row.get(k)
        if isinstance(v, list): out.append(" ".join(v))
        elif isinstance(v, str): out.append(v)
    return (" ".join(out)).lower()

def covers_musts(row: Dict[str,Any], p: Plan) -> Tuple[float, List[str], List[str]]:
    text = text_of(row)
    covered, missing = [], []
    for m in p.musts:
        fam_aliases = []
        for g in EQUIV.values():
            if g["canonical"] == m:
                fam_aliases = g["aliases"]; break
        ok = any((" "+a+" ") in (" "+text+" ") for a in (fam_aliases or [m]))
        (covered if ok else missing).append(m)
    ratio = (len(covered)/max(1,len(p.musts))) if p.musts else 1.0
    return ratio, covered, missing

def leadership_signal(txt: str) -> bool:
    return any(k in txt for k in ["lead","led","mentored","architect","ownership","stakeholder","managed"])
def comm_signal(txt: str) -> bool:
    return any(k in txt for k in ["communication","presented","client-facing","written","verbal","documentation","demo","stakeholder"])

def score_row(row: Dict[str,Any], p: Plan) -> Tuple[float, Dict[str,Any]]:
    txt = text_of(row)
    yrs = float(row.get("total_experience_years") or 0)
    s = 0.0
    if p.min_years is not None:
        s += 1.0 if yrs >= p.min_years else max(0.0, yrs/(p.min_years+1e-6))*0.7
    ratio, covered, missing = covers_musts(row, p)
    s += ratio * 2.0
    if p.leadership and leadership_signal(txt): s += 0.4
    if p.communication and comm_signal(txt):    s += 0.3
    return s, {"covered":covered, "missing":missing, "years":yrs}

def diversify(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    seen, out = set(), []
    for r in rows:
        tag, txt = None, text_of(r)
        for k in ["django","fastapi","flask","spring","spring boot","express","nest","asp.net","kubernetes","prometheus","datadog"]:
            if k in txt: tag = k; break
        if tag and tag in seen: continue
        if tag: seen.add(tag)
        out.append(r)
    if len(out) < len(rows):
        out += [r for r in rows if r not in out]
    return out

def embed_query(q: str) -> Optional[List[float]]:
    if _encoder is None: return None
    v = _encoder.encode(q or "", normalize_embeddings=True)
    return v.tolist() if hasattr(v,"tolist") else list(v)

def search(col: Collection, p: Plan, jd_text: str) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    expr, parts = build_expr(col, p)
    fields = [
        "candidate_id","name","location_city","location_state","location_country",
        "skills_extracted","tools_and_technologies","semantic_summary","employment_history",
        "top_titles_mentioned","total_experience_years","sources","last_updated"
    ]
    rows: List[Dict[str,Any]] = []
    if USE_VECTOR and _encoder is not None and "summary_embedding" in {f.name for f in col.schema.fields}:
        query_text = expand_jd(jd_text)
        v = embed_query(query_text)
        try:
            hits = col.search(
                data=[v],
                anns_field="summary_embedding",
                param={"metric_type":"COSINE","params":{"ef": EF_SEARCH}},
                limit=TOP_K*5,
                expr=expr if expr else None,
                partition_names=parts or None,
                output_fields=fields
            )
            for h in hits[0]:
                rows.append(h.entity.get("_raw_data", h.entity))
        except Exception as e:
            rows = col.query(expr=expr if expr else 'candidate_id != ""', output_fields=fields, limit=max(TOP_K*5, 40), partition_names=parts or None)
    else:
        rows = col.query(expr=expr if expr else 'candidate_id != ""', output_fields=fields, limit=max(TOP_K*5, 40), partition_names=parts or None)

    scored = []
    for r in rows:
        s, ev = score_row(r, p)
        scored.append((s, r, ev))
    scored.sort(key=lambda x: x[0], reverse=True)

    desired = 8
    take = min(len(scored), max(desired*2, 16))
    top = scored[:take]
    perfect, nearfits, seen = [], [], set()
    for s, r, ev in top:
        cid = r.get("candidate_id")
        if cid in seen: continue
        seen.add(cid)
        ratio, covered, missing = covers_musts(r, p)
        why = []
        if ev.get("years") is not None:
            try: why.append(f"{float(ev['years']):.1f}y total exp")
            except: pass
        for c in covered: why.append(f"{c} present")
        txt = text_of(r)
        if leadership_signal(txt): why.append("leadership signals")
        if comm_signal(txt):      why.append("communication signals")
        r["_why_"]  = why
        r["_gaps_"] = [m for m in missing] if missing else []
        if ratio >= 1.0: perfect.append(r)
        elif ratio >= 0.8: nearfits.append(r)

    nearfits = diversify(nearfits)[: max(0, desired - min(len(perfect), desired))]
    perfect  = perfect[: min(len(perfect), desired)]

    if HIDE_PII:
        for arr in (perfect, nearfits):
            for r in arr:
                r.pop("email", None)
                r.pop("phone", None)

    return perfect, nearfits

def fmt(arr: List[Dict[str,Any]]) -> List[str]:
    out = []
    for r in arr:
        name  = r.get("name","N/A")
        title = r.get("top_titles_mentioned","")
        loc   = ", ".join([x for x in [r.get("location_city",""), r.get("location_state",""), r.get("location_country","")] if x])
        why   = r.get("_why_", [])
        gaps  = r.get("_gaps_", [])
        out.append(f"- {name} | {title} | {loc}")
        if why:  out.append("  why: " + "; ".join(why[:4]))
        if gaps: out.append("  gaps: " + "; ".join(gaps[:3]))
    return out

def main():
    col = conn()
    print("Recruiter Brain — `new_candidate_pool`. Type JD (or /quit).")
    while True:
        jd = input("\nJD> ").strip()
        if jd.lower() in ("/q","/quit","exit"): break
        p = parse_jd(jd)
        perfect, near = search(col, p, jd)
        print("\nHere’s a slate (Perfect vs Near-fit):")
        if perfect:
            print("\nPerfect matches:")
            print("\n".join(fmt(perfect)))
        if near:
            print("\nNear-fits:")
            print("\n".join(fmt(near)))
        if not perfect and not near:
            print("\nNo confident matches yet. Quick clarifications:")
            print("- Which cloud(s) are in scope — AWS, GCP, Azure (hybrid okay)?")
            print("- Database preference — Postgres/MySQL/SQL Server — treat as strict or flexible?")
            print("- Work mode — Onsite/Hybrid/Remote — any location constraints?")

if __name__ == "__main__":
    main()
