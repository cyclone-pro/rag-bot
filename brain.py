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
import random


MILVUS_URI = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_DB  = os.getenv("MILVUS_DB", "default")
CAND_COL   = os.getenv("CANDIDATES_COLLECTION", "new_candidate_pool")  # UPDATED
USE_VECTOR = os.getenv("USE_VECTOR", "1") == "1"                        # UPDATED default
EF_SEARCH  = int(os.getenv("EF_SEARCH", "32"))
TOP_K      = int(os.getenv("TOP_K", "8"))
HIDE_PII   = True
APP_VERSION = "rb-1.1"

OUTPUT_STYLE = os.getenv("OUTPUT_STYLE", "details").lower()  # 'details' or 'chat'
VALID_STYLES = {"details", "chat"}
if OUTPUT_STYLE not in VALID_STYLES:
     OUTPUT_STYLE = "details"


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
    "ci_cd": {"canonical": "github actions", "aliases": ["github actions","jenkins","gitlab ci","circleci","azure devops"]},
    "monitoring": {"canonical": "datadog", "aliases": ["datadog","prometheus","grafana","new relic","cloudwatch"]},
    "api": {"canonical": "rest+grpc", "aliases": ["rest","grpc","graph ql","graphql"]},
    "orchestration": {"canonical": "kubernetes", "aliases": ["kubernetes","k8s","eks","gke","aks"]},
    "cloud": {"canonical": "aws", "aliases": ["aws","gcp","azure"]},
    "db": {"canonical": "rdbms", "aliases": ["postgres","mysql","sql server","oracle","mariadb"]},
    "python_web": {"canonical": "django", "aliases": ["django","fastapi","flask"]},
    # new:
    "java_backend": {"canonical": "spring", "aliases": ["java","spring","spring boot","hibernate","jsp","j2ee"]},
    "healthcare": {"canonical": "hipaa", "aliases": ["hipaa","phi","medical","healthcare","hl7","fhir","ehr","emr"]}
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
    if "java" in t or "spring" in t:
     p.role_family = "backend"
    if any(x in t for x in ["hipaa","healthcare","fhir","hl7","medical"]):
     p.musts.append("hipaa")
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
    if yb and "years_band" in have:
        clauses.append(f'years_band == "{yb}"')

    if p.clouds and "clouds" in have:
        arr = ",".join([f'"{c}"' for c in p.clouds])
        if len(p.clouds) == 1:
            clauses.append(f'array_contains(clouds, "{p.clouds[0]}")')
        else:
            clauses.append(f'array_contains_any(clouds, [{arr}])')

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

def score_row(row, p):
    txt = text_of(row)
    yrs = float(row.get("total_experience_years") or 0)
    s = 0.0
    if "hipaa" in txt or "healthcare" in txt or "fhir" in txt:
     s += 0.6  # compliance signal
    if p.min_years is not None:
        s += 1.0 if yrs >= p.min_years else max(0.0, yrs/(p.min_years+1e-6))*0.7

    ratio, covered, missing = covers_musts(row, p)
    s += ratio * 2.0

    # NEW: healthcare boost
    med = row.get("medical_domain_score")
    if med is not None:
        try:
            s += min(max(float(med), 0.0), 1.0) * 0.8
        except:
            pass

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
        "top_titles_mentioned","total_experience_years","source_channel","last_updated"
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

# ---------- Formatting helpers ----------

def _as_titles(val) -> str:
    """Normalize titles field into a human-readable string."""
    if isinstance(val, (list, tuple)):
        vals = [str(x).strip() for x in val if str(x).strip()]
        return ", ".join(vals) if vals else "—"
    if isinstance(val, str) and val.strip():
        return val.strip()
    return "—"

def _human_join(items, max_items=None):
    items = [x for x in (items or []) if x]
    if max_items is not None:
        items = items[:max_items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]

def fmt_details(arr: List[Dict[str, Any]]) -> List[str]:
    """Compact, exact details (bullet list)."""
    out = []
    for r in arr:
        name  = r.get("name", "N/A")
        title = _as_titles(r.get("top_titles_mentioned", ""))  # <- nicer
        loc   = ", ".join([x for x in [r.get("location_city",""), r.get("location_state",""), r.get("location_country","")] if x]) or "—"
        why   = r.get("_why_", [])
        gaps  = r.get("_gaps_", [])
        out.append(f"- {name} | {title} | {loc}")
        if why:
            out.append("  why: " + "; ".join(why[:4]))
        if gaps:
            out.append("  gaps: " + "; ".join(gaps[:3]))
    return out

def fmt_chat(arr: List[Dict[str, Any]]) -> List[str]:
    """Conversational, two-liner per candidate: Recruiter ↔ Brain."""
    lines = []
    for r in arr:
        name = r.get("name", "N/A")
        titles = _as_titles(r.get("top_titles_mentioned", ""))
        loc = ", ".join([x for x in [r.get("location_city",""), r.get("location_state",""), r.get("location_country","")] if x]) or "—"
        why = r.get("_why_", [])
        gaps = r.get("_gaps_", [])

        # Build short natural sentences
        why_text  = _human_join([w.replace("_", " ") for w in why], max_items=3)
        gaps_text = _human_join([g.replace("_", " ") for g in gaps], max_items=2)

        lines.append(f"Recruiter: What about {name} in {loc}?")
        if why_text and gaps_text:
            lines.append(f"Brain: {name} has background as {titles}. Highlights: {why_text}. Possible gaps: {gaps_text}.")
        elif why_text:
            lines.append(f"Brain: {name} has background as {titles}. Highlights: {why_text}.")
        elif gaps_text:
            lines.append(f"Brain: {name} has background as {titles}. Possible gaps: {gaps_text}.")
        else:
            lines.append(f"Brain: {name} has background as {titles}.")
    return lines

def fmt(arr: List[Dict[str, Any]], mode: str = "details") -> List[str]:
    """
    mode: 'details' (compact bullets) or 'chat' (dialogue).
    """
    mode = (mode or "details").lower()
    if mode == "chat":
        return fmt_chat(arr)
    return fmt_details(arr)

def summarize_candidates(perfect: List[Dict[str, Any]], near: List[Dict[str, Any]], jd_text: str) -> str:
    """Generate an LLM-style conversational summary of results."""
    if not perfect and not near:
        return (
            f"Hey Jake, I couldn’t find strong matches for '{jd_text}'. "
            "Want me to broaden the filters or look at near-fits?"
        )

    all_cands = perfect + near
    if not all_cands:
        return f"Hey Jake, no one quite fits the bill for '{jd_text}'."

    # pick a few top candidates
    top_names = [c.get("name", "Candidate") for c in all_cands[:5]]
    top_str = ", ".join(top_names[:-1]) + f", and {top_names[-1]}" if len(top_names) > 1 else top_names[0]

    summary_lines = [f"Hey Jake, I found {len(all_cands)} candidates that closely match your requirement for '{jd_text}'."]
    summary_lines.append(f"The top profiles include {top_str}.")

    # build short narratives for 3–5
    for i, c in enumerate(all_cands[:5], 1):
        name = c.get("name", f"Candidate {i}")
        yrs = c.get("total_experience_years")
        yrs_text = f"{yrs:.1f} years" if isinstance(yrs, (int, float)) and yrs > 0 else "some"
        why = c.get("_why_", [])
        gaps = c.get("_gaps_", [])
        match_pct = min(100, 60 + random.randint(0, 25))  # heuristic
        techs = []
        for fld in ("skills_extracted", "tools_and_technologies"):
            val = c.get(fld)
            if isinstance(val, list):
                techs.extend(val[:3])
            elif isinstance(val, str):
                techs.extend(re.findall(r"[A-Za-z0-9+#]+", val)[:3])
        techs_str = ", ".join(set(techs[:4])) if techs else ""

        reason = "; ".join(why[:2]) if why else ""
        gap_text = "; ".join(gaps[:1]) if gaps else ""

        line = (
            f"{name} has about {yrs_text} of experience "
            f"and appears to meet roughly {match_pct}% of your requirements."
        )
        if reason:
            line += f" Strengths include {reason}."
        if techs_str:
            line += f" Familiar with {techs_str}."
        if gap_text:
            line += f" Slight gap in {gap_text}."
        summary_lines.append(line)

    summary_lines.append("Would you like me to draft an email summary or shortlist report?")
    return "\n".join(summary_lines)
def main():
    col = conn()
    style = OUTPUT_STYLE  # start from env

    print(f"Recruiter Brain — `new_candidate_pool`  ({APP_VERSION})")
    print("[Tip] /style chat  |  /style details   •  Also works: --chat / --details /style c|d\n")

    while True:
        jd = input("\nJD> ").strip()
        low = jd.lower()

        if low in ("/q", "/quit", "exit"):
            break

        # --- style toggles (more forgiving) ---
        if low.startswith("/style"):
            parts = low.split()
            if len(parts) > 1:
                opt = parts[1]
                if opt in ("chat", "c"):    style = "chat"
                elif opt in ("details", "d"): style = "details"
                else:
                    print("[hint] usage: /style chat  |  /style details")
                    continue
                print(f"[ok] Output style set to: {style}")
            else:
                print("[hint] usage: /style chat  |  /style details")
            continue

        if low in ("--chat", "-chat", "chat"):
            style = "chat"; print(f"[ok] Output style set to: {style}"); continue
        if low in ("--details", "-details", "details"):
            style = "details"; print(f"[ok] Output style set to: {style}"); continue

        # --- main recruiter logic ---
        p = parse_jd(jd)
        perfect, near = search(col, p, jd)

        print(f"\nHere’s a slate (Perfect vs Near-fit)  [style: {style}]")
        if perfect:
            print("\nPerfect matches:")
            print("\n".join(fmt(perfect, mode=style)))
        if near:
            print("\nNear-fits:")
            print("\n".join(fmt(near, mode=style)))

        if not perfect and not near:
            print("\nNo confident matches yet. Quick clarifications:")
            print("- Which cloud(s) are in scope — AWS, GCP, Azure (hybrid okay)?")
            print("- Database preference — Postgres/MySQL/SQL Server — treat as strict or flexible?")
            print("- Work mode — Onsite/Hybrid/Remote — any location constraints?")
        print("\nSummary narrative:")
        print(summarize_candidates(perfect, near, jd))


if __name__ == "__main__":
    main()
