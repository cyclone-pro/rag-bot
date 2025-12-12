"""FastAPI surface for RecruiterBrain v2 - WITH FRESHNESS FILTERING."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymilvus import Collection

import v2brain as rbv2

logger = logging.getLogger(__name__)

app = FastAPI(title="RecruiterBrain v2 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cand_col: Optional[Collection] = None
_mem_col: Optional[Collection] = None


def _get_collections() -> Tuple[Collection, Collection]:
    """Lazy-connect to Milvus and return (candidate, client_memory) collections."""
    global _cand_col, _mem_col
    if _cand_col is not None and _mem_col is not None:
        return _cand_col, _mem_col
    try:
        _cand_col, _mem_col = rbv2.connect_milvus()
        logger.info("Connected to Milvus with collections %s / %s", _cand_col.name, _mem_col.name)
        return _cand_col, _mem_col
    except Exception as exc:  # pragma: no cover - only hit on startup failures
        logger.exception("Milvus connection failed")
        raise HTTPException(status_code=503, detail="Milvus not reachable") from exc


def collections_dep() -> Tuple[Collection, Collection]:
    return _get_collections()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    client_company: str = Field("UnknownCo", description="Client company name for memory tracking")
    contact_key: str = Field("unknown", description="Phone/email/name token for client memory")
    top_k: int = Field(default=rbv2.TOP_K, ge=1, le=200)
    need: Optional[List[str]] = Field(None, description="Skills/tools that must all be present")
    any_of: Optional[List[str]] = Field(None, description="Skills/tools where any match is OK")
    titles_any: Optional[List[str]] = None
    domains_any: Optional[List[str]] = None
    clouds_any: Optional[List[str]] = None
    
    # 🔥 NEW: Freshness parameter
    freshness_days: Optional[int] = Field(
        60,  # Default from config.DEFAULT_FRESHNESS_DAYS
        description="Days to look back (7/30/90/180/365 or null for all time)",
        ge=0
    )


class CandidateHit(BaseModel):
    data: Dict[str, Any]
    why: List[str]


class SearchResponse(BaseModel):
    client_id: str
    total_candidates: int
    slate: str
    candidates: List[CandidateHit]
    memory_snippet: Optional[str] = None
    parsed_jd: Dict[str, Any]


@app.on_event("startup")
def startup_connect() -> None:
    try:
        _get_collections()
    except HTTPException:
        # Keep app alive; /health will surface the error instead.
        logger.warning("Startup Milvus connection failed; will retry on demand.")


@app.get("/", tags=["health"])
def root() -> Dict[str, str]:
    return {"message": "RecruiterBrain v2 API is running. Try POST /search."}


@app.get("/health", tags=["health"])
def health(cols: Tuple[Collection, Collection] = Depends(collections_dep)) -> Dict[str, Any]:
    cand_col, mem_col = cols
    payload: Dict[str, Any] = {
        "status": "ok",
        "candidates_collection": cand_col.name,
        "memory_collection": mem_col.name,
    }
    try:
        payload["candidate_count"] = rbv2.get_candidate_count(cand_col)
    except Exception as exc:
        payload["candidate_count_error"] = str(exc)
    return payload


def _build_expr_from_jd(jd: Dict[str, Any]) -> Optional[str]:
    expr_parts: List[str] = []
    families = jd.get("role_families") or []
    if families:
        if len(families) == 1:
            expr_parts.append(f'role_family == "{families[0]}"')
        else:
            quoted = ", ".join(f'"{fam}"' for fam in families)
            expr_parts.append(f"role_family in [{quoted}]")
    if jd.get("years_band"):
        expr_parts.append(f'years_band == "{jd["years_band"]}"')
    return " and ".join(expr_parts) if expr_parts else None


def _format_slate_with_limit(rows: List[Dict[str, Any]], jd: Dict[str, Any], requested_top_k: int) -> str:
    base_lines = rbv2.format_slate(rows, jd).splitlines()
    role = "/".join(jd.get("role_families") or []) or (jd.get("role_family") or "the role")
    clouds = "/".join(jd.get("clouds", []) or []) or "n/a"
    band = jd.get("years_band") or "unspecified band"
    summary = (
        f"Summary: {len(rows)} candidates aligned to {role} "
        f"(band={band}, clouds={clouds}). Showing top {min(requested_top_k, len(rows))} "
        f"of requested {requested_top_k}."
    )
    if base_lines and base_lines[-1].startswith("Summary:"):
        base_lines[-1] = summary
    else:
        base_lines.append(summary)
    return "\n".join(base_lines)


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest, cols: Tuple[Collection, Collection] = Depends(collections_dep)) -> SearchResponse:
    cand_col, mem_col = cols
    jd = rbv2.parse_jd(req.query)

    expr = _build_expr_from_jd(jd)
    
    # 🔥 CALL UNIFIED_SEARCH WITH FRESHNESS
    rows = rbv2.unified_search(
        cand_col,
        jd_text=jd.get("raw"),
        expr=expr,
        need=req.need or jd.get("must_skills"),
        any_of=req.any_of,
        titles_any=req.titles_any,
        domains_any=req.domains_any,
        clouds_any=req.clouds_any or [c.lower() for c in jd.get("clouds", [])],
        top_k=req.top_k,
        freshness_days=req.freshness_days,  # 🔥 PASS FRESHNESS
    )
    chosen = rows[: req.top_k]

    client_id = rbv2.upsert_client(mem_col, req.client_company.strip() or "UnknownCo", req.contact_key.strip() or "unknown")
    slate_text = _format_slate_with_limit(chosen, jd, req.top_k)

    try:
        rbv2.append_interaction(mem_col, client_id, req.query, slate_text)
    except Exception as exc:
        logger.warning("Could not append client memory: %s", exc)

    memory_snippet = None
    try:
        memory_snippet = rbv2.latest_memory_snippet(mem_col, client_id)
    except Exception:
        pass

    hits: List[CandidateHit] = []
    for r in chosen:
        hits.append(CandidateHit(data=r, why=rbv2.evidence_lines(r, jd)))

    return SearchResponse(
        client_id=client_id,
        total_candidates=len(chosen),
        slate=slate_text,
        candidates=hits,
        memory_snippet=memory_snippet,
        parsed_jd=jd,
    )


@app.get("/memory/{client_id}", tags=["memory"])
def read_memory(client_id: str, cols: Tuple[Collection, Collection] = Depends(collections_dep)) -> Dict[str, Any]:
    _, mem_col = cols
    try:
        snippet = rbv2.latest_memory_snippet(mem_col, client_id)
        row = rbv2._get_mem_row(mem_col, client_id)  # type: ignore[attr-defined]
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Memory not found for {client_id}") from exc
    return {"client_id": client_id, "memory_snippet": snippet, "raw": row}


# 🔥 NEW ENDPOINT: Get freshness presets
@app.get("/freshness-presets", tags=["config"])
def get_freshness_presets():
    """Get available freshness filter presets for UI."""
    from .config import FRESHNESS_PRESETS
    return {"presets": FRESHNESS_PRESETS}