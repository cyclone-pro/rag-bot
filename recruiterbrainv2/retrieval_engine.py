"""Core V2 retrieval pipeline (vector search + hybrid ranking)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import (
    COLLECTION,
    EF_SEARCH,
    FINAL_RETURN,
    KEYWORD_WEIGHT,
    METRIC,
    SEARCH_OUTPUT_FIELDS,
    VECTOR_TOP_K,
    VECTOR_WEIGHT,
    get_encoder,
    get_milvus_client,
)
from .ranker import compute_match_details, hybrid_rank
from .skill_extractor import extract_requirements

logger = logging.getLogger(__name__)

# Assumes your Milvus collection stores the embedding under this field name.
VECTOR_FIELD = "summary_embedding"


def _vectorize(text: str) -> Optional[List[float]]:
    """Encode text to a normalized embedding."""
    try:
        enc = get_encoder()
        return enc.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    except Exception as exc:  # pragma: no cover - depends on env
        logger.warning("Embedding failed: %s", exc)
        return None


def _score_from_distance(distance: Any) -> float:
    """Convert Milvus distance to similarity (higher is better)."""
    try:
        return 1.0 - float(distance)
    except Exception:
        return 0.0


def _milvus_search(
    vector: List[float],
    career_stage: Optional[str],
    industry: Optional[str],
) -> List[Dict[str, Any]]:
    """Run a Milvus vector search and return raw candidate dicts."""
    client = get_milvus_client()
    search_params = {"metric_type": METRIC, "params": {"ef": EF_SEARCH}}

    filters = []
    if career_stage:
        filters.append(f'career_stage == "{career_stage}"')
    if industry:
        filters.append(f'primary_industry == "{industry}"')
    expr = " and ".join(filters) if filters else None

    try:
        res = client.search(
            collection_name=COLLECTION,
            data=[vector],
            anns_field=VECTOR_FIELD,
            search_params=search_params,
            limit=VECTOR_TOP_K,
            output_fields=SEARCH_OUTPUT_FIELDS,
            filter=expr,
        )
    except Exception as exc:
        logger.exception("Milvus search failed")
        raise

    # MilvusClient returns a list per query; we only send one vector
    hits = res[0] if res else []
    out: List[Dict[str, Any]] = []
    for hit in hits:
        entity = hit.get("entity") or {}
        if not isinstance(entity, dict):
            continue
        entity = dict(entity)
        entity["vector_score"] = _score_from_distance(hit.get("distance"))
        out.append(entity)
    return out


def search_candidates_v2(
    query: str,
    *,
    top_k: int = FINAL_RETURN,
    career_stage: Optional[str] = None,
    industry: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the V2 retrieval pipeline.

    - Extract structured requirements from the query (skills, seniority).
    - Vector search Milvus for ANN candidates.
    - Hybrid rank (vector + keyword overlap) and attach match diagnostics.
    """
    requirements = extract_requirements(query)
    vector = _vectorize(query)
    if vector is None:
        raise RuntimeError("Embedding model unavailable; cannot run V2 search.")

    candidates = _milvus_search(vector, career_stage, industry)

    ranked = hybrid_rank(
        candidates,
        requirements.get("must_have_skills", []),
        vector_weight=VECTOR_WEIGHT,
        keyword_weight=KEYWORD_WEIGHT,
    )

    top_ranked = [cand for cand, _ in ranked[: max(1, top_k or FINAL_RETURN)]]
    for cand in top_ranked:
        cand["match"] = compute_match_details(cand, requirements.get("must_have_skills", []))
        cand["summary"] = cand.get("semantic_summary") or cand.get("keywords_summary") or ""

    return {
        "query": query,
        "requirements": requirements,
        "search_mode": "vector",
        "total_found": len(candidates),
        "candidates": top_ranked,
    }
