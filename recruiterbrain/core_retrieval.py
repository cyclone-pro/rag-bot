"""Core ANN + scalar retrieval utilities for recruiter brain."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from recruiterbrain.shared_config import (
    COLLECTION,
    EF_SEARCH,
    EMBED_MODEL,
    FIELDS,
    METRIC,
    PREFILTER_LIMIT,
    TOP_K,
    VECTOR_FIELD_DEFAULT,
    get_encoder,
    get_milvus_client,
)
from recruiterbrain.shared_utils import (
    NETWORK_PAT,
    apply_model_prefix,
    attach_sim_scores,
    bag_from_entity,
)


def build_expr_from_plan(plan: Dict[str, Any]) -> str | None:
    """Build a Milvus filter expression for cheap server-side pruning."""
    exprs: List[str] = []
    industry = (plan.get("industry_equals") or "").strip()
    if industry:
        safe = industry.replace('"', '\"')
        exprs.append(f'primary_industry == "{safe}"')

    stage = plan.get("require_career_stage")
    if stage and stage not in ("", "Any"):
        safe = stage.replace('"', '\"')
        exprs.append(f'career_stage == "{safe}"')

    if not exprs:
        return None
    return " and ".join(exprs)


def post_filter(rows: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    keywords = [kw.lower() for kw in plan.get("must_have_keywords", []) if kw]
    require_domains = [dom.lower() for dom in plan.get("require_domains", []) if dom]
    networking_required = bool(plan.get("networking_required"))
    industry = (plan.get("industry_equals") or "").strip()
    stage = plan.get("require_career_stage")
    stage = None if not stage or stage == "Any" else stage

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if industry and (row.get("primary_industry") or "") != industry:
            continue
        if stage and (row.get("career_stage") or "") != stage:
            continue

        doms = str(row.get("domains_of_expertise", "")).lower()
        if require_domains and not any(dom in doms for dom in require_domains):
            continue

        bag = bag_from_entity(row).lower()
        if keywords and not all(kw in bag for kw in keywords):
            continue
        if networking_required and not NETWORK_PAT.search(bag):
            continue

        filtered.append(row)
    return filtered


def ann_search(plan: Dict[str, Any]):
    """Run ANN search plus scalar pre/post filters."""
    if "question" not in plan:
        raise ValueError("plan must include original question text under the 'question' key")

    vec_field = plan.get("vector_field") or VECTOR_FIELD_DEFAULT
    top_k = int(plan.get("top_k", TOP_K))

    cand_col = get_milvus_client()
    encoder = get_encoder()

    expr = build_expr_from_plan(plan)
    base_rows = cand_col.query(
        collection_name=COLLECTION,
        filter=expr,
        output_fields=FIELDS,
        limit=PREFILTER_LIMIT,
    ) or []
    base_by_id = {row["candidate_id"]: row for row in base_rows if row.get("candidate_id") is not None}

    qtext = apply_model_prefix(plan["question"], EMBED_MODEL, is_query=True)
    qvec = encoder.encode([qtext], normalize_embeddings=True)[0].tolist()

    ann_limit = max(top_k * 4, 100)
    hits = cand_col.search(
        collection_name=COLLECTION,
        data=[qvec],
        anns_field=vec_field,
        search_params={"metric_type": METRIC, "params": {"ef": EF_SEARCH}},
        limit=ann_limit,
        output_fields=["candidate_id"],
    )
    hit_list = hits[0] if hits else []

    vec_ids = []
    for hit in hit_list:
        if isinstance(hit, dict):
            entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
            cid = entity.get("candidate_id") or hit.get("candidate_id") or hit.get("id")
        else:
            cid = getattr(hit, "candidate_id", None) or getattr(hit, "id", None)
        if cid is None:
            continue
        vec_ids.append(cid)

    merged = [base_by_id[cid] for cid in vec_ids if cid in base_by_id]
    filtered = post_filter(merged, plan)
    limited = filtered[:top_k]
    return attach_sim_scores(limited, hit_list), len(filtered)


__all__ = ["ann_search", "build_expr_from_plan", "post_filter"]
