"""Core ANN + scalar retrieval utilities for recruiter brain."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from pymilvus import MilvusException

from recruiterbrain.shared_config import (
    COLLECTION,
    CORE_FIELDS,
    DOMAIN_SYNONYMS,
    EF_SEARCH,
    EMBED_MODEL,
    FIELDS,
    INSIGHT_DEFAULT_K,
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

logger = logging.getLogger(__name__)


def build_expr_from_plan(plan: Dict[str, Any]) -> str | None:
    """Build a Milvus filter expression for cheap server-side pruning.
    exprs: List[str] = []
    industry = (plan.get("industry_equals") or "").strip()
    if industry:
        safe = industry.replace('"', '\"')
        exprs.append(f'primary_industry == "{safe}"') """
    exprs: List[str] = []
    stage = plan.get("require_career_stage")
    if stage and stage not in ("", "Any"):
         safe = stage.replace('"', '\"')
         exprs.append(f'career_stage == "{safe}"')

    if not exprs:
        return None
    return " and ".join(exprs)

"""
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
        
        if keywords and not any(kw in bag for kw in keywords):
                 continue


        # Relaxed keyword filter:
        # - If only a few keywords (<= 4), require at least ONE to appear.
        # - If many keywords, don't hard-filter; let ANN similarity handle it.
        if networking_required and not NETWORK_PAT.search(bag):
            continue

        filtered.append(row)
    return filtered """
def post_filter(rows: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    # --- NORMALIZE required_tools ---
    req = plan.get("required_tools", [])
    if isinstance(req, str):
        # split by comma or whitespace
        req = [t.strip().lower() for t in re.split(r"[, ]+", req) if t.strip()]
        plan["required_tools"] = req

    # --- NORMALIZE must_have_keywords ---
    kws = plan.get("must_have_keywords", [])
    if isinstance(kws, str):
        kws = [k.strip().lower() for k in re.split(r"[, ]+", kws) if k.strip()]
        plan["must_have_keywords"] = kws
    keywords = [kw.lower() for kw in plan.get("must_have_keywords", []) if kw]
    rd = plan.get("require_domains", [])
    if isinstance(rd, str):
         rd = [t.strip().lower() for t in re.split(r"[, ]+", rd) if t.strip()]
         plan["require_domains"] = rd
    require_domains = [dom.lower() for dom in plan.get("require_domains", []) if dom]

    networking_required = bool(plan.get("networking_required"))

    # Case-insensitive, whitespace-trimmed industry & stage
    ind_val = plan.get("industry_equals")
    if isinstance(ind_val, list):
        # pick first or join; up to you
        if ind_val:
            industry = ind_val[0].strip().lower()
        else:
            industry = ""
    else:
        industry = (ind_val or "").strip().lower()

    stage = plan.get("require_career_stage")
    stage = (stage or "").strip()
    if not stage or stage == "Any":
        stage = None

    filtered: List[Dict[str, Any]] = []

    for row in rows:
        # === Industry filter (lenient) ===
        if industry:
            row_industry = (row.get("primary_industry") or "").strip().lower()
            # If no primary_industry, optionally fall back to sub_industries/domains
            if row_industry != industry:
                # Optional: check sub_industries/domains_of_expertise
                subs = str(row.get("sub_industries", "")).lower()
                doms_for_industry = str(row.get("domains_of_expertise", "")).lower()
                if industry not in subs and industry not in doms_for_industry:
                    continue

        # === Career stage filter ===
        if stage:
            row_stage = (row.get("career_stage") or "").strip()
            if row_stage != stage:
                continue

        # === Domain filter ===
        doms = str(row.get("domains_of_expertise", "")).lower()
        expanded_require_domains = set(require_domains)
        for d in require_domains:
            expanded_require_domains.update(
        synonym.lower() for synonym in DOMAIN_SYNONYMS.get(d, [])
    )

        if expanded_require_domains and not any(dom in doms for dom in expanded_require_domains):
             continue

        # === Keywords filter (relaxed ANY match) ===
        bag = bag_from_entity(row).lower()
        required_tools = [t.lower() for t in plan.get("required_tools", []) if t]
        if required_tools and not all(tool in bag for tool in required_tools):
             continue

        # === Keywords filter (relaxed ANY match) ===
        if keywords and not any(kw in bag for kw in keywords):
             continue

        # === Networking requirement ===
        if networking_required and not NETWORK_PAT.search(bag):
            continue

        filtered.append(row)

    return filtered


ACTIVE_FIELDS: List[str] = list(FIELDS)


def _query_with_fallback(client, *, expr: str | None):
    global ACTIVE_FIELDS
    try:
        return client.query(
            collection_name=COLLECTION,
            filter=expr,
            output_fields=ACTIVE_FIELDS,
            limit=PREFILTER_LIMIT,
        )
    except MilvusException as exc:
        if ACTIVE_FIELDS != CORE_FIELDS and "Field" in str(exc):
            logger.warning(
                "Milvus query failed with expanded fields (%s); retrying with core schema",
                exc,
            )
            ACTIVE_FIELDS = list(CORE_FIELDS)
            return client.query(
                collection_name=COLLECTION,
                filter=expr,
                output_fields=ACTIVE_FIELDS,
                limit=PREFILTER_LIMIT,
            )
        logger.exception("Milvus query failed")
        raise
def ann_search(plan: Dict[str, Any]):
    """Run ANN search plus scalar pre/post filters."""
    if "question" not in plan:
        raise ValueError("plan must include original question text under the 'question' key")

    vec_field = plan.get("vector_field") or VECTOR_FIELD_DEFAULT
    if (plan.get("intent") or "").lower() == "insight":
        top_k = int(plan.get("k") or INSIGHT_DEFAULT_K)
    else:
        top_k = int(plan.get("top_k") or plan.get("k") or TOP_K)

    cand_col = get_milvus_client()
    encoder = get_encoder()

    expr = build_expr_from_plan(plan)
    logger.debug("Scalar expr from plan: %s", expr)

    base_rows = _query_with_fallback(cand_col, expr=expr) or []
    logger.debug("Fetched %d scalar rows for plan", len(base_rows))

    base_by_id = {row["candidate_id"]: row for row in base_rows if row.get("candidate_id") is not None}

    embed_text = plan.get("embedding_query") or plan.get("question")
    if not embed_text:
        raise ValueError("plan must include 'embedding_query' or 'question' text for embedding")

    qtext = apply_model_prefix(embed_text, EMBED_MODEL, is_query=True)
    qvec = encoder.encode([qtext], normalize_embeddings=True)[0].tolist()
    ann_limit = max(top_k * 4, 100)
    logger.info(
        "Running ANN search (field=%s, top_k=%s, ann_limit=%s, embed_len=%d)",
        vec_field,
        top_k,
        ann_limit,
        len(embed_text),
    )

    hits = cand_col.search(
        collection_name=COLLECTION,
        data=[qvec],
        anns_field=vec_field,
        search_params={"metric_type": METRIC, "params": {"ef": EF_SEARCH}},
        limit=ann_limit,
        output_fields=["candidate_id"],
    )
    hit_list = hits[0] if hits else []
    logger.info("ann_search: raw ANN hits=%d", len(hit_list))

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

    logger.info("ann_search: unique candidate IDs from ANN=%d", len(vec_ids))

    merged = [base_by_id[cid] for cid in vec_ids if cid in base_by_id]
    logger.info("ann_search: merged ANN+scalar rows=%d", len(merged))

    filtered = post_filter(merged, plan)
    logger.info("ann_search: post_filter rows=%d", len(filtered))

    if not filtered and plan.get("must_have_keywords"):
        logger.info(
            "No rows after post_filter with keywords=%s; retrying without must_have_keywords",
            plan.get("must_have_keywords"),
        )
        plan_no_kw = dict(plan)
        plan_no_kw["must_have_keywords"] = []
        filtered = post_filter(merged, plan_no_kw)
        logger.debug("Post-filter (no keywords) rows=%d", len(filtered))
        if not filtered and plan.get("must_have_keywords"):
              logger.info(
              "No rows after post_filter with keywords=%s; retrying without must_have_keywords",
              plan.get("must_have_keywords"),
        )
        plan_no_kw = dict(plan)
        plan_no_kw["must_have_keywords"] = []
        filtered = post_filter(merged, plan_no_kw)
        logger.debug("Post-filter (no keywords) rows=%d", len(filtered))

    limited = filtered[:top_k]
    if not filtered and plan.get("require_domains"):
         logger.info(
          "No rows after post_filter with require_domains=%s; retrying without domain constraint",
          plan["require_domains"],
        )
         plan_no_dom = dict(plan)
         plan_no_dom["require_domains"] = []
         filtered = post_filter(merged, plan_no_dom)
         logger.debug("Post-filter (no domains) rows=%d", len(filtered))

    if not filtered and plan.get("industry_equals"):
        logger.info(
            "No rows after post_filter with industry_equals=%s; retrying without industry constraint",
            plan["industry_equals"],
        )
    return attach_sim_scores(limited, hit_list), len(filtered)



"""
def ann_search(plan: Dict[str, Any]):
    
    if "question" not in plan:
        raise ValueError("plan must include original question text under the 'question' key")

    vec_field = plan.get("vector_field") or VECTOR_FIELD_DEFAULT
    if (plan.get("intent") or "").lower() == "insight":
        top_k = int(plan.get("k") or INSIGHT_DEFAULT_K)
    else:
        top_k = int(plan.get("top_k") or plan.get("k") or TOP_K)

    cand_col = get_milvus_client()
    encoder = get_encoder()

    expr = build_expr_from_plan(plan)
    logger.debug(
        "Running scalar query with expr=%s, fields=%s", expr, ACTIVE_FIELDS
    )
    base_rows = _query_with_fallback(cand_col, expr=expr) or []
    logger.debug("Fetched %d scalar rows for plan", len(base_rows))
    base_by_id = {row["candidate_id"]: row for row in base_rows if row.get("candidate_id") is not None}

    embed_text = plan.get("embedding_query") or plan.get("question")
    if not embed_text:
        raise ValueError("plan must include 'embedding_query' or 'question' text for embedding")

    qtext = apply_model_prefix(embed_text, EMBED_MODEL, is_query=True)
    qvec = encoder.encode([qtext], normalize_embeddings=True)[0].tolist()
    ann_limit = max(top_k * 4, 100)
    logger.debug(
        "Running ANN search (field=%s, top_k=%s, ann_limit=%s)",
        vec_field,
        top_k,
        ann_limit,
    )

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
    logger.debug("Merged %d ANN hits with scalar rows", len(merged))
    filtered = post_filter(merged, plan)
    logger.debug("Post-filtered down to %d rows", len(filtered))
    limited = filtered[:top_k]
    return attach_sim_scores(limited, hit_list), len(filtered)
"""

__all__ = ["ann_search", "build_expr_from_plan", "post_filter"]
