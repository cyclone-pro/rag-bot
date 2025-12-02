"""Core ANN + scalar retrieval utilities for recruiter brain."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence
import re
from pymilvus import MilvusException

from recruiterbrain.shared_config import (
    COLLECTION,
    CORE_FIELDS,
    INDUSTRY_SYNONYMS,
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
STAGE_RANK = {
    "Entry": 1,
    "Mid": 2,
    "Senior": 3,
    "Lead/Manager": 4,
    "Director+": 5,
}

def _stage_rank(label: str) -> int:
    return STAGE_RANK.get((label or "").strip(), 0)


   
def build_expr_from_plan(plan: Dict[str, Any]) -> str | None:
    """Build Milvus scalar-filter expression based on career stage (min) and industry."""
    exprs: List[str] = []

    # --- Industry filter for primary_industry (string or list) ---
    ind_val = plan.get("industry_equals")
    industries: List[str] = []

    if isinstance(ind_val, str):
        ind_val = ind_val.strip()
        if ind_val:
            industries = [ind_val]
    elif isinstance(ind_val, list):
        industries = [x.strip() for x in ind_val if x and x.strip()]

    if industries:
        safe_industries = [ind.replace('"', '""') for ind in industries]
        safe_inds = ",".join(f'"{ind}"' for ind in safe_industries)
        exprs.append(f"primary_industry in [{safe_inds}]")

    # --- Career stage → minimum rank ---
    stage = (plan.get("require_career_stage") or "").strip()
    if stage and stage not in ("", "Any"):
        min_rank = _stage_rank(stage)
        if min_rank > 0:
            allowed = [s for s, r in STAGE_RANK.items() if r >= min_rank]
            if allowed:
                safe_labels = ",".join(f'"{s}"' for s in allowed)
                exprs.append(f"career_stage in [{safe_labels}]")

    if not exprs:
        return None
    return " and ".join(exprs)




def post_filter(rows: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    # --- NORMALIZE must_have_keywords ---
    kws = plan.get("must_have_keywords", [])
    if isinstance(kws, str):
        kws = [k.strip().lower() for k in re.split(r"[, ]+", kws) if k.strip()]
        plan["must_have_keywords"] = kws
    keywords = [kw.lower() for kw in plan.get("must_have_keywords", []) if kw]

    # --- NORMALIZE require_domains ---
    rd = plan.get("require_domains", [])
    if isinstance(rd, str):
        rd = [t.strip().lower() for t in re.split(r"[, ]+", rd) if t.strip()]
        plan["require_domains"] = rd
    require_domains = [dom.lower() for dom in plan.get("require_domains", []) if dom]

    networking_required = bool(plan.get("networking_required"))

    # --- NORMALIZE industry_equals (string or list) ---
    ind_val = plan.get("industry_equals")
    if isinstance(ind_val, list):
        if ind_val:
            industry = ind_val[0].strip()
        else:
            industry = ""
    else:
        industry = (ind_val or "").strip()

    # --- Normalize required career stage into a minimum rank ---
    required_stage = (plan.get("require_career_stage") or "").strip()
    if not required_stage or required_stage == "Any":
        min_stage_rank = None
    else:
        min_stage_rank = _stage_rank(required_stage)

    filtered: List[Dict[str, Any]] = []

    for row in rows:
        # === Industry filter (lenient, with synonyms) ===
        if industry:
            row_industry = (row.get("primary_industry") or "").strip()
            if row_industry != industry:
                # try industry synonyms
                synonyms = INDUSTRY_SYNONYMS.get(industry, [])
                if row_industry not in synonyms:
                    # optional: also look into sub_industries/domains
                    subs = str(row.get("sub_industries", "")).lower()
                    doms_for_industry = str(row.get("domains_of_expertise", "")).lower()
                    if industry.lower() not in subs and industry.lower() not in doms_for_industry:
                        continue

        # === Career stage filter (minimum threshold) ===
        if min_stage_rank is not None:
            row_stage_label = (row.get("career_stage") or "").strip()
            if _stage_rank(row_stage_label) < min_stage_rank:
                continue

        # === Domain filter with synonyms ===
        doms = str(row.get("domains_of_expertise", "")).lower()
        expanded_require_domains = set(require_domains)
        for d in require_domains:
            for synonym in DOMAIN_SYNONYMS.get(d, []):
                expanded_require_domains.add(synonym.lower())

        if expanded_require_domains and not any(dom in doms for dom in expanded_require_domains):
            continue

        # === Keyword filter (relaxed ANY match) ===
        bag = bag_from_entity(row).lower()
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
    """Run ANN search plus scalar post-filters with hybrid keyword boosting."""
    if "question" not in plan:
        raise ValueError("plan must include original question text under the 'question' key")

    vec_field = plan.get("vector_field") or VECTOR_FIELD_DEFAULT
    if (plan.get("intent") or "").lower() == "insight":
        top_k = int(plan.get("k") or INSIGHT_DEFAULT_K)
    else:
        top_k = int(plan.get("top_k") or plan.get("k") or TOP_K)

    cand_col = get_milvus_client()
    encoder = get_encoder()

    # Optional scalar constraint (industry, career_stage, etc.)
    expr = build_expr_from_plan(plan)
    logger.debug("Scalar expr from plan: %s", expr)

    # Check if this is a keyword-heavy query (like Salesforce with many specific tools)
    required_tools = plan.get("required_tools") or []
    is_keyword_heavy = len(required_tools) >= 5  # 5+ specific tools mentioned
    
    # HYBRID RETRIEVAL MODE
    if is_keyword_heavy:
        logger.info("Keyword-heavy query detected (%d tools) - using hybrid retrieval", len(required_tools))
        
        # Step 1: Get keyword-based candidates using scalar filter
        keyword_expr_parts = []
        for tool in required_tools[:7]:  # Use top 7 tools for keyword matching
            # Escape the tool name and search in both skills_extracted and tools_and_technologies
            safe_tool = tool.replace('"', '""')
            keyword_expr_parts.append(
                f'(skills_extracted like "%{safe_tool}%" or tools_and_technologies like "%{safe_tool}%")'
            )
        
        keyword_expr = " or ".join(keyword_expr_parts)
        if expr:
            keyword_expr = f"({expr}) and ({keyword_expr})"
        
        logger.info("Keyword filter: %s", keyword_expr[:200])
        
        # Query with keyword filter to get exact-match candidates
        try:
            keyword_candidates = _query_with_fallback(cand_col, expr=keyword_expr) or []
            logger.info("Keyword-based retrieval found %d candidates", len(keyword_candidates))
        except Exception as e:
            logger.warning("Keyword retrieval failed: %s, falling back to vector-only", e)
            keyword_candidates = []
        
        # Get candidate IDs from keyword search
        keyword_ids = {row["candidate_id"] for row in keyword_candidates}
        
    else:
        keyword_candidates = []
        keyword_ids = set()

    # --- Build query embedding for ANN search ---
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

    # --- ANN search, optionally with scalar filter ---
    search_kwargs: Dict[str, Any] = {
        "collection_name": COLLECTION,
        "data": [qvec],
        "anns_field": vec_field,
        "search_params": {"metric_type": METRIC, "params": {"ef": EF_SEARCH}},
        "limit": ann_limit,
        "output_fields": ["candidate_id"],
    }
    if expr:
        search_kwargs["filter"] = expr

    hits = cand_col.search(**search_kwargs)
    hit_list = hits[0] if hits else []
    logger.info("ann_search: raw ANN hits=%d", len(hit_list))

    # --- Extract candidate_ids from ANN hits ---
    vec_ids: List[Any] = []
    for hit in hit_list:
        if isinstance(hit, dict):
            entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
            cid = entity.get("candidate_id") or hit.get("candidate_id") or hit.get("id")
        else:
            cid = getattr(hit, "candidate_id", None) or getattr(hit, "id", None)
        if cid is None:
            continue
        vec_ids.append(cid)

    # preserve ANN order but dedupe
    unique_ann_ids: List[Any] = list(dict.fromkeys(vec_ids))
    logger.info("ann_search: unique candidate IDs from ANN=%d", len(unique_ann_ids))

    # --- MERGE STRATEGY: Keyword matches first, then ANN matches ---
    if is_keyword_heavy and keyword_ids:
        # Prioritize keyword matches (exact skill matches)
        # These go first because they're guaranteed to have the required skills
        merged_ids = []
        
        # 1. Add keyword-matched candidates first (they have exact skills)
        for row in keyword_candidates:
            cid = row["candidate_id"]
            if cid not in merged_ids:
                merged_ids.append(cid)
        
        # 2. Add ANN candidates that aren't already in keyword results
        for cid in unique_ann_ids:
            if cid not in keyword_ids and cid not in merged_ids:
                merged_ids.append(cid)
        
        # Limit to reasonable size for hydration
        merged_ids = merged_ids[:max(ann_limit * 2, 200)]
        logger.info("Hybrid merge: %d keyword + %d new from ANN = %d total", 
                    len(keyword_ids), len(merged_ids) - len(keyword_ids), len(merged_ids))
    else:
        # No keyword candidates or not keyword-heavy - use ANN results only
        merged_ids = unique_ann_ids

    if not merged_ids:
        return [], 0

    # --- Hydrate merged candidates by candidate_id ---
    id_literals = ",".join(
        f'"{cid}"' if isinstance(cid, str) else str(cid) for cid in merged_ids
    )
    hydrate_expr = f"candidate_id in [{id_literals}]"

    base_rows = _query_with_fallback(cand_col, expr=hydrate_expr) or []
    base_by_id = {
        row["candidate_id"]: row
        for row in base_rows
        if row.get("candidate_id") is not None
    }

    # Preserve the priority order (keyword matches first)
    merged = [base_by_id[cid] for cid in merged_ids if cid in base_by_id]
    logger.info("ann_search: merged ANN+scalar rows=%d", len(merged))

    # 1) normal post_filter
    filtered = post_filter(merged, plan)
    logger.info("ann_search: post_filter rows=%d", len(filtered))

    # 2) relax must_have_keywords if nothing
    if not filtered and plan.get("must_have_keywords"):
        logger.info(
            "No rows after post_filter with keywords=%s; retrying without must_have_keywords",
            plan.get("must_have_keywords"),
        )
        plan_no_kw = dict(plan)
        plan_no_kw["must_have_keywords"] = []
        filtered = post_filter(merged, plan_no_kw)
        logger.debug("Post-filter (no keywords) rows=%d", len(filtered))

    # 3) relax require_domains if still nothing
    if not filtered and plan.get("require_domains"):
        logger.info(
            "No rows after post_filter with require_domains=%s; retrying without domain constraint",
            plan["require_domains"],
        )
        plan_no_dom = dict(plan)
        plan_no_dom["require_domains"] = []
        filtered = post_filter(merged, plan_no_dom)
        logger.debug("Post-filter (no domains) rows=%d", len(filtered))

    # 4) relax industry_equals if still nothing
    if not filtered and plan.get("industry_equals"):
        logger.info(
            "No rows after post_filter with industry_equals=%s; retrying without industry constraint",
            plan["industry_equals"],
        )
        plan_no_ind = dict(plan)
        plan_no_ind["industry_equals"] = None
        filtered = post_filter(merged, plan_no_ind)
        logger.debug("Post-filter (no industry) rows=%d", len(filtered))

    limited = filtered[:top_k]
    return attach_sim_scores(limited, hit_list), len(filtered)




__all__ = ["ann_search", "build_expr_from_plan", "post_filter"]
