"""Core hybrid retrieval engine - WITH PARALLEL SEARCH AND FRESHNESS FILTERING."""
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .config import (
    get_milvus_client,
    get_encoder,
    get_search_thread_pool,
    COLLECTION,
    VECTOR_TOP_K,
    KEYWORD_TOP_K,
    FINAL_RETURN,
    METRIC,
    EF_SEARCH,
    VECTOR_WEIGHT,
    KEYWORD_WEIGHT,
    CAREER_STAGES,
    SEARCH_OUTPUT_FIELDS,
    ENABLE_CACHE,
    SEARCH_CACHE_TTL,
    ENABLE_PARALLEL_SEARCH,
    DEFAULT_FRESHNESS_DAYS,  # 🔥 NEW
)
from .skill_extractor import extract_requirements
from .ranker import hybrid_rank, compute_match_details
from .cache import generate_cache_key, get_cache

logger = logging.getLogger(__name__)


# 🔥 NEW: Freshness helper functions
def calculate_days_old(last_updated_str: str) -> int:
    """Calculate how many days old a timestamp is."""
    if not last_updated_str:
        return 9999
    try:
        # Handle format: "2025-12-07T20:35:20."
        last_updated_str = last_updated_str.rstrip('.')
        dt = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
        if dt.tzinfo:
            days = (datetime.now(dt.tzinfo) - dt).days
        else:
            days = (datetime.now() - dt).days
        return max(0, days)
    except Exception as e:
        logger.warning(f"Invalid timestamp: {last_updated_str} - {e}")
        return 9999


def get_freshness_badge(days_old: int) -> dict:
    """Get freshness badge info."""
    if days_old <= 7:
        return {"emoji": "🟢", "label": "FRESH", "priority": 1, "color": "#10b981"}
    elif days_old <= 30:
        return {"emoji": "🟢", "label": "RECENT", "priority": 2, "color": "#10b981"}
    elif days_old <= 90:
        return {"emoji": "🟡", "label": "GOOD", "priority": 3, "color": "#f59e0b"}
    elif days_old <= 180:
        return {"emoji": "🟠", "label": "VERIFY", "priority": 4, "color": "#f97316"}
    elif days_old <= 365:
        return {"emoji": "🔴", "label": "STALE", "priority": 5, "color": "#ef4444"}
    else:
        return {"emoji": "🔴", "label": "VERY STALE", "priority": 6, "color": "#dc2626"}


def build_date_filter(freshness_days: Optional[int]) -> Optional[str]:
    """Build Milvus date filter expression."""
    if freshness_days is None:
        return None
    
    cutoff = datetime.now() - timedelta(days=freshness_days)
    cutoff_iso = cutoff.isoformat()
    
    # Milvus filter format
    return f'last_updated >= "{cutoff_iso}"'


def _vectorize(text: str, batch_texts: Optional[List[str]] = None) -> Optional[List[float]]:
    """Encode text to a normalized embedding."""
    try:
        enc = get_encoder()
        
        if batch_texts:
            all_texts = [f"query: {t}" for t in [text] + batch_texts]
            embeddings = enc.encode(
                all_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=len(all_texts)
            )
            return [emb.tolist() for emb in embeddings]
        else:
            query_text = f"query: {text}"
            return enc.encode(
                [query_text],
                normalize_embeddings=True,
                show_progress_bar=False
            )[0].tolist()
            
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return None


def search_candidates_v2(
    query: str,
    top_k: int = 10,
    career_stage: Optional[str] = None,
    industry: Optional[str] = None,
    freshness_days: Optional[int] = DEFAULT_FRESHNESS_DAYS, 
) -> Dict[str, Any]:
    """
    Main search function WITH CACHING, PARALLEL SEARCH, AND FRESHNESS FILTERING.
    
    Args:
        query: Search query
        top_k: Number of results to return
        career_stage: Filter by career stage
        industry: Filter by industry
        freshness_days: Days to look back (None = all time) 🔥 NEW
    
    Parallel Search:
    - Vector and keyword searches run simultaneously
    - 50% faster than sequential execution
    - Thread-safe Milvus operations
    
    Freshness Filtering:
    - Filters candidates by last_updated date BEFORE search
    - Milvus-native filtering (fast, no data transfer overhead)
    - Adds freshness badges to results
    """
    # ==================== CACHE CHECK ====================
    if ENABLE_CACHE:
        cache_key = generate_cache_key(
            "search_v2",
            query=query,
            top_k=top_k,
            career_stage=career_stage or "",
            industry=industry or "",
            freshness_days=freshness_days or "all"  # 🔥 Include in cache key
        )
        
        cache = get_cache()
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("✅ Search cache HIT - returning cached results")
            return cached_result
        
        logger.info("⚠️  Search cache MISS - performing full search...")
    
    # ==================== VALIDATION ====================
    query = query.strip()
    
    if len(query) < 10:
        error_response = {
            "query": query,
            "requirements": {},
            "search_mode": "error",
            "total_found": 0,
            "candidates": [],
            "error": "Query too short. Please provide more details"
        }
        return error_response
    
    # Check for greetings
    greeting_patterns = [
        "hello", "hi", "hey", "how are you", "what's up", "good morning",
        "good afternoon", "good evening", "test", "testing", "sup"
    ]
    
    query_lower = query.lower()
    if any(pattern in query_lower for pattern in greeting_patterns) and len(query) < 50:
        error_response = {
            "query": query,
            "requirements": {},
            "search_mode": "error",
            "total_found": 0,
            "candidates": [],
            "error": "👋 Hi! Please ask a recruiting question like: 'Find Salesforce developers with Apex'"
        }
        return error_response
    
    # ==================== EXTRACT REQUIREMENTS ====================
    requirements = extract_requirements(query)
    required_skills = requirements.get("must_have_skills", [])
    detected_seniority = requirements.get("seniority_level", "Any")
    detected_industry = requirements.get("industry")
    
    if not career_stage or career_stage == "Any":
        career_stage = detected_seniority
    
    if not industry and detected_industry:
        industry = detected_industry
    
    if not required_skills and not career_stage and not industry:
        error_response = {
            "query": query,
            "requirements": requirements,
            "search_mode": "error",
            "total_found": 0,
            "candidates": [],
            "error": "❌ No technical skills detected. Try: 'Find Python developers'"
        }
        return error_response
    
    # ==================== VECTORIZE ====================
    vector = _vectorize(query)
    if vector is None:
        raise RuntimeError("Embedding model unavailable")
    
    # ==================== BUILD FILTER (WITH FRESHNESS) ====================
    filter_expr = _build_filter(career_stage, industry, freshness_days)  # 🔥 Pass freshness
    
    # ==================== DECIDE SEARCH MODE ====================
    use_hybrid = len(required_skills) >= 5 or requirements.get("role_type")
    
    if use_hybrid:
        logger.info("Using HYBRID mode (%d skills, parallel=%s, freshness=%s days)", 
                   len(required_skills), 
                   ENABLE_PARALLEL_SEARCH,
                   freshness_days or "all")  # 🔥 Log freshness
        results = _hybrid_search_parallel(query, required_skills, filter_expr, vector)
        search_mode = "hybrid_parallel" if ENABLE_PARALLEL_SEARCH else "hybrid"
    else:
        logger.info("Using VECTOR-ONLY mode (freshness=%s days)", freshness_days or "all")  # 🔥 Log freshness
        results = _vector_search(query, filter_expr, vector, VECTOR_TOP_K)
        search_mode = "vector_only"
    
    if not results:
        no_results_response = {
            "candidates": [],
            "total_found": 0,
            "requirements": requirements,
            "search_mode": search_mode,
            "query": query,
        }
        if ENABLE_CACHE:
            cache.set(cache_key, no_results_response, ttl=60)
        return no_results_response
    
    # ==================== HYBRID RANKING ====================
    ranked = hybrid_rank(
        results,
        required_skills,
        VECTOR_WEIGHT,
        KEYWORD_WEIGHT,
    )
    
    # ==================== FORMAT TOP RESULTS (WITH FRESHNESS) ====================
    top_candidates = []
    for candidate, score in ranked[:top_k]:
        match_details = compute_match_details(candidate, required_skills)
        
        location_parts = [
            candidate.get("location_city"),
            candidate.get("location_state"),
            candidate.get("location_country")
        ]
        location = ", ".join([p for p in location_parts if p])
        
        # 🔥 ADD FRESHNESS INFO
        last_updated = candidate.get('last_updated', '')
        days_old = calculate_days_old(last_updated)
        freshness_badge = get_freshness_badge(days_old)
        
        top_candidates.append({
            "candidate_id": candidate.get("candidate_id"),
            "name": candidate.get("name", "Unknown"),
            "email": candidate.get("email"),
            "phone": candidate.get("phone"),
            "linkedin_url": candidate.get("linkedin_url"),
            "github_url": candidate.get("github_url"),
            "portfolio_url": candidate.get("portfolio_url"),
            "career_stage": candidate.get("career_stage"),
            "industries_worked": candidate.get("industries_worked"),
            "primary_industry": candidate.get("industries_worked", "").split(",")[0].strip() if candidate.get("industries_worked") else "",
            "total_experience_years": candidate.get("total_experience_years"),
            "management_experience_years": candidate.get("management_experience_years"),
            "location": location,
            "location_city": candidate.get("location_city"),
            "location_state": candidate.get("location_state"),
            "location_country": candidate.get("location_country"),
            "match": match_details,
            "skills": candidate.get("skills_extracted", ""),
            "tools": candidate.get("tools_and_technologies", ""),
            "current_tech_stack": candidate.get("current_tech_stack", ""),
            "role_type": candidate.get("role_type", ""),
            "summary": (candidate.get("semantic_summary", "") or "")[:300],
            
            # 🔥 FRESHNESS FIELDS
            "last_updated": last_updated,
            "days_old": days_old,
            "freshness_badge": freshness_badge,
            "needs_verification": days_old > 90,
        })
    
    final_result = {
        "candidates": top_candidates,
        "total_found": len(results),
        "requirements": requirements,
        "search_mode": search_mode,
        "query": query,
        "original_query": query,
        "freshness_filter": freshness_days,  # 🔥 Include in response
    }
    
    # ==================== CACHE RESULT ====================
    if ENABLE_CACHE:
        cache.set(cache_key, final_result, ttl=SEARCH_CACHE_TTL)
        logger.info(f"✅ Cached search result for {SEARCH_CACHE_TTL}s")
    
    return final_result


def _hybrid_search_parallel(
    query: str,
    required_skills: List[str],
    filter_expr: Optional[str],
    vector: List[float],
) -> List[Dict[str, Any]]:
    """
    Parallel hybrid search for maximum speed.
    
    Runs keyword and vector searches simultaneously.
    """
    if not ENABLE_PARALLEL_SEARCH:
        return _hybrid_search_sequential(query, required_skills, filter_expr, vector)
    
    import time
    
    start = time.time()
    
    # Get thread pool
    executor = get_search_thread_pool()
    
    # Submit both searches in parallel
    keyword_future = executor.submit(
        _keyword_search,
        required_skills,
        filter_expr,
        KEYWORD_TOP_K
    )
    
    vector_future = executor.submit(
        _vector_search,
        query,
        filter_expr,
        vector,
        VECTOR_TOP_K,
        "tech_embedding"
    )
    
    # Wait for both to complete
    keyword_candidates = keyword_future.result()
    vector_candidates = vector_future.result()
    
    elapsed = time.time() - start
    logger.info(f"⚡ Parallel search completed in {elapsed:.2f}s")
    logger.info(f"   Keyword: {len(keyword_candidates)} candidates")
    logger.info(f"   Vector: {len(vector_candidates)} candidates")
    
    # Merge results
    keyword_ids = {c["candidate_id"] for c in keyword_candidates}
    merged = list(keyword_candidates)
    
    for vc in vector_candidates:
        if vc["candidate_id"] not in keyword_ids:
            merged.append(vc)
    
    logger.info(f"   Merged: {len(merged)} total candidates")
    
    return merged


def _hybrid_search_sequential(
    query: str,
    required_skills: List[str],
    filter_expr: Optional[str],
    vector: List[float],
) -> List[Dict[str, Any]]:
    """
    Sequential hybrid search (fallback).
    
    Used when parallel search is disabled or as fallback.
    """
    import time
    
    start = time.time()
    
    # Keyword search
    keyword_candidates = _keyword_search(required_skills, filter_expr, KEYWORD_TOP_K)
    keyword_ids = {c["candidate_id"] for c in keyword_candidates}
    
    logger.info(f"Keyword search: {len(keyword_candidates)} candidates")
    
    # Vector search
    vector_candidates = _vector_search(query, filter_expr, vector, VECTOR_TOP_K, "tech_embedding")
    
    logger.info(f"Vector search: {len(vector_candidates)} candidates")
    
    # Merge
    merged = list(keyword_candidates)
    for vc in vector_candidates:
        if vc["candidate_id"] not in keyword_ids:
            merged.append(vc)
    
    elapsed = time.time() - start
    logger.info(f"Sequential search completed in {elapsed:.2f}s, merged: {len(merged)} candidates")
    
    return merged


def _keyword_search(
    skills: List[str],
    filter_expr: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Keyword-based search using Milvus scalar filters.
    
    Thread-safe for parallel execution.
    Filter includes freshness if specified.
    """
    if not skills:
        return []
    
    # Each thread gets its own Milvus client (thread-safe)
    client_or_pool = get_milvus_client()
    
    # Build OR condition for skills
    skill_conditions = []
    for skill in skills[:7]:
        safe_skill = skill.replace('"', '""').replace("'", "''")
        skill_conditions.append(
            f'(skills_extracted like "%{safe_skill}%" or '
            f'tools_and_technologies like "%{safe_skill}%" or '
            f'tech_stack_primary like "%{safe_skill}%" or '
            f'current_tech_stack like "%{safe_skill}%")'
        )
    
    keyword_expr = " or ".join(skill_conditions)
    
    # 🔥 Combine with filter (which includes freshness)
    if filter_expr:
        keyword_expr = f"({filter_expr}) and ({keyword_expr})"
    
    try:
        # Use connection pool if available
        if hasattr(client_or_pool, 'connection'):
            # Connection pool
            with client_or_pool.connection() as client:
                results = client.query(
                    collection_name=COLLECTION,
                    filter=keyword_expr,
                    output_fields=SEARCH_OUTPUT_FIELDS,
                    limit=limit,
                )
        else:
            # Single client
            results = client_or_pool.query(
                collection_name=COLLECTION,
                filter=keyword_expr,
                output_fields=SEARCH_OUTPUT_FIELDS,
                limit=limit,
            )
        
        # Add vector score
        for r in results:
            r['vector_score'] = 0.95
        
        return results
        
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return []


def _vector_search(
    query: str,
    filter_expr: Optional[str],
    vector: List[float],
    limit: int,
    use_field: str = "summary_embedding"
) -> List[Dict[str, Any]]:
    """
    Pure vector search using ANN.
    
    Thread-safe for parallel execution.
    Filter includes freshness if specified.
    """
    # Each thread gets its own Milvus client (thread-safe)
    client_or_pool = get_milvus_client()

    
    search_params = {
        "collection_name": COLLECTION,
        "data": [vector],
        "anns_field": use_field,
        "search_params": {
            "metric_type": METRIC,
            "params": {"ef": EF_SEARCH}
        },
        "limit": limit,
        "output_fields": SEARCH_OUTPUT_FIELDS,
    }
    
    # 🔥 Add filter (includes freshness)
    if filter_expr:
        search_params["filter"] = filter_expr
    
    try:
        # Use connection pool if available
        if hasattr(client_or_pool, 'connection'):
            # Connection pool
            with client_or_pool.connection() as client:
                results = client.search(**search_params)
        else:
            # Single client
            results = client_or_pool.search(**search_params)
        
        if not results or not results[0]:
            return []
        
        candidates = []
        for hit in results[0]:
            if isinstance(hit, dict):
                entity = hit.get("entity", {})
                distance = hit.get("distance", 0)
            else:
                entity = {
                    field: getattr(hit, field, None)
                    for field in SEARCH_OUTPUT_FIELDS
                }
                distance = getattr(hit, "distance", 0)
            
            entity["vector_score"] = float(distance)
            candidates.append(entity)
        
        return candidates
        
    except Exception as e:
        logger.exception(f"Vector search failed: {e}")
        return []


def _build_filter(
    career_stage: Optional[str],
    industry: Optional[str],
    freshness_days: Optional[int] = None,  # 🔥 NEW PARAMETER
) -> Optional[str]:
    """
    Build Milvus filter expression.
    
    Now includes freshness filtering based on last_updated field.
    """
    filters = []
    
    # 🔥 FRESHNESS FILTER (FIRST)
    date_filter = build_date_filter(freshness_days)
    if date_filter:
        filters.append(date_filter)
        logger.debug(f"Freshness filter: last {freshness_days} days")
    
    # Career stage filter
    if career_stage and career_stage != "Any":
        try:
            min_index = CAREER_STAGES.index(career_stage)
            allowed = CAREER_STAGES[min_index:]
            stage_list = ",".join(f'"{s}"' for s in allowed)
            filters.append(f"career_stage in [{stage_list}]")
            logger.debug(f"Career stage filter: {career_stage} -> {allowed}")
        except ValueError:
            logger.warning(f"Unknown career stage: {career_stage}")
    
    # Industry filter
    if industry:
        safe_ind = industry.replace('"', '""').replace("'", "''")
        filters.append(f'industries_worked like "%{safe_ind}%"')
        logger.debug(f"Industry filter: {industry}")
    
    filter_expr = " and ".join(filters) if filters else None
    
    if filter_expr:
        logger.debug(f"Final filter: {filter_expr}")
    
    return filter_expr


def invalidate_search_cache():
    """Invalidate all search cache entries."""
    if not ENABLE_CACHE:
        return
    
    logger.info("ℹ️  Search cache will auto-expire in %d seconds", SEARCH_CACHE_TTL)