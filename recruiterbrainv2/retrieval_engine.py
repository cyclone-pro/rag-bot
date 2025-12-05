"""Core hybrid retrieval engine - optimized for candidates_v3 WITH CACHING."""
import logging
from typing import List, Dict, Any, Optional
from .config import (
    get_milvus_client,
    get_encoder,
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
)
from .skill_extractor import extract_requirements
from .ranker import hybrid_rank, compute_match_details
from .cache import generate_cache_key, get_cache

logger = logging.getLogger(__name__)


def _vectorize(text: str, batch_texts: Optional[List[str]] = None) -> Optional[List[float]]:
    """
    Encode text to a normalized embedding.
    
    Supports batch encoding for efficiency when multiple texts need embedding.
    
    Args:
        text: Primary text to encode
        batch_texts: Optional list of additional texts to encode in same batch
    
    Returns:
        Single embedding (or list if batch_texts provided)
    """
    try:
        enc = get_encoder()
        
        if batch_texts:
            # Batch encode multiple texts
            all_texts = [f"query: {t}" for t in [text] + batch_texts]
            embeddings = enc.encode(
                all_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=len(all_texts)
            )
            return [emb.tolist() for emb in embeddings]
        else:
            # Single text
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
) -> Dict[str, Any]:
    """
    Main search function for candidates_v3 collection WITH CACHING.
    
    Cache Strategy:
    - Cache key: hash(query + filters + top_k)
    - TTL: 5 minutes (configurable via SEARCH_CACHE_TTL)
    - Returns cached results instantly if available
    
    Uses 3 embeddings: summary, tech, role for optimal matching.
    """
    # ==================== CACHE CHECK ====================
    if ENABLE_CACHE:
        cache_key = generate_cache_key(
            "search_v2",
            query=query,
            top_k=top_k,
            career_stage=career_stage or "",
            industry=industry or ""
        )
        
        cache = get_cache()
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("✅ Search cache HIT - returning cached results instantly")
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
            "error": "Query too short. Please provide more details (e.g., 'Find Python developers with Django')"
        }
        # Don't cache errors
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
            "error": "👋 Hi! Please ask a recruiting question like: 'Find Salesforce developers with Apex experience'"
        }
        # Don't cache greetings
        return error_response
    
    # ==================== EXTRACT REQUIREMENTS (CACHED INTERNALLY) ====================
    requirements = extract_requirements(query)  # This is already cached in skill_extractor.py!
    required_skills = requirements.get("must_have_skills", [])
    detected_seniority = requirements.get("seniority_level", "Any")
    detected_industry = requirements.get("industry")
    
    # Use detected values if not explicitly provided
    if not career_stage or career_stage == "Any":
        career_stage = detected_seniority
    
    if not industry and detected_industry:
        industry = detected_industry
    
    # Require at least 1 skill (unless filters provided)
    if not required_skills and not career_stage and not industry:
        error_response = {
            "query": query,
            "requirements": requirements,
            "search_mode": "error",
            "total_found": 0,
            "candidates": [],
            "error": "❌ No technical skills detected. Try: 'Find Python developers' or 'Senior Salesforce engineers with Apex'"
        }
        # Don't cache errors
        return error_response
    
    # ==================== VECTORIZE ====================
    vector = _vectorize(query)
    if vector is None:
        raise RuntimeError("Embedding model unavailable; cannot run V2 search.")
    
    # ==================== BUILD FILTER ====================
    filter_expr = _build_filter(career_stage, industry)
    
    # ==================== DECIDE SEARCH MODE ====================
    # Use hybrid mode if 5+ skills OR if specific role type detected
    use_hybrid = len(required_skills) >= 5 or requirements.get("role_type")
    
    if use_hybrid:
        logger.info("Using HYBRID mode (%d skills, role=%s)", 
                   len(required_skills), 
                   requirements.get("role_type", "N/A"))
        results = _hybrid_search(query, required_skills, filter_expr, vector)
        search_mode = "hybrid"
    else:
        logger.info("Using VECTOR-ONLY mode (%d skills)", len(required_skills))
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
        # Cache "no results" for 1 minute (shorter TTL)
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
    
    # ==================== FORMAT TOP RESULTS ====================
    top_candidates = []
    for candidate, score in ranked[:top_k]:
        match_details = compute_match_details(candidate, required_skills)
        
        # Build clean location string
        location_parts = [
            candidate.get("location_city"),
            candidate.get("location_state"),
            candidate.get("location_country")
        ]
        location = ", ".join([p for p in location_parts if p])
        
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
        })
    
    final_result = {
        "candidates": top_candidates,
        "total_found": len(results),
        "requirements": requirements,
        "search_mode": search_mode,
        "query": query,
    }
    
    # ==================== CACHE RESULT ====================
    if ENABLE_CACHE:
        cache.set(cache_key, final_result, ttl=SEARCH_CACHE_TTL)
        logger.info(f"✅ Cached search result for {SEARCH_CACHE_TTL}s")
    
    return final_result


def _hybrid_search(
    query: str,
    required_skills: List[str],
    filter_expr: Optional[str],
    vector: List[float],
) -> List[Dict[str, Any]]:
    """
    Hybrid search: keyword + vector.
    
    Keyword search finds exact skill matches.
    Vector search finds semantic matches.
    """
    # Part 1: Keyword search
    keyword_candidates = _keyword_search(required_skills, filter_expr, KEYWORD_TOP_K)
    keyword_ids = {c["candidate_id"] for c in keyword_candidates}
    
    logger.info("Keyword search found %d candidates", len(keyword_candidates))
    
    # Part 2: Vector search (use tech_embedding for skill-heavy queries)
    vector_candidates = _vector_search(query, filter_expr, vector, VECTOR_TOP_K, use_field="tech_embedding")
    
    logger.info("Vector search found %d candidates", len(vector_candidates))
    
    # Part 3: Merge (keyword first - they have exact matches)
    merged = list(keyword_candidates)
    
    for vc in vector_candidates:
        if vc["candidate_id"] not in keyword_ids:
            merged.append(vc)
    
    logger.info("Hybrid merge: %d total candidates", len(merged))
    
    return merged


def _keyword_search(
    skills: List[str],
    filter_expr: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Keyword-based search using Milvus scalar filters.
    
    Searches across multiple fields:
    - skills_extracted
    - tools_and_technologies
    - tech_stack_primary
    - current_tech_stack (NEW in v3)
    """
    if not skills:
        return []
    
    client = get_milvus_client()
    
    # Build OR condition for skills (search top 7 most important)
    skill_conditions = []
    for skill in skills[:7]:
        # Escape quotes for Milvus filter syntax
        safe_skill = skill.replace('"', '""').replace("'", "''")
        
        # Search in multiple fields (candidates_v3 schema)
        skill_conditions.append(
            f'(skills_extracted like "%{safe_skill}%" or '
            f'tools_and_technologies like "%{safe_skill}%" or '
            f'tech_stack_primary like "%{safe_skill}%" or '
            f'current_tech_stack like "%{safe_skill}%")'
        )
    
    keyword_expr = " or ".join(skill_conditions)
    
    # Combine with filter expression (career stage, industry)
    if filter_expr:
        keyword_expr = f"({filter_expr}) and ({keyword_expr})"
    
    try:
        logger.debug(f"Keyword filter: {keyword_expr[:200]}...")
        
        results = client.query(
            collection_name=COLLECTION,
            filter=keyword_expr,
            output_fields=SEARCH_OUTPUT_FIELDS,
            limit=limit,
        )
        
        # Add high vector score for keyword matches (they're exact matches)
        for r in results:
            r['vector_score'] = 0.95  # High score for exact matches
        
        logger.debug(f"Keyword search returned {len(results)} results")
        
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
    Pure vector search using ANN (Approximate Nearest Neighbor).
    
    Args:
        query: Original search query (for logging)
        filter_expr: Milvus filter expression (career stage, industry)
        vector: Query embedding vector
        limit: Number of results to return
        use_field: Which embedding to search against
            - "summary_embedding": General semantic search
            - "tech_embedding": Skill-focused search (better for technical queries)
            - "role_embedding": Role/industry-focused search
    """
    client = get_milvus_client()
    
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
    
    if filter_expr:
        search_params["filter"] = filter_expr
    
    try:
        logger.debug(f"Vector search using {use_field}, limit={limit}")
        
        results = client.search(**search_params)
        
        if not results or not results[0]:
            logger.debug("Vector search returned no results")
            return []
        
        candidates = []
        for hit in results[0]:
            # Handle both dict and object responses from Milvus
            if isinstance(hit, dict):
                entity = hit.get("entity", {})
                distance = hit.get("distance", 0)
            else:
                # Extract fields from object
                entity = {
                    field: getattr(hit, field, None)
                    for field in SEARCH_OUTPUT_FIELDS
                }
                distance = getattr(hit, "distance", 0)
            
            entity["vector_score"] = float(distance)
            candidates.append(entity)
        
        logger.debug(f"Vector search returned {len(candidates)} candidates")
        
        return candidates
        
    except Exception as e:
        logger.exception(f"Vector search failed: {e}")
        return []


def _build_filter(
    career_stage: Optional[str],
    industry: Optional[str],
) -> Optional[str]:
    """
    Build Milvus filter expression for career stage and industry.
    
    Career stage uses seniority hierarchy:
    - If user asks for "Senior", also include "Lead/Manager" and "Director+"
    - This ensures we don't filter out over-qualified candidates
    
    Industry uses partial string matching:
    - Searches in industries_worked field (comma-separated list)
    """
    filters = []
    
    # Career stage filter (hierarchical)
    if career_stage and career_stage != "Any":
        try:
            min_index = CAREER_STAGES.index(career_stage)
            allowed = CAREER_STAGES[min_index:]
            stage_list = ",".join(f'"{s}"' for s in allowed)
            filters.append(f"career_stage in [{stage_list}]")
            logger.debug(f"Career stage filter: {career_stage} -> {allowed}")
        except ValueError:
            logger.warning(f"Unknown career stage: {career_stage}")
    
    # Industry filter (partial match in industries_worked)
    if industry:
        safe_ind = industry.replace('"', '""').replace("'", "''")
        filters.append(f'industries_worked like "%{safe_ind}%"')
        logger.debug(f"Industry filter: {industry}")
    
    filter_expr = " and ".join(filters) if filters else None
    
    if filter_expr:
        logger.debug(f"Final filter: {filter_expr}")
    
    return filter_expr


def invalidate_search_cache():
    """
    Invalidate all search cache entries.
    
    Call this when:
    - New resume uploaded
    - Candidate data updated
    - Collection modified
    """
    if not ENABLE_CACHE:
        return
    
    cache = get_cache()
    
    
    
    #  log (cache will auto-expire in 5 min)
    logger.info("ℹ️  Search cache will auto-expire in %d seconds", SEARCH_CACHE_TTL)
    
  