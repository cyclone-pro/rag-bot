"""Core hybrid retrieval engine - optimized for candidates_v3."""
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
)
from .skill_extractor import extract_requirements
from .ranker import hybrid_rank, compute_match_details

logger = logging.getLogger(__name__)


def _vectorize(text: str) -> Optional[List[float]]:
    """Encode text to a normalized embedding."""
    try:
        enc = get_encoder()
        # Add e5 query prefix
        query_text = f"query: {text}"
        return enc.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
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
    Main search function for candidates_v3 collection.
    
    Uses 3 embeddings: summary, tech, role for optimal matching.
    """
    # ==================== VALIDATION ====================
    query = query.strip()
    
    if len(query) < 10:
        return {
            "query": query,
            "requirements": {},
            "search_mode": "error",
            "total_found": 0,
            "candidates": [],
            "error": "Query too short. Please provide more details (e.g., 'Find Python developers with Django')"
        }
    
    # Check for greetings
    greeting_patterns = [
        "hello", "hi", "hey", "how are you", "what's up", "good morning",
        "good afternoon", "good evening", "test", "testing", "sup"
    ]
    
    query_lower = query.lower()
    if any(pattern in query_lower for pattern in greeting_patterns) and len(query) < 50:
        return {
            "query": query,
            "requirements": {},
            "search_mode": "error",
            "total_found": 0,
            "candidates": [],
            "error": "👋 Hi! Please ask a recruiting question like: 'Find Salesforce developers with Apex experience'"
        }
    
    # ==================== EXTRACT REQUIREMENTS ====================
    requirements = extract_requirements(query)
    required_skills = requirements.get("must_have_skills", [])
    detected_seniority = requirements.get("seniority_level", "Any")
    
    # Use detected seniority if not provided
    if not career_stage or career_stage == "Any":
        career_stage = detected_seniority
    
    # Require at least 1 skill (unless filters provided)
    if not required_skills and not career_stage and not industry:
        return {
            "query": query,
            "requirements": requirements,
            "search_mode": "error",
            "total_found": 0,
            "candidates": [],
            "error": "❌ No technical skills detected. Try: 'Find Python developers' or 'Senior Salesforce engineers with Apex'"
        }
    
    # ==================== VECTORIZE ====================
    vector = _vectorize(query)
    if vector is None:
        raise RuntimeError("Embedding model unavailable; cannot run V2 search.")
    
    # ==================== BUILD FILTER ====================
    filter_expr = _build_filter(career_stage, industry)
    
    # ==================== DECIDE SEARCH MODE ====================
    use_hybrid = len(required_skills) >= 5
    
    if use_hybrid:
        logger.info("Using HYBRID mode (%d skills)", len(required_skills))
        results = _hybrid_search(query, required_skills, filter_expr, vector)
        search_mode = "hybrid"
    else:
        logger.info("Using VECTOR-ONLY mode (%d skills)", len(required_skills))
        results = _vector_search(query, filter_expr, vector, VECTOR_TOP_K)
        search_mode = "vector_only"
    
    if not results:
        return {
            "candidates": [],
            "total_found": 0,
            "requirements": requirements,
            "search_mode": search_mode,
        }
    
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
    
    return {
        "candidates": top_candidates,
        "total_found": len(results),
        "requirements": requirements,
        "search_mode": search_mode,
        "query": query,
    }


def _hybrid_search(
    query: str,
    required_skills: List[str],
    filter_expr: Optional[str],
    vector: List[float],
) -> List[Dict[str, Any]]:
    """Hybrid: keyword + vector."""
    
    # Part 1: Keyword search
    keyword_candidates = _keyword_search(required_skills, filter_expr, KEYWORD_TOP_K)
    keyword_ids = {c["candidate_id"] for c in keyword_candidates}
    
    logger.info("Keyword search found %d candidates", len(keyword_candidates))
    
    # Part 2: Vector search (use tech_embedding for skill-heavy queries)
    vector_candidates = _vector_search(query, filter_expr, vector, VECTOR_TOP_K, use_field="tech_embedding")
    
    # Part 3: Merge (keyword first)
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
    """Keyword-based search using Milvus scalar filters."""
    if not skills:
        return []
    
    client = get_milvus_client()
    
    # Build OR condition for skills
    skill_conditions = []
    for skill in skills[:7]:  # Top 7 skills
        safe_skill = skill.replace('"', '""')
        # Search in multiple fields
        skill_conditions.append(
            f'(skills_extracted like "%{safe_skill}%" or '
            f'tools_and_technologies like "%{safe_skill}%" or '
            f'tech_stack_primary like "%{safe_skill}%")'
        )
    
    keyword_expr = " or ".join(skill_conditions)
    
    if filter_expr:
        keyword_expr = f"({filter_expr}) and ({keyword_expr})"
    
    try:
        results = client.query(
            collection_name=COLLECTION,
            filter=keyword_expr,
            output_fields=SEARCH_OUTPUT_FIELDS,
            limit=limit,
        )
        
        # Add dummy vector score
        for r in results:
            r['vector_score'] = 0.9  # High score for exact matches
        
        return results
        
    except Exception as e:
        logger.warning("Keyword search failed: %s", e)
        return []


def _vector_search(
    query: str,
    filter_expr: Optional[str],
    vector: List[float],
    limit: int,
    use_field: str = "summary_embedding"
) -> List[Dict[str, Any]]:
    """Pure vector search."""
    
    client = get_milvus_client()
    
    search_params = {
        "collection_name": COLLECTION,
        "data": [vector],
        "anns_field": use_field,  # Can be summary_embedding, tech_embedding, or role_embedding
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
        results = client.search(**search_params)
        
        if not results or not results[0]:
            return []
        
        candidates = []
        for hit in results[0]:
            # Handle both dict and object responses
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
        logger.exception("Vector search failed: %s", e)
        return []


def _build_filter(
    career_stage: Optional[str],
    industry: Optional[str],
) -> Optional[str]:
    """Build Milvus filter expression."""
    filters = []
    
    # Career stage
    if career_stage and career_stage != "Any":
        try:
            min_index = CAREER_STAGES.index(career_stage)
            allowed = CAREER_STAGES[min_index:]
            stage_list = ",".join(f'"{s}"' for s in allowed)
            filters.append(f"career_stage in [{stage_list}]")
        except ValueError:
            logger.warning("Unknown career stage: %s", career_stage)
    
    # Industry (search in industries_worked field)
    if industry:
        safe_ind = industry.replace('"', '""')
        filters.append(f'industries_worked like "%{safe_ind}%"')
    
    return " and ".join(filters) if filters else None