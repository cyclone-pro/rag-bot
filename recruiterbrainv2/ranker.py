"""Hybrid ranking: vector + keyword matching."""
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def hybrid_rank(
    candidates: List[Dict[str, Any]],
    required_skills: List[str],
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Rank candidates using hybrid scoring.
    
    Args:
        candidates: List with 'vector_score' already set
        required_skills: Required skill strings
        vector_weight: Weight for semantic similarity
        keyword_weight: Weight for keyword matching
    
    Returns:
        List of (candidate, hybrid_score) sorted descending
    """
    if not candidates:
        return []
    
    required_normalized = [s.lower().strip() for s in required_skills]
    
    ranked = []
    for candidate in candidates:
        vector_score = candidate.get('vector_score', 0.0)
        keyword_score = _keyword_score(candidate, required_normalized)
        
        hybrid_score = (
            vector_weight * vector_score + 
            keyword_weight * keyword_score
        )
        
        candidate['keyword_score'] = keyword_score
        candidate['hybrid_score'] = hybrid_score
        
        ranked.append((candidate, hybrid_score))
    
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(
        "Ranked %d candidates (avg_hybrid=%.3f)",
        len(ranked),
        sum(s for _, s in ranked) / len(ranked) if ranked else 0
    )
    
    return ranked


def _keyword_score(candidate: Dict[str, Any], required_skills: List[str]) -> float:
    """Compute keyword matching score (0.0 to 1.0)."""
    if not required_skills:
        return 1.0
    
    # Gather all text
    text = " ".join([
        str(candidate.get("skills_extracted", "")),
        str(candidate.get("tools_and_technologies", "")),
        str(candidate.get("domains_of_expertise", "")),
        str(candidate.get("semantic_summary", "")),
        str(candidate.get("keywords_summary", "")),
    ]).lower()
    
    # Count matches
    matched = sum(1 for skill in required_skills if skill in text)
    
    return matched / len(required_skills)


def compute_match_details(
    candidate: Dict[str, Any],
    required_skills: List[str],
) -> Dict[str, Any]:
    """Generate detailed match explanation."""
    text = " ".join([
        str(candidate.get("skills_extracted", "")),
        str(candidate.get("tools_and_technologies", "")),
    ]).lower()
    
    matched = []
    missing = []
    
    for skill in required_skills:
        if skill.lower() in text:
            matched.append(skill)
        else:
            missing.append(skill)
    
    match_pct = (len(matched) / len(required_skills) * 100) if required_skills else 100
    
    return {
        "match_percentage": round(match_pct),
        "matched_skills": matched,
        "missing_skills": missing,
        "total_required": len(required_skills),
        "vector_score": candidate.get('vector_score', 0),
        "keyword_score": candidate.get('keyword_score', 0),
        "hybrid_score": candidate.get('hybrid_score', 0),
    }