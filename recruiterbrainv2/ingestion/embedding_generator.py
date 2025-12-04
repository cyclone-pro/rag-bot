"""Generate embeddings for candidate (OPTIMIZED - Batch Processing)."""
import logging
from typing import Dict, List
from ..config import get_encoder

logger = logging.getLogger(__name__)


def generate_embeddings(candidate_data: Dict) -> Dict[str, List[float]]:
    """
    Generate 3 embeddings in ONE BATCH (3x faster).
    
    OLD: 3 separate encode calls = 9 seconds
    NEW: 1 batch encode call = 3 seconds
    
    Returns:
        {
            "summary_embedding": [...],
            "tech_embedding": [...],
            "role_embedding": [...]
        }
    """
    encoder = get_encoder()
    
    logger.info("   Preparing 3 text inputs for batch encoding...")
    
    # ==================== PREPARE 3 TEXTS ====================
    
    # 1. Summary text (career narrative)
    summary_text = " ".join([
        candidate_data.get("semantic_summary", ""),
        candidate_data.get("keywords_summary", ""),
        candidate_data.get("employment_history", "")[:1000],  # Truncate to avoid token limits
    ]).strip()
    
    # 2. Tech text (skills & tools)
    tech_text = " ".join([
        candidate_data.get("skills_extracted", ""),
        candidate_data.get("tools_and_technologies", ""),
        candidate_data.get("tech_stack_primary", ""),
        candidate_data.get("programming_languages", ""),
        candidate_data.get("certifications", ""),
        candidate_data.get("current_tech_stack", ""),
    ]).strip()
    
    # 3. Role text (industry & expertise)
    role_text = " ".join([
        candidate_data.get("role_type", ""),
        candidate_data.get("industries_worked", ""),
        candidate_data.get("domain_expertise", ""),
        candidate_data.get("top_3_titles", ""),
        candidate_data.get("evidence_leadership", ""),
        candidate_data.get("verticals_experience", ""),
    ]).strip()
    
    # Fallback for empty texts
    if not summary_text:
        summary_text = candidate_data.get("name", "Unknown candidate")
    if not tech_text:
        tech_text = "General skills"
    if not role_text:
        role_text = "General role"
    
    # ==================== ADD E5 PREFIX ====================
    # E5 models need "passage:" prefix for documents (vs "query:" for search)
    texts_with_prefix = [
        f"passage: {summary_text}",
        f"passage: {tech_text}",
        f"passage: {role_text}"
    ]
    
    logger.info(f"   Text lengths: summary={len(summary_text)}, tech={len(tech_text)}, role={len(role_text)}")
    
    # ==================== BATCH ENCODE (KEY OPTIMIZATION) ====================
    logger.info("   🚀 Batch encoding 3 texts in parallel...")
    
    embeddings = encoder.encode(
        texts_with_prefix,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=3,  # Process all 3 at once
        convert_to_numpy=True
    )
    
    logger.info("   ✅ Batch encoding complete (3 embeddings in one pass)")
    
    # ==================== RETURN ====================
    return {
        "summary_embedding": embeddings[0].tolist(),
        "tech_embedding": embeddings[1].tolist(),
        "role_embedding": embeddings[2].tolist(),
    }