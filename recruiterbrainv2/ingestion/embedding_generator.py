"""Generate 3 embeddings for candidate."""
import logging
from typing import Dict, List
from ..config import get_encoder

logger = logging.getLogger(__name__)


def generate_embeddings(candidate_data: Dict) -> Dict[str, List[float]]:
    """
    Generate 3 embeddings: summary, tech, role.
    
    Returns:
        {
            "summary_embedding": [...],
            "tech_embedding": [...],
            "role_embedding": [...]
        }
    """
    encoder = get_encoder()
    
    # 1. Summary embedding
    summary_text = " ".join([
        candidate_data.get("semantic_summary", ""),
        candidate_data.get("keywords_summary", ""),
        candidate_data.get("employment_history", "")[:1000],  # Truncate long history
    ]).strip()
    
    # 2. Tech embedding
    tech_text = " ".join([
        candidate_data.get("skills_extracted", ""),
        candidate_data.get("tools_and_technologies", ""),
        candidate_data.get("tech_stack_primary", ""),
        candidate_data.get("programming_languages", ""),
        candidate_data.get("certifications", ""),
    ]).strip()
    
    # 3. Role embedding
    role_text = " ".join([
        candidate_data.get("role_type", ""),
        candidate_data.get("industries_worked", ""),
        candidate_data.get("domain_expertise", ""),
        candidate_data.get("top_3_titles", ""),
        candidate_data.get("evidence_leadership", ""),
    ]).strip()
    
    # Add e5 prefix for query/document distinction
    summary_text = f"passage: {summary_text}"
    tech_text = f"passage: {tech_text}"
    role_text = f"passage: {role_text}"
    
    # Generate embeddings
    texts = [summary_text, tech_text, role_text]
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    logger.info("Generated 3 embeddings (768d each)")
    
    return {
        "summary_embedding": embeddings[0].tolist(),
        "tech_embedding": embeddings[1].tolist(),
        "role_embedding": embeddings[2].tolist(),
    }