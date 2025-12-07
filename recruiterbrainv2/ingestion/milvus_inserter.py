"""Insert candidate data into Milvus."""
import logging
import uuid
from datetime import datetime
from typing import Dict
from ..config import get_milvus_client, COLLECTION
ENABLE_CONNECTION_POOL=False

logger = logging.getLogger(__name__)


def generate_candidate_id() -> str:
    """Generate unique 6-character candidate ID."""
    return uuid.uuid4().hex[:6].upper()


def insert_candidate(candidate_data: Dict, embeddings: Dict, source_channel: str = "Upload") -> str:
    """
    Insert candidate into Milvus.
    
    Returns:
        candidate_id
    """
    client = get_milvus_client()
    
    # Generate candidate_id if not exists
    candidate_id = candidate_data.get("candidate_id") or generate_candidate_id()
    
    # Prepare data for Milvus
    milvus_data = {
        # Identity
        "candidate_id": candidate_id,
        "name": candidate_data.get("name", "Unknown"),
        "email": candidate_data.get("email", ""),
        "phone": candidate_data.get("phone", ""),
        "linkedin_url": candidate_data.get("linkedin_url", ""),
        "portfolio_url": candidate_data.get("portfolio_url", ""),
        "github_url": candidate_data.get("github_url", ""),
        
        # Location
        "location_city": candidate_data.get("location_city", ""),
        "location_state": candidate_data.get("location_state", ""),
        "location_country": candidate_data.get("location_country", ""),
        "relocation_willingness": candidate_data.get("relocation_willingness", "Unknown"),
        "remote_preference": candidate_data.get("remote_preference", "Unknown"),
        "availability_status": candidate_data.get("availability_status", "Unknown"),
        
        # Experience
        "total_experience_years": float(candidate_data.get("total_experience_years", 0.0)),
        "education_level": candidate_data.get("education_level", ""),
        "degrees": candidate_data.get("degrees", ""),
        "institutions": candidate_data.get("institutions", ""),
        "languages_spoken": candidate_data.get("languages_spoken", ""),
        "management_experience_years": float(candidate_data.get("management_experience_years", 0.0)),
        
        # Career
        "career_stage": candidate_data.get("career_stage", "Unknown"),
        "years_in_current_role": float(candidate_data.get("years_in_current_role", 0.0)),
        "top_3_titles": candidate_data.get("top_3_titles", ""),
        "role_type": candidate_data.get("role_type", ""),
        
        # Industry
        "industries_worked": candidate_data.get("industries_worked", ""),
        "domain_expertise": candidate_data.get("domain_expertise", ""),
        "verticals_experience": candidate_data.get("verticals_experience", ""),
        
        # Skills
        "skills_extracted": candidate_data.get("skills_extracted", ""),
        "tools_and_technologies": candidate_data.get("tools_and_technologies", ""),
        "certifications": candidate_data.get("certifications", ""),
        "tech_stack_primary": candidate_data.get("tech_stack_primary", ""),
        "programming_languages": candidate_data.get("programming_languages", ""),
        
        # Evidence
        "employment_history": candidate_data.get("employment_history", ""),
        "semantic_summary": candidate_data.get("semantic_summary", ""),
        "keywords_summary": candidate_data.get("keywords_summary", ""),
        "evidence_skills": candidate_data.get("evidence_skills", ""),
        "evidence_projects": candidate_data.get("evidence_projects", ""),
        "evidence_leadership": candidate_data.get("evidence_leadership", ""),
        
        # Recency
        "current_tech_stack": candidate_data.get("current_tech_stack", ""),
        "years_since_last_update": 0.0,  # Fresh upload
        "top_5_skills_with_years": candidate_data.get("top_5_skills_with_years", ""),
        
        # Metadata
        "source_channel": source_channel,
        "hiring_manager_notes": "",
        "interview_feedback": "",
        "offer_status": "Unknown",
        "assigned_recruiter": "",
        "resume_embedding_version": "e5-base-v2-v3",
        "last_updated": datetime.utcnow().isoformat() + "Z",
        
        # Embeddings
        "summary_embedding": embeddings["summary_embedding"],
        "tech_embedding": embeddings["tech_embedding"],
        "role_embedding": embeddings["role_embedding"],
    }
    
    # Insert into Milvus
    try:
        client.insert(
            collection_name=COLLECTION,
            data=[milvus_data]
        )
        logger.info(f"✅ Inserted candidate {candidate_id} into Milvus")
        return candidate_id
        
    except Exception as e:
        logger.exception(f"Failed to insert candidate {candidate_id}")
        raise RuntimeError(f"Milvus insertion failed: {e}")