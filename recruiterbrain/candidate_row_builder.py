# candidate_row_builder.py
from typing import Dict, Any, List
from utils_resume_ingest import PIIInfo
from llm_resume_extractor import generate_candidate_id, current_date_str


def build_candidate_row(
    llm_json: Dict[str, Any],
    pii: PIIInfo,
    summary_embedding: List[float],
    skills_embedding: List[float],
    raw_resume_uri: str,
) -> Dict[str, Any]:
    """Return one row ready for insertion into new_candidate_pool."""

    # prefer PII extractor, fall back to LLM if empty
    name = pii.name or llm_json.get("name")
    email = pii.email or llm_json.get("email")
    phone = pii.phone or llm_json.get("phone")
    linkedin_url = pii.linkedin_url or llm_json.get("linkedin_url")
    portfolio_url = pii.portfolio_url or llm_json.get("portfolio_url")

    candidate_id = generate_candidate_id()

    row = {
        "candidate_id": candidate_id,
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin_url": linkedin_url,
        "portfolio_url": portfolio_url,

        "location_city": llm_json.get("location_city"),
        "location_state": llm_json.get("location_state"),
        "location_country": llm_json.get("location_country"),

        "relocation_willingness": llm_json.get("relocation_willingness") or "Unknown",
        "remote_preference": llm_json.get("remote_preference") or "Unknown",
        "availability_status": llm_json.get("availability_status") or "Unknown",

        "total_experience_years": llm_json.get("total_experience_years"),
        "education_level": llm_json.get("education_level") or "Unknown",
        "degrees": llm_json.get("degrees") or [],
        "institutions": llm_json.get("institutions") or [],
        "languages_spoken": llm_json.get("languages_spoken") or [],

        "primary_industry": llm_json.get("primary_industry"),
        "sub_industries": llm_json.get("sub_industries") or [],

        "skills_extracted": llm_json.get("skills_extracted") or [],
        "tools_and_technologies": llm_json.get("tools_and_technologies") or [],
        "certifications": llm_json.get("certifications") or [],
        "top_titles_mentioned": llm_json.get("top_titles_mentioned") or [],
        "domains_of_expertise": llm_json.get("domains_of_expertise") or [],
        "employment_history": llm_json.get("employment_history") or [],

        "semantic_summary": llm_json.get("semantic_summary"),
        "keywords_summary": llm_json.get("keywords_summary") or [],
        "career_stage": llm_json.get("career_stage"),

        "genai_relevance_score": llm_json.get("genai_relevance_score"),
        "medical_domain_score": llm_json.get("medical_domain_score"),
        "construction_domain_score": llm_json.get("construction_domain_score"),
        "cad_relevance_score": llm_json.get("cad_relevance_score"),
        "nlp_relevance_score": llm_json.get("nlp_relevance_score"),
        "computer_vision_relevance_score": llm_json.get("computer_vision_relevance_score"),
        "data_engineering_relevance_score": llm_json.get("data_engineering_relevance_score"),
        "mlops_relevance_score": llm_json.get("mlops_relevance_score"),

        "evidence_skills": llm_json.get("evidence_skills") or [],
        "evidence_domains": llm_json.get("evidence_domains") or [],
        "evidence_certifications": llm_json.get("evidence_certifications") or [],
        "evidence_tools": llm_json.get("evidence_tools") or [],

        "source_channel": llm_json.get("source_channel") or "Unknown",
        "hiring_manager_notes": llm_json.get("hiring_manager_notes") or "unknown",
        "interview_feedback": llm_json.get("interview_feedback") or "unknown",
        "offer_status": llm_json.get("offer_status") or "Unknown",
        "assigned_recruiter": llm_json.get("assigned_recruiter") or "unknown",

        "clouds": llm_json.get("clouds") or [],
        "role_family": llm_json.get("role_family") or "other",
        "years_band": llm_json.get("years_band") or "mid",

        "summary_embedding": summary_embedding,
        "skills_embedding": skills_embedding,
        "resume_embedding_version": "v1",
        "last_updated": current_date_str(),

        # store raw file URI if you want it in this collection
        "raw_resume_uri": raw_resume_uri,
    }

    return row
