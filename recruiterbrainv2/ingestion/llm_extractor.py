"""Extract structured data from resume using LLM."""
import json
import logging
from typing import Dict, Any
from ..config import get_openai_client, OPENAI_MODEL

logger = logging.getLogger(__name__)


def extract_resume_data(sanitized_text: str) -> Dict[str, Any]:
    """
    Extract structured data from sanitized resume text.
    
    Args:
        sanitized_text: Resume text with PII removed
    
    Returns:
        Dictionary with all extracted fields
    """
    client = get_openai_client()
    if not client:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY.")
    
    system_prompt = """You are a resume parsing expert. Extract structured data from resumes.

IMPORTANT: Return ONLY valid JSON. No markdown, no preamble, no explanation.

Extract these fields:
- name: Full name (string)
- location_city, location_state, location_country: Current location
- relocation_willingness, remote_preference, availability_status: Work preferences
- total_experience_years: Total years (float)
- education_level: Highest degree
- degrees: All degrees (string, semicolon-separated)
- institutions: Schools attended (string, semicolon-separated)
- languages_spoken: Languages (string, comma-separated)
- management_experience_years: Years managing people (float, 0 if none)
- career_stage: Entry/Mid/Senior/Lead/Director+
- years_in_current_role: Years in current job (float)
- top_3_titles: Top 3 job titles (string, semicolon-separated)
- role_type: Role categories (string, pipe-separated) e.g., "Backend Engineering | Cloud Architecture"
- industries_worked: Industries (string, comma-separated)
- domain_expertise: Domains (string, comma-separated)
- verticals_experience: Company types (string, comma-separated)
- skills_extracted: All technical skills (string, comma-separated)
- tools_and_technologies: Tools/platforms (string, comma-separated)
- certifications: Certifications (string, comma-separated)
- tech_stack_primary: Top 10 technologies (string, comma-separated)
- programming_languages: Languages (string, comma-separated)
- employment_history: JSON string of jobs
- semantic_summary: Rich 2-3 paragraph summary
- keywords_summary: Keyword-based summary
- evidence_skills: Evidence of skills from resume
- evidence_projects: Key projects (string)
- evidence_leadership: Leadership evidence (string)
- current_tech_stack: Currently using (string, comma-separated)
- top_5_skills_with_years: Format "Skill:Years" (string) e.g., "Python:7, AWS:5, Java:10"

Set empty string "" for missing fields. Set 0.0 for missing numeric fields."""

    user_prompt = f"""Extract data from this resume:

{sanitized_text[:8000]}

Return JSON only."""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        data = json.loads(content)
        
        logger.info(f"Extracted data for: {data.get('name', 'Unknown')}")
        
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}")
        logger.error(f"Raw response: {content[:500]}")
        raise ValueError(f"LLM returned invalid JSON: {e}")
    
    except Exception as e:
        logger.exception("LLM extraction failed")
        raise RuntimeError(f"Failed to extract resume data: {e}")