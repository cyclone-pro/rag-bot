"""Dynamic skill extraction using LLM."""
import logging
import json
import re
from typing import Dict, List, Any
from .config import get_openai_client, OPENAI_MODEL

logger = logging.getLogger(__name__)


def extract_requirements(query_text: str) -> Dict[str, Any]:
    """
    Extract structured requirements from query/JD.
    
    Returns:
        {
            "must_have_skills": ["skill1", "skill2", ...],
            "nice_to_have_skills": ["skill3", ...],
            "seniority_level": "Entry|Mid|Senior|Lead|Director+|Any",
            "industry": str or None,
            "years_experience_min": int or None
        }
    """
    client = get_openai_client()
    if not client:
        logger.info("OpenAI unavailable, using fallback")
        return _fallback_extraction(query_text)
    
    system_prompt = """Extract technical requirements from job queries.
Return ONLY valid JSON.

Rules:
- Extract concrete technical skills (tools, languages, frameworks, platforms)
- No soft skills (communication, teamwork)
- Normalize: "LWC" → "lightning web components", "k8s" → "kubernetes"
- Max 15 must-have skills
- Detect seniority from: "5+ years", "senior", "lead", etc."""

    user_prompt = f"""Analyze this job requirement:

{query_text[:3000]}

Return JSON:
{{
  "must_have_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill3"],
  "seniority_level": "Entry|Mid|Senior|Lead|Director+|Any",
  "industry": "Healthcare|Finance|Technology|etc or null",
  "years_experience_min": 5 or null
}}"""

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
        
        result = json.loads(content)
        
        # Normalize skills to lowercase
        result["must_have_skills"] = [
            s.lower().strip() for s in result.get("must_have_skills", [])
        ]
        result["nice_to_have_skills"] = [
            s.lower().strip() for s in result.get("nice_to_have_skills", [])
        ]
        
        logger.info(
            "Extracted: %d must-have, %d nice-to-have, seniority=%s",
            len(result["must_have_skills"]),
            len(result["nice_to_have_skills"]),
            result.get("seniority_level", "Any")
        )
        
        return result
        
    except Exception as e:
        logger.warning("LLM extraction failed: %s, using fallback", e)
        return _fallback_extraction(query_text)


def _fallback_extraction(text: str) -> Dict[str, Any]:
    """Regex-based extraction when LLM unavailable."""
    text_lower = text.lower()
    
    skills = set()
    
    # Common tech patterns
    patterns = [
        # Salesforce
        r'\b(salesforce|apex|visualforce|lightning|lwc|soql|cpq|service cloud|sales cloud)\b',
        # Languages
        r'\b(python|java|javascript|typescript|go|rust|c\+\+|c#)\b',
        # Frameworks
        r'\b(django|flask|fastapi|spring|react|angular|vue|node\.?js)\b',
        # Databases
        r'\b(postgresql|mysql|mongodb|redis|snowflake|bigquery)\b',
        # Cloud
        r'\b(aws|azure|gcp|kubernetes|docker|terraform)\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        skills.update(matches)
    
    # Detect seniority
    seniority = "Any"
    if re.search(r'\b(senior|8\+|10\+|staff|principal)\b', text_lower):
        seniority = "Senior"
    elif re.search(r'\b(lead|manager|engineering manager)\b', text_lower):
        seniority = "Lead/Manager"
    elif re.search(r'\b(junior|entry|0-2 years)\b', text_lower):
        seniority = "Entry"
    elif re.search(r'\b(mid|3-5 years|intermediate)\b', text_lower):
        seniority = "Mid"
    
    return {
        "must_have_skills": list(skills)[:15],
        "nice_to_have_skills": [],
        "seniority_level": seniority,
        "industry": None,
        "years_experience_min": None
    }