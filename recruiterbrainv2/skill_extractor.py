"""Dynamic skill extraction with intelligent caching and optimized LLM prompting."""
import logging
import json
import re
from typing import Dict, List, Any, Optional
from .config import get_openai_client, OPENAI_MODEL, ENABLE_CACHE, LLM_CACHE_TTL
from .cache import cached
from .rate_limiter import global_limiter
from .rate_limiter import RateLimitExceeded

logger = logging.getLogger(__name__)

# Seniority detection patterns (ordered by specificity - MOST specific first)
SENIORITY_PATTERNS = {
    "Director+": ["director", "vp", "vice president", "c-level", "cto", "cio", "chief", "head of engineering"],
    "Lead/Manager": ["lead", "manager", "head", "principal", "staff engineer", "architect", "engineering manager", "tech lead"],
    "Senior": ["senior", "sr", "sr.", "experienced", "expert", "8+ years", "10+ years", "5+ years", "7+ years"],
    "Mid": ["mid", "mid-level", "intermediate", "3-5 years", "4-6 years", "3+ years"],
    "Entry": ["entry", "junior", "jr", "jr.", "graduate", "associate", "0-2 years", "1-3 years", "new grad", "entry-level"],
}

# Comprehensive tech skills database (aligned with candidates_v3 schema)
COMMON_TECH_SKILLS = {
    # Salesforce ecosystem (high priority)
    "salesforce", "apex", "visualforce", "lightning web components", "lwc", 
    "soql", "sosl", "cpq", "service cloud", "sales cloud", "marketing cloud",
    "salesforce commerce cloud", "einstein analytics", "mulesoft", "field service",
    
    # Programming languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust", "c++", "c#", 
    "ruby", "php", "swift", "kotlin", "scala", "perl", "r", "matlab", "objective-c",
    
    # Frontend frameworks/libraries
    "react", "angular", "vue", "vue.js", "svelte", "next.js", "nuxt", "ember",
    "jquery", "backbone", "redux", "mobx", "tailwind css", "bootstrap", "material-ui",
    
    # Backend frameworks
    "django", "flask", "fastapi", "spring", "spring boot", "node.js", 
    "express", "express.js", "nestjs", ".net", "asp.net", "rails", "ruby on rails",
    "laravel", "symfony", "gin", "echo",
    
    # Databases (SQL and NoSQL)
    "postgresql", "mysql", "mongodb", "redis", "cassandra", "dynamodb",
    "snowflake", "bigquery", "oracle", "sql server", "elasticsearch",
    "mariadb", "couchdb", "neo4j", "influxdb", "timescaledb",
    
    # Cloud platforms
    "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud platform", 
    "google cloud", "heroku", "digitalocean", "ibm cloud", "oracle cloud",
    "cloudflare", "vercel", "netlify",
    
    # AWS Services (specific)
    "ec2", "s3", "lambda", "rds", "dynamodb", "cloudfront", "route 53",
    "ecs", "eks", "fargate", "sagemaker", "redshift", "athena",
    
    # Azure Services
    "azure functions", "cosmos db", "azure sql", "azure devops", "azure ad",
    
    # DevOps/Infrastructure
    "docker", "kubernetes", "k8s", "terraform", "ansible", "puppet", "chef",
    "jenkins", "github actions", "gitlab ci", "circleci", "argocd", "helm",
    "vagrant", "packer", "consul", "vault", "prometheus", "grafana",
    
    # CI/CD
    "jenkins", "travis ci", "bamboo", "teamcity", "octopus deploy",
    
    # Data Engineering & Big Data
    "spark", "apache spark", "hadoop", "airflow", "kafka", "flink", "storm",
    "pandas", "numpy", "dask", "pyspark", "databricks", "presto", "hive",
    
    # Machine Learning / AI
    "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm",
    "mlflow", "kubeflow", "hugging face", "langchain", "llama", "openai",
    
    # Message Queues / Streaming
    "rabbitmq", "apache kafka", "amazon sqs", "google pub/sub", "nats", "zeromq",
    
    # Monitoring & Observability
    "datadog", "new relic", "splunk", "elk stack", "prometheus", "grafana",
    "sentry", "pagerduty", "logstash", "kibana",
    
    # Testing
    "pytest", "jest", "mocha", "chai", "selenium", "cypress", "junit",
    "testng", "mockito", "postman", "insomnia",
    
    # Version Control
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
    
    # API & Integration
    "rest", "restful", "graphql", "grpc", "soap", "websocket", "api gateway",
    
    # Security
    "oauth", "jwt", "saml", "ssl/tls", "penetration testing", "owasp",
    "security scanning", "vulnerability assessment",
    
    # Project Management / Collaboration
    "jira", "confluence", "asana", "trello", "monday.com", "slack",
    
    # Healthcare Specific
    "hl7", "fhir", "hipaa", "epic", "cerner", "meditech", "dicom",
    
    # Finance Specific
    "fintech", "blockchain", "solidity", "web3", "defi", "cryptocurrency",
    "trading systems", "risk management", "compliance",
}

# Skill normalization map (handle common aliases)
SKILL_ALIASES = {
    # Kubernetes
    "k8s": "kubernetes",
    "kube": "kubernetes",
    
    # JavaScript ecosystem
    "js": "javascript",
    "ts": "typescript",
    "react.js": "react",
    "vue.js": "vue",
    "node": "node.js",
    "nodejs": "node.js",
    "express.js": "express",
    
    # Salesforce
    "lwc": "lightning web components",
    "sfdc": "salesforce",
    
    # Databases
    "postgres": "postgresql",
    "psql": "postgresql",
    "mongo": "mongodb",
    "mssql": "sql server",
    
    # Cloud
    "aws lambda": "lambda",
    "amazon s3": "s3",
    "amazon ec2": "ec2",
    
    # DevOps
    "tf": "terraform",
    "k8": "kubernetes",
    "gh actions": "github actions",
    
    # Languages
    "py": "python",
    "golang": "go",
    "c sharp": "c#",
    "c plus plus": "c++",
}


def extract_requirements(query_text: str) -> Dict[str, Any]:
    """
    Extract structured requirements from job query or description.
    
    Optimized for candidates_v3 schema with intelligent caching.
    
    Args:
        query_text: Job description or search query
        
    Returns:
        {
            "must_have_skills": ["python", "django", "aws"],
            "nice_to_have_skills": ["redis", "docker"],
            "seniority_level": "Senior",
            "industry": "Healthcare",
            "years_experience_min": 5,
            "role_type": "Backend Engineering",
            "remote_preference": "Hybrid"
        }
    """
    # VALIDATION: Check query length
    if len(query_text.strip()) < 10:
        return {
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "seniority_level": "Any",
            "industry": None,
            "years_experience_min": None,
            "role_type": None,
            "query_text": query_text,
            "error": "Query too short. Please provide more details about requirements."
        }
      # VALIDATION: Check for generic terms only
    generic_only = all(word in query_text.lower() for word in ["find", "developer", "engineer"])
    word_count = len(query_text.split())
    
    if generic_only and word_count < 5:
        return {
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "seniority_level": "Any",
            "industry": None,
            "years_experience_min": None,
            "role_type": None,
            "query_text": query_text,
            "error": "Query too generic. Please specify skills, technologies, or requirements."
        }
    # Quick seniority detection
    seniority = _detect_seniority_fast(query_text)
    
    # Extract skills using cached LLM call
    if ENABLE_CACHE:
        skill_data = _extract_skills_cached(query_text)
    else:
        skill_data = _extract_skills_llm(query_text)
    
    # Merge quick detection with LLM results (LLM takes priority if different)
    if skill_data.get("seniority_level") == "Any" and seniority != "Any":
        skill_data["seniority_level"] = seniority
    
    return {
        "must_have_skills": skill_data.get("must_have_skills", []),
        "nice_to_have_skills": skill_data.get("nice_to_have_skills", []),
        "seniority_level": skill_data.get("seniority_level", "Any"),
        "industry": skill_data.get("industry"),
        "years_experience_min": skill_data.get("years_experience_min"),
        "role_type": skill_data.get("role_type"),  
        "remote_preference": skill_data.get("remote_preference"),  
    }


@cached("skill_extract", ttl=LLM_CACHE_TTL)
def _extract_skills_cached(query_text: str) -> Dict[str, Any]:
    """
    Cached wrapper for LLM skill extraction.
    
    Cache key: "skill_extract:{hash(query_text)}"
    TTL: Configurable (default 1 hour)
    
    This avoids repeated LLM calls for identical queries.
    """
    logger.info("🤖 LLM extraction (will be cached for %d seconds)", LLM_CACHE_TTL)
    return _extract_skills_llm(query_text)


def _extract_skills_llm(query_text: str) -> Dict[str, Any]:
    """
    Extract skills and requirements using LLM with optimized prompting.
    
    Aligned with candidates_v3 schema for perfect matching.
    """
    # Check global rate limit BEFORE calling OpenAI
    if not global_limiter.check_openai_limit():
        logger.warning("OpenAI global rate limit exceeded, using fallback")
        return _fallback_extraction(query_text)
    client = get_openai_client()
    
    if not client:
        logger.warning("OpenAI client unavailable, using regex fallback")
        return _fallback_extraction(query_text)
    
    # Optimized system prompt (aligned with candidates_v3 fields)
    system_prompt = """You are an expert technical recruiter and skills analyst. Extract structured job requirements from text.

CRITICAL RULES:
1. Return ONLY valid JSON - no markdown, no explanations, no preamble
2. Extract TECHNICAL skills only (languages, frameworks, tools, platforms, databases, certifications)
3. NO soft skills (communication, teamwork, leadership, problem-solving)
4. Normalize skill names: "k8s" → "kubernetes", "lwc" → "lightning web components", "react.js" → "react"
5. Distinguish must-have (required/critical) from nice-to-have (preferred/bonus/plus)
6. Detect seniority from years of experience and job titles
7. Extract role type (e.g., "Backend Engineering", "Frontend Development", "Cloud Architecture")
8. Maximum 15 must-have skills (prioritize most important)
9. Lowercase all skills for consistency"""

    # Optimized user prompt with clear examples
    user_prompt = f"""Analyze this job requirement and extract structured data:

{query_text[:3000]}

Return this EXACT JSON format:
{{
  "must_have_skills": ["python", "django", "postgresql", "aws"],
  "nice_to_have_skills": ["redis", "docker", "kubernetes"],
  "seniority_level": "Senior",
  "industry": "Healthcare",
  "years_experience_min": 5,
  "role_type": "Backend Engineering",
  "remote_preference": "Remote"
}}

Field definitions:
- must_have_skills: Required technical skills (lowercase, normalized)
- nice_to_have_skills: Preferred/bonus skills (lowercase, normalized)
- seniority_level: MUST be one of: "Entry", "Mid", "Senior", "Lead/Manager", "Director+", "Any"
- industry: Only if explicitly mentioned (Healthcare, Finance, Technology, etc.) or null
- years_experience_min: Minimum years required (integer) or null
- role_type: Role category like "Backend Engineering", "Frontend Development", "DevOps", "Data Engineering", etc. or null
- remote_preference: "Remote", "Hybrid", "Onsite", or null if not mentioned"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,  # Deterministic output for better caching
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=15,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown code blocks if present
        content = _clean_json_response(content)
        
        # Parse JSON
        result = json.loads(content)
        
        # Normalize and validate skills
        result = _normalize_extraction_result(result)
        
        logger.info(
            "✅ LLM extracted: %d must-have, %d nice-to-have, seniority=%s, role=%s",
            len(result["must_have_skills"]),
            len(result["nice_to_have_skills"]),
            result.get("seniority_level", "Any"),
            result.get("role_type", "N/A")
        )
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from LLM: {e}\nContent: {content[:300]}")
        return _fallback_extraction(query_text)
        
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}, using fallback")
        return _fallback_extraction(query_text)


def _fallback_extraction(query_text: str) -> Dict[str, Any]:
    """
    Regex-based extraction when LLM is unavailable.
    
    Uses pattern matching and keyword detection for basic extraction.
    """
    text_lower = query_text.lower()
    skills = set()
    
    # Match against known skills (exact word boundaries)
    for skill in COMMON_TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            skills.add(skill)
    
    # Normalize found skills
    normalized_skills = []
    for skill in skills:
        normalized = SKILL_ALIASES.get(skill.lower(), skill)
        if normalized not in normalized_skills:
            normalized_skills.append(normalized)
    
    # Detect seniority
    seniority = _detect_seniority_fast(query_text)
    
    # Extract years of experience
    years_match = re.search(r'(\d+)\+?\s*years?', text_lower)
    years_min = int(years_match.group(1)) if years_match else None
    
    # Detect industry keywords
    industry = None
    industry_keywords = {
        "Healthcare": ["healthcare", "health", "medical", "hospital", "clinical", "hipaa", "hl7", "fhir"],
        "Finance": ["finance", "fintech", "banking", "trading", "investment", "blockchain"],
        "Technology": ["tech", "software", "saas", "b2b"],
        "Retail": ["retail", "e-commerce", "ecommerce", "shopping"],
        "Education": ["education", "edtech", "learning", "university"],
    }
    
    for ind, keywords in industry_keywords.items():
        if any(kw in text_lower for kw in keywords):
            industry = ind
            break
    
    # Detect role type
    role_type = None
    role_keywords = {
        "Backend Engineering": ["backend", "server-side", "api development"],
        "Frontend Development": ["frontend", "ui", "client-side"],
        "Full Stack": ["full stack", "fullstack", "full-stack"],
        "DevOps": ["devops", "infrastructure", "sre", "site reliability"],
        "Data Engineering": ["data engineering", "etl", "data pipeline"],
        "Cloud Architecture": ["cloud architect", "cloud infrastructure"],
    }
    
    for role, keywords in role_keywords.items():
        if any(kw in text_lower for kw in keywords):
            role_type = role
            break
    
    # Detect remote preference
    remote_pref = None
    if any(word in text_lower for word in ["remote", "work from home", "wfh"]):
        remote_pref = "Remote"
    elif any(word in text_lower for word in ["hybrid"]):
        remote_pref = "Hybrid"
    elif any(word in text_lower for word in ["onsite", "on-site", "in-office"]):
        remote_pref = "Onsite"
    
    return {
        "must_have_skills": sorted(normalized_skills)[:15],
        "nice_to_have_skills": [],
        "seniority_level": seniority,
        "industry": industry,
        "years_experience_min": years_min,
        "role_type": role_type,
        "remote_preference": remote_pref,
    }


def _detect_seniority_fast(query_text: str) -> str:
    """
    Fast seniority detection using pattern matching.
    
    Checks patterns in order of specificity to avoid misclassification.
    Director+ patterns checked first to avoid matching "Senior Director" as "Senior".
    """
    text_lower = query_text.lower()
    
    # Check each level in order of specificity (most specific first)
    for level, patterns in SENIORITY_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                logger.debug(f"Detected seniority: {level} (matched: '{pattern}')")
                return level
    
    return "Any"


def _clean_json_response(content: str) -> str:
    """
    Clean LLM response to extract valid JSON.
    
    Removes markdown code blocks, extra whitespace, and common formatting issues.
    """
    # Remove markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Find and remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        elif lines and "```" in lines[-1]:
            lines[-1] = lines[-1].replace("```", "")
        content = "\n".join(lines)
    
    return content.strip()


def _normalize_extraction_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate extraction results for candidates_v3 schema.
    
    - Lowercase all skills
    - Apply skill aliases/normalization
    - Remove duplicates
    - Validate data types
    - Ensure schema compatibility
    """
    # Normalize must-have skills
    must_have = result.get("must_have_skills", [])
    if isinstance(must_have, list):
        normalized = []
        for skill in must_have:
            if isinstance(skill, str) and skill.strip():
                # Lowercase and normalize
                normalized_skill = SKILL_ALIASES.get(
                    skill.lower().strip(),
                    skill.lower().strip()
                )
                if normalized_skill not in normalized:
                    normalized.append(normalized_skill)
        result["must_have_skills"] = normalized[:15]  # Max 15
    else:
        result["must_have_skills"] = []
    
    # Normalize nice-to-have skills
    nice_to_have = result.get("nice_to_have_skills", [])
    if isinstance(nice_to_have, list):
        normalized = []
        for skill in nice_to_have:
            if isinstance(skill, str) and skill.strip():
                normalized_skill = SKILL_ALIASES.get(
                    skill.lower().strip(),
                    skill.lower().strip()
                )
                # Don't add if already in must-have
                if normalized_skill not in result["must_have_skills"] and normalized_skill not in normalized:
                    normalized.append(normalized_skill)
        result["nice_to_have_skills"] = normalized
    else:
        result["nice_to_have_skills"] = []
    
    # Validate seniority level
    valid_seniority = ["Entry", "Mid", "Senior", "Lead/Manager", "Director+", "Any"]
    if result.get("seniority_level") not in valid_seniority:
        result["seniority_level"] = "Any"
    
    # Ensure years is int or None
    years = result.get("years_experience_min")
    if years is not None:
        try:
            result["years_experience_min"] = int(years)
        except (ValueError, TypeError):
            result["years_experience_min"] = None
    
    # Validate remote preference
    valid_remote = ["Remote", "Hybrid", "Onsite", None]
    if result.get("remote_preference") not in valid_remote:
        result["remote_preference"] = None
    
    # Ensure role_type and industry are strings or None
    if result.get("role_type") and not isinstance(result["role_type"], str):
        result["role_type"] = None
    
    if result.get("industry") and not isinstance(result["industry"], str):
        result["industry"] = None
    
    return result


# Backward compatibility function
def extract_skills_only(query_text: str) -> Dict[str, List[str]]:
    """
    Simplified interface for just skill extraction.
    
    Returns:
        {"must_have": [...], "nice_to_have": [...]}
    """
    result = extract_requirements(query_text)
    return {
        "must_have": result["must_have_skills"],
        "nice_to_have": result["nice_to_have_skills"],
    }