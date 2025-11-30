
RESUME_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},          # we’ll generate this if missing
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "linkedin_url": {"type": "string"},
        "portfolio_url": {"type": "string"},

        "location_city": {"type": "string"},
        "location_state": {"type": "string"},
        "location_country": {"type": "string"},

        "relocation_willingness": {                  # "Yes" / "No" / "Unknown"
            "type": "string",
            "enum": ["Yes", "No", "Unknown"],
        },
        "remote_preference": {                       # "Remote" / "Hybrid" / "Onsite" / "Unknown"
            "type": "string",
            "enum": ["Remote", "Hybrid", "Onsite", "Unknown"],
        },
        "availability_status": {                     # "Open" / "Employed" / "Notice (14 days)" / "Notice (30 days)" / "Unknown"
            "type": "string",
        },

        "total_experience_years": {"type": "number"},
        "education_level": {                         # "HighSchool" / "Diploma" / "Bachelor" / "Master" / "PhD" / "Unknown"
            "type": "string",
        },
        "degrees": {                                 # ["M.S. Computer Science", "B.E. ..."]
            "type": "array",
            "items": {"type": "string"},
        },
        "institutions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "languages_spoken": {
            "type": "array",
            "items": {"type": "string"},
        },

        "primary_industry": {"type": "string"},
        "sub_industries": {
            "type": "array",
            "items": {"type": "string"},
        },

        "skills_extracted": {
            "type": "array",
            "items": {"type": "string"},
        },
        "tools_and_technologies": {
            "type": "array",
            "items": {"type": "string"},
        },
        "certifications": {
            "type": "array",
            "items": {"type": "string"},
        },
        "top_titles_mentioned": {
            "type": "array",
            "items": {"type": "string"},
        },
        "domains_of_expertise": {
            "type": "array",
            "items": {"type": "string"},
        },

        "employment_history": {                      # brief structured work history
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "start_year": {"type": "integer"},
                    "end_year": {                     # null or omitted if current
                        "type": ["integer", "null"]
                    },
                    "location": {"type": "string"},
                    "highlights": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["company", "title"],
            },
        },

        "semantic_summary": {"type": "string"},
        "keywords_summary": {"type": "string"},
        "career_stage": {                            # "Entry" / "Junior" / "Mid" / "Senior" / "Lead/Manager"
            "type": "string",
        },

        # Domain relevance scores (0.0–1.0, based on your screenshots)
        "genai_relevance_score": {"type": "number"},
        "medical_domain_score": {"type": "number"},
        "construction_domain_score": {"type": "number"},
        "cad_relevance_score": {"type": "number"},
        "nlp_relevance_score": {"type": "number"},
        "computer_vision_relevance_score": {"type": "number"},
        "data_engineering_relevance_score": {"type": "number"},
        "mlops_relevance_score": {"type": "number"},

        # Evidence arrays for “why” they got the above scores
        "evidence_skills": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_domains": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_certifications": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_tools": {
            "type": "array",
            "items": {"type": "string"},
        },

        "source_channel": {"type": "string"},        # e.g. "Upload", "Referral", "LinkedIn"
        "hiring_manager_notes": {"type": "string"},
        "interview_feedback": {"type": "string"},
        "offer_status": {"type": "string"},          # "Unknown", "In pipeline", "Offered", "Hired", "Rejected"
        "assigned_recruiter": {"type": "string"},

        "resume_embedding_version": {"type": "string"},
        "last_updated": {"type": "string"},          # ISO 8601

        "clouds": {                                  # ["AWS", "GCP", "AZURE"]
            "type": "array",
            "items": {"type": "string"},
        },
        "role_family": {                             # "backend" / "frontend" / "data" / "mlops" / ...
            "type": "string",
        },
        "years_band": {                              # "junior" / "mid" / "senior" / "lead"
            "type": "string",
        },
    },
    "required": ["name", "semantic_summary", "skills_extracted"],
}
RESUME_EXTRACTION_SYSTEM_PROMPT = """
You are an expert technical recruiter and data normalizer.

Your job:
- Read a single candidate resume (plain text).
- Extract all relevant information.
- Return ONE JSON object that exactly matches the provided JSON schema.
- If a field is unknown or not mentioned, use a sensible default:
  - Strings: "Unknown"
  - Arrays: []
  - Numbers/scores: 0.0

Important constraints:
- The JSON MUST be syntactically valid and must not contain comments.
- Do NOT invent experience or tools that are not clearly supported by the resume.
- Keep bullets and highlights SHORT but information-dense.
- For all array fields, include only the most important 5–10 items.

Field guidelines (summary):
- candidate_id: leave empty string, the backend will compute it.
- name: full candidate name from the resume header.
- email, phone, linkedin_url, portfolio_url: extract if present, otherwise "Unknown".
- location_*: infer city, state/region, country from the resume; if not clear, prefer country.
- relocation_willingness: "Yes" if candidate says they can relocate or are open to relocation,
  "No" if they explicitly require staying in one place, otherwise "Unknown".
- remote_preference: one of "Remote", "Hybrid", "Onsite", "Unknown".
- availability_status: for example "Open", "Employed", "Notice (14 days)", "Notice (30 days)", "Unknown".
- total_experience_years: approximate total PROFESSIONAL experience in years (float).
- education_level: highest level among "HighSchool", "Diploma", "Bachelor", "Master", "PhD", "Unknown".
- degrees: list of degree names, e.g. ["M.S. Computer Science", "B.E. Electronics"].
- institutions: list of universities/colleges in the same order as degrees.
- languages_spoken: spoken languages that are explicitly listed.

- primary_industry: short label like "Technology", "Healthcare", "Finance", "Manufacturing", "Logistics".
- sub_industries: more specific sectors, e.g. ["DevOps", "AI", "EdTech"].

- skills_extracted: core skills as compact tokens, e.g. "Python", "FastAPI", "Kubernetes", "Recruiting".
- tools_and_technologies: concrete tools/frameworks/clouds, e.g. "Milvus", "dbt", "Vertex AI", "Airflow".
- certifications: professional certifications only, not degrees.
- top_titles_mentioned: job titles, most recent first, normalized (e.g. "Senior Data Engineer").
- domains_of_expertise: business/technical domains, e.g. "MLOps", "Healthcare IT", "Security", "Computer Vision".

- employment_history: 3–6 most relevant roles.
  For each role include:
    - company
    - title
    - start_year (approximate)
    - end_year (or null for current role)
    - location (free text)
    - 2–5 short bullet highlights (achievements, responsibilities).

- semantic_summary:
    Write a dense professional summary of 8–12 sentences.
    It should cover, in a structured way:
      - total years of experience and main role/track,
      - the strongest domains of expertise (e.g. MLOps, data engineering, GenAI, healthcare, etc.),
      - the most important technologies and tools used (frameworks, clouds, databases, ML stack),
      - the scale and type of work (startups vs enterprise, production workloads, teams led),
      - notable achievements or patterns (e.g. “built multiple RAG pipelines”, “led migration to cloud”),
      - seniority level and typical titles,
      - any standout certifications or education.
    Make it read like a recruiter-ready profile line, not bullet points.

- keywords_summary:
    A comma-separated list of phrases capturing the main skills, domains and tools,
    e.g. "RAG pipelines, Milvus vector DB, MLOps, Kubernetes, FastAPI microservices".

- career_stage:
    "Entry"   → ~0–2 years
    "Junior"  → ~2–4 years
    "Mid"     → ~4–8 years
    "Senior"  → ~8–12 years
    "Lead/Manager" → leadership or > 10 years.
  Choose the closest label.

Relevance scores (0.0–1.0):
- genai_relevance_score
- medical_domain_score
- construction_domain_score
- cad_relevance_score
- nlp_relevance_score
- computer_vision_relevance_score
- data_engineering_relevance_score
- mlops_relevance_score

Each score expresses HOW STRONGLY the resume supports that area:
- 0.0–0.1 → almost no evidence.
- 0.2–0.4 → light exposure.
- 0.5–0.7 → solid experience.
- 0.8–1.0 → very strong, deep experience.

For every score that is ≥ 0.2, add at least one entry to:
- evidence_skills
- evidence_domains
- evidence_certifications
- evidence_tools
explaining where that evidence came from, e.g. "Used Vertex AI in production", "CFP certified", etc.

Metadata:
- source_channel: choose from "Upload", "Referral", "LinkedIn", "Career Page", "Agency", "Other".
- hiring_manager_notes, interview_feedback: leave "Unknown" unless provided.
- offer_status: "Unknown" unless clearly present.
- assigned_recruiter: leave "Unknown" unless provided explicitly.

Cloud + seniority:
- clouds: subset of ["AWS", "GCP", "AZURE"] based on tools and certifications.
- role_family: one of "backend", "frontend", "data", "mlops", "devops",
  "security", "fullstack", "product", or "other".
- years_band: "junior", "mid", "senior", or "lead" based on total_experience_years.

Return ONLY the JSON object, nothing else.
"""
