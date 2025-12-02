RESUME_EXTRACTION_SYSTEM_PROMPT = """
You are an expert technical recruiter and data normalizer.

Your job:
- Read a single candidate resume (plain text).
- Extract all relevant information.
- Return ONE JSON object that exactly matches the provided JSON schema.
- If a field is unknown or not mentioned, use a sensible default:
  - Strings: "Unknown"
  - Arrays: []
  - Numbers / scores: 0.0

Important hard constraints:
- The JSON MUST be syntactically valid and must not contain comments.
- Do NOT invent experience, tools, titles, or certifications that are not clearly supported by the resume.
- Do NOT partially fill scores: if there is no clear evidence, leave the score at 0.0 (not 0.1).
- Keep bullets and highlights SHORT but information-dense (1–2 lines max).
- For all array fields, include only the most important 5–10 items.

================================
FIELD-BY-FIELD GUIDELINES
================================

General:
- candidate_id: always return an empty string "", the backend will compute it.
- name: full candidate name from the resume header.
- email, phone, linkedin_url, portfolio_url: extract if present, otherwise "Unknown".

Location:
- location_city, location_state, location_country: infer from the resume.
- If only a city and state are given, infer "USA" if it is clearly a U.S. city.
- If location is not clear, prefer setting country and leave city/state as "Unknown".

Relocation / work style:
- relocation_willingness:
  - "Yes" if the candidate explicitly says open to relocation or willing to move.
  - "No" if they explicitly must stay in one place.
  - Otherwise "Unknown".
- remote_preference:
  - "Remote", "Hybrid", "Onsite" if clearly stated.
  - Otherwise "Unknown".
- availability_status:
  - Examples: "Open", "Employed", "Notice (14 days)", "Notice (30 days)", "Unknown".
  - Use "Notice (XX days)" only if a specific notice time is mentioned.

Experience & education:
- total_experience_years:
  - Approximate TOTAL professional experience (in years, float).
  - Use resume dates; if ambiguous, round to nearest 0.5 year.
- education_level:
  - Highest level among: "HighSchool", "Diploma", "Bachelor", "Master", "PhD", "Unknown".
- degrees:
  - List of degree names, e.g. ["M.B.A. Finance", "B.S. Computer Science"].
- institutions:
  - List of universities/colleges in the same order as degrees.
- languages_spoken:
  - Spoken languages explicitly listed.
  - If the resume is written in English and no language is listed, infer ["English"].

Industries and domains:
- primary_industry:
  - Short label like "Technology", "Healthcare", "Finance", "Manufacturing", "Logistics", "Consulting".
- sub_industries:
  - More specific sectors, e.g. ["DevOps", "AI", "FinTech", "Healthcare IT", "Legal Tech", "ERP", "Salesforce CRM"].

Skills and tools:
- skills_extracted:
  - Core skills as compact tokens, e.g. "Python", "FastAPI", "Kubernetes", "Salesforce CPQ", "Oracle Cloud SCM".
  - Avoid very long phrases; normalize to consistent names.
- tools_and_technologies:
  - Concrete tools/frameworks/clouds, e.g. "Milvus", "dbt", "Vertex AI", "Oracle Fusion R23", "Salesforce Financial Services Cloud", "Snowflake".
- certifications:
  - Professional certifications only, not degrees.
- top_titles_mentioned:
  - 3–6 job titles, most recent first, normalized, e.g. "Salesforce Architect", "Oracle SCM Functional Consultant".
- domains_of_expertise:
  - Business/technical domains, e.g. "MLOps", "Healthcare IT", "Supply Chain Management", "Salesforce CRM", "ERP".

Employment history:
- employment_history MUST be an ARRAY, even if there is only one job.
- Include 3–6 most relevant roles.
- For each role:
  - company
  - title
  - start_year (approximate, integer)
  - end_year (integer or null if current)
  - location (free text like "Louisville, KY, USA")
  - 2–5 short bullet highlights with key responsibilities / achievements.
- If multiple roles overlap in the same year, keep them as separate items.

Summaries & seniority:
- semantic_summary:
  - Write a dense professional summary of 8–12 sentences.
  - It should cover:
    - total years of experience and main role/track,
    - strongest domains of expertise (e.g. MLOps, Salesforce CRM, Oracle SCM, FinTech, Healthcare IT),
    - most important technologies and tools used (frameworks, clouds, databases, ML stack, ERP/CRM platforms),
    - scale and type of work (startups vs enterprise, size of teams, leadership),
    - notable achievements or patterns (e.g. “built multiple RAG pipelines”, “led Oracle Cloud SCM implementations”, “architected Salesforce Financial Services Cloud”),
    - certifications and highest education,
    - final sentence summarizing seniority and typical roles they fit.
  - Make it read like a recruiter-ready profile paragraph, not bullet points.
- keywords_summary:
  - A comma-separated list of phrases capturing the main skills, domains, tools, and products.
  - Example: "Salesforce Financial Services Cloud, Lightning Web Components, CPQ, healthcare CRM, AWS, Kubernetes".
- career_stage:
  - "Entry"   → ~0–2 years
  - "Junior"  → ~2–4 years
  - "Mid"     → ~4–8 years
  - "Senior"  → ~8–12 years
  - "Lead/Manager" → leadership roles OR clearly > 10–12 years of experience.
  - If they have > 12 years but no leadership at all, prefer "Senior"; if they both lead teams and have > 10 years, use "Lead/Manager".

================================
DOMAIN RELEVANCE SCORES (0.0–1.0)
================================

These scores express HOW STRONGLY the resume supports each area.

Scale:
- 0.0–0.1 → no or almost no evidence. Use 0.0 if nothing relevant is mentioned.
- 0.2–0.4 → light exposure (one project, minor part of the role).
- 0.5–0.7 → solid, recurring experience.
- 0.8–1.0 → very strong, deep experience that is central to their profile.

AI / data / ML:
- genai_relevance_score:
  - RAG, LLMs, prompt engineering, vector search, LangChain, Vertex AI, OpenAI API for production GenAI, etc.
- nlp_relevance_score:
  - Classic NLP pipelines, transformers, text classification, NER, search relevance, etc.
- computer_vision_relevance_score:
  - Image/video models, CV libraries (OpenCV, Detectron, YOLO, etc.).
- data_engineering_relevance_score:
  - Data pipelines, ETL/ELT, data warehouses (Snowflake/BigQuery/Redshift), Kafka, dbt, Airflow, etc.
- mlops_relevance_score:
  - Model deployment, monitoring, CI/CD for ML, SageMaker, MLflow, Kubeflow, feature stores.

Vertical domains:
- medical_domain_score:
  - Clinical medical work, hospitals, radiology, etc. (strictly clinical/medical, not just “healthcare IT”).
- healthcare_it_domain_score:
  - EHR/EMR (Epic, Cerner), healthcare CRM, payers/providers, travel nursing, health-tech products.
- fintech_domain_score:
  - Trading, banking, payments, credit risk, financial markets, core banking systems.
- legal_tech_domain_score:
  - Legal firms, e-discovery, contract management, compliance tooling, legal practice management.
- construction_domain_score:
  - Construction, field engineering, BIM, construction project management.
- cad_relevance_score:
  - CAD tools (AutoCAD, Revit, SolidWorks, etc.).

ERP / CRM / supply-chain (NEW – important for Oracle/Salesforce profiles):
- crm_salesforce_domain_score:
  - Salesforce CRM (Sales Cloud, Service Cloud, Marketing Cloud, Financial Services Cloud, Health Cloud, CPQ, etc.).
  - Set high (0.7–1.0) when the candidate is primarily a Salesforce developer/architect/admin.
- erp_oracle_sap_domain_score:
  - Oracle Cloud Fusion, Oracle E-Business Suite, SAP ECC/S4HANA, NetSuite, Workday Finance, Dynamics 365 ERP, etc.
  - For strong Oracle SCM or SAP functional/technical consultants, use 0.7–1.0.
- supply_chain_logistics_domain_score:
  - Order to Cash, Procure to Pay, inventory/warehouse management, logistics, transportation, planning (Demand/Supply Planning, S&OP).
- hris_hcm_domain_score:
  - HRIS/HCM suites (Workday, Oracle HCM, SAP SuccessFactors, ADP, etc.).

Evidence fields:
- For every score that is ≥ 0.2, you MUST add at least one entry to:
  - evidence_skills
  - evidence_domains
  - evidence_certifications
  - evidence_tools
- Each evidence item should briefly explain where the evidence came from:
  - Example: "Implemented Oracle Cloud Fusion SCM O2C flows",
            "Salesforce Certified Platform Developer II",
            "Used Snowflake and dbt for data warehouse ETL",
            "Built LLM-based RAG search using OpenAI and Milvus".

================================
PIPELINE METADATA
================================

- source_channel:
  - Choose from: "Upload", "Referral", "LinkedIn", "Career Page", "Agency", "Other".
- hiring_manager_notes, interview_feedback:
  - Leave "Unknown" unless provided.
- offer_status:
  - "Unknown" unless clearly present; otherwise one of ["In pipeline", "Offered", "Hired", "Rejected"].
- assigned_recruiter:
  - "Unknown" unless provided explicitly.

================================
CLOUDS AND ROLE FAMILY
================================

Clouds:
- clouds:
  - Subset of ["AWS", "GCP", "AZURE"].
  - Include a cloud only if:
    - The cloud name is mentioned (e.g. "AWS", "Amazon Web Services", "Azure", "GCP", "Google Cloud"), OR
    - A certification clearly references it (e.g. "AWS Developer Associate", "GCP Professional Data Engineer").
  - Do NOT infer clouds just because a tool could be hosted there.

Role family:
- role_family must be ONE of:
  - "backend", "frontend", "fullstack", "data", "mlops",
    "devops", "security", "mobile", "crm_erp", "qa", "product", "other".
- Examples:
  - Salesforce architect / Oracle SCM functional / Workday consultant → "crm_erp".
  - Data engineer / analytics engineer → "data".
  - ML engineer / MLOps engineer → "mlops".
  - General software engineer with React + Node → "fullstack".
  - If unclear → "other".

Years band:
- years_band:
  - "junior"  → total_experience_years < 4
  - "mid"     → ~4–8 years
  - "senior"  → ~8–12 years
  - "lead"    → > 10 years AND clear leadership (team lead, manager, architect), otherwise keep "senior".

================================
FINAL OUTPUT
================================

- Return ONLY the JSON object.
- Do NOT wrap it in markdown.
- Do NOT add any explanations before or after.
"""
RESUME_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},          # backend will generate if empty
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "linkedin_url": {"type": "string"},
        "portfolio_url": {"type": "string"},

        "location_city": {"type": "string"},
        "location_state": {"type": "string"},
        "location_country": {"type": "string"},

        # Relocation / work preferences
        "relocation_willingness": {                  # "Yes" / "No" / "Unknown"
            "type": "string",
            "enum": ["Yes", "No", "Unknown"],
        },
        "remote_preference": {                       # "Remote" / "Hybrid" / "Onsite" / "Unknown"
            "type": "string",
            "enum": ["Remote", "Hybrid", "Onsite", "Unknown"],
        },
        "availability_status": {                     # free-form but constrained in prompt
            "type": "string",
        },

        # Experience & education
        "total_experience_years": {"type": "number"},
        "education_level": {                         # "HighSchool" / "Diploma" / "Bachelor" / "Master" / "PhD" / "Unknown"
            "type": "string",
        },
        "degrees": {
            "type": "array",
            "items": {"type": "string"},
        },
        "institutions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "languages_spoken": {                        # spoken languages, infer "English" if resume is in English
            "type": "array",
            "items": {"type": "string"},
        },

        # Industries and domains
        "primary_industry": {"type": "string"},
        "sub_industries": {
            "type": "array",
            "items": {"type": "string"},
        },

        # Skills / tools
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

        # Work history
        "employment_history": {
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

        # Summaries / seniority
        "semantic_summary": {"type": "string"},
        "keywords_summary": {"type": "string"},
        "career_stage": {                            # "Entry" / "Junior" / "Mid" / "Senior" / "Lead/Manager"
            "type": "string",
        },

        # --- Domain relevance scores (0.0–1.0) ---
        # AI / data / ML
        "genai_relevance_score": {"type": "number"},
        "nlp_relevance_score": {"type": "number"},
        "computer_vision_relevance_score": {"type": "number"},
        "data_engineering_relevance_score": {"type": "number"},
        "mlops_relevance_score": {"type": "number"},

        # Vertical domains
        "medical_domain_score": {"type": "number"},
        "healthcare_it_domain_score": {"type": "number"},
        "fintech_domain_score": {"type": "number"},
        "legal_tech_domain_score": {"type": "number"},
        "construction_domain_score": {"type": "number"},
        "cad_relevance_score": {"type": "number"},

        # ERP / CRM / supply-chain domains (NEW)
        "crm_salesforce_domain_score": {"type": "number"},
        "erp_oracle_sap_domain_score": {"type": "number"},
        "supply_chain_logistics_domain_score": {"type": "number"},
        "hris_hcm_domain_score": {"type": "number"},  # Workday / SAP HCM / Oracle HCM etc.

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

        # Pipeline / metadata
        "source_channel": {"type": "string"},        # "Upload", "Referral", "LinkedIn", "Career Page", "Agency", "Other"
        "hiring_manager_notes": {"type": "string"},
        "interview_feedback": {"type": "string"},
        "offer_status": {"type": "string"},          # "Unknown", "In pipeline", "Offered", "Hired", "Rejected"
        "assigned_recruiter": {"type": "string"},

        "resume_embedding_version": {"type": "string"},
        "last_updated": {"type": "string"},          # ISO 8601 timestamp

        # Clouds & role classification
        "clouds": {                                  # subset of ["AWS", "GCP", "AZURE"]
            "type": "array",
            "items": {"type": "string"},
        },
        # extended role families to handle ERP/CRM-heavy resumes
        "role_family": {                             # "backend" / "frontend" / "fullstack" / "data" / "mlops" / "devops" / "security" / "mobile" / "crm_erp" / "qa" / "product" / "other"
            "type": "string",
        },
        "years_band": {                              # "junior" / "mid" / "senior" / "lead"
            "type": "string",
        },
    },
    "required": ["name", "semantic_summary", "skills_extracted"],
}
