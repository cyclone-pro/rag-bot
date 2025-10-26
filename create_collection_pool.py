# ---------- Milvus collection setup (HNSW + COSINE) ----------
# Requires: pip install pymilvus==2.4.4 sentence-transformers==2.7.0 torch --extra-index-url https://download.pytorch.org/whl/cpu
# Local embedding model: intfloat/e5-base-v2 (768d)

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

# 1) Connect to Milvus / Zilliz
# Adjust to your deployment
connections.connect(
    alias="default",
    uri="http://34.135.232.156:19530",   # or "https://YOUR-ZILLIZ-ENDPOINT"
    token="",                       # if using Zilliz Cloud, set your API key here
)

# 2) Define schema
# We'll store two vector fields to support different retrieval flavors:
#   - summary_embedding: embedding of the candidate's summary/profile text
#   - skills_embedding:  embedding of the candidate's skills/tools text
dim = 768
candidate_pool = "candidate_pool"

if utility.has_collection(candidate_pool):
    utility.drop_collection(candidate_pool)

fields = [
    FieldSchema(name="candidate_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64, description="Primary key"),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="phone", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="linkedin_url", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="portfolio_url", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="location_city", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location_state", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location_country", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="relocation_willingness", dtype=DataType.VARCHAR, max_length=16),
    FieldSchema(name="remote_preference", dtype=DataType.VARCHAR, max_length=16),
    FieldSchema(name="availability_status", dtype=DataType.VARCHAR, max_length=32),

    FieldSchema(name="total_experience_years", dtype=DataType.FLOAT),
    FieldSchema(name="education_level", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="degrees", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="institutions", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="languages_spoken", dtype=DataType.VARCHAR, max_length=512),

    FieldSchema(name="primary_industry", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="sub_industries", dtype=DataType.VARCHAR, max_length=1024),

    FieldSchema(name="skills_extracted", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="tools_and_technologies", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="certifications", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="top_titles_mentioned", dtype=DataType.VARCHAR, max_length=1024),

    FieldSchema(name="domains_of_expertise", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="employment_history", dtype=DataType.VARCHAR, max_length=8192),
    FieldSchema(name="semantic_summary", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="keywords_summary", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="career_stage", dtype=DataType.VARCHAR, max_length=32),

    FieldSchema(name="genai_relevance_score", dtype=DataType.FLOAT),
    FieldSchema(name="medical_domain_score", dtype=DataType.FLOAT),
    FieldSchema(name="construction_domain_score", dtype=DataType.FLOAT),
    FieldSchema(name="cad_relevance_score", dtype=DataType.FLOAT),
    FieldSchema(name="nlp_relevance_score", dtype=DataType.FLOAT),
    FieldSchema(name="computer_vision_relevance_score", dtype=DataType.FLOAT),
    FieldSchema(name="data_engineering_relevance_score", dtype=DataType.FLOAT),
    FieldSchema(name="mlops_relevance_score", dtype=DataType.FLOAT),

    FieldSchema(name="evidence_skills", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="evidence_domains", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="evidence_certifications", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="evidence_tools", dtype=DataType.VARCHAR, max_length=1024),

    FieldSchema(name="source_channel", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="hiring_manager_notes", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="interview_feedback", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="offer_status", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="assigned_recruiter", dtype=DataType.VARCHAR, max_length=64),

    FieldSchema(name="resume_embedding_version", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="last_updated", dtype=DataType.VARCHAR, max_length=32),

    # Vector fields
    FieldSchema(name="summary_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="skills_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
]

schema = CollectionSchema(fields=fields, description="Candidate pool with dual vector fields (summary & skills)")

col = Collection(
    name=candidate_pool,
    schema=schema,
    using="default",
    shards_num=2
)

# 3) Create HNSW indexes (COSINE)
hnsw_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
col.create_index(field_name="summary_embedding", index_params=hnsw_params)
col.create_index(field_name="skills_embedding", index_params=hnsw_params)

# 4) Load for search
col.load()
print(f"Collection `{candidate_pool}` is ready.")
