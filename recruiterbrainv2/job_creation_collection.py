from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType, Collection, utility
)
import os

# ==================== CONFIGURATION ====================
MILVUS_HOST = os.getenv("MILVUS_HOST", "34.55.41.188")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
EMBEDDING_DIM = 768

# ==================== CONNECT FUNCTION ====================
def connect_milvus():
    """Connect to Milvus instance."""
    if MILVUS_TOKEN:
        connections.connect(
            alias="default",
            uri=f"{MILVUS_HOST}:{MILVUS_PORT}",
            token=MILVUS_TOKEN
        )
    else:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
    print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

# ---------------------- CALL IT ----------------------
connect_milvus()
# -----------------------------------------------------

# If collection already exists, drop it first (optional)
if utility.has_collection("job_postings"):
    print("⚠️ job_postings already exists. Dropping...")
    utility.drop_collection("job_postings")

# ==================== DEFINE FIELDS ====================
job_id = FieldSchema(
    name="job_id",
    dtype=DataType.VARCHAR,
    is_primary=True,
    max_length=64,
)

title = FieldSchema(
    name="title",
    dtype=DataType.VARCHAR,
    max_length=256,
)

company = FieldSchema(
    name="company",
    dtype=DataType.VARCHAR,
    max_length=256,
)

location = FieldSchema(
    name="location",
    dtype=DataType.VARCHAR,
    max_length=256,
)

posted_ts = FieldSchema(
    name="posted_ts",
    dtype=DataType.INT64,
)

status = FieldSchema(
    name="status",
    dtype=DataType.VARCHAR,
    max_length=64,
)

jd_text = FieldSchema(
    name="jd_text",
    dtype=DataType.VARCHAR,
    max_length=65535,
)

jd_embedding = FieldSchema(
    name="jd_embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=768,
)

# ==================== CREATE COLLECTION ====================
job_postings_schema = CollectionSchema(
    fields=[job_id, title, company, location, posted_ts, status, jd_text, jd_embedding],
    description="Job postings collection"
)

job_postings = Collection(
    name="job_postings",
    schema=job_postings_schema,
    using="default",
    shards_num=2
)

print("✅ Collection created:", job_postings.name)

# ==================== CREATE VECTOR INDEX ====================
index_params = {
    "metric_type": "IP",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64}
}

job_postings.create_index(
    field_name="jd_embedding",
    index_params=index_params
)

print("✅ Index created on jd_embedding")
