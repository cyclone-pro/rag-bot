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

def connect_milvus():
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

connect_milvus()

# Drop old collection if exists
if utility.has_collection("job_postings"):
    print("⚠️ job_postings already exists. Dropping…")
    utility.drop_collection("job_postings")

# ==================== DEFINE FIELDS ====================
job_id         = FieldSchema(name="job_id",         dtype=DataType.VARCHAR, is_primary=True, max_length=64)
title          = FieldSchema(name="title",          dtype=DataType.VARCHAR, max_length=256)
company        = FieldSchema(name="company",        dtype=DataType.VARCHAR, max_length=256)
department     = FieldSchema(name="department",     dtype=DataType.VARCHAR, max_length=128)
location       = FieldSchema(name="location",       dtype=DataType.VARCHAR, max_length=256)
employment_type = FieldSchema(name="employment_type", dtype=DataType.VARCHAR, max_length=32)  # full_time/contract/etc
tax_term       = FieldSchema(name="tax_term",       dtype=DataType.VARCHAR, max_length=16)   # w2/c2c/1099/na
salary_range   = FieldSchema(name="salary_range",   dtype=DataType.VARCHAR, max_length=128)
req_id         = FieldSchema(name="req_id",         dtype=DataType.VARCHAR, max_length=64)
status         = FieldSchema(name="status",         dtype=DataType.VARCHAR, max_length=64)
posted_ts      = FieldSchema(name="posted_ts",      dtype=DataType.INT64)
jd_text        = FieldSchema(name="jd_text",        dtype=DataType.VARCHAR, max_length=65535)
jd_embedding   = FieldSchema(name="jd_embedding",   dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

job_postings_schema = CollectionSchema(
    fields=[
        job_id,
        title,
        company,
        department,
        location,
        employment_type,
        tax_term,
        salary_range,
        req_id,
        status,
        posted_ts,
        jd_text,
        jd_embedding,
    ],
    description="Job postings collection"
)

job_postings = Collection(
    name="job_postings",
    schema=job_postings_schema,
    using="default",
    shards_num=2,
)

print("✅ Collection created:", job_postings.name)

index_params = {
    "metric_type": "IP",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}

job_postings.create_index(
    field_name="jd_embedding",
    index_params=index_params
)

print("✅ Index created on jd_embedding")
