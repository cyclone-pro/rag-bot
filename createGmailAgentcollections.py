"""
Create Milvus collections for Gmail agent:
- conv_recruiter
- conv_candidate
- applications

Requires:
    pip install pymilvus
"""

from pymilvus import (
    connections,
    db,
    FieldSchema, CollectionSchema, DataType, Collection
)

# ==========================
# CONFIG
# ==========================
MILVUS_ALIAS = "default"
MILVUS_HOST = "localhost"      # TODO: change to your Milvus host / IP
MILVUS_PORT = "19530"          # default port
MILVUS_DB   = "default"        # database name

EMBEDDING_DIM = 768

INDEX_TYPE = "HNSW"
METRIC_TYPE = "COSINE"
INDEX_PARAMS = {"M": 32, "efConstruction": 200}


def connect_milvus():
    connections.connect(
        alias=MILVUS_ALIAS,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    try:
        db.using_database(MILVUS_DB, using=MILVUS_ALIAS)
    except Exception:
        db.create_database(MILVUS_DB, using=MILVUS_ALIAS)
        db.using_database(MILVUS_DB, using=MILVUS_ALIAS)


def create_conv_recruiter():
    name = "conv_recruiter"
    if Collection.exists(name, using=MILVUS_ALIAS):
        print(f"[{name}] already exists, skipping.")
        return

    fields = [
        FieldSchema("conv_id", DataType.VARCHAR, is_primary=True, auto_id=False,
                    max_length=64, description="Unique conversation message id"),
        FieldSchema("vendor_id", DataType.VARCHAR, max_length=64,
                    description="Hashed vendor/company id"),
        FieldSchema("recruiter_id", DataType.VARCHAR, max_length=64,
                    description="Hashed recruiter id"),
        FieldSchema("job_id", DataType.VARCHAR, max_length=128,
                    description="Internal job id"),
        FieldSchema("req_key", DataType.VARCHAR, max_length=512,
                    description="sender_email::gmail_thread_id::normalized_title"),
        FieldSchema("role", DataType.VARCHAR, max_length=16,
                    description="'user' or 'agent'"),
        FieldSchema("ts_utc", DataType.INT64,
                    description="Unix timestamp (ms or s)"),
        FieldSchema("text_redacted", DataType.VARCHAR, max_length=4000,
                    description="Email text with PII masked"),
        FieldSchema("conv_embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM,
                    description="Embedding of redacted text"),
        FieldSchema("expires_at_utc", DataType.INT64,
                    description="Unix timestamp when to purge"),
    ]

    schema = CollectionSchema(fields, description="Recruiter/vendor conv turns (redacted)")
    col = Collection(name=name, schema=schema, using=MILVUS_ALIAS, shards_num=2)

    col.create_index(
        field_name="conv_embedding",
        index_params={
            "index_type": INDEX_TYPE,
            "metric_type": METRIC_TYPE,
            "params": INDEX_PARAMS,
        },
    )
    print(f"[{name}] created with index on conv_embedding.")


def create_conv_candidate():
    name = "conv_candidate"
    if Collection.exists(name, using=MILVUS_ALIAS):
        print(f"[{name}] already exists, skipping.")
        return

    fields = [
        FieldSchema("conv_id", DataType.VARCHAR, is_primary=True, auto_id=False,
                    max_length=64, description="Unique conversation message id"),
        FieldSchema("candidate_id", DataType.VARCHAR, max_length=64,
                    description="Hashed candidate id"),
        FieldSchema("job_id", DataType.VARCHAR, max_length=128,
                    description="Internal job id"),
        FieldSchema("role", DataType.VARCHAR, max_length=16,
                    description="'candidate' or 'agent'"),
        FieldSchema("ts_utc", DataType.INT64,
                    description="Unix timestamp"),
        FieldSchema("text_redacted", DataType.VARCHAR, max_length=4000,
                    description="Email text with PII masked"),
        FieldSchema("conv_embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM,
                    description="Embedding of redacted text"),
        FieldSchema("expires_at_utc", DataType.INT64,
                    description="Unix timestamp when to purge"),
    ]

    schema = CollectionSchema(fields, description="Candidate conv turns (redacted)")
    col = Collection(name=name, schema=schema, using=MILVUS_ALIAS, shards_num=2)

    col.create_index(
        field_name="conv_embedding",
        index_params={
            "index_type": INDEX_TYPE,
            "metric_type": METRIC_TYPE,
            "params": INDEX_PARAMS,
        },
    )
    print(f"[{name}] created with index on conv_embedding.")


def create_applications():
    name = "applications"
    if Collection.exists(name, using=MILVUS_ALIAS):
        print(f"[{name}] already exists, skipping.")
        return

    fields = [
        FieldSchema("application_id", DataType.VARCHAR, is_primary=True, auto_id=False,
                    max_length=64, description="Unique application id"),
        FieldSchema("candidate_id", DataType.VARCHAR, max_length=64,
                    description="Candidate id (hash)"),
        FieldSchema("job_id", DataType.VARCHAR, max_length=128,
                    description="Job id"),
        FieldSchema("vendor_id", DataType.VARCHAR, max_length=64,
                    description="Vendor/company id"),
        FieldSchema("current_stage", DataType.VARCHAR, max_length=32,
                    description="applied | phone_screening | video_screening | submitted_to_client | accepted"),
        FieldSchema("stage_history", DataType.VARCHAR, max_length=2000,
                    description="JSON string of stage transitions (no PII)"),
        FieldSchema("notes", DataType.VARCHAR, max_length=2000,
                    description="Redacted summary about candidate for this job"),
        FieldSchema("app_embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM,
                    description="Embedding of JD summary + candidate summary + stage + notes"),
        FieldSchema("created_at_utc", DataType.INT64,
                    description="Creation time (unix epoch)"),
        FieldSchema("updated_at_utc", DataType.INT64,
                    description="Last update time (unix epoch)"),
        FieldSchema("expires_at_utc", DataType.INT64,
                    description="When to purge (unix epoch)"),
    ]

    schema = CollectionSchema(fields, description="Semantic view of job applications")
    col = Collection(name=name, schema=schema, using=MILVUS_ALIAS, shards_num=2)

    col.create_index(
        field_name="app_embedding",
        index_params={
            "index_type": INDEX_TYPE,
            "metric_type": METRIC_TYPE,
            "params": INDEX_PARAMS,
        },
    )
    print(f"[{name}] created with index on app_embedding.")


if __name__ == "__main__":
    connect_milvus()
    create_conv_recruiter()
    create_conv_candidate()
    create_applications()
    print("✅ All collections checked/created.")
