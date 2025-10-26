#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime
from pymilvus import (
    connections, db, utility,
    FieldSchema, CollectionSchema, DataType, Collection
)

MILVUS_URI   = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_DB    = os.getenv("MILVUS_DB", "default")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", None)

COLLECTION = "client_memory"
DESC = "Client contact + preferences (scalar fields + tiny dummy vector)"
SHARDS = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("client_memory_setup")

def connect():
    connections.connect("default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    if MILVUS_DB not in db.list_database():
        db.create_database(MILVUS_DB)
    db.using_database(MILVUS_DB)
    log.info("Connected to %s / DB=%s", MILVUS_URI, MILVUS_DB)

def build_schema() -> CollectionSchema:
    fields = [
        # Primary key
        FieldSchema("client_id", DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),

        # Scalars
        FieldSchema("client_company", DataType.VARCHAR, max_length=256),
        FieldSchema("phone", DataType.VARCHAR, max_length=48),
        FieldSchema("email", DataType.VARCHAR, max_length=256),

        FieldSchema("contact_name", DataType.VARCHAR, max_length=128),
        FieldSchema("notes", DataType.VARCHAR, max_length=4096),
        FieldSchema("preferences", DataType.VARCHAR, max_length=8192),
        FieldSchema("last_queries", DataType.VARCHAR, max_length=8192),

        FieldSchema("created_at", DataType.VARCHAR, max_length=64),
        FieldSchema("updated_at", DataType.VARCHAR, max_length=64),

        FieldSchema("searches_count", DataType.INT64),
        FieldSchema("desired_headcount", DataType.INT64),
        FieldSchema("status", DataType.VARCHAR, max_length=32),

        # 🔸 Required in Milvus 2.5+: at least one vector field.
        # Keep it tiny; we won't index or search it.
        FieldSchema("cm_dummy_vec", DataType.FLOAT_VECTOR, dim=2),
    ]
    return CollectionSchema(fields=fields, description=DESC)

def ensure_collection() -> Collection:
    from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType

    if utility.has_collection(COLLECTION):
        coll = Collection(COLLECTION)
        print(f"Collection {COLLECTION} exists.")
    else:
        schema = build_schema()
        coll = Collection(
            name=COLLECTION,
            schema=schema,
            shards_num=SHARDS,
            consistency_level="Bounded",
        )
        print(f"Collection {COLLECTION} created.")

    # ---- scalar inverted indexes ----
    for field in ["client_company", "phone", "email", "status"]:
        try:
            coll.create_index(field_name=field, index_params={"index_type": "INVERTED"})
        except Exception as e:
            print(f"[warn] scalar index on {field}: {e}")

    # ---- REQUIRED: vector index on the dummy vector ----
    try:
        vec_idx = {
            "index_type": "HNSW",
            "metric_type": "COSINE",            # or "L2" — either is fine
            "params": {"M": 4, "efConstruction": 8}  # very small, low-cost
        }
        coll.create_index(field_name="cm_dummy_vec", index_params=vec_idx)
        print("Vector index created on cm_dummy_vec (HNSW, tiny).")
    except Exception as e:
        print(f"[warn] vector index on cm_dummy_vec: {e}")

    # Now load safely
    coll.load()
    print(f"{COLLECTION} loaded.")
    return coll

def insert_one(coll, row: dict):
    """
    Insert ONE row using column-major format, aligned to the collection's schema order.
    """
    field_order = [f.name for f in coll.schema.fields]
    # make sure all fields exist in row
    missing = [k for k in field_order if k not in row]
    if missing:
        raise ValueError(f"Row missing fields: {missing}")

    # each inner list is a column; wrap each value in a list for single-row insert
    data = [[row[name]] for name in field_order]
    coll.insert(data)


def insert_example(coll: Collection):
    import uuid
    from datetime import datetime

    row = {
        "client_id": str(uuid.uuid4()),
        "client_company": "Acme Health",
        "phone": "+1-555-1212",
        "email": "cto@acme.example",
        "contact_name": "Jordan",
        "notes": "",
        "preferences": "",
        "last_queries": "",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "searches_count": 0,
        "desired_headcount": 0,
        "status": "Active",
        # REQUIRED dummy vector (dim=2)
        "cm_dummy_vec": [0.0, 0.0],
    }

    insert_one(coll, row)
    coll.flush()
    print(f"Inserted sample client_id={row['client_id']}")
    log.info("Inserted sample client_id=%s", row["client_id"])

if __name__ == "__main__":
    connect()
    coll = ensure_collection()
    # Optional: test insert once
    insert_example(coll)              
    log.info("client_memory ready. Entities=%s", coll.num_entities)
