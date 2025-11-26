#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import uuid
from typing import Optional
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

        # Dummy vector to satisfy Milvus 2.5 (we won't actually search it)
        FieldSchema("cm_dummy_vec", DataType.FLOAT_VECTOR, dim=2),
    ]
    return CollectionSchema(fields=fields, description=DESC)

def ensure_scalar_indexes(coll: Collection):
    for field in ["client_company", "phone", "email", "status"]:
        try:
            coll.create_index(field_name=field, index_params={"index_type": "INVERTED"})
        except Exception as e:
            log.warning("Scalar index on %s: %s", field, e)

def ensure_dummy_vec_index(coll: Collection):
    # Create tiny HNSW index on cm_dummy_vec if missing (required before load in 2.5)
    try:
        has_vec_index = any(ix.field_name == "cm_dummy_vec" for ix in coll.indexes)
    except Exception:
        has_vec_index = False
    if not has_vec_index:
        coll.create_index(
            field_name="cm_dummy_vec",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 4, "efConstruction": 8}},
        )
        log.info("Vector index created on cm_dummy_vec (HNSW tiny).")

def ensure_client_memory() -> Collection:
    if utility.has_collection(COLLECTION):
        coll = Collection(COLLECTION)
        log.info("Collection %s exists.", COLLECTION)

        # Guard: make sure cm_dummy_vec is there; if not, schema mismatch
        field_names = [f.name for f in coll.schema.fields]
        if "cm_dummy_vec" not in field_names:
            raise RuntimeError(
                "Existing client_memory lacks cm_dummy_vec. Drop it once (utility.drop_collection) and recreate."
            )

        ensure_scalar_indexes(coll)
        ensure_dummy_vec_index(coll)
    else:
        log.info("Creating collection %s ...", COLLECTION)
        coll = Collection(
            name=COLLECTION,
            schema=build_schema(),
            shards_num=SHARDS,
            consistency_level="Bounded",
        )
        ensure_scalar_indexes(coll)
        ensure_dummy_vec_index(coll)

    coll.load()
    log.info("%s loaded.", COLLECTION)
    return coll

def insert_one(coll: Collection, row: dict):
    """Insert ONE row using column-major order matching the schema."""
    field_order = [f.name for f in coll.schema.fields]
    missing = [k for k in field_order if k not in row]
    if missing:
        raise ValueError(f"Row missing fields: {missing}")
    data = [[row[name]] for name in field_order]  # column-major
    coll.insert(data)

def insert_example(coll: Collection):
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
        "cm_dummy_vec": [0.0, 0.0],   
    }
    insert_one(coll, row)
    coll.flush()
    log.info("Inserted sample client_id=%s", row["client_id"])

# Optional: use this from your terminal recruiter script too
def upsert_client(mem: Collection, company: str, phone: str, email: Optional[str], contact_name: Optional[str]) -> str:
    try:
        rows = mem.query(
            expr=f'client_company == "{company}" and phone == "{phone}"',
            output_fields=["client_id"],
            limit=1
        )
    except Exception:
        rows = []

    if rows:
        cid = rows[0]["client_id"]
        mem.update(expr=f'client_id == "{cid}"', field_name="updated_at", value=datetime.utcnow().isoformat())
        return cid

    cid = str(uuid.uuid4())
    row = {
        "client_id": cid,
        "client_company": company,
        "phone": phone,
        "email": email or "",
        "contact_name": contact_name or "",
        "notes": "",
        "preferences": "",
        "last_queries": "",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "searches_count": 0,
        "desired_headcount": 0,
        "status": "Active",
        "cm_dummy_vec": [0.0, 0.0],
    }
    insert_one(mem, row)
    mem.flush()
    return cid

if __name__ == "__main__":
    connect()
    coll = ensure_client_memory()
    # demo insert (comment out if you don’t want a sample row)
    insert_example(coll)
    log.info("client_memory ready. Entities=%s", coll.num_entities)
