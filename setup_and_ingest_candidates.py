#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup + Ingest for candidate_pool (terminal-only)

What this does:
- Connects to Milvus
- Ensures vector indexes: HNSW(COSINE) with M=32, efConstruction=280 on summary_embedding + skills_embedding
- Ensures scalar INVERTED indexes: role_family, years_band, clouds, location_country/state/city, last_updated
- Ensures partitions by role family: backend, frontend, devops, security, data, mlops, cloud, systems, mobile, blockchain
- Reads a CSV (default: /mnt/data/candidate_pool_1000.csv)
- Derives role_family, years_band, clouds
- Upserts rows; for NEW rows, inserts into the appropriate partition by role_family
- OPTIONAL: builds 768-dim embeddings for summary if USE_EMBED=1 (requires sentence-transformers)

Usage:
  export MILVUS_URI="http://<host>:19530"
  export MILVUS_DB="default"
  export COLLECTION="candidate_pool"
  export CSV_PATH="/mnt/data/candidate_pool_1000.csv"
  export USE_EMBED="0"    # set to "1" to compute 768-dim embeddings
  python setup_and_ingest_candidates.py
"""

import os
import csv
import sys
import time
import json
import math
import logging
from typing import Dict, Any, List, Optional

from pymilvus import (
    connections, db, utility,
    FieldSchema, CollectionSchema, DataType, Collection
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("setup_ingest")

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB  = os.getenv("MILVUS_DB", "default")
COLLECTION = os.getenv("COLLECTION", "candidate_pool")
CSV_PATH   = os.getenv("CSV_PATH", "/mnt/data/candidate_pool_1000.csv")
USE_EMBED  = os.getenv("USE_EMBED", "0") == "1"

# Optional embedding (same 768-dim family)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/e5-base-v2")
_encoder = None
if USE_EMBED:
    try:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(EMBED_MODEL_NAME)
        log.info("Loaded embedding model: %s", EMBED_MODEL_NAME)
    except Exception as e:
        log.warning("Failed to load embedding model (%s). Continuing without embeddings. %s", EMBED_MODEL_NAME, e)
        _encoder = None

ROLE_PARTITIONS = [
    "backend","frontend","devops","security","data","mlops","cloud","systems","mobile","blockchain"
]

def connect():
    connections.connect("default", uri=MILVUS_URI)
    try:
        db.using_database(MILVUS_DB)
    except Exception:
        pass
    log.info("Connected to Milvus at %s (DB=%s)", MILVUS_URI, MILVUS_DB)

def ensure_collection_exists() -> Collection:
    if not utility.has_collection(COLLECTION):
        raise SystemExit(f"Collection `{COLLECTION}` not found. Create it first (schema with summary_embedding/skills_embedding).")

    col = Collection(COLLECTION)
    col.load()  # for index checks
    return col

def ensure_vector_indexes(col: Collection):
    # Re-create / ensure HNSW(COSINE) M=32, efConstruction=280
    vec_fields = []
    for f in col.schema.fields:
        if f.dtype == DataType.FLOAT_VECTOR:
            vec_fields.append(f.name)
    target_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 32, "efConstruction": 280}}

    existing = {ix.field_name: ix.params for ix in col.indexes} if hasattr(col, "indexes") else {}
    for vf in vec_fields:
        try:
            # If different index, (re)create
            recreate = True
            if vf in existing:
                p = existing[vf]
                # a light check
                if str(p.get("index_type","")).upper()=="HNSW" and str(p.get("metric_type","")).upper()=="COSINE":
                    recreate = False
            if recreate:
                log.info("Creating vector index on %s: %s", vf, target_params)
                col.create_index(field_name=vf, index_params=target_params)
        except Exception as e:
            log.warning("Index ensure failed for %s: %s", vf, e)

def ensure_scalar_indexes(col: Collection):
    scalar_fields = ["role_family","years_band","clouds","location_country","location_state","location_city","last_updated"]
    for sf in scalar_fields:
        # Only index if field exists
        if sf not in [f.name for f in col.schema.fields]:
            log.info("Field %s not in schema; skip INVERTED index", sf)
            continue
        try:
            col.create_index(field_name=sf, index_params={"index_type": "INVERTED"})
            log.info("INVERTED index created on %s", sf)
        except Exception as e:
            log.warning("INVERTED index on %s: %s", sf, e)

def ensure_partitions(col: Collection):
    # create partitions if missing
    for p in ROLE_PARTITIONS:
        try:
            if not utility.has_partition(COLLECTION, p):
                utility.create_partition(COLLECTION, p)
                log.info("Created partition: %s", p)
        except Exception as e:
            log.warning("Partition %s: %s", p, e)

def role_from_text(title: str, skills: str) -> str:
    t = (title or "").lower() + " " + (skills or "").lower()
    if any(k in t for k in ["react","angular","vue","frontend","front end"]): return "frontend"
    if any(k in t for k in ["django","spring","fastapi","flask","node","express","nest","backend","back end",".net","asp.net","rust"]): return "backend"
    if any(k in t for k in ["sre","devops","platform","terraform","ansible","kubernetes","k8s","helm","docker"]): return "devops"
    if any(k in t for k in ["security","soc","siem","iam","pentest","zero trust","cyber"]): return "security"
    if any(k in t for k in ["data engineer","etl","dbt","spark","hadoop","snowflake","bigquery"]): return "data"
    if any(k in t for k in ["mlops","ml ops","model serving","feature store"]): return "mlops"
    if any(k in t for k in ["aws","gcp","azure","cloud engineer","cloud architect"]): return "cloud"
    if any(k in t for k in ["vmware","nutanix","olvm","sccm","dns","dhcp","ad","rhel","linux admin","systems"]): return "systems"
    if any(k in t for k in ["android","ios","mobile"]): return "mobile"
    if any(k in t for k in ["blockchain","solidity","evm","web3"]): return "blockchain"
    return "backend"  # default bias for software resumes

def years_band_from_years(x: Optional[float]) -> Optional[str]:
    if x is None: return None
    try:
        y = float(x)
    except Exception:
        return None
    if y < 3: return "junior"
    if y < 6: return "mid"
    return "senior"

def clouds_from_text(skills: str) -> List[str]:
    t = (skills or "").lower()
    clouds = []
    if "aws" in t: clouds.append("AWS")
    if "gcp" in t: clouds.append("GCP")
    if "azure" in t: clouds.append("AZURE")
    return list(dict.fromkeys(clouds))

def embed(text: str, fallback_dim: int = 768):
    # returns a 768-d vector (or zeros if encoder unavailable)
    if _encoder is None:
        return [0.0]*fallback_dim
    v = _encoder.encode(text or "", normalize_embeddings=True)
    return v.tolist() if hasattr(v, "tolist") else list(v)

def read_csv_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            yield row

def upsert_rows(col: Collection, rows: List[Dict[str,Any]], batch: int = 256):
    """
    Upsert strategy:
      - If PK exists: update scalar fields (role_family, years_band, clouds, last_updated).
      - If PK not found: INSERT into target partition using role_family.
    NOTE: Moving existing entities between partitions requires delete+insert; we avoid mass moves here.
    New data will land in correct partitions.
    """
    # Cache field order
    field_names = [f.name for f in col.schema.fields]
    has_summary = "summary_embedding" in field_names
    has_skills  = "skills_embedding" in field_names

    # Small helper: existence check
    def exists(pk: str) -> bool:
        try:
            r = col.query(expr=f'candidate_id == "{pk}"', output_fields=["candidate_id"], limit=1)
            return bool(r)
        except Exception:
            return False

    buf_new_by_partition: Dict[str, Dict[str, List[Any]]] = {}

    def push_new(part: str, entity: Dict[str,Any]):
        if part not in buf_new_by_partition:
            buf_new_by_partition[part] = {fn: [] for fn in field_names}
        for fn in field_names:
            buf_new_by_partition[part][fn].append(entity.get(fn))

    count_new, count_update = 0, 0
    for row in rows:
        pk = row.get("candidate_id") or row.get("id") or row.get("uid")
        if not pk:
            # create a deterministic pk if missing (NOT RECOMMENDED if you already inserted)
            continue

        name   = row.get("name","")
        titles = row.get("top_titles_mentioned","") or row.get("titles","")
        skills = row.get("skills_extracted","") or row.get("skills","")
        tools  = row.get("tools_and_technologies","") or row.get("tools","")
        summary= row.get("semantic_summary","") or row.get("summary","")
        exp    = row.get("total_experience_years") or row.get("experience_years")

        # derive fields
        role   = role_from_text(titles, skills + " " + tools)
        yband  = years_band_from_years(exp if exp not in (None, "") else None)
        clouds = clouds_from_text(skills + " " + tools)
        last_updated = row.get("last_updated") or row.get("updated_at") or ""

        if exists(pk):
            # update scalar fields (safe)
            try:
                col.update(expr=f'candidate_id == "{pk}"', field_name="role_family",   value=role)
                if yband: col.update(expr=f'candidate_id == "{pk}"', field_name="years_band",   value=yband)
                if clouds: col.update(expr=f'candidate_id == "{pk}"', field_name="clouds", value=clouds)
                if last_updated:
                    col.update(expr=f'candidate_id == "{pk}"', field_name="last_updated", value=last_updated)
                count_update += 1
            except Exception as e:
                log.warning("Update failed for %s: %s (continuing)", pk, e)
        else:
            # prepare a new insert with required fields present
            entity: Dict[str,Any] = {}
            for fn in field_names:
                entity[fn] = row.get(fn, None)

            # derive / fill
            entity["candidate_id"] = pk
            entity["role_family"]  = role
            if "years_band" in field_names: entity["years_band"] = yband or ""
            if "clouds" in field_names: entity["clouds"] = clouds

            # ensure vectors exist
            if has_summary and (not entity.get("summary_embedding")):
                # embed from the richest available text
                text = " ".join([summary, skills, tools, titles]).strip()
                entity["summary_embedding"] = embed(text)
            if has_skills and (not entity.get("skills_embedding")):
                entity["skills_embedding"] = embed(skills or tools)

            # choose partition
            part = role if role in ROLE_PARTITIONS else "backend"
            push_new(part, entity)
            count_new += 1

            # flush inserts in batches
            for part, columns in list(buf_new_by_partition.items()):
                if len(next(iter(columns.values()))) >= batch:
                    log.info("Inserting batch into partition=%s size=%d", part, len(next(iter(columns.values()))))
                    col.insert(columns, partition_name=part)
                    for k in columns.keys(): columns[k].clear()

    # flush remaining
    for part, columns in buf_new_by_partition.items():
        n = len(next(iter(columns.values()))) if columns else 0
        if n > 0:
            log.info("Inserting final batch into partition=%s size=%d", part, n)
            col.insert(columns, partition_name=part)

    col.flush()
    log.info("Upsert complete. new=%d updated=%d", count_new, count_update)

def main():
    connect()
    col = ensure_collection_exists()
    ensure_vector_indexes(col)
    ensure_scalar_indexes(col)
    ensure_partitions(col)

    if not os.path.exists(CSV_PATH):
        log.warning("CSV not found at %s; skipping ingest.", CSV_PATH)
        return

    # stream CSV
    rows = list(read_csv_rows(CSV_PATH))
    log.info("Read %d rows from CSV", len(rows))
    upsert_rows(col, rows, batch=256)
    log.info("Done. Entities=%s", col.num_entities)

if __name__ == "__main__":
    main()
