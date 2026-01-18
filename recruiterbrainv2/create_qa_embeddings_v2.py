"""
Create qa_embeddings_v2 collection with job_id + interview_date
and optionally migrate data from qa_embeddings.

Usage:
  python create_qa_embeddings_v2.py --migrate
  python create_qa_embeddings_v2.py --drop-existing --migrate --mapping mapping.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


def connect_milvus() -> None:
    uri = os.getenv("MILVUS_URI")
    token = os.getenv("MILVUS_TOKEN", "")
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))

    if uri:
        connections.connect(alias="default", uri=uri, token=token or None)
        logger.info("Connected to Milvus via MILVUS_URI")
        return

    if token:
        connections.connect(alias="default", uri=f"{host}:{port}", token=token)
        logger.info("Connected to Milvus with token auth at %s:%s", host, port)
        return

    connections.connect(alias="default", host=host, port=port)
    logger.info("Connected to Milvus at %s:%s", host, port)


def create_collection(name: str, drop_existing: bool) -> Collection:
    if utility.has_collection(name):
        if not drop_existing:
            logger.info("Collection %s already exists. Using existing collection.", name)
            return Collection(name)
        logger.warning("Dropping existing collection: %s", name)
        utility.drop_collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="interview_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="candidate_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="job_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="job_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="job_description", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="question_index", dtype=DataType.INT64),
        FieldSchema(name="answer_snippet", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="interview_date", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="qa_embeddings v2 with job_id and interview_date",
        enable_dynamic_field=False,
    )

    collection = Collection(
        name=name,
        schema=schema,
        using="default",
        consistency_level="Strong",
    )

    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
        index_name="idx_embedding",
    )

    for field in ("interview_id", "candidate_id", "job_id"):
        collection.create_index(
            field_name=field,
            index_params={"index_type": "INVERTED"},
            index_name=f"idx_{field}",
        )

    collection.create_index(
        field_name="interview_date",
        index_params={"index_type": "STL_SORT"},
        index_name="idx_interview_date",
    )

    collection.create_index(
        field_name="question_index",
        index_params={"index_type": "STL_SORT"},
        index_name="idx_question_index",
    )

    collection.load()
    logger.info("Collection created: %s", name)
    return collection


def _parse_timestamp(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        value_int = int(value)
        return int(value_int / 1000) if value_int > 1_000_000_000_000 else value_int
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return 0
        if raw.isdigit():
            return _parse_timestamp(int(raw))
        try:
            raw = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except Exception:
            return 0
    return 0


def _iter_mapping_rows(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, dict):
                    yield value
            return
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield row
        return

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _extract_job_id(row: Dict[str, Any]) -> str:
    if "job_id" in row and row["job_id"]:
        return str(row["job_id"]).strip()
    job = row.get("job")
    if isinstance(job, dict) and job.get("job_id"):
        return str(job["job_id"]).strip()
    return ""


def _extract_job_title(row: Dict[str, Any]) -> str:
    if "job_title" in row and row["job_title"]:
        return str(row["job_title"]).strip()
    job = row.get("job")
    if isinstance(job, dict) and job.get("title"):
        return str(job["title"]).strip()
    return ""


def _extract_job_description(row: Dict[str, Any]) -> str:
    if "job_description" in row and row["job_description"]:
        return str(row["job_description"]).strip()
    job = row.get("job")
    if isinstance(job, dict) and job.get("description"):
        return str(job["description"]).strip()
    return ""


def _extract_interview_date(row: Dict[str, Any]) -> int:
    for key in (
        "interview_date",
        "scheduled_time",
        "call_initiated_at",
        "call_answered_at",
        "created_at",
        "start_time",
    ):
        if key in row and row[key]:
            return _parse_timestamp(row[key])
    return 0


def load_mapping(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}

    mapping: Dict[str, Dict[str, Any]] = {}
    for row in _iter_mapping_rows(path):
        interview_id = row.get("interview_id") or row.get("id")
        if not interview_id:
            continue
        interview_id = str(interview_id).strip()
        mapping[interview_id] = {
            "job_id": _extract_job_id(row),
            "job_title": _extract_job_title(row),
            "job_description": _extract_job_description(row),
            "interview_date": _extract_interview_date(row),
        }
    logger.info("Loaded mapping rows: %d", len(mapping))
    return mapping


def migrate_data(
    source_name: str,
    target_name: str,
    mapping: Dict[str, Dict[str, Any]],
    batch_size: int,
    limit: Optional[int],
    dry_run: bool,
) -> Tuple[int, int]:
    if not utility.has_collection(source_name):
        raise RuntimeError(f"Source collection not found: {source_name}")

    source = Collection(source_name)
    target = Collection(target_name)
    source.load()
    target.load()

    total = source.num_entities
    if limit is not None:
        total = min(total, limit)

    logger.info("Migrating %d rows from %s to %s", total, source_name, target_name)

    output_fields = [
        "id",
        "interview_id",
        "candidate_id",
        "question_index",
        "answer_snippet",
        "embedding",
    ]

    migrated = 0
    offset = 0
    expr = 'id != ""'

    while offset < total:
        batch_limit = min(batch_size, total - offset)
        rows = source.query(
            expr=expr,
            output_fields=output_fields,
            limit=batch_limit,
            offset=offset,
        )
        if not rows:
            break

        payload: List[Dict[str, Any]] = []
        for row in rows:
            interview_id = (row.get("interview_id") or "").strip()
            meta = mapping.get(interview_id, {})
            payload.append(
                {
                    "id": row.get("id"),
                    "interview_id": interview_id,
                    "candidate_id": (row.get("candidate_id") or "").strip(),
                    "job_id": meta.get("job_id", ""),
                    "job_title": meta.get("job_title", ""),
                    "job_description": meta.get("job_description", ""),
                    "question_index": int(row.get("question_index") or 0),
                    "answer_snippet": row.get("answer_snippet") or "",
                    "interview_date": int(meta.get("interview_date") or 0),
                    "embedding": row.get("embedding"),
                }
            )

        if not dry_run:
            target.insert(payload)

        migrated += len(payload)
        offset += len(rows)
        logger.info("Migrated %d/%d rows", migrated, total)

    if not dry_run:
        target.flush()
        target.load()

    return migrated, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Create qa_embeddings_v2 and migrate data.")
    parser.add_argument("--source", default="qa_embeddings", help="Source collection name")
    parser.add_argument("--target", default="qa_embeddings_v2", help="Target collection name")
    parser.add_argument("--drop-existing", action="store_true", help="Drop target if exists")
    parser.add_argument("--migrate", action="store_true", help="Copy data from source to target")
    parser.add_argument("--mapping", help="JSON/JSONL/CSV mapping file for job_id/interview_date")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    connect_milvus()
    create_collection(args.target, args.drop_existing)

    if args.migrate:
        mapping = load_mapping(args.mapping)
        migrated, total = migrate_data(
            source_name=args.source,
            target_name=args.target,
            mapping=mapping,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        logger.info("Migration complete: %d/%d rows", migrated, total)
    else:
        logger.info("Collection ready. Run with --migrate to copy data.")


if __name__ == "__main__":
    main()
