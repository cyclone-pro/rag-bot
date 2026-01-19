"""
Diagnostic tool for qa_embeddings migration.

Compares interview_ids between a mapping file and a Milvus collection.
Writes a JSON report and prints a short summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pymilvus import Collection, connections


def _load_env_from_path(path: Optional[str]) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


def _load_mapping_ids(path: str) -> List[str]:
    ids: List[str] = []
    for row in _iter_mapping_rows(path):
        interview_id = row.get("interview_id") or row.get("id")
        if not interview_id:
            continue
        ids.append(str(interview_id).strip())
    return ids


def _connect_milvus(env_path: Optional[str]) -> None:
    _load_env_from_path(env_path)
    uri = os.getenv("MILVUS_URI")
    token = os.getenv("MILVUS_TOKEN") or None
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")

    if uri:
        connections.connect(alias="default", uri=uri, token=token)
        return

    connections.connect(alias="default", host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose qa_embeddings migration coverage.")
    parser.add_argument("--mapping", required=True, help="Mapping file (jsonl/json/csv)")
    parser.add_argument("--collection", default="qa_embeddings", help="Source collection name")
    parser.add_argument("--env", default=None, help="Optional .env file to load")
    parser.add_argument("--output", default="logs/qa_migration_diagnostic.json")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    _connect_milvus(args.env)
    mapping_ids = _load_mapping_ids(args.mapping)
    mapping_set = set(mapping_ids)

    collection = Collection(args.collection)
    collection.load()

    interview_counts: Counter[str] = Counter()
    total_rows = 0
    iterator = collection.query_iterator(
        expr='interview_id != ""',
        output_fields=["interview_id"],
        batch_size=args.batch_size,
    )

    while True:
        rows = iterator.next()
        if not rows:
            break
        total_rows += len(rows)
        for row in rows:
            interview_id = (row.get("interview_id") or "").strip()
            if interview_id:
                interview_counts[interview_id] += 1

    milvus_ids = set(interview_counts.keys())
    found = sorted(mapping_set.intersection(milvus_ids))
    missing = sorted(mapping_set.difference(milvus_ids))
    extra = sorted(milvus_ids.difference(mapping_set))
    mapped_rows = sum(interview_counts[i] for i in found)

    report = {
        "collection": args.collection,
        "total_rows": total_rows,
        "unique_interview_ids": len(milvus_ids),
        "mapping_ids": len(mapping_set),
        "mapping_found": len(found),
        "mapping_missing": len(missing),
        "rows_for_mapping_ids": mapped_rows,
        "missing_interview_ids": missing,
        "extra_interview_ids": extra,
        "rows_by_interview_id": interview_counts,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)

    print(f"Collection rows: {total_rows}")
    print(f"Mapping ids: {len(mapping_set)}")
    print(f"Mapping found: {len(found)}")
    print(f"Mapping missing: {len(missing)}")
    print(f"Rows for mapping ids: {mapped_rows}")
    print(f"Wrote report: {output_path}")


if __name__ == "__main__":
    main()
