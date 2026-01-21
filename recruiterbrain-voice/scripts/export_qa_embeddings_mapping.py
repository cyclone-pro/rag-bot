"""
Export interview mapping data from Postgres for qa_embeddings_v2 migration.

Outputs: interview_id, job_id, job_title, job_description, scheduled_time
Optional: created_at for fallback.

Usage:
  python scripts/export_qa_embeddings_mapping.py --output mapping.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import os
from pathlib import Path

import asyncpg


def _serialize_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _load_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


async def _fetch_rows(limit: Optional[int]) -> List[Dict[str, Any]]:
    _load_env()
    host = _get_env("POSTGRES_HOST", "postgres_host")
    port = int(_get_env("POSTGRES_PORT", "postgres_port", default="5432"))
    database = _get_env("POSTGRES_DB", "postgres_db")
    user = _get_env("POSTGRES_USER", "postgres_user")
    password = _get_env("POSTGRES_PASSWORD", "postgres_password")

    if not all([host, database, user, password]):
        raise RuntimeError("Missing Postgres env vars. Set POSTGRES_HOST/DB/USER/PASSWORD.")

    conn = await asyncpg.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )
    try:
        query = """
            SELECT interview_id,
                   job_id,
                   job_title,
                   job_description,
                   scheduled_time,
                   created_at
            FROM interviews
            WHERE interview_id IS NOT NULL
            ORDER BY created_at DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = await conn.fetch(query)
        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "interview_id": row["interview_id"],
                    "job_id": row["job_id"],
                    "job_title": row["job_title"],
                    "job_description": row["job_description"],
                    "scheduled_time": _serialize_timestamp(row["scheduled_time"]),
                    "created_at": _serialize_timestamp(row["created_at"]),
                }
            )
        return results
    finally:
        await conn.close()


def _write_jsonl(rows: Iterable[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(rows: Iterable[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=True, indent=2)


def _write_csv(rows: Iterable[Dict[str, Any]], output_path: str) -> None:
    rows = list(rows)
    if not rows:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export interview mapping data for qa_embeddings_v2.")
    parser.add_argument("--output", required=True, help="Output path (.jsonl, .json, .csv)")
    parser.add_argument("--format", choices=["jsonl", "json", "csv"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_format = args.format
    if not output_format:
        if args.output.endswith(".jsonl"):
            output_format = "jsonl"
        elif args.output.endswith(".json"):
            output_format = "json"
        elif args.output.endswith(".csv"):
            output_format = "csv"
        else:
            raise SystemExit("Unknown output extension. Use --format or a .jsonl/.json/.csv suffix.")

    rows = asyncio.run(_fetch_rows(args.limit))

    if output_format == "jsonl":
        _write_jsonl(rows, args.output)
    elif output_format == "json":
        _write_json(rows, args.output)
    elif output_format == "csv":
        _write_csv(rows, args.output)

    print(f"Exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
