from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg.types.json import Jsonb


# ---------------------------
# Enum value sets (from your schema)
# ---------------------------
SENIORITY_LEVEL_ENUM = {
    "Entry",
    "Mid",
    "Senior",
    "Lead",
    "Architect",
    "Manager",
    "Director",
    "VP",
    "C-level",
    "unspecified",
}

JOB_TYPE_ENUM = {
    "Contract",
    "Contract-to-hire",
    "Full-time",
    "Part-time",
    "Internship",
    "Other",
    "unspecified",
}

SUBMISSION_URGENCY_ENUM = {"normal", "urgent", "flexible"}

WORK_MODEL_ENUM = {"onsite", "remote", "hybrid", "flexible", "unspecified"}

RATE_UNIT_ENUM = {"hour", "day", "week", "month", "year", "unspecified"}

EMPLOYMENT_TYPE_ENUM = {"C2C", "W2", "1099", "unspecified"}

WORK_AUTH_ENUM = {
    "USC",
    "GC",
    "H1B",
    "H4-EAD",
    "L1",
    "L2-EAD",
    "TN",
    "E3",
    "F1-OPT",
    "F1-CPT",
    "STEM-OPT",
    "J1",
    "O1",
    "EAD",
    "Asylum-EAD",
    "GC-EAD",
    "Any",
    "unsp",
}

LOG_LEVEL_ENUM = {"debug", "info", "warning", "error"}

PROCESSING_STAGE_ENUM = {
    "webhook_received",
    "transcript_cleaned",
    "llm_started",
    "llm_complete",
    "llm_failed",
    "embedding_started",
    "embedding_complete",
    "embedding_failed",
    "similarity_check_started",
    "similarity_check_complete",
    "similarity_check_failed",
    "postgres_insert_started",
    "postgres_insert_complete",
    "postgres_insert_skipped",
    "postgres_insert_failed",
    "milvus_insert_started",
    "milvus_insert_complete",
    "milvus_insert_skipped",
    "milvus_insert_failed",
    "processing_complete",
    "processing_failed",
}


# ---------------------------
# Output-schema defaults (match your prompt)
# ---------------------------
DEFAULTS: Dict[str, Any] = {
    "seniority_level": "unspecified",
    "job_type": "unspecified",
    "submission_urgency": "normal",
    "work_model": "unspecified",
    "pay_rate_unit": "unspecified",
    "employment_type": "unspecified",
    "status": "active",
    "has_equity": False,
    "relocation_assistance": False,
    "positions_available": 1,
    "allowed_work_auth": ["Any"],
    "not_allowed_work_auth": ["Any"],
    "source_role_index": 0,
}

CALL_TRANSCRIPTS_TABLE = "call_transcripts"
PROCESSING_LOGS_TABLE = "processing_logs"

CALL_TRANSCRIPT_JSON_COLUMNS = {
    "raw_payload",
    "messages",
    "evaluation",
    "parsed_requirements",
    "tags",
}

logger = logging.getLogger("bey_db")


def _log_event(level: str, message: str, **fields: Any) -> None:
    payload = {"message": message, **fields}
    record = json.dumps(payload, ensure_ascii=True)
    if level == "warning":
        logger.warning(record)
    elif level == "error":
        logger.error(record)
    else:
        logger.info(record)


# ---------------------------
# Helpers
# ---------------------------

def _db_env_status() -> Dict[str, bool]:
    return {
        "DATABASE_URL_set": bool(os.getenv("DATABASE_URL")),
        "PGHOST_set": bool(os.getenv("PGHOST")),
        "PGDATABASE_set": bool(os.getenv("PGDATABASE")),
        "PGUSER_set": bool(os.getenv("PGUSER")),
        "PGPASSWORD_set": bool(os.getenv("PGPASSWORD")),
        "PGPORT_set": bool(os.getenv("PGPORT")),
    }


def _db_info_from_url(db_url: str) -> Dict[str, Optional[str]]:
    try:
        parsed = urlparse(db_url)
    except Exception:
        return {"db_url_set": bool(db_url)}
    return {
        "db_scheme": parsed.scheme or None,
        "db_host": parsed.hostname,
        "db_port": str(parsed.port) if parsed.port else None,
        "db_name": parsed.path.lstrip("/") if parsed.path else None,
        "db_user": parsed.username,
    }


def _db_url_from_env() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    host = os.getenv("PGHOST")
    db = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    pw = os.getenv("PGPASSWORD")
    port = os.getenv("PGPORT", "5432")

    if not (host and db and user and pw):
        _log_event("error", "db_env_missing", **_db_env_status())
        raise ValueError(
            "Set DATABASE_URL or PGHOST/PGDATABASE/PGUSER/PGPASSWORD (and optional PGPORT)"
        )

    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


async def check_db_connection(timeout: float = 5.0) -> Tuple[bool, str]:
    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    try:
        try:
            conn = await AsyncConnection.connect(db_url, connect_timeout=timeout)
        except TypeError:
            conn = await AsyncConnection.connect(db_url)
        async with conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
        return True, "ok"
    except Exception as exc:
        _log_event("error", "db_health_failed", error=str(exc), **db_info)
        return False, str(exc)


def _strip_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v)


def _to_bool_or_none(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    return None


def _ensure_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def _coerce_enum(v: Any, allowed: set, default: str) -> str:
    s = _strip_or_none(v)
    if not s:
        return default
    return s if s in allowed else default


def _coerce_work_auth_list(v: Any) -> List[str]:
    items = _ensure_list(v)
    out: List[str] = []
    for it in items:
        s = _strip_or_none(it)
        if not s:
            continue
        if s == "unspecified":
            s = "unsp"
        if s in WORK_AUTH_ENUM:
            out.append(s)
    return out


def _iso_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            if "T" in s:
                s = s.split("T", 1)[0]
            return date.fromisoformat(s)
        except Exception:
            return None
    return None


def _iso_timestamptz(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def _clean_locations(cities: Any, states: Any) -> Tuple[List[str], List[str]]:
    c = [x for x in (_strip_or_none(i) for i in _ensure_list(cities)) if x]
    s = [x for x in (_strip_or_none(i) for i in _ensure_list(states)) if x]
    if len(c) != len(s):
        return ([], [])
    return (c, s)


@dataclass
class PreparedRow:
    columns: List[str]
    placeholders: List[str]
    values: List[Any]


# Columns we accept from the model JSON (output schema) + meta fields
MODEL_COLUMNS = {
    "source_call_id",
    "source_role_index",
    "job_id",
    "external_requisition_id",
    "job_title",
    "seniority_level",
    "job_type",
    "end_client_name",
    "client_name",
    "industry",
    "positions_available",
    "submission_cutoff_date",
    "submission_urgency",
    "max_candidates_allowed",
    "location_cities",
    "location_states",
    "location_country",
    "work_model",
    "work_model_details",
    "pay_rate_min",
    "pay_rate_max",
    "pay_rate_currency",
    "pay_rate_unit",
    "employment_type",
    "is_rate_strict",
    "pay_rate_notes",
    "salary_min",
    "salary_max",
    "salary_currency",
    "bonus_percentage_min",
    "bonus_percentage_max",
    "bonus_type",
    "bonus_notes",
    "has_equity",
    "equity_type",
    "equity_details",
    "pto_days",
    "health_insurance_provided",
    "retirement_matching",
    "retirement_matching_details",
    "benefits_summary",
    "sign_on_bonus",
    "relocation_assistance",
    "relocation_amount",
    "contract_duration_text",
    "contract_start_date",
    "contract_end_date",
    "contract_can_extend",
    "allowed_work_auth",
    "not_allowed_work_auth",
    "citizenship_required",
    "work_auth_notes",
    "background_check_required",
    "background_check_details",
    "security_clearance_required",
    "security_clearance_level",
    "overall_min_years",
    "primary_role_min_years",
    "management_experience_required",
    "must_have_skills",
    "nice_to_have_skills",
    "primary_technologies",
    "certifications_required",
    "certifications_preferred",
    "domains",
    "responsibilities",
    "day_to_day",
    "other_constraints",
    "work_hours",
    "time_zone",
    "travel_required",
    "travel_details",
    "interview_process",
    "vendor_name",
    "vendor_contact_name",
    "vendor_contact_email",
    "vendor_contact_phone",
    "raw_role_title_block",
    "status",
    # Similarity tracking
    "similar_jobs",
    "similarity_score",
}


# For some columns, we must explicitly cast to enum/enum[] types.
CASTS = {
    "seniority_level": "::seniority_level_enum",
    "job_type": "::job_type_enum",
    "submission_urgency": "::submission_urgency_enum",
    "work_model": "::work_model_enum",
    "pay_rate_unit": "::rate_unit_enum",
    "employment_type": "::employment_type_enum",
    "allowed_work_auth": "::work_authorization_enum[]",
    "not_allowed_work_auth": "::work_authorization_enum[]",
}


def _prepare_role(role: Mapping[str, Any], metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Normalize + validate a single role object against DB constraints."""
    data: Dict[str, Any] = {}

    # Start with defaults
    for k, v in DEFAULTS.items():
        data[k] = v

    # Copy over known keys only
    for k, v in role.items():
        if k in MODEL_COLUMNS:
            data[k] = v

    # Normalize enums
    data["seniority_level"] = _coerce_enum(data.get("seniority_level"), SENIORITY_LEVEL_ENUM, DEFAULTS["seniority_level"])
    data["job_type"] = _coerce_enum(data.get("job_type"), JOB_TYPE_ENUM, DEFAULTS["job_type"])
    data["submission_urgency"] = _coerce_enum(data.get("submission_urgency"), SUBMISSION_URGENCY_ENUM, DEFAULTS["submission_urgency"])
    data["work_model"] = _coerce_enum(data.get("work_model"), WORK_MODEL_ENUM, DEFAULTS["work_model"])
    data["pay_rate_unit"] = _coerce_enum(data.get("pay_rate_unit"), RATE_UNIT_ENUM, DEFAULTS["pay_rate_unit"])
    data["employment_type"] = _coerce_enum(data.get("employment_type"), EMPLOYMENT_TYPE_ENUM, DEFAULTS["employment_type"])

    # Locations
    cities, states = _clean_locations(data.get("location_cities"), data.get("location_states"))
    data["location_cities"] = cities
    data["location_states"] = states

    # Work auth lists
    allowed = _coerce_work_auth_list(data.get("allowed_work_auth"))
    not_allowed = _coerce_work_auth_list(data.get("not_allowed_work_auth"))

    if not allowed:
        allowed = ["Any"]
    if not not_allowed:
        not_allowed = ["Any"]

    data["allowed_work_auth"] = allowed
    data["not_allowed_work_auth"] = not_allowed

    # Booleans that are nullable
    for bk in (
        "is_rate_strict",
        "health_insurance_provided",
        "retirement_matching",
        "background_check_required",
        "security_clearance_required",
        "management_experience_required",
        "travel_required",
    ):
        if bk in data:
            data[bk] = _to_bool_or_none(data[bk])

    # Booleans with defaults
    data["has_equity"] = bool(data.get("has_equity", False))
    data["relocation_assistance"] = bool(data.get("relocation_assistance", False))

    # Dates
    data["contract_start_date"] = _iso_date(data.get("contract_start_date"))
    data["contract_end_date"] = _iso_date(data.get("contract_end_date"))
    data["submission_cutoff_date"] = _iso_timestamptz(data.get("submission_cutoff_date"))

    # Trim strings
    for sk in (
        "source_call_id",
        "job_id",
        "external_requisition_id",
        "job_title",
        "end_client_name",
        "client_name",
        "industry",
        "location_country",
        "work_model_details",
        "pay_rate_currency",
        "pay_rate_notes",
        "salary_currency",
        "bonus_type",
        "bonus_notes",
        "equity_type",
        "equity_details",
        "retirement_matching_details",
        "benefits_summary",
        "contract_duration_text",
        "citizenship_required",
        "work_auth_notes",
        "background_check_details",
        "security_clearance_level",
        "work_hours",
        "time_zone",
        "travel_details",
        "interview_process",
        "vendor_name",
        "vendor_contact_name",
        "vendor_contact_email",
        "vendor_contact_phone",
        "raw_role_title_block",
        "status",
    ):
        if sk in data:
            data[sk] = _strip_or_none(data[sk])

    # Arrays (text[] columns)
    for ak in (
        "must_have_skills",
        "nice_to_have_skills",
        "primary_technologies",
        "certifications_required",
        "certifications_preferred",
        "domains",
        "responsibilities",
        "day_to_day",
        "other_constraints",
    ):
        if ak in data:
            arr = [x for x in (_strip_or_none(i) for i in _ensure_list(data[ak])) if x]
            data[ak] = arr

    # Ensure source_role_index is an integer
    if "source_role_index" in data:
        try:
            data["source_role_index"] = int(data["source_role_index"])
        except (TypeError, ValueError):
            data["source_role_index"] = 0

    # Ensure similar_jobs is a list
    if "similar_jobs" in data and data["similar_jobs"] is not None:
        if not isinstance(data["similar_jobs"], list):
            data["similar_jobs"] = []

    # Merge metadata if present
    if metadata:
        for k, v in metadata.items():
            data[k] = v

    return data


def _build_insert(row: Mapping[str, Any]) -> PreparedRow:
    """Build INSERT column list + placeholders for non-null values."""
    cols: List[str] = []
    ph: List[str] = []
    vals: List[Any] = []

    for k, v in row.items():
        if v is None:
            continue

        cols.append(k)

        cast = CASTS.get(k, "")
        ph.append(f"%s{cast}")

        if k in ("raw_json_input", "similar_jobs"):
            vals.append(Jsonb(v))
        else:
            vals.append(v)

    return PreparedRow(columns=cols, placeholders=ph, values=vals)


# ---------------------------
# Processing Logs Functions
# ---------------------------

async def insert_processing_log(
    call_id: str,
    stage: str,
    *,
    level: str = "info",
    role_index: Optional[int] = None,
    job_id: Optional[str] = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[int] = None,
) -> None:
    """Insert a log entry into the processing_logs table."""
    if stage not in PROCESSING_STAGE_ENUM:
        _log_event("warning", "invalid_processing_stage", stage=stage, call_id=call_id)
        return
    
    if level not in LOG_LEVEL_ENUM:
        level = "info"

    data: Dict[str, Any] = {
        "call_id": call_id,
        "stage": stage,
        "level": level,
        "role_index": role_index,
        "job_id": job_id,
        "message": message,
        "metadata": metadata or {},
        "duration_ms": duration_ms,
        "created_at": datetime.now(tz=timezone.utc),
    }

    cols: List[str] = []
    ph: List[str] = []
    vals: List[Any] = []

    for k, v in data.items():
        if v is None:
            continue
        cols.append(k)
        if k == "metadata":
            ph.append("%s")
            vals.append(Jsonb(v))
        elif k == "stage":
            ph.append("%s::processing_stage_enum")
            vals.append(v)
        elif k == "level":
            ph.append("%s::log_level_enum")
            vals.append(v)
        else:
            ph.append("%s")
            vals.append(v)

    if not cols:
        return

    query = SQL("INSERT INTO {table} ({cols}) VALUES ({vals})").format(
        table=Identifier(PROCESSING_LOGS_TABLE),
        cols=SQL(", ").join(Identifier(c) for c in cols),
        vals=SQL(", ").join(SQL(p) for p in ph),
    )

    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, vals)
                await conn.commit()
    except Exception as exc:
        # Log to stdout but don't raise - logging shouldn't break processing
        _log_event("error", "insert_processing_log_failed", call_id=call_id, stage=stage, error=str(exc))


async def fetch_processing_logs(
    call_id: str,
    *,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch processing logs for a call, ordered by created_at."""
    query = SQL(
        "SELECT id, call_id, role_index, job_id, level, stage, message, metadata, created_at, duration_ms "
        "FROM {table} WHERE call_id = %s ORDER BY created_at ASC LIMIT %s"
    ).format(table=Identifier(PROCESSING_LOGS_TABLE))

    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (call_id, limit))
                rows = await cur.fetchall()
                return [dict(row) for row in rows]
    except Exception as exc:
        _log_event("error", "fetch_processing_logs_failed", call_id=call_id, error=str(exc))
        return []


# ---------------------------
# Job Requirements Functions
# ---------------------------

async def insert_job_requirements(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    created_by: str = "ava_ai_recruiter",
    source_type: str = "phone_interview",
) -> List[Tuple[str, str]]:
    """Insert 1..N role payloads into job_requirements.

    Uses ON CONFLICT (source_call_id, source_role_index) DO NOTHING for deduplication.
    Returns a list of (id, job_id) tuples for each inserted role.
    Skipped (duplicate) roles are not included in the results.
    """

    roles: List[Dict[str, Any]]
    if isinstance(payload, list):
        roles = payload
    else:
        roles = [payload]

    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    _log_event("info", "db_connecting", **db_info)
    results: List[Tuple[str, str]] = []

    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                for idx, role in enumerate(roles):
                    meta = dict(metadata or {})
                    meta.setdefault("created_by", created_by)
                    meta.setdefault("source_type", source_type)
                    
                    # Ensure source_role_index is set
                    if "source_role_index" not in role:
                        role["source_role_index"] = idx

                    if "raw_json_input" not in meta:
                        role_payload = role if isinstance(role, dict) else {}
                        meta["raw_json_input"] = role_payload.get("raw_json_input", role)

                    prepared = _prepare_role(role, meta)
                    ins = _build_insert(prepared)

                    if not ins.columns:
                        continue

                    # Use ON CONFLICT for deduplication
                    query = SQL(
                        "INSERT INTO job_requirements ({cols}) VALUES ({vals}) "
                        "ON CONFLICT (source_call_id, source_role_index) DO NOTHING "
                        "RETURNING id, job_id"
                    ).format(
                        cols=SQL(", ").join(Identifier(c) for c in ins.columns),
                        vals=SQL(", ").join(SQL(p) for p in ins.placeholders),
                    )

                    await cur.execute(query, ins.values)
                    row = await cur.fetchone()
                    
                    if row:
                        # Row was inserted
                        results.append((str(row["id"]), row["job_id"]))
                        _log_event(
                            "info",
                            "db_insert_job_requirements_row",
                            job_id=row["job_id"],
                            job_title=prepared.get("job_title"),
                            role_index=idx,
                            **db_info,
                        )
                    else:
                        # Row was skipped (duplicate)
                        _log_event(
                            "info",
                            "db_insert_job_requirements_skipped",
                            source_call_id=prepared.get("source_call_id"),
                            source_role_index=prepared.get("source_role_index"),
                            role_index=idx,
                            **db_info,
                        )

                await conn.commit()

        _log_event("info", "db_insert_job_requirements_ok", rows=len(results), **db_info)
        return results
    except Exception as exc:
        _log_event("error", "db_insert_job_requirements_failed", error=str(exc), **db_info)
        raise


async def fetch_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a job requirement by job_id."""
    query = SQL("SELECT * FROM job_requirements WHERE job_id = %s")
    
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (job_id,))
                row = await cur.fetchone()
                return dict(row) if row else None
    except Exception as exc:
        _log_event("error", "fetch_job_by_id_failed", job_id=job_id, error=str(exc))
        return None


# ---------------------------
# Call Transcripts Functions
# ---------------------------

async def insert_call_transcript(
    payload: Mapping[str, Any],
    *,
    table: str = CALL_TRANSCRIPTS_TABLE,
) -> None:
    data = dict(payload)
    if not data.get("call_id"):
        raise ValueError("call_id is required for call_transcripts insert")

    data.setdefault("received_at", datetime.now(tz=timezone.utc))

    cols: List[str] = []
    ph: List[str] = []
    vals: List[Any] = []
    for k, v in data.items():
        if v is None:
            continue
        cols.append(k)
        ph.append("%s")
        if k in CALL_TRANSCRIPT_JSON_COLUMNS:
            vals.append(Jsonb(v))
        else:
            vals.append(v)

    if not cols:
        return

    query = SQL("INSERT INTO {table} ({cols}) VALUES ({vals}) ON CONFLICT (call_id) DO NOTHING").format(
        table=Identifier(table),
        cols=SQL(", ").join(Identifier(c) for c in cols),
        vals=SQL(", ").join(SQL(p) for p in ph),
    )

    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    _log_event("info", "db_connecting", **db_info)
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, vals)
                await conn.commit()
        _log_event("info", "db_insert_call_transcript_ok", call_id=data.get("call_id"), **db_info)
    except Exception as exc:
        _log_event(
            "error",
            "db_insert_call_transcript_failed",
            call_id=data.get("call_id"),
            error=str(exc),
            **db_info,
        )
        raise


async def append_call_message(
    call_id: str,
    *,
    message: Mapping[str, Any],
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_name: Optional[str] = None,
    event_type: str = "message",
    call_started_at: Optional[str] = None,
    call_ended_at: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    raw_payload: Optional[Any] = None,
    table: str = CALL_TRANSCRIPTS_TABLE,
) -> None:
    if not call_id:
        raise ValueError("call_id is required for call_transcripts append")

    data: Dict[str, Any] = {
        "call_id": call_id,
        "agent_id": agent_id,
        "session_id": session_id,
        "user_name": user_name,
        "event_type": event_type,
        "call_started_at": call_started_at,
        "call_ended_at": call_ended_at,
        "messages": [dict(message)],
        "raw_payload": raw_payload,
        "tags": tags,
        "status": "streaming",
        "received_at": datetime.now(tz=timezone.utc),
    }

    cols: List[str] = []
    ph: List[str] = []
    vals: List[Any] = []
    for k, v in data.items():
        if v is None:
            continue
        cols.append(k)
        ph.append("%s")
        if k in CALL_TRANSCRIPT_JSON_COLUMNS:
            vals.append(Jsonb(v))
        else:
            vals.append(v)

    query = SQL(
        "INSERT INTO {table} ({cols}) VALUES ({vals}) "
        "ON CONFLICT (call_id) DO UPDATE SET "
        "messages = COALESCE({table}.messages, '[]'::jsonb) || EXCLUDED.messages, "
        "updated_at = NOW()"
    ).format(
        table=Identifier(table),
        cols=SQL(", ").join(Identifier(c) for c in cols),
        vals=SQL(", ").join(SQL(p) for p in ph),
    )

    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    _log_event("info", "db_connecting", **db_info)
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, vals)
                await conn.commit()
        _log_event("info", "db_append_call_message_ok", call_id=call_id, **db_info)
    except Exception as exc:
        _log_event(
            "error",
            "db_append_call_message_failed",
            call_id=call_id,
            error=str(exc),
            **db_info,
        )
        raise


async def fetch_call_messages(
    call_id: str,
    *,
    table: str = CALL_TRANSCRIPTS_TABLE,
) -> List[Any]:
    if not call_id:
        return []
    query = SQL("SELECT messages FROM {table} WHERE call_id = %s").format(table=Identifier(table))
    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    _log_event("info", "db_connecting", **db_info)
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (call_id,))
                row = await cur.fetchone()
                if not row:
                    return []
                messages = row.get("messages") or []
                return messages if isinstance(messages, list) else []
    except Exception as exc:
        _log_event("error", "db_fetch_call_messages_failed", call_id=call_id, error=str(exc), **db_info)
        return []


async def fetch_call_transcript(
    call_id: str,
    *,
    table: str = CALL_TRANSCRIPTS_TABLE,
) -> Optional[Dict[str, Any]]:
    """Fetch full call transcript record."""
    if not call_id:
        return None
    query = SQL("SELECT * FROM {table} WHERE call_id = %s").format(table=Identifier(table))
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (call_id,))
                row = await cur.fetchone()
                return dict(row) if row else None
    except Exception as exc:
        _log_event("error", "db_fetch_call_transcript_failed", call_id=call_id, error=str(exc))
        return None


async def update_call_transcript(
    call_id: str,
    *,
    parsed_requirements: Optional[Any] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None,
    table: str = CALL_TRANSCRIPTS_TABLE,
) -> None:
    updates: Dict[str, Any] = {}
    if parsed_requirements is not None:
        updates["parsed_requirements"] = parsed_requirements
    if status is not None:
        updates["status"] = status
    if error_message is not None:
        updates["error_message"] = error_message

    if not updates:
        return

    updates["updated_at"] = datetime.now(tz=timezone.utc)

    cols = list(updates.keys())
    set_clause = SQL(", ").join(
        SQL("{} = {}").format(
            Identifier(c),
            SQL("%s"),
        )
        for c in cols
    )

    values: List[Any] = []
    for c in cols:
        if c in CALL_TRANSCRIPT_JSON_COLUMNS:
            values.append(Jsonb(updates[c]))
        else:
            values.append(updates[c])
    values.append(call_id)

    query = SQL("UPDATE {table} SET {set_clause} WHERE call_id = %s").format(
        table=Identifier(table),
        set_clause=set_clause,
    )

    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    _log_event("info", "db_connecting", **db_info)
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, values)
                await conn.commit()
        _log_event("info", "db_update_call_transcript_ok", call_id=call_id, **db_info)
    except Exception as exc:
        _log_event(
            "error",
            "db_update_call_transcript_failed",
            call_id=call_id,
            error=str(exc),
            **db_info,
        )
        raise