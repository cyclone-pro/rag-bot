"""Database operations for Beyond Presence webhook processing."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg.types.json import Jsonb

logger = logging.getLogger("bey_db")


@dataclass
class PreparedRow:
    columns: List[str]
    placeholders: List[str]
    values: List[Any]


# ============================================================
# ENUMS (must match Postgres exactly)
# ============================================================

SENIORITY_LEVEL_ENUM = {"Entry", "Mid", "Senior", "Lead", "Architect", "Manager", "Director", "VP", "C-level", "unspecified"}
JOB_TYPE_ENUM = {"Contract", "Contract-to-hire", "Full-time", "Part-time", "Internship", "Other", "unspecified"}
SUBMISSION_URGENCY_ENUM = {"normal", "urgent", "flexible"}
WORK_MODEL_ENUM = {"onsite", "remote", "hybrid", "flexible", "unspecified"}
RATE_UNIT_ENUM = {"hour", "day", "week", "month", "year", "unspecified"}
EMPLOYMENT_TYPE_ENUM = {"C2C", "W2", "1099", "unspecified"}
WORK_AUTH_ENUM = {"USC", "GC", "H1B", "H4-EAD", "L1", "L2-EAD", "TN", "E3", "F1-OPT", "F1-CPT", "STEM-OPT", "J1", "O1", "EAD", "Asylum-EAD", "GC-EAD", "Any", "unsp"}
LOG_LEVEL_ENUM = {"debug", "info", "warning", "error"}
PROCESSING_STAGE_ENUM = {
    "webhook_received", "call_classified", "transcript_cleaned", "llm_started", "llm_complete", "llm_failed",
    "validation_complete", "validation_failed", "embedding_started", "embedding_complete", "embedding_failed",
    "similarity_check_started", "similarity_check_complete", "similarity_check_failed",
    "postgres_insert_started", "postgres_insert_complete", "postgres_insert_skipped", "postgres_insert_failed",
    "milvus_insert_started", "milvus_insert_complete", "milvus_insert_skipped", "milvus_insert_failed",
    "processing_complete", "processing_failed", "processing_skipped"
}

DEFAULTS: Dict[str, Any] = {
    "seniority_level": "unspecified", "job_type": "unspecified", "submission_urgency": "normal",
    "work_model": "unspecified", "pay_rate_unit": "unspecified", "employment_type": "unspecified",
    "status": "active", "has_equity": False, "relocation_assistance": False, "positions_available": 1,
    "allowed_work_auth": ["Any"], "not_allowed_work_auth": ["Any"], "source_role_index": 0,
}

CASTS = {
    "seniority_level": "::seniority_level_enum", "job_type": "::job_type_enum",
    "submission_urgency": "::submission_urgency_enum", "work_model": "::work_model_enum",
    "pay_rate_unit": "::rate_unit_enum", "employment_type": "::employment_type_enum",
    "allowed_work_auth": "::work_authorization_enum[]", "not_allowed_work_auth": "::work_authorization_enum[]",
}

MODEL_COLUMNS = {
    "source_call_id", "source_role_index", "job_id", "external_requisition_id", "job_title",
    "seniority_level", "job_type", "end_client_name", "client_name", "industry", "positions_available",
    "submission_cutoff_date", "submission_urgency", "max_candidates_allowed", "location_cities",
    "location_states", "location_country", "work_model", "work_model_details", "pay_rate_min",
    "pay_rate_max", "pay_rate_currency", "pay_rate_unit", "employment_type", "is_rate_strict",
    "pay_rate_notes", "salary_min", "salary_max", "salary_currency", "bonus_percentage_min",
    "bonus_percentage_max", "bonus_type", "bonus_notes", "has_equity", "equity_type", "equity_details",
    "pto_days", "health_insurance_provided", "retirement_matching", "retirement_matching_details",
    "benefits_summary", "sign_on_bonus", "relocation_assistance", "relocation_amount",
    "contract_duration_text", "contract_start_date", "contract_end_date", "contract_can_extend",
    "allowed_work_auth", "not_allowed_work_auth", "citizenship_required", "work_auth_notes",
    "background_check_required", "background_check_details", "security_clearance_required",
    "security_clearance_level", "overall_min_years", "primary_role_min_years",
    "management_experience_required", "must_have_skills", "nice_to_have_skills", "primary_technologies",
    "certifications_required", "certifications_preferred", "domains", "responsibilities", "day_to_day",
    "other_constraints", "work_hours", "time_zone", "travel_required", "travel_details",
    "interview_process", "vendor_name", "vendor_contact_name", "vendor_contact_email",
    "vendor_contact_phone", "raw_role_title_block", "raw_json_input", "status", "similar_jobs", 
    "similarity_score", "milvus_synced", "milvus_synced_at", "notes", "created_by", "source_type",
    "phone_call_duration",
}

CALL_TRANSCRIPT_JSON_COLUMNS = {"raw_payload", "messages", "evaluation", "parsed_requirements", "tags"}


def _log_event(level: str, message: str, **fields: Any) -> None:
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


def _db_env_status() -> Dict[str, bool]:
    return {
        "DATABASE_URL_set": bool(os.getenv("DATABASE_URL")),
        "PGHOST_set": bool(os.getenv("PGHOST")),
        "PGDATABASE_set": bool(os.getenv("PGDATABASE")),
        "PGUSER_set": bool(os.getenv("PGUSER")),
        "PGPASSWORD_set": bool(os.getenv("PGPASSWORD")),
    }


def _db_info_from_url(db_url: str) -> Dict[str, Optional[str]]:
    try:
        parsed = urlparse(db_url)
        return {"db_scheme": parsed.scheme, "db_host": parsed.hostname,
                "db_port": str(parsed.port) if parsed.port else None,
                "db_name": parsed.path.lstrip("/") if parsed.path else None, "db_user": parsed.username}
    except Exception:
        return {"db_url_set": bool(db_url)}


def _db_url_from_env() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host, db = os.getenv("PGHOST"), os.getenv("PGDATABASE")
    user, pw = os.getenv("PGUSER"), os.getenv("PGPASSWORD")
    port = os.getenv("PGPORT", "5432")
    if not (host and db and user and pw):
        _log_event("error", "db_env_missing", **_db_env_status())
        raise ValueError("Set DATABASE_URL or PGHOST/PGDATABASE/PGUSER/PGPASSWORD")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


async def check_db_connection(timeout: float = 5.0) -> Tuple[bool, str]:
    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        return True, "ok"
    except Exception as e:
        _log_event("error", "db_health_failed", error=str(e), **db_info)
        return False, str(e)


# ============================================================
# HELPERS
# ============================================================

def _strip_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1", "t", "y")
    return bool(value)


def _to_bool_or_none(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
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
    if s in allowed:
        return s
    for item in allowed:
        if item.lower() == s.lower():
            return item
    return default


def _coerce_work_auth(values: Any) -> List[str]:
    if not values:
        return ["Any"]
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return ["Any"]
    mapping = {"us citizen": "USC", "green card": "GC", "h1-b": "H1B", "h-1b": "H1B", "unspecified": "unsp"}
    result = []
    for v in values:
        if not isinstance(v, str):
            continue
        v_lower = v.strip().lower()
        if v_lower in mapping:
            result.append(mapping[v_lower])
        elif v.strip() in WORK_AUTH_ENUM:
            result.append(v.strip())
    return result if result else ["Any"]


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


def _prepare_role(role: Mapping[str, Any], metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Normalize + validate a single role object against DB constraints."""
    data: Dict[str, Any] = dict(DEFAULTS)
    
    for k, v in role.items():
        if k in MODEL_COLUMNS:
            data[k] = v
    
    # Coerce enums
    data["seniority_level"] = _coerce_enum(data.get("seniority_level"), SENIORITY_LEVEL_ENUM, "unspecified")
    data["job_type"] = _coerce_enum(data.get("job_type"), JOB_TYPE_ENUM, "unspecified")
    data["submission_urgency"] = _coerce_enum(data.get("submission_urgency"), SUBMISSION_URGENCY_ENUM, "normal")
    data["work_model"] = _coerce_enum(data.get("work_model"), WORK_MODEL_ENUM, "unspecified")
    data["pay_rate_unit"] = _coerce_enum(data.get("pay_rate_unit"), RATE_UNIT_ENUM, "unspecified")
    data["employment_type"] = _coerce_enum(data.get("employment_type"), EMPLOYMENT_TYPE_ENUM, "unspecified")
    
    # Locations
    cities, states = _clean_locations(data.get("location_cities"), data.get("location_states"))
    data["location_cities"] = cities
    data["location_states"] = states
    
    # Work auth
    data["allowed_work_auth"] = _coerce_work_auth(data.get("allowed_work_auth"))
    data["not_allowed_work_auth"] = _coerce_work_auth(data.get("not_allowed_work_auth"))
    
    # Booleans (nullable)
    for bk in ("is_rate_strict", "health_insurance_provided", "retirement_matching", "background_check_required",
               "security_clearance_required", "management_experience_required", "travel_required", "contract_can_extend"):
        if bk in data:
            data[bk] = _to_bool_or_none(data[bk])
    
    # Booleans with defaults
    data["has_equity"] = _to_bool(data.get("has_equity"), False)
    data["relocation_assistance"] = _to_bool(data.get("relocation_assistance"), False)
    data.setdefault("milvus_synced", False)
    
    # Dates
    data["contract_start_date"] = _iso_date(data.get("contract_start_date"))
    data["contract_end_date"] = _iso_date(data.get("contract_end_date"))
    data["submission_cutoff_date"] = _iso_timestamptz(data.get("submission_cutoff_date"))
    
    # String arrays
    for ak in ("must_have_skills", "nice_to_have_skills", "primary_technologies", "certifications_required",
               "certifications_preferred", "domains", "responsibilities", "day_to_day", "other_constraints"):
        if ak in data:
            data[ak] = [x for x in (_strip_or_none(i) for i in _ensure_list(data[ak])) if x]
    
    # Job title
    title = _strip_or_none(data.get("job_title"))
    if not title:
        data["job_title"] = "Unknown role"
    elif len(title) > 200:
        data["job_title"] = title[:200]
    else:
        data["job_title"] = title
    
    # Ensure source_role_index is int
    if "source_role_index" in data:
        try:
            data["source_role_index"] = int(data["source_role_index"])
        except (TypeError, ValueError):
            data["source_role_index"] = 0
    
    # Merge metadata
    if metadata:
        for k, v in metadata.items():
            if k in MODEL_COLUMNS:
                data[k] = v
    
    return data


def _build_insert(row: Mapping[str, Any]) -> PreparedRow:
    cols, ph, vals = [], [], []
    for k, v in row.items():
        if v is None or k not in MODEL_COLUMNS:
            continue
        cols.append(k)
        ph.append(f"%s{CASTS.get(k, '')}")
        if k in ("raw_json_input", "similar_jobs"):
            vals.append(Jsonb(v))
        else:
            vals.append(v)
    return PreparedRow(columns=cols, placeholders=ph, values=vals)


# ============================================================
# CALL TRANSCRIPTS
# ============================================================

async def insert_call_transcript(payload: Mapping[str, Any]) -> bool:
    """Insert or update call transcript from Beyond Presence webhook."""
    call_id = payload.get("call_id")
    if not call_id:
        return False
    
    call_data = payload.get("call_data", {})
    evaluation = payload.get("evaluation", {})
    messages = payload.get("messages", [])
    
    data = {
        "call_id": call_id,
        "agent_id": call_data.get("agentId") or payload.get("agent_id") or "unknown",
        "session_id": payload.get("session_id") or call_id,
        "user_name": call_data.get("userName") or payload.get("user_name"),
        "event_type": payload.get("event_type"),
        "call_started_at": _iso_timestamptz(call_data.get("startedAt") or payload.get("call_started_at")),
        "call_ended_at": _iso_timestamptz(call_data.get("endedAt") or payload.get("call_ended_at")),
        "received_at": datetime.now(tz=timezone.utc),
        "evaluation": evaluation,
        "messages": messages,
        "raw_payload": payload,
        "status": "received",
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tags": payload.get("tags"),
    }
    
    cols, ph, vals = [], [], []
    for k, v in data.items():
        if v is None:
            continue
        cols.append(k)
        ph.append("%s")
        vals.append(Jsonb(v) if k in CALL_TRANSCRIPT_JSON_COLUMNS else v)
    
    query = SQL(
        "INSERT INTO call_transcripts ({cols}) VALUES ({vals}) "
        "ON CONFLICT (call_id) DO UPDATE SET "
        "event_type=EXCLUDED.event_type, call_ended_at=EXCLUDED.call_ended_at, "
        "evaluation=EXCLUDED.evaluation, messages=EXCLUDED.messages, "
        "raw_payload=EXCLUDED.raw_payload, message_count=EXCLUDED.message_count, updated_at=NOW()"
    ).format(cols=SQL(", ").join(Identifier(c) for c in cols), vals=SQL(", ").join(SQL(p) for p in ph))
    
    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, vals)
            await conn.commit()
        _log_event("info", "db_insert_call_transcript_ok", call_id=call_id, **db_info)
        return True
    except Exception as e:
        _log_event("error", "insert_call_transcript_failed", call_id=call_id, error=str(e), **db_info)
        raise


async def append_call_message(
    call_id: str, *, message: Mapping[str, Any], agent_id: Optional[str] = None,
    session_id: Optional[str] = None, user_name: Optional[str] = None, event_type: str = "message",
    tags: Optional[Dict[str, Any]] = None, raw_payload: Optional[Any] = None,
) -> None:
    """Append a single message to call_transcripts.messages array (for streaming)."""
    if not call_id:
        raise ValueError("call_id required")
    
    data = {
        "call_id": call_id, "agent_id": agent_id, "session_id": session_id, "user_name": user_name,
        "event_type": event_type, "messages": [dict(message)], "raw_payload": raw_payload,
        "tags": tags, "status": "streaming", "received_at": datetime.now(tz=timezone.utc),
    }
    
    cols, ph, vals = [], [], []
    for k, v in data.items():
        if v is None:
            continue
        cols.append(k)
        ph.append("%s")
        vals.append(Jsonb(v) if k in CALL_TRANSCRIPT_JSON_COLUMNS else v)
    
    query = SQL(
        "INSERT INTO call_transcripts ({cols}) VALUES ({vals}) "
        "ON CONFLICT (call_id) DO UPDATE SET "
        "messages = COALESCE(call_transcripts.messages, '[]'::jsonb) || EXCLUDED.messages, "
        "message_count = COALESCE(call_transcripts.message_count, 0) + 1, updated_at = NOW()"
    ).format(cols=SQL(", ").join(Identifier(c) for c in cols), vals=SQL(", ").join(SQL(p) for p in ph))
    
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, vals)
            await conn.commit()
        _log_event("info", "db_append_call_message_ok", call_id=call_id)
    except Exception as e:
        _log_event("error", "db_append_call_message_failed", call_id=call_id, error=str(e))
        raise


async def fetch_call_messages(call_id: str) -> List[Any]:
    """Fetch messages array from call_transcripts."""
    if not call_id:
        return []
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT messages FROM call_transcripts WHERE call_id = %s", (call_id,))
                row = await cur.fetchone()
                if not row:
                    return []
                messages = row.get("messages") or []
                return messages if isinstance(messages, list) else []
    except Exception as e:
        _log_event("error", "db_fetch_call_messages_failed", call_id=call_id, error=str(e))
        return []


async def fetch_call_transcript(call_id: str) -> Optional[Dict[str, Any]]:
    """Fetch full call transcript record."""
    if not call_id:
        return None
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT * FROM call_transcripts WHERE call_id = %s", (call_id,))
                row = await cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        _log_event("error", "fetch_call_transcript_failed", call_id=call_id, error=str(e))
        return None


async def update_call_transcript(
    call_id: str, *, status: Optional[str] = None,
    parsed_requirements: Optional[Dict] = None, error_message: Optional[str] = None,
) -> bool:
    """Update call transcript status and parsed requirements."""
    updates: Dict[str, Any] = {"updated_at": datetime.now(tz=timezone.utc)}
    if status:
        updates["status"] = status
    if parsed_requirements is not None:
        updates["parsed_requirements"] = parsed_requirements
    if error_message is not None:
        updates["error_message"] = error_message
    
    set_parts, vals = [], []
    for c, v in updates.items():
        set_parts.append(SQL("{} = %s").format(Identifier(c)))
        vals.append(Jsonb(v) if c in CALL_TRANSCRIPT_JSON_COLUMNS else v)
    vals.append(call_id)
    
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(SQL("UPDATE call_transcripts SET {sets} WHERE call_id = %s").format(
                    sets=SQL(", ").join(set_parts)), vals)
            await conn.commit()
        return True
    except Exception as e:
        _log_event("error", "update_call_transcript_failed", call_id=call_id, error=str(e))
        return False


# ============================================================
# JOB REQUIREMENTS
# ============================================================

async def insert_job_requirement(role: Dict[str, Any], call_id: str, role_index: int) -> Optional[Tuple[str, str]]:
    """Insert a single job requirement. Returns (id, job_id) or None if skipped."""
    role["source_call_id"] = call_id
    role["source_role_index"] = role_index
    data = _prepare_role(role)
    ins = _build_insert(data)
    
    if not ins.columns:
        return None
    
    query = SQL(
        "INSERT INTO job_requirements ({cols}) VALUES ({vals}) "
        "ON CONFLICT (source_call_id, source_role_index) DO NOTHING RETURNING id, job_id"
    ).format(
        cols=SQL(", ").join(Identifier(c) for c in ins.columns),
        vals=SQL(", ").join(SQL(p) for p in ins.placeholders),
    )
    
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, ins.values)
                row = await cur.fetchone()
            await conn.commit()
            if row:
                _log_event("info", "db_insert_job_ok", job_id=row["job_id"], job_title=data.get("job_title"))
                return (str(row["id"]), row["job_id"])
            _log_event("info", "db_insert_job_skipped", source_call_id=call_id, source_role_index=role_index)
            return None
    except Exception as e:
        _log_event("error", "insert_job_requirement_failed", call_id=call_id, role_index=role_index, error=str(e))
        raise


async def insert_job_requirements(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]], *,
    metadata: Optional[Mapping[str, Any]] = None,
    created_by: str = "ava_ai_recruiter", source_type: str = "beyond_presence",
) -> List[Tuple[str, str]]:
    """Insert 1..N role payloads into job_requirements. Returns list of (id, job_id) tuples."""
    roles = payload if isinstance(payload, list) else [payload]
    
    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    results: List[Tuple[str, str]] = []
    
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                for idx, role in enumerate(roles):
                    meta = dict(metadata or {})
                    meta.setdefault("created_by", created_by)
                    meta.setdefault("source_type", source_type)
                    if "source_role_index" not in role:
                        role["source_role_index"] = idx
                    
                    prepared = _prepare_role(role, meta)
                    ins = _build_insert(prepared)
                    if not ins.columns:
                        continue
                    
                    query = SQL(
                        "INSERT INTO job_requirements ({cols}) VALUES ({vals}) "
                        "ON CONFLICT (source_call_id, source_role_index) DO NOTHING RETURNING id, job_id"
                    ).format(
                        cols=SQL(", ").join(Identifier(c) for c in ins.columns),
                        vals=SQL(", ").join(SQL(p) for p in ins.placeholders),
                    )
                    
                    await cur.execute(query, ins.values)
                    row = await cur.fetchone()
                    if row:
                        results.append((str(row["id"]), row["job_id"]))
                        _log_event("info", "db_insert_job_ok", job_id=row["job_id"], **db_info)
                    else:
                        _log_event("info", "db_insert_job_skipped", source_call_id=prepared.get("source_call_id"), **db_info)
                
                await conn.commit()
        return results
    except Exception as e:
        _log_event("error", "insert_job_requirements_failed", error=str(e), **db_info)
        raise


async def update_job_milvus_status(job_id: str, synced: bool) -> bool:
    """Update milvus_synced status for a job."""
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE job_requirements SET milvus_synced=%s, milvus_synced_at=%s WHERE job_id=%s",
                    (synced, datetime.now(tz=timezone.utc) if synced else None, job_id))
            await conn.commit()
        return True
    except Exception as e:
        _log_event("error", "update_job_milvus_status_failed", job_id=job_id, error=str(e))
        return False


async def fetch_unsynced_jobs(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch jobs not yet synced to Milvus."""
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM job_requirements WHERE milvus_synced=FALSE ORDER BY created_at LIMIT %s", (limit,))
                return [dict(r) for r in await cur.fetchall()]
    except Exception as e:
        _log_event("error", "fetch_unsynced_jobs_failed", error=str(e))
        return []


async def fetch_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a job by job_id."""
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT * FROM job_requirements WHERE job_id=%s", (job_id,))
                row = await cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        _log_event("error", "fetch_job_by_id_failed", job_id=job_id, error=str(e))
        return None


# ============================================================
# PROCESSING LOGS
# ============================================================

async def insert_processing_log(
    call_id: str, stage: str, *, level: str = "info",
    role_index: Optional[int] = None, job_id: Optional[str] = None,
    message: Optional[str] = None, metadata: Optional[Dict] = None, duration_ms: Optional[int] = None,
) -> None:
    """Insert a processing log entry."""
    if stage not in PROCESSING_STAGE_ENUM or level not in LOG_LEVEL_ENUM:
        return
    
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO processing_logs (call_id, stage, level, role_index, job_id, message, metadata, duration_ms, created_at)
                       VALUES (%s, %s::processing_stage_enum, %s::log_level_enum, %s, %s, %s, %s, %s, %s)""",
                    (call_id, stage, level, role_index, job_id, message, Jsonb(metadata or {}),
                     duration_ms, datetime.now(tz=timezone.utc)))
            await conn.commit()
    except Exception as e:
        _log_event("error", "insert_processing_log_failed", call_id=call_id, stage=stage, error=str(e))


async def fetch_processing_logs(call_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch processing logs for a call."""
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM processing_logs WHERE call_id=%s ORDER BY created_at LIMIT %s", (call_id, limit))
                return [dict(r) for r in await cur.fetchall()]
    except Exception as e:
        _log_event("error", "fetch_processing_logs_failed", call_id=call_id, error=str(e))
        return []


# ============================================================
# STATISTICS
# ============================================================

async def fetch_stats() -> Dict[str, Any]:
    """Fetch processing statistics."""
    now = datetime.now(tz=timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())
    month_start = today_start.replace(day=1)
    
    stats = {
        "total_calls": 0, "calls_today": 0, "calls_this_week": 0, "calls_this_month": 0,
        "minutes_used": {"today": 0, "this_week": 0, "this_month": 0, "total": 0},
        "status_breakdown": {"received": 0, "processing": 0, "parsed": 0, "failed": 0, 
                            "skipped": 0, "product_inquiry": 0},
        "jobs_created": {"total": 0, "today": 0, "this_week": 0, "this_month": 0},
        "avg_processing_time_ms": 0, "avg_roles_per_call": 0, "avg_call_duration_minutes": 0,
        "similarity_matches": {"total": 0, "above_95_percent": 0, "above_90_percent": 0},
        "error_breakdown": {"llm_failed": 0, "milvus_failed": 0, "postgres_failed": 0, "validation_failed": 0},
        "milvus_sync": {"synced": 0, "unsynced": 0},
    }
    
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                # Calls
                await cur.execute("""
                    SELECT COUNT(*) as total,
                        COUNT(*) FILTER (WHERE received_at >= %s) as today,
                        COUNT(*) FILTER (WHERE received_at >= %s) as this_week,
                        COUNT(*) FILTER (WHERE received_at >= %s) as this_month,
                        COUNT(*) FILTER (WHERE status='received') as s_received,
                        COUNT(*) FILTER (WHERE status='processing') as s_processing,
                        COUNT(*) FILTER (WHERE status='parsed') as s_parsed,
                        COUNT(*) FILTER (WHERE status='failed' OR status LIKE 'failed_%%') as s_failed,
                        COUNT(*) FILTER (WHERE status='skipped') as s_skipped,
                        COUNT(*) FILTER (WHERE status='product_inquiry') as s_product_inquiry
                    FROM call_transcripts WHERE event_type='call_ended'
                """, (today_start, week_start, month_start))
                r = await cur.fetchone()
                if r:
                    stats.update({"total_calls": r["total"] or 0, "calls_today": r["today"] or 0,
                                  "calls_this_week": r["this_week"] or 0, "calls_this_month": r["this_month"] or 0,
                                  "status_breakdown": {"received": r["s_received"] or 0, "processing": r["s_processing"] or 0,
                                                       "parsed": r["s_parsed"] or 0, "failed": r["s_failed"] or 0,
                                                       "skipped": r["s_skipped"] or 0, "product_inquiry": r["s_product_inquiry"] or 0}})
                
                # Minutes
                await cur.execute("""
                    SELECT COALESCE(SUM((evaluation->>'duration_minutes')::float), 0) as total,
                        COALESCE(SUM((evaluation->>'duration_minutes')::float) FILTER (WHERE received_at >= %s), 0) as today,
                        COALESCE(SUM((evaluation->>'duration_minutes')::float) FILTER (WHERE received_at >= %s), 0) as this_week,
                        COALESCE(SUM((evaluation->>'duration_minutes')::float) FILTER (WHERE received_at >= %s), 0) as this_month
                    FROM call_transcripts WHERE event_type='call_ended' AND evaluation->>'duration_minutes' IS NOT NULL
                """, (today_start, week_start, month_start))
                r = await cur.fetchone()
                if r:
                    stats["minutes_used"] = {"today": round(r["today"] or 0, 2), "this_week": round(r["this_week"] or 0, 2),
                                              "this_month": round(r["this_month"] or 0, 2), "total": round(r["total"] or 0, 2)}
                    if stats["total_calls"]:
                        stats["avg_call_duration_minutes"] = round(r["total"] / stats["total_calls"], 2)
                
                # Jobs
                await cur.execute("""
                    SELECT COUNT(*) as total,
                        COUNT(*) FILTER (WHERE created_at >= %s) as today,
                        COUNT(*) FILTER (WHERE created_at >= %s) as this_week,
                        COUNT(*) FILTER (WHERE created_at >= %s) as this_month,
                        COUNT(*) FILTER (WHERE milvus_synced=TRUE) as synced,
                        COUNT(*) FILTER (WHERE milvus_synced=FALSE OR milvus_synced IS NULL) as unsynced,
                        COUNT(*) FILTER (WHERE similarity_score IS NOT NULL) as with_similar,
                        COUNT(*) FILTER (WHERE similarity_score >= 0.95) as above_95,
                        COUNT(*) FILTER (WHERE similarity_score >= 0.90) as above_90
                    FROM job_requirements
                """, (today_start, week_start, month_start))
                r = await cur.fetchone()
                if r:
                    stats["jobs_created"] = {"total": r["total"] or 0, "today": r["today"] or 0,
                                              "this_week": r["this_week"] or 0, "this_month": r["this_month"] or 0}
                    stats["milvus_sync"] = {"synced": r["synced"] or 0, "unsynced": r["unsynced"] or 0}
                    stats["similarity_matches"] = {"total": r["with_similar"] or 0, "above_95_percent": r["above_95"] or 0,
                                                    "above_90_percent": r["above_90"] or 0}
                
                if stats["total_calls"] and stats["jobs_created"]["total"]:
                    stats["avg_roles_per_call"] = round(stats["jobs_created"]["total"] / stats["total_calls"], 2)
                
                # Errors
                await cur.execute("""
                    SELECT COUNT(*) FILTER (WHERE stage='llm_failed') as llm,
                        COUNT(*) FILTER (WHERE stage='milvus_insert_failed') as milvus,
                        COUNT(*) FILTER (WHERE stage='postgres_insert_failed') as postgres,
                        COUNT(*) FILTER (WHERE stage='validation_failed') as validation
                    FROM processing_logs WHERE level='error'
                """)
                r = await cur.fetchone()
                if r:
                    stats["error_breakdown"] = {"llm_failed": r["llm"] or 0, "milvus_failed": r["milvus"] or 0,
                                                 "postgres_failed": r["postgres"] or 0, "validation_failed": r["validation"] or 0}
                
                # Avg processing time
                await cur.execute("SELECT AVG(duration_ms) as avg FROM processing_logs WHERE stage='processing_complete' AND duration_ms IS NOT NULL")
                r = await cur.fetchone()
                if r and r["avg"]:
                    stats["avg_processing_time_ms"] = round(r["avg"], 0)
        
        return stats
    except Exception as e:
        _log_event("error", "fetch_stats_failed", error=str(e))
        return stats


async def fetch_failed_calls(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch failed calls for reprocessing."""
    db_url = _db_url_from_env()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT call_id, error_message, updated_at FROM call_transcripts WHERE status='failed' OR status LIKE 'failed_%%' ORDER BY updated_at DESC LIMIT %s",
                    (limit,))
                return [dict(r) for r in await cur.fetchall()]
    except Exception as e:
        _log_event("error", "fetch_failed_calls_failed", error=str(e))
        return []