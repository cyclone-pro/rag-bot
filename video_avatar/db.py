from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
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

# Your work_authorization_enum list includes 'unsp' (not 'unspecified').
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
}

CALL_TRANSCRIPTS_TABLE = "call_transcripts"
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

    # NOTE: basic URL; if you have special chars in password, url-encode it.
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
    # occasionally model sends tuple
    if isinstance(v, tuple):
        return list(v)
    return [v]


def _coerce_enum(v: Any, allowed: set, default: str) -> str:
    s = _strip_or_none(v)
    if not s:
        return default
    # exact match only (case-sensitive enums)
    return s if s in allowed else default


def _coerce_work_auth_list(v: Any) -> List[str]:
    items = _ensure_list(v)
    out: List[str] = []
    for it in items:
        s = _strip_or_none(it)
        if not s:
            continue
        # model might send "unspecified"; DB uses "unsp"
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
        # accept YYYY-MM-DD or full ISO; take date portion
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
    # enforce same length; if mismatch, drop both to avoid insert errors
    if len(c) != len(s):
        return ([], [])
    return (c, s)


@dataclass
class PreparedRow:
    columns: List[str]
    placeholders: List[str]
    values: List[Any]


# Columns we accept from the model JSON (output schema) + a few meta fields
MODEL_COLUMNS = {
    "source_call_id",
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

    # Start with defaults for enum-ish fields and required defaults.
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

    # Work auth lists (enum arrays)
    allowed = _coerce_work_auth_list(data.get("allowed_work_auth"))
    not_allowed = _coerce_work_auth_list(data.get("not_allowed_work_auth"))

    # If vendor didn't specify anything, enforce both to ["Any"]
    if not allowed:
        allowed = ["Any"]
    if not not_allowed:
        not_allowed = ["Any"]

    data["allowed_work_auth"] = allowed
    data["not_allowed_work_auth"] = not_allowed

    # Booleans that are nullable in DB
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

    # submission_cutoff_date is timestamptz
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

    # Numeric-ish values: leave as-is (psycopg will coerce strings for numeric); optionally strip.

    # Merge metadata (DB columns) if present
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
        # Pattern A: omit nulls so DEFAULT applies.
        if v is None:
            continue

        cols.append(k)

        cast = CASTS.get(k, "")
        ph.append(f"%s{cast}")

        if k == "raw_json_input":
            vals.append(Jsonb(v))
        else:
            vals.append(v)

    return PreparedRow(columns=cols, placeholders=ph, values=vals)


async def insert_job_requirements(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    created_by: str = "ava_ai_recruiter",
    source_type: str = "phone_interview",
    dedupe_call_id: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Insert 1..N role payloads into job_requirements.

    Returns a list of (id, job_id) tuples for each processed role.
    """

    roles: List[Dict[str, Any]]
    if isinstance(payload, list):
        roles = payload
    else:
        roles = [payload]

    # best-effort idempotency: if dedupe_call_id exists, skip if already inserted
    dedupe_key = None
    if dedupe_call_id:
        dedupe_key = f"bey_call_id:{dedupe_call_id}"

    db_url = _db_url_from_env()
    db_info = _db_info_from_url(db_url)
    _log_event("info", "db_connecting", **db_info)
    results: List[Tuple[str, str]] = []

    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
            for idx, role in enumerate(roles):
                # store raw payload JSON in raw_json_input always
                meta = dict(metadata or {})
                meta.setdefault("created_by", created_by)
                meta.setdefault("source_type", source_type)
                if dedupe_call_id:
                    meta.setdefault("source_call_id", dedupe_call_id)
                if "raw_json_input" not in meta:
                    role_payload = role if isinstance(role, dict) else {}
                    meta["raw_json_input"] = role_payload.get("raw_json_input", role)

                # For multi-role calls with a single dedupe_call_id, keep unique-ish notes.
                # If you want true idempotency per-role, include stable role id in model output.
                if dedupe_key:
                    role_note = dedupe_key if idx == 0 else f"{dedupe_key}#role{idx}"
                    meta["notes"] = role_note
                    await cur.execute(
                        "SELECT id, job_id FROM job_requirements WHERE notes = %s ORDER BY created_at DESC LIMIT 1",
                        (role_note,),
                    )
                    existing = await cur.fetchone()
                    if existing:
                        _log_event(
                            "info",
                            "db_insert_job_requirements_deduped",
                            dedupe_key=role_note,
                            job_id=existing["job_id"],
                            role_index=idx,
                            **db_info,
                        )
                        continue

                prepared = _prepare_role(role, meta)

                    ins = _build_insert(prepared)

                    if not ins.columns:
                        # Should never happen; but avoid invalid SQL.
                        continue

                    query = SQL("INSERT INTO job_requirements ({cols}) VALUES ({vals}) RETURNING id, job_id").format(
                        cols=SQL(", ").join(Identifier(c) for c in ins.columns),
                        vals=SQL(", ").join(SQL(p) for p in ins.placeholders),
                    )

                await cur.execute(query, ins.values)
                row = await cur.fetchone()
                results.append((row["id"], row["job_id"]))
                _log_event(
                    "info",
                    "db_insert_job_requirements_row",
                    job_id=prepared.get("job_id"),
                    job_title=prepared.get("job_title"),
                    role_index=idx,
                    **db_info,
                )

                await conn.commit()

        _log_event("info", "db_insert_job_requirements_ok", rows=len(results), **db_info)
        return results
    except Exception as exc:
        _log_event("error", "db_insert_job_requirements_failed", error=str(exc), **db_info)
        raise


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
