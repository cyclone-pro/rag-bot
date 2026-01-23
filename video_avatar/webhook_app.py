import asyncio
import json
import os
import re
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

load_dotenv(Path(__file__).with_name(".env"))

from db import check_db_connection, insert_call_transcript, insert_job_requirements, update_call_transcript
from milvus_job_postings import check_milvus_connection, insert_job_postings

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="Bey Webhook")

_client: Optional[AsyncOpenAI] = None

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(message)s",
)
logger = logging.getLogger("bey_webhook")

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST",
    "Access-Control-Allow-Headers": "Content-Type",
}


def _log_event(level: str, message: str, **fields: Any) -> None:
    payload = {"message": message, **fields}
    record = json.dumps(payload, ensure_ascii=True)
    if level == "warning":
        logger.warning(record)
    elif level == "error":
        logger.error(record)
    else:
        logger.info(record)


@app.on_event("startup")
async def startup_checks() -> None:
    async def _run() -> None:
        ok, detail = await asyncio.to_thread(check_milvus_connection)
        if ok:
            _log_event("info", "milvus_health_check_ok")
        else:
            _log_event("error", "milvus_health_check_failed", error=detail)

    asyncio.create_task(_run())


def _openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM parsing")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


STRING_LIMITS = {
    "job_id": 100,
    "external_requisition_id": 100,
    "job_title": 200,
    "end_client_name": 200,
    "client_name": 200,
    "industry": 100,
    "location_country": 3,
    "pay_rate_currency": 3,
    "salary_currency": 3,
    "bonus_type": 50,
    "equity_type": 50,
    "contract_duration_text": 100,
    "citizenship_required": 50,
    "security_clearance_level": 50,
    "work_hours": 100,
    "time_zone": 50,
    "vendor_name": 200,
    "vendor_contact_name": 200,
    "vendor_contact_email": 200,
    "vendor_contact_phone": 20,
    "email_subject": 500,
    "email_sender": 200,
    "created_by": 100,
    "source_type": 50,
    "status": 50,
}

RATE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


ENUMS = {
    "seniority_level": [
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
    ],
    "job_type": [
        "Contract",
        "Contract-to-hire",
        "Full-time",
        "Part-time",
        "Internship",
        "Other",
        "unspecified",
    ],
    "submission_urgency": ["normal", "urgent", "flexible"],
    "work_model": ["onsite", "remote", "hybrid", "flexible", "unspecified"],
    "pay_rate_unit": ["hour", "day", "week", "month", "year", "unspecified"],
    "employment_type": ["C2C", "W2", "1099", "unspecified"],
    "work_authorization": [
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
    ],
}


SYSTEM_PROMPT = f"""
You extract structured job requirements from a transcript.

Return ONLY valid JSON. No markdown. No commentary.
Always return a JSON object with:
- multi_role: boolean
- roles: array of role objects (at least one)
- parse_warnings: array (can be empty)

If job_title is missing, infer it from the transcript. If still unknown, use "Unknown role".
job_title max length is 200 chars; if longer, return the first 200 chars.

If pay rate is stated as a single value (e.g., "$60/hr"), set BOTH pay_rate_min and
pay_rate_max to that value. If a range is stated (e.g., "$50-$60/hr"), set pay_rate_min
to the lower value and pay_rate_max to the higher value. Always set pay_rate_currency
and pay_rate_unit when pay rate is mentioned.

Always populate these arrays (use [] if none):
- must_have_skills
- nice_to_have_skills
- primary_technologies
- responsibilities
- day_to_day
- other_constraints
- certifications_required
- certifications_preferred
- domains

If the transcript mentions a minimum years-of-experience requirement, populate
overall_min_years (and primary_role_min_years if it is role-specific).

For any other string field that would exceed its column limit, return null.
Do not truncate other fields.

Enum values MUST match exactly:
- seniority_level: {", ".join(ENUMS["seniority_level"])}
- job_type: {", ".join(ENUMS["job_type"])}
- submission_urgency: {", ".join(ENUMS["submission_urgency"])}
- work_model: {", ".join(ENUMS["work_model"])}
- pay_rate_unit: {", ".join(ENUMS["pay_rate_unit"])}
- employment_type: {", ".join(ENUMS["employment_type"])}
- work_authorization enum values: {", ".join(ENUMS["work_authorization"])}

Defaults when unknown:
- seniority_level: "unspecified"
- job_type: "unspecified"
- submission_urgency: "normal"
- work_model: "unspecified"
- pay_rate_unit: "unspecified"
- employment_type: "unspecified"
- status: "active"
- allowed_work_auth: ["Any"]
- not_allowed_work_auth: ["Any"]

Column length limits (do not exceed; use null instead):
{json.dumps(STRING_LIMITS, indent=2)}

Important:
- Do NOT invent data.
- Arrays must be JSON arrays (even if empty).
- Use null for missing optional fields.
- parse_warnings must be present (use [] when none).
""".strip()

FILLER_RE = re.compile(
    r"(?i)\b(?:um+|uh+|ah+|er+|hmm+|okay|alright|yeah|yep|so|like)\b(?=\s*(?:[.,!?]|$))"
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
)


def _remove_fillers(text: str) -> str:
    cleaned = FILLER_RE.sub("", text)
    return " ".join(cleaned.split())


def _redact_pii(text: str) -> str:
    redacted = EMAIL_RE.sub("[redacted]", text)
    redacted = PHONE_RE.sub("[redacted]", redacted)
    return redacted


def _clean_transcript(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    lines: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = msg.get("message")
        if text is None:
            text = msg.get("content")
        if text is None:
            text = msg.get("text")
        if not isinstance(text, str):
            continue
        cleaned = " ".join(text.split())
        cleaned = _remove_fillers(cleaned)
        if not cleaned:
            continue
        if (cleaned.startswith("{") and cleaned.endswith("}")) or (
            cleaned.startswith("[") and cleaned.endswith("]")
        ):
            continue
        sender = msg.get("sender")
        if sender is None:
            sender = msg.get("role")
        if isinstance(sender, str) and sender.lower() == "ai":
            sender = "agent"
        if isinstance(sender, str) and sender.strip():
            lines.append(f"{sender.strip()}: {cleaned}")
        else:
            lines.append(cleaned)
    return "\n".join(lines)


def _coerce_enum_value(value: Any, allowed: List[str], default: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return default
    if value not in allowed:
        return default
    return value


def _normalize_role_enums(role: Dict[str, Any], *, role_index: int) -> List[str]:
    warnings: List[str] = []

    for field, allowed, default in (
        ("seniority_level", ENUMS["seniority_level"], "unspecified"),
        ("job_type", ENUMS["job_type"], "unspecified"),
        ("submission_urgency", ENUMS["submission_urgency"], "normal"),
        ("work_model", ENUMS["work_model"], "unspecified"),
        ("pay_rate_unit", ENUMS["pay_rate_unit"], "unspecified"),
        ("employment_type", ENUMS["employment_type"], "unspecified"),
    ):
        raw_value = role.get(field)
        coerced = _coerce_enum_value(raw_value, allowed, default)
        if raw_value is not None and coerced != raw_value:
            warnings.append(f"role[{role_index}] {field} invalid; set to {coerced}")
            role[field] = coerced

    for field in ("allowed_work_auth", "not_allowed_work_auth"):
        raw_values = role.get(field)
        if raw_values is None:
            continue
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        if not isinstance(raw_values, list):
            warnings.append(f"role[{role_index}] {field} invalid; set to Any")
            role[field] = ["Any"]
            continue
        cleaned = [val for val in raw_values if isinstance(val, str) and val in ENUMS["work_authorization"]]
        if not cleaned:
            warnings.append(f"role[{role_index}] {field} invalid; set to Any")
            role[field] = ["Any"]
        else:
            if len(cleaned) != len(raw_values):
                warnings.append(f"role[{role_index}] {field} invalid entries removed")
            role[field] = cleaned

    return warnings


def _normalize_pay_rates(role: Dict[str, Any], *, role_index: int) -> List[str]:
    warnings: List[str] = []
    pay_min = role.get("pay_rate_min")
    pay_max = role.get("pay_rate_max")
    pay_currency = role.get("pay_rate_currency")
    pay_unit = role.get("pay_rate_unit")

    rate_hint = role.get("pay_rate")
    if rate_hint is None:
        rate_hint = role.get("pay_rate_range")
    if rate_hint is None:
        rate_hint = role.get("pay_rate_text")

    parsed = _parse_pay_rate_hint(rate_hint)
    if parsed:
        parsed_min, parsed_max, parsed_currency, parsed_unit = parsed
        if pay_min is None and parsed_min is not None:
            role["pay_rate_min"] = parsed_min
            pay_min = parsed_min
        if pay_max is None and parsed_max is not None:
            role["pay_rate_max"] = parsed_max
            pay_max = parsed_max
        if parsed_currency and not pay_currency:
            role["pay_rate_currency"] = parsed_currency
            pay_currency = parsed_currency
        if parsed_unit and (pay_unit is None or pay_unit == "unspecified"):
            role["pay_rate_unit"] = parsed_unit
            pay_unit = parsed_unit

    if pay_min is None and pay_max is not None:
        role["pay_rate_min"] = pay_max
        warnings.append(f"role[{role_index}] pay_rate_min missing; set to pay_rate_max")
    elif pay_max is None and pay_min is not None:
        role["pay_rate_max"] = pay_min
        warnings.append(f"role[{role_index}] pay_rate_max missing; set to pay_rate_min")
    return warnings


def _parse_pay_rate_hint(value: Any) -> Optional[Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value), float(value), None, None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    numbers = [float(x) for x in RATE_NUMBER_RE.findall(text)]
    if not numbers:
        return None
    if len(numbers) >= 2:
        low = min(numbers[0], numbers[1])
        high = max(numbers[0], numbers[1])
    else:
        low = high = numbers[0]

    lowered = text.lower()
    currency = None
    if "$" in text or "usd" in lowered:
        currency = "USD"
    elif "eur" in lowered:
        currency = "EUR"
    elif "gbp" in lowered or "pound" in lowered:
        currency = "GBP"

    unit = None
    if re.search(r"/\s*hr|\bhr\b|hourly|per\s*hour", lowered):
        unit = "hour"
    elif "day" in lowered:
        unit = "day"
    elif "week" in lowered:
        unit = "week"
    elif "month" in lowered:
        unit = "month"
    elif "year" in lowered or "annual" in lowered:
        unit = "year"

    return low, high, currency, unit


def _parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    # Best-effort fallback: extract outermost JSON block.
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        return None
    end = max(text.rfind("}"), text.rfind("]"))
    if end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _apply_length_rules(role: Dict[str, Any], *, role_index: int) -> List[str]:
    warnings: List[str] = []

    raw_title = role.get("job_title")
    title = str(raw_title).strip() if raw_title is not None else ""
    if not title:
        role["job_title"] = "Unknown role"
        warnings.append(f"role[{role_index}] job_title missing; set to Unknown role")
    elif len(title) > STRING_LIMITS["job_title"]:
        role["job_title"] = title[: STRING_LIMITS["job_title"]]
        warnings.append(f"role[{role_index}] job_title truncated to 200 chars")
    else:
        role["job_title"] = title

    for field, limit in STRING_LIMITS.items():
        if field == "job_title":
            continue
        value = role.get(field)
        if not isinstance(value, str):
            continue
        if len(value) > limit:
            role[field] = None
            warnings.append(f"role[{role_index}] {field} exceeded {limit}; set to null")
        else:
            role[field] = value.strip()

    return warnings


def _build_notes(call_id: str, warnings: List[str]) -> str:
    notes = [f"bey_call_id:{call_id}"]
    if warnings:
        notes.append("WARNINGS:")
        notes.extend(warnings)
    return "\n".join(notes)


async def _call_llm(cleaned_transcript: str) -> str:
    client = _openai_client()
    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cleaned_transcript},
        ],
    )
    return resp.choices[0].message.content or ""


async def _process_call(payload: Dict[str, Any]) -> None:
    call_id = payload.get("call_id")
    if not call_id:
        return

    messages = payload.get("messages") or []
    cleaned = _clean_transcript(messages)
    redacted = _redact_pii(cleaned)
    _log_event("info", "call_processing_started", call_id=call_id, message_count=len(messages))

    try:
        llm_text = await _call_llm(cleaned)
    except Exception as exc:
        _log_event("error", "llm_call_failed", call_id=call_id, error=str(exc))
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message=f"LLM call failed: {exc}",
        )
        return

    parsed = _parse_json(llm_text)
    if parsed is None:
        _log_event("error", "llm_invalid_json", call_id=call_id)
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message="LLM returned invalid JSON",
        )
        return

    if not isinstance(parsed, dict):
        _log_event("error", "llm_invalid_wrapper", call_id=call_id)
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message="LLM output JSON must be a wrapper object",
        )
        return

    roles_raw = parsed.get("roles")
    if not isinstance(roles_raw, list) or not roles_raw:
        _log_event("error", "llm_missing_roles", call_id=call_id)
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message="LLM output missing roles list",
        )
        return

    parse_warnings = parsed.get("parse_warnings")
    if not isinstance(parse_warnings, list):
        parse_warnings = ["parse_warnings missing or invalid; initialized empty"]

    roles: List[Dict[str, Any]] = []
    for idx, role in enumerate(roles_raw):
        if not isinstance(role, dict):
            parse_warnings.append(f"role[{idx}] is not an object; skipped")
            continue
        role_copy = dict(role)
        role_copy["job_id"] = f"{call_id}_{idx}"
        parse_warnings.extend(_apply_length_rules(role_copy, role_index=idx))
        parse_warnings.extend(_normalize_role_enums(role_copy, role_index=idx))
        parse_warnings.extend(_normalize_pay_rates(role_copy, role_index=idx))
        roles.append(role_copy)

    if not roles:
        _log_event("error", "llm_no_valid_roles", call_id=call_id)
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message="LLM output contained no valid role objects",
        )
        return

    multi_role = parsed.get("multi_role")
    if not isinstance(multi_role, bool):
        multi_role = len(roles) > 1

    wrapper_roles = [dict(r) for r in roles]
    wrapper = {
        "multi_role": multi_role,
        "roles": wrapper_roles,
        "parse_warnings": parse_warnings,
        "cleaned_transcript": redacted,
        "source_call_id": call_id,
        "extraction_version": "v1.5",
    }

    role_payloads: List[Dict[str, Any]] = []
    for idx, role in enumerate(roles):
        role_payload = dict(role)
        role_payload["raw_json_input"] = {
            **wrapper,
            "role_index": idx,
            "role_count": len(wrapper_roles),
        }
        role_payloads.append(role_payload)

    notes = _build_notes(call_id, parse_warnings)
    metadata = {
        "notes": notes,
        "phone_call_duration": None,
        "source_call_id": call_id,
    }

    evaluation = payload.get("evaluation") or {}
    if isinstance(evaluation, dict):
        duration_minutes = evaluation.get("duration_minutes")
        if isinstance(duration_minutes, (int, float)):
            metadata["phone_call_duration"] = int(duration_minutes * 60)

    try:
        await insert_job_requirements(
            role_payloads,
            metadata=metadata,
            created_by="ava_ai_recruiter",
            source_type="beyond_presence",
            dedupe_call_id=call_id,
        )
    except Exception as exc:
        _log_event("error", "job_insert_failed", call_id=call_id, error=str(exc))
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message=f"DB insert failed: {exc}",
        )
        return

    milvus_error = None
    try:
        inserted = insert_job_postings(role_payloads)
        _log_event("info", "milvus_job_postings_inserted", call_id=call_id, count=inserted)
    except Exception as exc:
        milvus_error = f"Milvus insert failed: {exc}"
        _log_event("error", "milvus_insert_failed", call_id=call_id, error=str(exc))

    await update_call_transcript(
        call_id,
        parsed_requirements=wrapper,
        status="parsed",
        error_message=milvus_error,
    )
    _log_event(
        "info",
        "call_processing_complete",
        call_id=call_id,
        role_count=len(role_payloads),
        warning_count=len(parse_warnings),
    )


@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    event_type = payload.get("event_type")

    if event_type == "test":
        _log_event("info", "webhook_test", event_type=event_type)
        return JSONResponse({"status": "ok"}, headers=CORS_HEADERS)

    if event_type != "call_ended":
        _log_event("info", "webhook_ignored", event_type=event_type)
        return JSONResponse({"status": "ignored"}, headers=CORS_HEADERS)

    call_id = payload.get("call_id")
    if not call_id:
        _log_event("warning", "webhook_missing_call_id")
        return JSONResponse({"status": "missing_call_id"}, headers=CORS_HEADERS)

    agent_id = payload.get("agent_id")
    tags = payload.get("tags") if isinstance(payload.get("tags"), dict) else None
    if not agent_id and isinstance(tags, dict):
        agent_id = tags.get("agent_id")
    if not agent_id:
        agent_id = "unknown"

    record = {
        "call_id": call_id,
        "agent_id": agent_id,
        "session_id": payload.get("session_id") or (tags.get("session_id") if isinstance(tags, dict) else None),
        "user_name": payload.get("user_name"),
        "event_type": event_type,
        "call_started_at": payload.get("call_started_at") or payload.get("started_at"),
        "call_ended_at": payload.get("call_ended_at") or payload.get("ended_at"),
        "evaluation": payload.get("evaluation"),
        "messages": payload.get("messages"),
        "raw_payload": payload,
        "tags": tags,
        "status": "received",
        "received_at": datetime.now(tz=timezone.utc),
    }

    try:
        await insert_call_transcript(record)
    except Exception as exc:
        _log_event("error", "call_transcript_insert_failed", call_id=call_id, error=str(exc))
        return JSONResponse({"status": "db_error", "error": str(exc)}, headers=CORS_HEADERS)

    _log_event("info", "webhook_received", call_id=call_id, event_type=event_type)
    asyncio.create_task(_process_call(payload))
    return JSONResponse({"status": "accepted"}, headers=CORS_HEADERS)


@app.get("/webhook")
async def webhook_get() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "message": "Webhook endpoint is ready. Send POST events to /webhook.",
        },
        headers=CORS_HEADERS,
    )


@app.options("/webhook")
async def webhook_options() -> Response:
    return Response(status_code=204, headers=CORS_HEADERS)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/health/db")
async def health_db() -> JSONResponse:
    ok, detail = await check_db_connection()
    status = "ok" if ok else "error"
    status_code = 200 if ok else 503
    return JSONResponse({"status": status, "detail": detail}, status_code=status_code)


@app.get("/health/milvus")
async def health_milvus() -> JSONResponse:
    ok, detail = await asyncio.to_thread(check_milvus_connection)
    status = "ok" if ok else "error"
    status_code = 200 if ok else 503
    return JSONResponse({"status": status, "detail": detail}, status_code=status_code)
