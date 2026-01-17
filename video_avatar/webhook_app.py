import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from db import insert_call_transcript, insert_job_requirements, update_call_transcript


load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="Bey Webhook")

_client: Optional[AsyncOpenAI] = None


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
Output either:
- ONE JSON object (single role), OR
- a JSON array of objects (multiple roles).

If job_title is missing, infer it from the transcript. If still unknown, use "Unknown role".
job_title max length is 200 chars; if longer, return the first 200 chars.

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
""".strip()


def _clean_transcript(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    lines: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = msg.get("message")
        if not isinstance(text, str):
            continue
        cleaned = " ".join(text.split())
        if not cleaned:
            continue
        if (cleaned.startswith("{") and cleaned.endswith("}")) or (
            cleaned.startswith("[") and cleaned.endswith("]")
        ):
            continue
        sender = msg.get("sender")
        if isinstance(sender, str) and sender.strip():
            lines.append(f"{sender.strip()}: {cleaned}")
        else:
            lines.append(cleaned)
    return "\n".join(lines)


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

    try:
        llm_text = await _call_llm(cleaned)
    except Exception as exc:
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message=f"LLM call failed: {exc}",
        )
        return

    parsed = _parse_json(llm_text)
    if parsed is None:
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message="LLM returned invalid JSON",
        )
        return

    roles: List[Dict[str, Any]]
    if isinstance(parsed, list):
        roles = [r for r in parsed if isinstance(r, dict)]
    elif isinstance(parsed, dict):
        roles = [parsed]
    else:
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message="LLM output JSON must be object or array",
        )
        return

    warnings: List[str] = []
    for idx, role in enumerate(roles):
        warnings.extend(_apply_length_rules(role, role_index=idx))

    notes = _build_notes(call_id, warnings)
    metadata = {
        "raw_json_input": {"cleaned_transcript": cleaned},
        "notes": notes,
        "phone_call_duration": None,
    }

    evaluation = payload.get("evaluation") or {}
    if isinstance(evaluation, dict):
        duration_minutes = evaluation.get("duration_minutes")
        if isinstance(duration_minutes, (int, float)):
            metadata["phone_call_duration"] = int(duration_minutes * 60)

    try:
        await insert_job_requirements(
            roles,
            metadata=metadata,
            created_by="ava_ai_recruiter",
            source_type="phone_interview",
        )
    except Exception as exc:
        await update_call_transcript(
            call_id,
            status="failed_parse",
            error_message=f"DB insert failed: {exc}",
        )
        return

    await update_call_transcript(
        call_id,
        parsed_requirements=parsed,
        status="parsed",
        error_message=None,
    )


@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    event_type = payload.get("event_type")

    if event_type == "test":
        return JSONResponse({"status": "ok"})

    if event_type != "call_ended":
        return JSONResponse({"status": "ignored"})

    call_id = payload.get("call_id")
    if not call_id:
        return JSONResponse({"status": "missing_call_id"})

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
        return JSONResponse({"status": "db_error", "error": str(exc)})

    asyncio.create_task(_process_call(payload))
    return JSONResponse({"status": "accepted"})
