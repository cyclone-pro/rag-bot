import argparse
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

API_URL = "https://api.bey.dev/v1"

MAX_ROLES_PER_CALL = 5

REQUIRED_FIELDS = {"job_title", "job_type", "seniority_level", "work_model"}

ENUMS = {
    "seniority_level": {
        "Entry","Mid","Senior","Lead","Architect","Manager",
        "Director","VP","C-level","unspecified"
    },
    "job_type": {
        "Contract","Contract-to-hire","Full-time",
        "Part-time","Internship","Other","unspecified"
    },
    "employment_type": {"C2C","W2","1099","unspecified"},
    "pay_rate_unit": {"hour","day","week","month","year","unspecified"},
    "submission_urgency": {"normal","urgent","flexible"},
    "priority_level": {"low","medium","high"},
    "work_model": {"onsite","remote","hybrid","flexible","unspecified"},
}

DEFAULTS = {
    "seniority_level": "unspecified",
    "job_type": "unspecified",
    "work_model": "unspecified",
    "pay_rate_unit": "unspecified",
    "employment_type": "unspecified",
    "submission_urgency": "normal",
    "priority_level": "medium",
    "location_country": "US",
    "pay_rate_currency": "USD",
    "salary_currency": "USD",
    "has_equity": False,
    "relocation_assistance": False,
    "status": "active",
}

JOB_SCHEMA = {
    "type": "object",
    "required": [
        "job_title",
        "seniority_level",
        "job_type",
        "location_cities",
        "location_states",
        "work_model",
        "allowed_work_auth",
        "not_allowed_work_auth",
        "status",
    ],
    "additionalProperties": True,
}

def _parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def _get_message_text(m: Dict[str, Any]) -> Optional[str]:
    """
    Tries many possible shapes Bey might use.
    """
    # common flat keys
    for k in ("message", "text", "content", "body"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # content could be dict
    content = m.get("content")
    if isinstance(content, dict):
        for k in ("text", "message", "body", "value"):
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # parts could be [{"text": "..."}]
    parts = m.get("parts")
    if isinstance(parts, list):
        chunks = []
        for p in parts:
            if isinstance(p, dict):
                t = p.get("text") or p.get("content") or p.get("message")
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
        if chunks:
            return "\n".join(chunks).strip()

    # sometimes nested under "data" or "payload"
    for root in ("data", "payload", "response", "delta"):
        obj = m.get(root)
        if isinstance(obj, dict):
            for k in ("text", "message", "content", "body"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    return None

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if "```" in t:
            t = t.rsplit("```", 1)[0]
        t = t.strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t.strip()

def _iter_json_candidates(text: str) -> List[Any]:
    """
    Extract all JSON candidates from text by scanning with JSONDecoder.
    """
    s = _strip_code_fences(text)
    out: List[Any] = []
    decoder = json.JSONDecoder()

    # full-string attempt
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            out.append(json.loads(s))
            return out
        except Exception:
            pass

    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
            out.append(obj)
        except Exception:
            continue

    return out

def _looks_like_role(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("job_title"), str)

def _looks_like_roles_array(obj: Any) -> bool:
    return isinstance(obj, list) and len(obj) > 0 and all(isinstance(x, dict) and isinstance(x.get("job_title"), str) for x in obj)

def _normalize_payload(obj: Any) -> Optional[Any]:
    if _looks_like_role(obj):
        return obj
    if _looks_like_roles_array(obj):
        return obj
    if isinstance(obj, dict) and "roles" in obj and _looks_like_roles_array(obj.get("roles")):
        print("[INFO] Detected wrapped roles payload; auto-unwrapping.")
        return obj["roles"]
    return None

def _extract_final_payload(messages: List[Dict[str, Any]]) -> Optional[Any]:
    def score(p: Any) -> int:
        # prefer full arrays over single objects
        if isinstance(p, list):
            return 3
        if isinstance(p, dict):
            return 2
        return 0

    best = None
    best_score = 0

    for m in messages:
        text = _get_message_text(m)
        if not text:
            continue
        for cand in _iter_json_candidates(text):
            payload = _normalize_payload(cand)
            if payload is None:
                continue
            s = score(payload)
            if s > best_score:
                best = payload
                best_score = s

    return best

def enforce_role_count(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Invalid payload type: {type(payload)}")
    if len(payload) == 0:
        raise ValueError("Zero roles detected")
    if len(payload) > MAX_ROLES_PER_CALL:
        raise ValueError(f"Too many roles ({len(payload)}) > {MAX_ROLES_PER_CALL}")
    for i, r in enumerate(payload):
        if not isinstance(r, dict):
            raise ValueError(f"role[{i}] is not an object: {type(r)}")
    return payload

def validate_enums(role: Dict[str, Any], idx: int) -> Dict[str, Any]:
    cleaned = dict(role)
    for field, allowed in ENUMS.items():
        value = cleaned.get(field)
        if value is None:
            continue
        if value not in allowed:
            print(f"[ENUM WARNING] role[{idx}] {field}='{value}' invalid -> defaulting")
            cleaned[field] = DEFAULTS.get(field, "unspecified")
    return cleaned

def apply_db_safe_defaults(role: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(role)
    for k, v in DEFAULTS.items():
        if k not in r or r[k] is None:
            r[k] = v

    # arrays must exist
    for k in ("location_cities", "location_states"):
        if k not in r or r[k] is None:
            r[k] = []
        if not isinstance(r[k], list):
            r[k] = [str(r[k])]

    # work auth required for DB
    if not isinstance(r.get("allowed_work_auth"), list) or not r.get("allowed_work_auth"):
        r["allowed_work_auth"] = ["Any"]
    if not isinstance(r.get("not_allowed_work_auth"), list) or not r.get("not_allowed_work_auth"):
        r["not_allowed_work_auth"] = ["Any"]

    # sanity: city/state mismatch
    if len(r["location_cities"]) != len(r["location_states"]):
        print("[WARN] location_cities/state mismatch -> clearing both")
        r["location_cities"] = []
        r["location_states"] = []

    return r

def is_role_complete(role: Dict[str, Any], idx: int) -> bool:
    missing = [f for f in REQUIRED_FIELDS if role.get(f) in (None, "", [])]
    if missing:
        print(f"[DISCARD] role[{idx}] missing critical fields: {missing}")
        return False
    return True

def schema_validate(role: Dict[str, Any], idx: int) -> bool:
    try:
        validate(instance=role, schema=JOB_SCHEMA)
        return True
    except ValidationError as e:
        print(f"[SCHEMA FAIL] role[{idx}] {e.message}")
        return False

async def _insert_roles(roles: List[Dict[str, Any]], *, call_id: str, call_started_at: Optional[str], duration_sec: Optional[int]):
    from db import insert_job_requirements
    meta = {
        "phone_call_duration": duration_sec,
        "email_received_at": call_started_at,
        "source_type": "phone_interview",
        "created_by": "ava_ai_recruiter",
        "source_call_id": call_id,
    }
    return await insert_job_requirements(
        roles,
        metadata=meta,
        created_by="ava_ai_recruiter",
        source_type="phone_interview",
        dedupe_call_id=call_id,
    )

def _dump_messages(call_id: str, messages: Any) -> str:
    fname = f"debug_messages_{call_id}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    return fname

def main(api_key: str, agent_id: str, dump_on_fail: bool = True) -> None:
    resp = requests.get(
        f"{API_URL}/calls",
        headers={"x-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()

    calls = resp.json().get("data", [])

    for call in calls:
        if call.get("agent_id") != agent_id:
            continue

        call_id = call.get("id")
        call_started_at = call.get("started_at")
        call_ended_at = call.get("ended_at")

        print(f"\n=== Call {call_id} ===")

        msg_resp = requests.get(
            f"{API_URL}/calls/{call_id}/messages",
            headers={"x-api-key": api_key},
            timeout=30,
        )
        msg_resp.raise_for_status()

        raw = msg_resp.json()
        messages = raw.get("data", raw) if isinstance(raw, dict) else raw

        # DEBUG: print shape once
        if isinstance(messages, list):
            print(f"[INFO] messages_count={len(messages)} first_keys={list(messages[0].keys()) if messages else []}")
        else:
            print(f"[INFO] messages_type={type(messages)} keys={list(messages.keys()) if isinstance(messages, dict) else None}")

        payload = _extract_final_payload(messages if isinstance(messages, list) else [])
        if payload is None:
            print("No final JSON payload found; skipping.")
            if dump_on_fail:
                path = _dump_messages(str(call_id), messages)
                print(f"[DEBUG] Dumped raw messages -> {path}")

            # Also show last few message key summaries
            if isinstance(messages, list) and messages:
                tail = messages[-5:]
                for i, m in enumerate(tail):
                    sender = m.get("sender", m.get("role", "unknown"))
                    text = _get_message_text(m)
                    snip = (text[:140] + "...") if isinstance(text, str) and len(text) > 140 else text
                    print(f"  tail[{i}] sender={sender} keys={list(m.keys())} text_snip={snip!r}")
            continue

        try:
            roles = enforce_role_count(payload)
        except ValueError as e:
            print(f"[ROLE COUNT FAIL] {e}")
            continue

        start = _parse_iso_dt(call_started_at)
        end = _parse_iso_dt(call_ended_at)
        duration = int((end - start).total_seconds()) if start and end else None

        final_roles: List[Dict[str, Any]] = []
        for i, role in enumerate(roles):
            role = validate_enums(role, i)
            role = apply_db_safe_defaults(role)

            if not is_role_complete(role, i):
                continue
            if not schema_validate(role, i):
                continue

            final_roles.append(role)

        if not final_roles:
            print("All roles discarded after validation.")
            continue

        inserted = asyncio.run(
            _insert_roles(
                final_roles,
                call_id=str(call_id),
                call_started_at=call_started_at,
                duration_sec=duration,
            )
        )

        for i, (row_id, job_id) in enumerate(inserted):
            print(f"Inserted role[{i}] -> id={row_id} job_id={job_id}")

if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("BEY_API_KEY")
    if not api_key:
        raise ValueError("Missing BEY_API_KEY")

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--no-dump", action="store_true", help="disable dumping raw messages on failure")
    args = parser.parse_args()

    main(api_key, args.agent_id, dump_on_fail=not args.no_dump)
