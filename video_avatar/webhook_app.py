"""Beyond Presence Webhook Handler for RecruiterBrain."""

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from db import (
    check_db_connection, insert_call_transcript, fetch_call_transcript, update_call_transcript,
    insert_job_requirement, update_job_milvus_status, fetch_unsynced_jobs,
    insert_processing_log, fetch_processing_logs, fetch_stats, fetch_failed_calls,
)
from milvus_job_postings import (
    check_milvus_connection, generate_embedding, search_similar_jobs, 
    insert_job_posting, _build_jd_text, sync_jobs_to_milvus,
)
from bey_client import build_webhook_payload

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))
MAX_SIMILAR_JOBS = int(os.getenv("MAX_SIMILAR_JOBS", "3"))
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

app = FastAPI(title="Beyond Presence Webhook", version="2.0.0")
_openai: Optional[AsyncOpenAI] = None

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
logger = logging.getLogger("bey_webhook")

CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Admin-Key"}

def _log(level: str, msg: str, **kw: Any) -> None:
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps({"message": msg, **kw}))

def _openai_client() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai

SYSTEM_PROMPT = '''You extract structured job requirements from recruitment call transcripts.
CRITICAL: Return ONLY valid JSON. No markdown, no commentary.

OUTPUT SCHEMA:
{
  "multi_role": boolean,
  "roles": [
    {
      "job_title": "string REQUIRED max 200 chars",
      "external_requisition_id": "string or null",
      "positions_available": integer default 1,
      "max_candidates_allowed": integer or null,
      
      "seniority_level": "Entry|Mid|Senior|Lead|Architect|Manager|Director|VP|C-level|unspecified",
      "job_type": "Contract|Contract-to-hire|Full-time|Part-time|Internship|Other|unspecified",
      "submission_urgency": "normal|urgent|flexible",
      "work_model": "onsite|remote|hybrid|flexible|unspecified",
      "pay_rate_unit": "hour|day|week|month|year|unspecified",
      "employment_type": "C2C|W2|1099|unspecified",
      
      "location_cities": ["array of city names"],
      "location_states": ["array of state codes - MUST match cities length"],
      "location_country": "string default US",
      "work_model_details": "string or null",
      
      "pay_rate_min": number or null,
      "pay_rate_max": number or null,
      "pay_rate_currency": "USD",
      "is_rate_strict": boolean or null,
      "pay_rate_notes": "string or null",
      
      "salary_min": number or null,
      "salary_max": number or null,
      "salary_currency": "USD",
      
      "bonus_percentage_min": number or null,
      "bonus_percentage_max": number or null,
      "bonus_type": "string or null",
      "bonus_notes": "string or null",
      "has_equity": boolean default false,
      "equity_type": "string or null",
      "equity_details": "string or null",
      
      "pto_days": integer or null,
      "health_insurance_provided": boolean or null,
      "retirement_matching": boolean or null,
      "retirement_matching_details": "string or null",
      "benefits_summary": "string or null",
      "sign_on_bonus": number or null,
      "relocation_assistance": boolean default false,
      "relocation_amount": number or null,
      
      "contract_duration_text": "string e.g. 6 months",
      "contract_start_date": "YYYY-MM-DD or null",
      "contract_end_date": "YYYY-MM-DD or null",
      "contract_can_extend": boolean or null,
      
      "allowed_work_auth": ["USC|GC|H1B|H4-EAD|L1|L2-EAD|TN|E3|F1-OPT|F1-CPT|STEM-OPT|J1|O1|EAD|Asylum-EAD|GC-EAD|Any|unsp"],
      "not_allowed_work_auth": ["same values as allowed"],
      "citizenship_required": "string or null",
      "work_auth_notes": "string or null",
      
      "background_check_required": boolean default false,
      "background_check_details": "string or null",
      "security_clearance_required": boolean default false,
      "security_clearance_level": "string or null",
      
      "overall_min_years": number or null,
      "primary_role_min_years": number or null,
      "management_experience_required": boolean default false,
      
      "must_have_skills": ["required skills array"],
      "nice_to_have_skills": ["preferred skills array"],
      "primary_technologies": ["main tech stack"],
      "certifications_required": ["required certs"],
      "certifications_preferred": ["preferred certs"],
      "domains": ["industry domains"],
      
      "responsibilities": ["job responsibilities"],
      "day_to_day": ["daily activities"],
      "other_constraints": ["other requirements"],
      
      "work_hours": "string e.g. 9am-5pm EST",
      "time_zone": "string e.g. EST",
      "travel_required": boolean default false,
      "travel_details": "string or null",
      
      "interview_process": "string describing interview rounds",
      "submission_cutoff_date": "ISO datetime or null",
      
      "end_client_name": "string or null",
      "client_name": "string or null",
      "industry": "string or null",
      "vendor_name": "string or null",
      "vendor_contact_name": "string or null",
      "vendor_contact_email": "string or null",
      "vendor_contact_phone": "string or null"
    }
  ],
  "parse_warnings": ["array of any parsing issues"]
}

RULES:
1. job_title is REQUIRED - infer from context if not stated, use "Unknown role" as last resort
2. For pay rates: if single value like "$60/hr", set BOTH pay_rate_min and pay_rate_max to 60
3. For salary: if "100k to 130k", set salary_min=100000, salary_max=130000
4. Arrays must be arrays even if empty: []
5. Enums must match EXACTLY (case-sensitive)
6. Do NOT invent data - use null for unknown fields
7. If multiple roles discussed, create separate role objects
8. location_cities and location_states arrays MUST have same length
'''

FILLER_RE = re.compile(r"(?i)\b(?:um+|uh+|ah+|er+|hmm+|okay|alright|yeah|yep|so|like)\b(?=\s*[.,!?]|\s*$)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")

def _clean_transcript(messages: List[Dict]) -> str:
    lines = []
    for msg in messages:
        text = msg.get("message") or msg.get("content") or msg.get("text", "")
        if not text or not isinstance(text, str):
            continue
        text = " ".join(text.split())
        text = FILLER_RE.sub("", text).strip()
        if not text:
            continue
        sender = msg.get("sender") or msg.get("role", "")
        if sender.lower() == "ai":
            sender = "agent"
        lines.append(f"{sender}: {text}" if sender else text)
    return "\n".join(lines)

def _redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)

def _parse_json(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except:
        pass
    # Try to extract JSON from text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                continue
    return None

async def _log_proc(call_id: str, stage: str, level: str = "info", **kw) -> None:
    _log(level, stage, call_id=call_id, **kw)
    await insert_processing_log(call_id, stage, level=level, **kw)

def _validate_webhook(payload: Dict) -> tuple:
    if not payload.get("call_id"):
        return False, "missing_call_id"
    if not payload.get("event_type"):
        return False, "missing_event_type"
    if payload.get("event_type") == "call_ended":
        if not payload.get("messages"):
            return False, "missing_messages"
        if not isinstance(payload.get("messages"), list):
            return False, "invalid_messages"
        if not payload.get("call_data", {}).get("agentId"):
            return False, "missing_agent_id"
    return True, "ok"

def _verify_admin(key: Optional[str]) -> bool:
    if not ADMIN_API_KEY:
        return True
    return key == ADMIN_API_KEY

async def _process_call(call_id: str, payload: Dict) -> None:
    """Background task to process a call."""
    start = time.time()
    await _log_proc(call_id, "webhook_received", message="Processing started")
    await update_call_transcript(call_id, status="processing")
    
    messages = payload.get("messages", [])
    if not messages:
        transcript = await fetch_call_transcript(call_id)
        messages = transcript.get("messages", []) if transcript else []
    
    # Clean transcript
    t0 = time.time()
    cleaned = _clean_transcript(messages)
    redacted = _redact_pii(cleaned)
    await _log_proc(call_id, "transcript_cleaned", message="Cleaned", 
                    metadata={"chars": len(cleaned), "messages": len(messages)},
                    duration_ms=int((time.time()-t0)*1000))
    
    if not cleaned.strip():
        await _log_proc(call_id, "llm_failed", level="error", message="Empty transcript")
        await update_call_transcript(call_id, status="failed", error_message="Empty transcript")
        return
    
    # Call LLM
    await _log_proc(call_id, "llm_started", message="Calling LLM", metadata={"model": OPENAI_MODEL})
    t0 = time.time()
    try:
        client = _openai_client()
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.1,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": cleaned}]
        )
        llm_text = resp.choices[0].message.content or ""
        llm_ms = int((time.time()-t0)*1000)
    except Exception as e:
        await _log_proc(call_id, "llm_failed", level="error", message=str(e), duration_ms=int((time.time()-t0)*1000))
        await update_call_transcript(call_id, status="failed", error_message=f"LLM error: {e}")
        await _log_proc(call_id, "processing_failed", level="error", duration_ms=int((time.time()-start)*1000))
        return
    
    # Parse LLM response
    parsed = _parse_json(llm_text)
    if not parsed or not isinstance(parsed.get("roles"), list) or not parsed["roles"]:
        await _log_proc(call_id, "llm_failed", level="error", message="Invalid JSON", duration_ms=llm_ms)
        await update_call_transcript(call_id, status="failed", error_message="LLM returned invalid JSON")
        await _log_proc(call_id, "processing_failed", level="error", duration_ms=int((time.time()-start)*1000))
        return
    
    await _log_proc(call_id, "llm_complete", message="LLM done", 
                    metadata={"roles": len(parsed["roles"])}, duration_ms=llm_ms)
    
    # Process roles
    roles = parsed["roles"]
    warnings = parsed.get("parse_warnings", [])
    inserted = []
    
    for idx, role in enumerate(roles):
        if not isinstance(role, dict):
            warnings.append(f"role[{idx}] not a dict, skipped")
            continue
        
        # Generate embedding
        await _log_proc(call_id, "embedding_started", role_index=idx, message="Generating embedding")
        t0 = time.time()
        try:
            jd_text = _build_jd_text(role)
            embedding = await asyncio.to_thread(generate_embedding, jd_text)
            await _log_proc(call_id, "embedding_complete", role_index=idx, duration_ms=int((time.time()-t0)*1000))
        except Exception as e:
            await _log_proc(call_id, "embedding_failed", level="error", role_index=idx, message=str(e))
            embedding = None
        
        # Similarity search
        similar_jobs, similarity_score = [], None
        if embedding:
            await _log_proc(call_id, "similarity_check_started", role_index=idx)
            t0 = time.time()
            try:
                similar_jobs = await asyncio.to_thread(search_similar_jobs, embedding, SIMILARITY_THRESHOLD, MAX_SIMILAR_JOBS)
                if similar_jobs:
                    similarity_score = similar_jobs[0]["score"]
                await _log_proc(call_id, "similarity_check_complete", role_index=idx,
                               metadata={"matches": len(similar_jobs), "top_score": similarity_score},
                               duration_ms=int((time.time()-t0)*1000))
            except Exception as e:
                await _log_proc(call_id, "similarity_check_failed", level="warning", role_index=idx, message=str(e))
        
        # Prepare role for insert
        role["similar_jobs"] = similar_jobs
        role["similarity_score"] = similarity_score
        role["milvus_synced"] = False
        
        # Insert to Postgres
        await _log_proc(call_id, "postgres_insert_started", role_index=idx)
        t0 = time.time()
        try:
            result = await insert_job_requirement(role, call_id, idx)
            if result:
                db_id, job_id = result
                await _log_proc(call_id, "postgres_insert_complete", role_index=idx, job_id=job_id,
                               metadata={"title": role.get("job_title")}, duration_ms=int((time.time()-t0)*1000))
                inserted.append({"idx": idx, "job_id": job_id, "embedding": embedding, "role": role})
            else:
                await _log_proc(call_id, "postgres_insert_skipped", role_index=idx, message="Duplicate")
        except Exception as e:
            await _log_proc(call_id, "postgres_insert_failed", level="error", role_index=idx, message=str(e))
    
    # Insert to Milvus
    for item in inserted:
        await _log_proc(call_id, "milvus_insert_started", role_index=item["idx"], job_id=item["job_id"])
        t0 = time.time()
        try:
            item["role"]["job_id"] = item["job_id"]
            success = await asyncio.to_thread(insert_job_posting, item["role"], item["embedding"])
            if success:
                await update_job_milvus_status(item["job_id"], True)
                await _log_proc(call_id, "milvus_insert_complete", role_index=item["idx"], job_id=item["job_id"],
                               duration_ms=int((time.time()-t0)*1000))
            else:
                await _log_proc(call_id, "milvus_insert_skipped", role_index=item["idx"], job_id=item["job_id"])
        except Exception as e:
            await _log_proc(call_id, "milvus_insert_failed", level="error", role_index=item["idx"], 
                           job_id=item["job_id"], message=str(e))
    
    # Update call transcript
    wrapper = {
        "multi_role": parsed.get("multi_role", len(roles) > 1),
        "roles": [i["role"] for i in inserted],
        "parse_warnings": warnings,
        "cleaned_transcript": redacted,
        "source_call_id": call_id,
        "extraction_version": "v2.0",
        "jobs_created": [i["job_id"] for i in inserted],
    }
    await update_call_transcript(call_id, status="parsed", parsed_requirements=wrapper)
    await _log_proc(call_id, "processing_complete", message="Done",
                   metadata={"roles_found": len(roles), "jobs_created": len(inserted)},
                   duration_ms=int((time.time()-start)*1000))


# ============================================================
# WEBHOOK ENDPOINTS
# ============================================================

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    
    valid, reason = _validate_webhook(payload)
    if not valid:
        _log("warning", "webhook_invalid", reason=reason)
        return JSONResponse({"status": "invalid", "reason": reason}, status_code=400, headers=CORS)
    
    event_type = payload.get("event_type")
    call_id = payload.get("call_id")
    
    if event_type == "test":
        return JSONResponse({"status": "ok"}, headers=CORS)
    
    if event_type != "call_ended":
        return JSONResponse({"status": "ignored", "event_type": event_type}, headers=CORS)
    
    # Store and process
    try:
        await insert_call_transcript(payload)
    except Exception as e:
        _log("error", "webhook_db_error", call_id=call_id, error=str(e))
        return JSONResponse({"status": "db_error", "error": str(e)}, status_code=500, headers=CORS)
    
    asyncio.create_task(_process_call(call_id, payload))
    return JSONResponse({"status": "accepted", "call_id": call_id}, headers=CORS)


@app.options("/webhook")
async def webhook_options():
    return JSONResponse({}, headers=CORS)


# ============================================================
# HEALTH ENDPOINTS
# ============================================================

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "timestamp": datetime.now(tz=timezone.utc).isoformat()})

@app.get("/health/db")
async def health_db():
    ok, msg = await check_db_connection()
    return JSONResponse({"status": "ok" if ok else "error", "detail": msg}, status_code=200 if ok else 503)

@app.get("/health/milvus")
async def health_milvus():
    ok, msg = await asyncio.to_thread(check_milvus_connection)
    return JSONResponse({"status": "ok" if ok else "error", "detail": msg}, status_code=200 if ok else 503)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/stats")
async def api_stats():
    stats = await fetch_stats()
    return JSONResponse(stats, headers=CORS)

@app.get("/api/calls/{call_id}/status")
async def api_call_status(call_id: str):
    transcript = await fetch_call_transcript(call_id)
    if not transcript:
        raise HTTPException(404, "Call not found")
    logs = await fetch_processing_logs(call_id)
    return JSONResponse({
        "call_id": call_id,
        "status": transcript.get("status"),
        "error_message": transcript.get("error_message"),
        "message_count": transcript.get("message_count"),
        "processing_logs": logs,
        "parsed_requirements": transcript.get("parsed_requirements"),
    }, headers=CORS)

@app.get("/api/calls/{call_id}/logs")
async def api_call_logs(call_id: str):
    logs = await fetch_processing_logs(call_id)
    return JSONResponse({"call_id": call_id, "logs": logs}, headers=CORS)

@app.post("/api/calls/{call_id}/reprocess")
async def api_reprocess(call_id: str, x_admin_key: Optional[str] = Header(None)):
    if not _verify_admin(x_admin_key):
        raise HTTPException(401, "Invalid admin key")
    
    transcript = await fetch_call_transcript(call_id)
    if not transcript:
        raise HTTPException(404, "Call not found")
    if transcript.get("status") != "failed":
        raise HTTPException(400, f"Call status is '{transcript.get('status')}', not 'failed'")
    
    payload = transcript.get("raw_payload", {})
    payload["call_id"] = call_id
    asyncio.create_task(_process_call(call_id, payload))
    return JSONResponse({"status": "reprocessing", "call_id": call_id}, headers=CORS)

@app.post("/api/calls/ingest")
async def api_ingest(request: Request, x_admin_key: Optional[str] = Header(None)):
    if not _verify_admin(x_admin_key):
        raise HTTPException(401, "Invalid admin key")
    
    body = await request.json()
    call_id = body.get("call_id")
    if not call_id:
        raise HTTPException(400, "call_id required")
    
    # Check if already exists
    existing = await fetch_call_transcript(call_id)
    if existing:
        raise HTTPException(409, f"Call {call_id} already exists")
    
    # Fetch from Beyond Presence API
    payload = await asyncio.to_thread(build_webhook_payload, call_id)
    if not payload:
        raise HTTPException(404, f"Call {call_id} not found in Beyond Presence")
    
    try:
        await insert_call_transcript(payload)
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")
    
    asyncio.create_task(_process_call(call_id, payload))
    return JSONResponse({"status": "ingesting", "call_id": call_id}, headers=CORS)

@app.post("/api/milvus/sync")
async def api_milvus_sync(x_admin_key: Optional[str] = Header(None)):
    if not _verify_admin(x_admin_key):
        raise HTTPException(401, "Invalid admin key")
    
    jobs = await fetch_unsynced_jobs(100)
    if not jobs:
        return JSONResponse({"status": "ok", "synced": 0, "failed": 0}, headers=CORS)
    
    success, fail = await asyncio.to_thread(sync_jobs_to_milvus, jobs)
    
    # Update synced status
    for job in jobs[:success]:
        await update_job_milvus_status(job["job_id"], True)
    
    return JSONResponse({"status": "ok", "synced": success, "failed": fail, "total": len(jobs)}, headers=CORS)

@app.get("/api/failed")
async def api_failed():
    calls = await fetch_failed_calls()
    return JSONResponse({"failed_calls": calls}, headers=CORS)