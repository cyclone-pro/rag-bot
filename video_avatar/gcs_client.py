"""Google Cloud Storage client for call history management."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound

logger = logging.getLogger("gcs_client")

BUCKET_NAME = os.getenv("GCS_BUCKET", "rcrtrai-call-history")
HISTORY_FILE = "call_history.json"
MAX_HISTORY_SIZE = 10


def _log(level: str, msg: str, **kw: Any) -> None:
    getattr(logger, level if level in ("warning", "error") else "info")(
        json.dumps({"message": msg, **kw})
    )


def _get_client() -> storage.Client:
    """Get GCS client."""
    return storage.Client()


def _get_bucket(client: storage.Client) -> storage.Bucket:
    """Get or create the bucket."""
    try:
        bucket = client.get_bucket(BUCKET_NAME)
    except NotFound:
        _log("info", "creating_bucket", bucket=BUCKET_NAME)
        bucket = client.create_bucket(BUCKET_NAME, location="us-central1")
    return bucket


def get_call_history() -> List[Dict[str, Any]]:
    """Fetch call history from GCS."""
    try:
        client = _get_client()
        bucket = _get_bucket(client)
        blob = bucket.blob(HISTORY_FILE)
        
        if not blob.exists():
            _log("info", "call_history_not_found", creating_empty=True)
            return []
        
        content = blob.download_as_text()
        data = json.loads(content)
        calls = data.get("calls", [])
        _log("info", "call_history_loaded", count=len(calls))
        return calls
    
    except Exception as e:
        _log("error", "get_call_history_failed", error=str(e))
        return []


def save_call_history(calls: List[Dict[str, Any]]) -> bool:
    """Save call history to GCS, keeping only last MAX_HISTORY_SIZE entries."""
    try:
        # Keep only most recent calls
        trimmed = calls[-MAX_HISTORY_SIZE:] if len(calls) > MAX_HISTORY_SIZE else calls
        
        client = _get_client()
        bucket = _get_bucket(client)
        blob = bucket.blob(HISTORY_FILE)
        
        data = {
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "calls": trimmed,
        }
        
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str),
            content_type="application/json"
        )
        
        _log("info", "call_history_saved", count=len(trimmed))
        return True
    
    except Exception as e:
        _log("error", "save_call_history_failed", error=str(e))
        return False


def add_call_to_history(
    call_id: str,
    agent_id: str,
    username: str = "vendor",
    job_summary: Optional[Dict[str, Any]] = None,
    status: str = "completed",
) -> bool:
    """Add a call to history after processing."""
    try:
        calls = get_call_history()
        
        # Check if call already exists (dedupe)
        existing_ids = {c.get("call_id") for c in calls}
        if call_id in existing_ids:
            _log("info", "call_already_in_history", call_id=call_id)
            # Update existing entry
            for call in calls:
                if call.get("call_id") == call_id:
                    call["job_summary"] = job_summary
                    call["status"] = status
                    call["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
                    break
        else:
            # Add new entry
            calls.append({
                "call_id": call_id,
                "agent_id": agent_id,
                "username": username,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "job_summary": job_summary or {},
                "status": status,
            })
        
        return save_call_history(calls)
    
    except Exception as e:
        _log("error", "add_call_to_history_failed", call_id=call_id, error=str(e))
        return False


def update_call_status(call_id: str, status: str, job_summary: Optional[Dict[str, Any]] = None) -> bool:
    """Update an existing call's status and job summary."""
    try:
        calls = get_call_history()
        
        for call in calls:
            if call.get("call_id") == call_id:
                call["status"] = status
                call["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
                if job_summary:
                    call["job_summary"] = job_summary
                _log("info", "call_status_updated", call_id=call_id, status=status)
                return save_call_history(calls)
        
        _log("warning", "call_not_found_for_update", call_id=call_id)
        return False
    
    except Exception as e:
        _log("error", "update_call_status_failed", call_id=call_id, error=str(e))
        return False


def get_recent_job_summaries(limit: int = 5) -> List[Dict[str, Any]]:
    """Get recent job summaries for context building."""
    calls = get_call_history()
    
    summaries = []
    for call in reversed(calls):
        if call.get("job_summary") and call.get("status") == "completed":
            summary = call.get("job_summary", {})
            if summary.get("job_title"):
                summaries.append({
                    "call_id": call.get("call_id"),
                    "job_title": summary.get("job_title"),
                    "job_type": summary.get("job_type"),
                    "must_have_skills": summary.get("must_have_skills", []),
                    "work_model": summary.get("work_model"),
                    "pay_rate_min": summary.get("pay_rate_min"),
                    "pay_rate_max": summary.get("pay_rate_max"),
                    "salary_min": summary.get("salary_min"),
                    "salary_max": summary.get("salary_max"),
                    "location_cities": summary.get("location_cities", []),
                })
                if len(summaries) >= limit:
                    break
    
    return summaries


def mark_call_started(call_id: str, agent_id: str, username: str = "vendor") -> bool:
    """Mark a call as started (before processing completes)."""
    return add_call_to_history(
        call_id=call_id,
        agent_id=agent_id,
        username=username,
        job_summary=None,
        status="started",
    )