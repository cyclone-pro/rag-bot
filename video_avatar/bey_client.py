"""Beyond Presence API client for fetching calls and messages."""

import os
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("bey_client")

BEY_API_URL = os.getenv("BEY_API_URL", "https://api.bey.dev/v1")
BEY_API_KEY = os.getenv("BEY_API_KEY")


def _log_event(level: str, message: str, **fields: Any) -> None:
    import json
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


def _get_headers() -> Dict[str, str]:
    """Get API headers with authentication."""
    if not BEY_API_KEY:
        raise ValueError("BEY_API_KEY environment variable is not set")
    return {
        "x-api-key": BEY_API_KEY,
        "Content-Type": "application/json",
    }


def fetch_call(call_id: str) -> Optional[Dict[str, Any]]:
    """Fetch call details from Beyond Presence API.
    
    GET /v1/calls/{id}
    
    Returns call object with:
    - id: string
    - agent_id: string
    - started_at: string (ISO 8601)
    - ended_at: string | null
    - tags: object
    """
    try:
        response = requests.get(
            f"{BEY_API_URL}/calls/{call_id}",
            headers=_get_headers(),
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        _log_event("info", "bey_fetch_call_ok", call_id=call_id)
        return data
    except requests.exceptions.HTTPError as e:
        _log_event("error", "bey_fetch_call_failed", call_id=call_id, 
                   status_code=e.response.status_code if e.response else None,
                   error=str(e))
        return None
    except Exception as e:
        _log_event("error", "bey_fetch_call_error", call_id=call_id, error=str(e))
        return None


def fetch_messages(call_id: str) -> List[Dict[str, Any]]:
    """Fetch messages for a call from Beyond Presence API.
    
    GET /v1/calls/{id}/messages
    
    Returns list of message objects:
    - message: string
    - sent_at: string (ISO 8601)
    - sender: 'ai' | 'user'
    """
    try:
        response = requests.get(
            f"{BEY_API_URL}/calls/{call_id}/messages",
            headers=_get_headers(),
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # API might return list directly or wrapped in object
        messages = data if isinstance(data, list) else data.get("messages", data.get("data", []))
        
        _log_event("info", "bey_fetch_messages_ok", call_id=call_id, count=len(messages))
        return messages
    except requests.exceptions.HTTPError as e:
        _log_event("error", "bey_fetch_messages_failed", call_id=call_id,
                   status_code=e.response.status_code if e.response else None,
                   error=str(e))
        return []
    except Exception as e:
        _log_event("error", "bey_fetch_messages_error", call_id=call_id, error=str(e))
        return []


def build_webhook_payload(call_id: str) -> Optional[Dict[str, Any]]:
    """Build a webhook-like payload by fetching from Beyond Presence API.
    
    This is used for manual ingestion when webhook was missed.
    """
    call = fetch_call(call_id)
    if not call:
        return None
    
    messages = fetch_messages(call_id)
    
    # Build payload similar to webhook format
    payload = {
        "event_type": "call_ended",
        "call_id": call.get("id", call_id),
        "call_data": {
            "agentId": call.get("agent_id"),
            "userName": None,
            "startedAt": call.get("started_at"),
            "endedAt": call.get("ended_at"),
        },
        "evaluation": {
            "topic": None,
            "user_sentiment": None,
            "duration_minutes": None,
            "messages_count": str(len(messages)),
        },
        "messages": messages,
        "source": "manual_ingest",  # Mark as manual ingest
    }
    
    # Calculate duration if we have timestamps
    started_at = call.get("started_at")
    ended_at = call.get("ended_at")
    if started_at and ended_at:
        try:
            from datetime import datetime
            start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            duration_minutes = (end - start).total_seconds() / 60
            payload["evaluation"]["duration_minutes"] = str(round(duration_minutes, 2))
        except Exception:
            pass
    
    _log_event("info", "bey_build_payload_ok", call_id=call_id, message_count=len(messages))
    return payload