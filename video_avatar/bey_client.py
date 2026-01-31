"""Beyond Presence API client for agents, calls, and messages."""

import os
import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("bey_client")

BEY_API_URL = os.getenv("BEY_API_URL", "https://api.bey.dev/v1")
BEY_API_KEY = os.getenv("BEY_API_KEY")

DEFAULT_AVATAR_ID = "b63ba4e6-d346-45d0-ad28-5ddffaac0bd0_v2"


def _log_event(level: str, message: str, **fields: Any) -> None:
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


def _get_headers() -> Dict[str, str]:
    """Get API headers with authentication."""
    api_key = BEY_API_KEY or os.getenv("BEY_API_KEY")
    if not api_key:
        raise ValueError("BEY_API_KEY environment variable is not set")
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }


# ============================================================
# AGENT OPERATIONS
# ============================================================

def create_agent(
    name: str,
    system_prompt: str,
    greeting: str,
    avatar_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create a new agent with Just-in-Time context.
    
    POST /v1/agent
    
    Returns agent object with id.
    """
    # Validate inputs
    if not system_prompt:
        _log_event("error", "bey_create_agent_error", error="system_prompt is empty")
        return None
    
    if len(system_prompt) > 10000:
        _log_event("error", "bey_create_agent_error", 
                   error=f"system_prompt too long: {len(system_prompt)} chars (max 10000)")
        return None
    
    try:
        payload = {
            "name": name,
            "system_prompt": system_prompt,
            "greeting": greeting,
            "avatar_id": avatar_id or DEFAULT_AVATAR_ID,
        }
        
        _log_event("info", "bey_create_agent_request",
                   name=name,
                   prompt_length=len(system_prompt),
                   greeting_length=len(greeting),
                   avatar_id=avatar_id or DEFAULT_AVATAR_ID)
        
        response = requests.post(
            f"{BEY_API_URL}/agent",
            headers=_get_headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        _log_event("info", "bey_create_agent_ok", 
                   agent_id=data.get("id"), 
                   prompt_length=len(system_prompt))
        return data
    
    except requests.exceptions.HTTPError as e:
        error_detail = {
            "status_code": e.response.status_code if e.response else None,
            "response_text": e.response.text[:500] if e.response and e.response.text else None,
            "error": str(e),
            "prompt_length": len(system_prompt),
            "avatar_id": avatar_id or DEFAULT_AVATAR_ID,
        }
        _log_event("error", "bey_create_agent_failed", **error_detail)
        return None
    except requests.exceptions.Timeout as e:
        _log_event("error", "bey_create_agent_timeout", error=str(e))
        return None
    except requests.exceptions.ConnectionError as e:
        _log_event("error", "bey_create_agent_connection_error", error=str(e))
        return None
    except Exception as e:
        import traceback
        _log_event("error", "bey_create_agent_error", 
                   error=str(e), 
                   traceback=traceback.format_exc()[:500])
        return None


def delete_agent(agent_id: str) -> bool:
    """Delete a disposable agent after call ends.
    
    DELETE /v1/agent/{id}
    """
    try:
        response = requests.delete(
            f"{BEY_API_URL}/agent/{agent_id}",
            headers=_get_headers(),
            timeout=30,
        )
        response.raise_for_status()
        _log_event("info", "bey_delete_agent_ok", agent_id=agent_id)
        return True
    
    except requests.exceptions.HTTPError as e:
        # 404 is okay - agent might already be deleted
        if e.response and e.response.status_code == 404:
            _log_event("info", "bey_delete_agent_not_found", agent_id=agent_id)
            return True
        _log_event("error", "bey_delete_agent_failed",
                   agent_id=agent_id,
                   status_code=e.response.status_code if e.response else None,
                   error=str(e))
        return False
    except Exception as e:
        _log_event("error", "bey_delete_agent_error", agent_id=agent_id, error=str(e))
        return False


def get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    """Get agent details.
    
    GET /v1/agent/{id}
    """
    try:
        response = requests.get(
            f"{BEY_API_URL}/agent/{agent_id}",
            headers=_get_headers(),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        _log_event("error", "bey_get_agent_error", agent_id=agent_id, error=str(e))
        return None


# ============================================================
# CALL OPERATIONS
# ============================================================

def create_call(
    agent_id: str,
    username: str = "User",
    tags: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Create a new call session.
    
    POST /v1/calls
    
    Returns call object with livekit_url and livekit_token.
    """
    try:
        payload = {
            "agent_id": agent_id,
            "livekit_username": username,
        }
        if tags:
            payload["tags"] = tags
        
        response = requests.post(
            f"{BEY_API_URL}/calls",
            headers=_get_headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        _log_event("info", "bey_create_call_ok",
                   call_id=data.get("id"),
                   agent_id=agent_id)
        return data
    
    except requests.exceptions.HTTPError as e:
        _log_event("error", "bey_create_call_failed",
                   agent_id=agent_id,
                   status_code=e.response.status_code if e.response else None,
                   response_text=e.response.text if e.response else None,
                   error=str(e))
        return None
    except Exception as e:
        _log_event("error", "bey_create_call_error", agent_id=agent_id, error=str(e))
        return None


def fetch_call(call_id: str) -> Optional[Dict[str, Any]]:
    """Fetch call details from Beyond Presence API.
    
    GET /v1/calls/{id}
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
        "source": "manual_ingest",
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