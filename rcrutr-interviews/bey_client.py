"""Bey API client for creating agents and managing calls."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import requests

from config import BEY_API_KEY, BEY_API_URL, AVATARS, DEFAULT_AVATAR, BEY_LLM_API_ID, BEY_LLM_MODEL, BEY_LLM_TEMPERATURE
from models import BeyAgent, BeyCall, BeySendToExternalResponse

logger = logging.getLogger("rcrutr_interviews_bey")


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


def _get_headers() -> Dict[str, str]:
    """Get headers for Bey API calls."""
    if not BEY_API_KEY:
        raise ValueError("BEY_API_KEY not configured")
    return {
        "x-api-key": BEY_API_KEY,
        "Content-Type": "application/json",
    }


def get_avatar_config(avatar_key: str) -> Dict[str, Any]:
    """Get avatar configuration by key."""
    key = avatar_key.lower() if avatar_key else DEFAULT_AVATAR
    return AVATARS.get(key, AVATARS[DEFAULT_AVATAR])


# =============================================================================
# AGENT OPERATIONS
# =============================================================================

def create_agent(
    name: str,
    system_prompt: str,
    greeting: str,
    avatar_id: Optional[str] = None,
    use_external_llm: bool = True,
) -> Optional[BeyAgent]:
    """
    Create a Bey agent for the interview.
    
    Args:
        name: Agent name
        system_prompt: The interview prompt
        greeting: Initial greeting message
        avatar_id: Bey avatar ID
        use_external_llm: If True and OPENAI_LLM_API_ID is set, use GPT-4o-mini
    
    Returns:
        BeyAgent object or None if failed
    """
    try:
        headers = _get_headers()
        
        # Validate prompt length
        if len(system_prompt) > 10000:
            _log_event("error", "bey_create_agent_prompt_too_long",
                       length=len(system_prompt), max=10000)
            return None
        
        # Use default avatar if not specified
        if not avatar_id:
            avatar_id = AVATARS[DEFAULT_AVATAR]["id"]
        
        payload = {
            "name": name,
            "system_prompt": system_prompt,
            "greeting": greeting,
            "avatar_id": avatar_id,
        }
        
        # Add external LLM configuration if available
        if use_external_llm and BEY_LLM_API_ID:
            payload["llm"] = {
                "type": "openai_compatible",
                "api_id": BEY_LLM_API_ID,
                "model": BEY_LLM_MODEL,
                "temperature": BEY_LLM_TEMPERATURE,
            }
            _log_event("info", "bey_create_agent_with_external_llm",
                       model=BEY_LLM_MODEL, temperature=BEY_LLM_TEMPERATURE)
        
        _log_event("info", "bey_create_agent_request",
                   name=name, prompt_length=len(system_prompt), avatar_id=avatar_id,
                   external_llm=bool(BEY_LLM_API_ID and use_external_llm))
        
        response = requests.post(
            f"{BEY_API_URL}/agent",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        
        data = response.json()
        agent = BeyAgent(
            id=data["id"],
            name=name,
        )
        
        _log_event("info", "bey_create_agent_ok", agent_id=agent.id)
        return agent
        
    except requests.exceptions.HTTPError as e:
        _log_event("error", "bey_create_agent_failed",
                   status_code=e.response.status_code if e.response else None,
                   response=e.response.text[:500] if e.response and e.response.text else None,
                   error=str(e))
        return None
    except Exception as e:
        _log_event("error", "bey_create_agent_error", error=str(e))
        return None


def delete_agent(agent_id: str) -> bool:
    """Delete a Bey agent."""
    try:
        headers = _get_headers()
        
        response = requests.delete(
            f"{BEY_API_URL}/agent/{agent_id}",
            headers=headers,
            timeout=30,
        )
        
        if response.status_code in (200, 204, 404):
            _log_event("info", "bey_delete_agent_ok", agent_id=agent_id)
            return True
        
        response.raise_for_status()
        return True
        
    except Exception as e:
        _log_event("error", "bey_delete_agent_failed", agent_id=agent_id, error=str(e))
        return False


# =============================================================================
# CALL OPERATIONS
# =============================================================================

def create_call(agent_id: str, username: str = "Candidate") -> Optional[BeyCall]:
    """
    Create a call session with an agent.
    
    Args:
        agent_id: The Bey agent ID
        username: Display name for the user in LiveKit
    
    Returns:
        BeyCall object with call_id, livekit_url, livekit_token
    """
    try:
        headers = _get_headers()
        
        payload = {
            "agent_id": agent_id,
            "livekit_username": username,
        }
        
        _log_event("info", "bey_create_call_request", agent_id=agent_id, username=username)
        
        response = requests.post(
            f"{BEY_API_URL}/calls",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        
        data = response.json()
        call = BeyCall(
            id=data["id"],
            agent_id=agent_id,
            livekit_url=data["livekit_url"],
            livekit_token=data["livekit_token"],
        )
        
        _log_event("info", "bey_create_call_ok", call_id=call.id)
        return call
        
    except requests.exceptions.HTTPError as e:
        _log_event("error", "bey_create_call_failed",
                   status_code=e.response.status_code if e.response else None,
                   response=e.response.text[:500] if e.response and e.response.text else None,
                   error=str(e))
        return None
    except Exception as e:
        _log_event("error", "bey_create_call_error", error=str(e))
        return None


def end_call(call_id: str) -> bool:
    """
    End a call session.
    
    This will disconnect the avatar from the meeting.
    """
    try:
        headers = _get_headers()
        
        _log_event("info", "bey_end_call_request", call_id=call_id)
        
        response = requests.post(
            f"{BEY_API_URL}/calls/{call_id}/end",
            headers=headers,
            timeout=30,
        )
        
        # Accept 200, 204, or 404 (already ended)
        if response.status_code in (200, 204, 404):
            _log_event("info", "bey_end_call_ok", call_id=call_id)
            return True
        
        response.raise_for_status()
        return True
        
    except Exception as e:
        _log_event("error", "bey_end_call_failed", call_id=call_id, error=str(e))
        return False


def send_to_external_meeting(
    call_id: str,
    meeting_url: str,
    livekit_url: str,
    livekit_token: str,
    bot_name: str = "RCRUTR AI Interviewer",
) -> Optional[BeySendToExternalResponse]:
    """
    Send the avatar to an external meeting (Zoom, Teams, etc.)
    
    Args:
        call_id: The Bey call ID from create_call
        meeting_url: The Zoom/Teams meeting join URL
        livekit_url: The LiveKit WebSocket URL
        livekit_token: The LiveKit token
        bot_name: Display name for the avatar in the meeting
    
    Returns:
        BeySendToExternalResponse with bot_id and status
    """
    try:
        headers = _get_headers()
        
        payload = {
            "meeting_url": meeting_url,
            "call_id": call_id,
            "url": livekit_url,
            "token": livekit_token,
            "bot_name": bot_name,
        }
        
        _log_event("info", "bey_send_to_external_request",
                   call_id=call_id, meeting_url=meeting_url[:50] + "...", bot_name=bot_name)
        
        response = requests.post(
            f"{BEY_API_URL}/calls/send-to-external",
            headers=headers,
            json=payload,
            timeout=60,  # Longer timeout for joining meeting
        )
        response.raise_for_status()
        
        data = response.json()
        result = BeySendToExternalResponse(
            bot_id=data.get("bot_id", ""),
            status=data.get("status", "unknown"),
        )
        
        _log_event("info", "bey_send_to_external_ok",
                   call_id=call_id, bot_id=result.bot_id, status=result.status)
        return result
        
    except requests.exceptions.HTTPError as e:
        _log_event("error", "bey_send_to_external_failed",
                   call_id=call_id,
                   status_code=e.response.status_code if e.response else None,
                   response=e.response.text[:500] if e.response and e.response.text else None,
                   error=str(e))
        return None
    except Exception as e:
        _log_event("error", "bey_send_to_external_error", call_id=call_id, error=str(e))
        return None


def get_call(call_id: str) -> Optional[Dict[str, Any]]:
    """Get call details."""
    try:
        headers = _get_headers()
        
        response = requests.get(
            f"{BEY_API_URL}/calls/{call_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        _log_event("error", "bey_get_call_failed", call_id=call_id, error=str(e))
        return None


def end_call(call_id: str) -> bool:
    """End an active call."""
    try:
        headers = _get_headers()
        
        response = requests.post(
            f"{BEY_API_URL}/calls/{call_id}/end",
            headers=headers,
            timeout=30,
        )
        
        if response.status_code in (200, 204, 404):
            _log_event("info", "bey_end_call_ok", call_id=call_id)
            return True
        
        response.raise_for_status()
        return True
        
    except Exception as e:
        _log_event("error", "bey_end_call_failed", call_id=call_id, error=str(e))
        return False


# =============================================================================
# HEALTH CHECK
# =============================================================================

def check_bey_connection() -> tuple[bool, str]:
    """Check Bey API connection health."""
    try:
        headers = _get_headers()
        
        # Try to list calls (or any lightweight endpoint)
        response = requests.get(
            f"{BEY_API_URL}/calls",
            headers=headers,
            timeout=10,
        )
        
        if response.status_code in (200, 401, 403):
            # 401/403 means API is reachable but auth issue
            if response.status_code != 200:
                return (False, f"API reachable but auth failed: {response.status_code}")
            return (True, "ok")
        
        return (False, f"unexpected status: {response.status_code}")
        
    except ValueError as e:
        return (False, f"credentials not configured: {e}")
    except Exception as e:
        return (False, str(e))
