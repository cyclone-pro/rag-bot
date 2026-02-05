"""Zoom API client for creating and managing meetings."""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET,
    ZOOM_API_BASE, ZOOM_OAUTH_URL, NAME_MATCH_THRESHOLD,
)
from models import ZoomMeeting, ZoomParticipant, ParticipantAction

logger = logging.getLogger("rcrutr_interviews_zoom")

# Token cache
_access_token: Optional[str] = None
_token_expires_at: Optional[datetime] = None


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


# =============================================================================
# AUTHENTICATION
# =============================================================================

def _get_access_token() -> str:
    """Get Zoom OAuth access token using Server-to-Server OAuth."""
    global _access_token, _token_expires_at
    
    # Return cached token if still valid
    if _access_token and _token_expires_at and datetime.utcnow() < _token_expires_at:
        return _access_token
    
    if not all([ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET]):
        raise ValueError("Zoom credentials not configured. Set ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET")
    
    # Build authorization header
    credentials = f"{ZOOM_CLIENT_ID}:{ZOOM_CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    data = {
        "grant_type": "account_credentials",
        "account_id": ZOOM_ACCOUNT_ID,
    }
    
    _log_event("info", "zoom_oauth_request", account_id=ZOOM_ACCOUNT_ID[:8] + "...")
    
    response = requests.post(ZOOM_OAUTH_URL, headers=headers, data=data, timeout=30)
    response.raise_for_status()
    
    token_data = response.json()
    _access_token = token_data["access_token"]
    expires_in = token_data.get("expires_in", 3600)
    _token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)  # Refresh 1 min early
    
    _log_event("info", "zoom_oauth_success", expires_in=expires_in)
    return _access_token


def _get_headers() -> Dict[str, str]:
    """Get headers with authorization for Zoom API calls."""
    token = _get_access_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# =============================================================================
# MEETING OPERATIONS
# =============================================================================

def create_meeting(
    topic: str,
    start_time: datetime,
    duration_minutes: int = 30,
    timezone: str = "UTC",
    *,
    waiting_room: bool = True,
    auto_recording: str = "none",  # "none", "local", "cloud"
    password: Optional[str] = None,
    join_before_host: bool = False,  # Allow participants before host
) -> Optional[ZoomMeeting]:
    """
    Create a Zoom meeting.
    
    Args:
        topic: Meeting topic/title
        start_time: When the meeting should start
        duration_minutes: Expected duration
        timezone: Timezone for the meeting
        waiting_room: Enable waiting room (set False for fully autonomous)
        auto_recording: Recording setting
        password: Optional meeting password
        join_before_host: Allow participants to join before host
    
    Returns:
        ZoomMeeting object or None if failed
    """
    try:
        headers = _get_headers()
        
        # Format start time for Zoom API
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        payload = {
            "topic": topic,
            "type": 2,  # Scheduled meeting
            "start_time": start_time_str,
            "duration": duration_minutes,
            "timezone": timezone,
            "settings": {
                "waiting_room": waiting_room,
                "join_before_host": join_before_host,
                "mute_upon_entry": True,
                "auto_recording": auto_recording,
                "meeting_authentication": False,
                "participant_video": True,
                "host_video": True,
                # Alternative host can admit from waiting room
                "alternative_hosts_email_notification": True,
            },
        }
        
        if password:
            payload["password"] = password
        
        _log_event("info", "zoom_create_meeting", 
                   topic=topic, 
                   start_time=start_time_str,
                   waiting_room=waiting_room,
                   join_before_host=join_before_host)
        
        response = requests.post(
            f"{ZOOM_API_BASE}/users/me/meetings",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        
        data = response.json()
        
        meeting = ZoomMeeting(
            id=str(data["id"]),
            join_url=data["join_url"],
            start_url=data["start_url"],  # Host URL - bypasses waiting room
            password=data.get("password"),
            topic=data["topic"],
            start_time=start_time,
            duration=duration_minutes,
            timezone=timezone,
        )
        
        _log_event("info", "zoom_meeting_urls",
                   join_url=meeting.join_url[:60],
                   start_url=meeting.start_url[:60] if meeting.start_url else "None")
        
        _log_event("info", "zoom_meeting_created", 
                   meeting_id=meeting.id, 
                   join_url=meeting.join_url[:50] + "...")
        
        return meeting
        
    except requests.exceptions.HTTPError as e:
        _log_event("error", "zoom_create_meeting_failed", 
                   status_code=e.response.status_code if e.response else None,
                   response=e.response.text[:500] if e.response else None,
                   error=str(e))
        return None
    except Exception as e:
        _log_event("error", "zoom_create_meeting_error", error=str(e))
        return None


def get_meeting(meeting_id: str) -> Optional[Dict[str, Any]]:
    """Get meeting details."""
    try:
        headers = _get_headers()
        
        response = requests.get(
            f"{ZOOM_API_BASE}/meetings/{meeting_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        _log_event("error", "zoom_get_meeting_failed", meeting_id=meeting_id, error=str(e))
        return None


def delete_meeting(meeting_id: str) -> bool:
    """Delete/cancel a meeting."""
    try:
        headers = _get_headers()
        
        response = requests.delete(
            f"{ZOOM_API_BASE}/meetings/{meeting_id}",
            headers=headers,
            timeout=30,
        )
        
        if response.status_code in (204, 200):
            _log_event("info", "zoom_meeting_deleted", meeting_id=meeting_id)
            return True
        
        response.raise_for_status()
        return True
    except Exception as e:
        _log_event("error", "zoom_delete_meeting_failed", meeting_id=meeting_id, error=str(e))
        return False


# =============================================================================
# PARTICIPANT MANAGEMENT
# =============================================================================

def get_waiting_room_participants(meeting_id: str) -> List[ZoomParticipant]:
    """
    Get participants in the waiting room.
    
    Uses the Zoom Dashboard API to get live meeting participants.
    Requires: dashboard_meetings:read:admin scope
    """
    try:
        headers = _get_headers()
        
        # Method 1: Try Dashboard API (more reliable for live meetings)
        # GET /metrics/meetings/{meetingId}/participants
        response = requests.get(
            f"{ZOOM_API_BASE}/metrics/meetings/{meeting_id}/participants",
            headers=headers,
            params={"type": "waiting", "page_size": 30},
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            participants = []
            for p in data.get("participants", []):
                if p.get("status") == "waiting" or p.get("in_waiting_room"):
                    participants.append(ZoomParticipant(
                        id=p.get("id", p.get("user_id", p.get("participant_user_id", ""))),
                        user_name=p.get("user_name", p.get("name", "")),
                        email=p.get("email"),
                        status="waiting",
                    ))
            
            _log_event("info", "zoom_waiting_room_participants_dashboard", 
                       meeting_id=meeting_id, count=len(participants))
            return participants
        
        # Method 2: Try live meeting participants endpoint
        response = requests.get(
            f"{ZOOM_API_BASE}/live_meetings/{meeting_id}/participants",
            headers=headers,
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            participants = []
            for p in data.get("participants", []):
                if p.get("status") == "waiting":
                    participants.append(ZoomParticipant(
                        id=p.get("id", p.get("user_id", "")),
                        user_name=p.get("user_name", p.get("name", "")),
                        email=p.get("email"),
                        status="waiting",
                    ))
            
            _log_event("info", "zoom_waiting_room_participants_live", 
                       meeting_id=meeting_id, count=len(participants))
            return participants
        
        if response.status_code == 404:
            # Meeting not started yet
            return []
        
        _log_event("warning", "zoom_waiting_room_api_failed",
                   meeting_id=meeting_id, status=response.status_code)
        return []
        
    except Exception as e:
        _log_event("error", "zoom_get_waiting_room_failed", meeting_id=meeting_id, error=str(e))
        return []


def admit_participant(meeting_id: str, participant_id: str) -> bool:
    """
    Admit a participant from waiting room to the meeting.
    
    Uses PUT /live_meetings/{meetingId}/participants/{participantId}
    Requires: meeting:write:admin scope
    """
    try:
        headers = _get_headers()
        
        # Method 1: Live meetings API
        payload = {"action": "admit"}
        
        response = requests.put(
            f"{ZOOM_API_BASE}/live_meetings/{meeting_id}/participants/{participant_id}",
            headers=headers,
            json=payload,
            timeout=30,
        )
        
        if response.status_code in (200, 204):
            _log_event("info", "zoom_participant_admitted", 
                       meeting_id=meeting_id, participant_id=participant_id)
            return True
        
        # Method 2: Alternative endpoint
        payload = {"method": "admit"}
        response = requests.put(
            f"{ZOOM_API_BASE}/meetings/{meeting_id}/participants/{participant_id}/status",
            headers=headers,
            json=payload,
            timeout=30,
        )
        
        if response.status_code in (200, 204):
            _log_event("info", "zoom_participant_admitted_alt", 
                       meeting_id=meeting_id, participant_id=participant_id)
            return True
        
        _log_event("warning", "zoom_admit_failed",
                   meeting_id=meeting_id, 
                   participant_id=participant_id,
                   status=response.status_code,
                   response=response.text[:200] if response.text else None)
        return False
        
    except Exception as e:
        _log_event("error", "zoom_admit_participant_failed", 
                   meeting_id=meeting_id, participant_id=participant_id, error=str(e))
        return False


def remove_participant(meeting_id: str, participant_id: str) -> bool:
    """Remove/reject a participant from the meeting."""
    try:
        headers = _get_headers()
        
        response = requests.delete(
            f"{ZOOM_API_BASE}/meetings/{meeting_id}/participants/{participant_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        
        _log_event("info", "zoom_participant_removed", 
                   meeting_id=meeting_id, participant_id=participant_id)
        return True
        
    except Exception as e:
        _log_event("error", "zoom_remove_participant_failed", 
                   meeting_id=meeting_id, participant_id=participant_id, error=str(e))
        return False


# =============================================================================
# NAME MATCHING
# =============================================================================

def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    if not name:
        return ""
    # Lowercase, strip, remove extra spaces
    return " ".join(name.lower().strip().split())


def calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names using fuzzy matching.
    Returns a score between 0.0 and 1.0.
    """
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    if not n1 or not n2:
        return 0.0
    
    # Exact match
    if n1 == n2:
        return 1.0
    
    # Check if one contains the other (partial match)
    if n1 in n2 or n2 in n1:
        return 0.9
    
    # Check first name match
    n1_parts = n1.split()
    n2_parts = n2.split()
    
    if n1_parts and n2_parts and n1_parts[0] == n2_parts[0]:
        return 0.85
    
    # Sequence matcher for fuzzy comparison
    return SequenceMatcher(None, n1, n2).ratio()


def should_admit_participant(
    participant_name: str,
    expected_candidate_name: str,
    threshold: float = NAME_MATCH_THRESHOLD,
) -> Tuple[ParticipantAction, float]:
    """
    Determine if a participant should be admitted based on name matching.
    
    Returns:
        (action, similarity_score)
    """
    similarity = calculate_name_similarity(participant_name, expected_candidate_name)
    
    _log_event("info", "zoom_name_match_check",
               participant_name=participant_name,
               expected_name=expected_candidate_name,
               similarity=similarity,
               threshold=threshold)
    
    if similarity >= threshold:
        return (ParticipantAction.ADMIT, similarity)
    else:
        return (ParticipantAction.REJECT, similarity)


# =============================================================================
# WEBHOOK SIGNATURE VERIFICATION
# =============================================================================

def verify_webhook_signature(
    payload: bytes,
    signature: str,
    timestamp: str,
    secret: str,
) -> bool:
    """
    Verify Zoom webhook signature.
    
    Zoom uses HMAC-SHA256 for webhook signature verification.
    """
    import hashlib
    import hmac
    
    message = f"v0:{timestamp}:{payload.decode()}"
    expected_signature = "v0=" + hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


# =============================================================================
# HEALTH CHECK
# =============================================================================

def check_zoom_connection() -> Tuple[bool, str]:
    """Check Zoom API connection health."""
    try:
        # Try to get access token
        token = _get_access_token()
        
        # Try a simple API call
        headers = _get_headers()
        response = requests.get(
            f"{ZOOM_API_BASE}/users/me",
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        
        user_data = response.json()
        return (True, f"ok, user: {user_data.get('email', 'unknown')}")
        
    except ValueError as e:
        return (False, f"credentials not configured: {e}")
    except requests.exceptions.HTTPError as e:
        return (False, f"API error: {e.response.status_code if e.response else 'unknown'}")
    except Exception as e:
        return (False, str(e))
