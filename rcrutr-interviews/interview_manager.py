#!/usr/bin/env python3
"""
Interview Manager - Handles concurrent interviews, timeouts, and edge cases.

Key Features:
1. Prevents overlapping interviews (configurable buffer time)
2. Timeout for candidate no-show
3. Graceful cleanup of abandoned interviews
4. Tracking of active interviews
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from config import SERVICE_NAME
from db import (
    get_interview, 
    update_interview_status, 
    update_interview,
    list_interviews,
)
from models import InterviewStatus
from bey_client import delete_agent, end_call

logger = logging.getLogger(f"{SERVICE_NAME}_interview_manager")

# Configuration
CANDIDATE_JOIN_TIMEOUT_MINUTES = 10  # How long to wait for candidate
INTERVIEW_MAX_DURATION_MINUTES = 30  # Maximum interview length
INTERVIEW_BUFFER_MINUTES = 5  # Buffer between interviews


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


@dataclass
class ActiveInterview:
    """Tracks an active interview session."""
    interview_id: str
    agent_id: str
    call_id: str
    started_at: datetime
    candidate_joined: bool = False
    candidate_joined_at: Optional[datetime] = None


# In-memory tracking of active interviews
# In production, use Redis for distributed tracking
_active_interviews: Dict[str, ActiveInterview] = {}


# =============================================================================
# CONFLICT DETECTION
# =============================================================================

async def check_scheduling_conflict(
    scheduled_time: datetime,
    duration_minutes: int = 30,
    buffer_minutes: int = INTERVIEW_BUFFER_MINUTES,
) -> Optional[Dict[str, Any]]:
    """
    Check if there's a scheduling conflict with existing interviews.
    
    Returns:
        Conflicting interview dict if conflict exists, None otherwise
    """
    # Calculate time window
    start_window = scheduled_time - timedelta(minutes=buffer_minutes)
    end_window = scheduled_time + timedelta(minutes=duration_minutes + buffer_minutes)
    
    # Get interviews in this time range
    interviews, _ = await list_interviews(
        from_date=start_window,
        to_date=end_window,
        limit=10,
    )
    
    for interview in interviews:
        status = interview.get("interview_status")
        
        # Skip completed/cancelled interviews
        if status in ("completed", "cancelled", "failed", "incomplete"):
            continue
        
        interview_time = interview.get("scheduled_time")
        if interview_time:
            # Check for overlap
            interview_end = interview_time + timedelta(minutes=duration_minutes)
            
            if (scheduled_time < interview_end and 
                scheduled_time + timedelta(minutes=duration_minutes) > interview_time):
                _log_event("warning", "scheduling_conflict_detected",
                           new_time=scheduled_time.isoformat(),
                           conflicting_interview=interview.get("interview_id"),
                           conflicting_time=interview_time.isoformat())
                return interview
    
    return None


async def get_active_interviews_count() -> int:
    """Get count of currently active (in_progress) interviews."""
    interviews, _ = await list_interviews(status="in_progress", limit=100)
    return len(interviews)


# =============================================================================
# TIMEOUT MANAGEMENT
# =============================================================================

async def start_candidate_timeout(
    interview_id: str,
    timeout_minutes: int = CANDIDATE_JOIN_TIMEOUT_MINUTES,
):
    """
    Start a timeout timer for candidate to join.
    If candidate doesn't join within timeout, end the interview.
    """
    _log_event("info", "candidate_timeout_started",
               interview_id=interview_id, timeout_minutes=timeout_minutes)
    
    await asyncio.sleep(timeout_minutes * 60)
    
    # Check if candidate has joined
    interview = await get_interview(interview_id)
    if not interview:
        return
    
    status = interview.get("interview_status")
    candidate_joined_at = interview.get("candidate_joined_at")
    
    if status == "waiting_for_candidate" and not candidate_joined_at:
        _log_event("warning", "candidate_no_show_timeout",
                   interview_id=interview_id, timeout_minutes=timeout_minutes)
        
        await handle_candidate_no_show(interview_id)


async def start_interview_max_duration_timeout(
    interview_id: str,
    max_minutes: int = INTERVIEW_MAX_DURATION_MINUTES,
):
    """
    Start a maximum duration timeout for the interview.
    Safety mechanism to prevent runaway interviews.
    """
    _log_event("info", "max_duration_timeout_started",
               interview_id=interview_id, max_minutes=max_minutes)
    
    await asyncio.sleep(max_minutes * 60)
    
    # Check if interview is still in progress
    interview = await get_interview(interview_id)
    if not interview:
        return
    
    status = interview.get("interview_status")
    
    if status == "in_progress":
        _log_event("warning", "interview_max_duration_reached",
                   interview_id=interview_id, max_minutes=max_minutes)
        
        await force_end_interview(interview_id, reason="Maximum duration reached")


# =============================================================================
# INTERVIEW LIFECYCLE
# =============================================================================

async def handle_candidate_no_show(interview_id: str):
    """Handle case where candidate didn't join within timeout."""
    
    interview = await get_interview(interview_id)
    if not interview:
        return
    
    agent_id = interview.get("agent_id")
    call_id = interview.get("call_id")
    
    _log_event("info", "handling_candidate_no_show", 
               interview_id=interview_id, agent_id=agent_id)
    
    # End the call session
    if call_id:
        try:
            end_call(call_id)
        except Exception as e:
            _log_event("warning", "failed_to_end_call", call_id=call_id, error=str(e))
    
    # Delete the agent
    if agent_id:
        try:
            delete_agent(agent_id)
        except Exception as e:
            _log_event("warning", "failed_to_delete_agent", agent_id=agent_id, error=str(e))
    
    # Update interview status
    await update_interview(
        interview_id,
        interview_status=InterviewStatus.INCOMPLETE.value,
        error_message="Candidate did not join within timeout period",
        interview_ended_at=datetime.now(tz=timezone.utc),
        completed_at=datetime.now(tz=timezone.utc),
    )
    
    # Remove from active tracking
    if interview_id in _active_interviews:
        del _active_interviews[interview_id]
    
    _log_event("info", "candidate_no_show_handled", interview_id=interview_id)


async def force_end_interview(interview_id: str, reason: str = "Forced end"):
    """Force end an interview (timeout, error, etc.)."""
    
    interview = await get_interview(interview_id)
    if not interview:
        return
    
    agent_id = interview.get("agent_id")
    call_id = interview.get("call_id")
    
    _log_event("info", "forcing_interview_end",
               interview_id=interview_id, reason=reason)
    
    # End the call
    if call_id:
        try:
            end_call(call_id)
        except Exception as e:
            _log_event("warning", "failed_to_end_call", call_id=call_id, error=str(e))
    
    # Delete the agent
    if agent_id:
        try:
            delete_agent(agent_id)
        except Exception as e:
            _log_event("warning", "failed_to_delete_agent", agent_id=agent_id, error=str(e))
    
    # Update status
    current_status = interview.get("interview_status")
    if current_status == "in_progress":
        new_status = InterviewStatus.INCOMPLETE.value
    else:
        new_status = InterviewStatus.CANCELLED.value
    
    await update_interview(
        interview_id,
        interview_status=new_status,
        error_message=reason,
        interview_ended_at=datetime.now(tz=timezone.utc),
        completed_at=datetime.now(tz=timezone.utc),
    )
    
    # Remove from active tracking
    if interview_id in _active_interviews:
        del _active_interviews[interview_id]


async def mark_candidate_joined(interview_id: str):
    """Mark that the candidate has joined the interview."""
    
    now = datetime.now(tz=timezone.utc)
    
    await update_interview(
        interview_id,
        candidate_joined_at=now,
        interview_started_at=now,
        interview_status=InterviewStatus.IN_PROGRESS.value,
    )
    
    # Update active tracking
    if interview_id in _active_interviews:
        _active_interviews[interview_id].candidate_joined = True
        _active_interviews[interview_id].candidate_joined_at = now
    
    _log_event("info", "candidate_joined", interview_id=interview_id)


def register_active_interview(
    interview_id: str,
    agent_id: str,
    call_id: str,
):
    """Register an interview as active (avatar has joined)."""
    
    _active_interviews[interview_id] = ActiveInterview(
        interview_id=interview_id,
        agent_id=agent_id,
        call_id=call_id,
        started_at=datetime.now(tz=timezone.utc),
    )
    
    _log_event("info", "interview_registered_active",
               interview_id=interview_id, agent_id=agent_id)


def unregister_active_interview(interview_id: str):
    """Remove interview from active tracking."""
    if interview_id in _active_interviews:
        del _active_interviews[interview_id]


def get_active_interview_ids() -> List[str]:
    """Get list of currently active interview IDs."""
    return list(_active_interviews.keys())


# =============================================================================
# CLEANUP
# =============================================================================

async def cleanup_stale_interviews():
    """
    Clean up interviews that are stuck in intermediate states.
    Run this periodically (e.g., every 5 minutes).
    """
    now = datetime.now(tz=timezone.utc)
    stale_threshold = now - timedelta(hours=2)
    
    _log_event("info", "cleanup_stale_interviews_started")
    
    # Find interviews stuck in waiting_for_candidate or in_progress
    for status in ["waiting_for_candidate", "in_progress"]:
        interviews, _ = await list_interviews(status=status, limit=50)
        
        for interview in interviews:
            interview_id = interview.get("interview_id")
            avatar_joined_at = interview.get("avatar_joined_at")
            
            if avatar_joined_at and avatar_joined_at < stale_threshold:
                _log_event("warning", "cleaning_up_stale_interview",
                           interview_id=interview_id,
                           status=status,
                           avatar_joined_at=avatar_joined_at.isoformat() if avatar_joined_at else None)
                
                await force_end_interview(
                    interview_id,
                    reason=f"Cleanup: Interview stuck in {status} for >2 hours"
                )
    
    _log_event("info", "cleanup_stale_interviews_completed")


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Interview Manager CLI')
    parser.add_argument('--cleanup', action='store_true', help='Run cleanup of stale interviews')
    parser.add_argument('--active', action='store_true', help='Show active interviews')
    parser.add_argument('--end', type=str, help='Force end an interview by ID')
    
    args = parser.parse_args()
    
    async def main():
        if args.cleanup:
            print("Running cleanup...")
            await cleanup_stale_interviews()
            print("Done!")
        
        elif args.active:
            count = await get_active_interviews_count()
            print(f"Active interviews (in_progress): {count}")
            
            interviews, _ = await list_interviews(status="in_progress", limit=20)
            for i in interviews:
                print(f"  - {i['interview_id']}: {i.get('candidate_name')} (started: {i.get('interview_started_at')})")
        
        elif args.end:
            print(f"Force ending interview: {args.end}")
            await force_end_interview(args.end, reason="Manual force end via CLI")
            print("Done!")
        
        else:
            parser.print_help()
    
    asyncio.run(main())
