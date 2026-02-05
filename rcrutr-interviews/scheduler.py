"""
Scheduler for triggering interviews at their scheduled times.

This module ensures avatars join meetings at the correct scheduled time,
not immediately when the interview is created.

Can be run as:
1. Standalone worker: python scheduler.py --loop
2. One-time check: python scheduler.py --once
3. Cloud Scheduler trigger: POST /api/scheduler/check
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List

from config import SERVICE_NAME
from db import get_pending_interviews, update_interview_status, get_interview
from models import InterviewStatus

logger = logging.getLogger(f"{SERVICE_NAME}_scheduler")


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


async def check_and_start_interviews():
    """
    Check for interviews that should start NOW and trigger them.
    
    This function looks for interviews:
    - With status 'scheduled' or 'meeting_created'
    - With scheduled_time in the past or within next 2 minutes
    
    It does NOT start interviews scheduled for the future.
    """
    now = datetime.now(tz=timezone.utc)
    window_start = now - timedelta(hours=1)  # Don't process very old ones
    window_end = now + timedelta(minutes=2)   # Start up to 2 min early
    
    _log_event("info", "scheduler_check_start", 
               current_time=now.isoformat(),
               window_start=window_start.isoformat(),
               window_end=window_end.isoformat())
    
    try:
        # Get pending interviews within the window
        interviews = await get_pending_interviews(
            before_time=window_end,
            after_time=window_start,
        )
        
        if not interviews:
            _log_event("info", "scheduler_no_pending_interviews")
            return []
        
        _log_event("info", "scheduler_found_interviews", count=len(interviews))
        
        started = []
        for interview in interviews:
            interview_id = interview["interview_id"]
            scheduled_time = interview.get("scheduled_time")
            status = interview.get("interview_status")
            
            # Only start if it's time
            if scheduled_time and scheduled_time <= window_end:
                _log_event("info", "scheduler_starting_interview",
                           interview_id=interview_id,
                           scheduled_time=scheduled_time.isoformat() if scheduled_time else None,
                           current_status=status)
                
                # Import here to avoid circular imports
                from start_interview import start_interview_async
                
                # Start the interview (sends avatar to meeting)
                asyncio.create_task(start_interview_async(interview_id))
                started.append(interview_id)
            else:
                _log_event("info", "scheduler_interview_not_ready",
                           interview_id=interview_id,
                           scheduled_time=scheduled_time.isoformat() if scheduled_time else None,
                           message="Scheduled for future")
        
        return started
        
    except Exception as e:
        _log_event("error", "scheduler_check_failed", error=str(e))
        return []


async def get_upcoming_interviews(hours: int = 24) -> List[dict]:
    """
    Get interviews scheduled in the next N hours.
    Also includes any past interviews still in 'scheduled' status (missed).
    """
    now = datetime.now(tz=timezone.utc)
    future = now + timedelta(hours=hours)
    past = now - timedelta(hours=1)  # Include interviews up to 1 hour overdue
    
    return await get_pending_interviews(
        before_time=future,
        after_time=past,  # Changed from 'now' to include missed interviews
    )


async def run_scheduler_loop(interval_seconds: int = 30):
    """
    Run the scheduler in a continuous loop.
    
    This is useful for local development or running as a standalone worker.
    In production, use Cloud Scheduler to call the /api/scheduler/check endpoint.
    """
    _log_event("info", "scheduler_loop_starting", interval_seconds=interval_seconds)
    
    while True:
        try:
            started = await check_and_start_interviews()
            if started:
                _log_event("info", "scheduler_loop_started_interviews", 
                           count=len(started), interview_ids=started)
        except Exception as e:
            _log_event("error", "scheduler_loop_error", error=str(e))
        
        await asyncio.sleep(interval_seconds)


# =============================================================================
# CLOUD TASKS INTEGRATION
# =============================================================================

def create_interview_task(
    interview_id: str,
    scheduled_time: datetime,
    service_url: str,
    project_id: str = "taqforce",
    location: str = "us-central1",
    queue: str = "interview-scheduler",
) -> str:
    """
    Create a Cloud Task to start an interview at the scheduled time.
    
    Args:
        interview_id: The interview to start
        scheduled_time: When to execute the task
        service_url: The Cloud Run service URL
        project_id: GCP project ID
        location: GCP region
        queue: Cloud Tasks queue name
    
    Returns:
        Task name if created successfully
    """
    try:
        from google.cloud import tasks_v2
        from google.protobuf import timestamp_pb2
        
        client = tasks_v2.CloudTasksClient()
        
        # Build the task queue path
        parent = client.queue_path(project_id, location, queue)
        
        # Build the task
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{service_url}/api/interview/{interview_id}/start",
                "headers": {
                    "Content-Type": "application/json",
                },
                "oidc_token": {
                    "service_account_email": f"{project_id}@appspot.gserviceaccount.com",
                },
            },
        }
        
        # Set schedule time
        if scheduled_time > datetime.now(tz=timezone.utc):
            timestamp = timestamp_pb2.Timestamp()
            timestamp.FromDatetime(scheduled_time)
            task["schedule_time"] = timestamp
        
        # Create the task
        response = client.create_task(parent=parent, task=task)
        
        _log_event("info", "cloud_task_created",
                   interview_id=interview_id,
                   task_name=response.name,
                   scheduled_time=scheduled_time.isoformat())
        
        return response.name
        
    except ImportError:
        _log_event("warning", "cloud_tasks_not_available")
        return ""
    except Exception as e:
        _log_event("error", "cloud_task_creation_failed",
                   interview_id=interview_id, error=str(e))
        return ""


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RCRUTR Interview Scheduler")
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (useful for cron)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
    )
    
    if args.once:
        # Run once
        asyncio.run(check_and_start_interviews())
    else:
        # Run continuous loop
        asyncio.run(run_scheduler_loop(interval_seconds=args.interval))
