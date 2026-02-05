#!/usr/bin/env python3
"""
Scheduler for auto-starting interviews at their scheduled times.

IMPORTANT: This scheduler needs to run continuously to auto-start interviews!

Run modes:
1. Background loop: python scheduler.py --loop
2. One-time check: python scheduler.py --once  
3. API trigger: POST /api/scheduler/check

The scheduler checks every 30 seconds for interviews that need to start.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

from config import SERVICE_NAME
from db import get_pending_interviews, update_interview_status, get_interview, update_interview
from models import InterviewStatus

logger = logging.getLogger(f"{SERVICE_NAME}_scheduler")

# How often to check for interviews (seconds)
CHECK_INTERVAL = 30

# How early to start (minutes before scheduled time)
START_EARLY_MINUTES = 1

# How late is too late (don't start if more than this many minutes overdue)
MAX_OVERDUE_MINUTES = 30


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


def _print_status(message: str, status: str = "info"):
    """Print status message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icon = "✅" if status == "success" else "❌" if status == "error" else "⏳" if status == "pending" else "ℹ️"
    print(f"[{timestamp}] {icon} {message}")


async def _log_to_db(interview_id: str, event_type: str, status: str, message: str, data: dict = None):
    """Log event to database if logger module is available."""
    try:
        from interview_logger import log_event
        await log_event(
            interview_id=interview_id,
            event_type=event_type,
            status=status,
            component="scheduler",
            message=message,
            data=data,
        )
    except Exception:
        pass  # Logger not available, continue


async def check_and_start_interviews() -> List[str]:
    """
    Check for interviews due to start and trigger them.
    
    Returns:
        List of interview IDs that were started
    """
    now = datetime.now(tz=timezone.utc)
    
    # Window: from MAX_OVERDUE ago to START_EARLY from now
    window_start = now - timedelta(minutes=MAX_OVERDUE_MINUTES)
    window_end = now + timedelta(minutes=START_EARLY_MINUTES)
    
    _print_status(f"Checking for interviews between {window_start.strftime('%H:%M')} and {window_end.strftime('%H:%M')} UTC")
    
    try:
        # Get pending interviews
        interviews = await get_pending_interviews(
            before_time=window_end,
            after_time=window_start,
        )
        
        if not interviews:
            _print_status("No interviews ready to start", "info")
            return []
        
        _print_status(f"Found {len(interviews)} interview(s) ready to start", "pending")
        
        started = []
        for interview in interviews:
            interview_id = interview["interview_id"]
            scheduled_time = interview.get("scheduled_time")
            candidate_name = interview.get("candidate_name", "Unknown")
            status = interview.get("interview_status")
            
            # Skip if already being processed
            if status not in ("scheduled", "meeting_created"):
                _print_status(f"Skipping {interview_id} - status is {status}", "info")
                continue
            
            # Calculate how overdue
            if scheduled_time:
                if scheduled_time.tzinfo is None:
                    scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
                
                delta = now - scheduled_time
                minutes_overdue = int(delta.total_seconds() / 60)
                
                if minutes_overdue > 0:
                    _print_status(f"Interview {interview_id} ({candidate_name}) is {minutes_overdue}min overdue - starting now!", "pending")
                else:
                    _print_status(f"Interview {interview_id} ({candidate_name}) is due in {-minutes_overdue}min - starting!", "pending")
            
            # Log to database
            await _log_to_db(
                interview_id=interview_id,
                event_type="scheduler_triggering",
                status="pending",
                message=f"Scheduler triggering interview start for {candidate_name}",
                data={"scheduled_time": scheduled_time.isoformat() if scheduled_time else None}
            )
            
            # Actually start the interview
            try:
                success = await start_interview_now(interview_id)
                
                if success:
                    started.append(interview_id)
                    _print_status(f"✅ Started interview {interview_id} ({candidate_name})", "success")
                    await _log_to_db(
                        interview_id=interview_id,
                        event_type="scheduler_started",
                        status="success",
                        message=f"Interview started successfully",
                    )
                else:
                    _print_status(f"❌ Failed to start interview {interview_id}", "error")
                    await _log_to_db(
                        interview_id=interview_id,
                        event_type="scheduler_failed",
                        status="failed",
                        message=f"Failed to start interview",
                    )
                    
            except Exception as e:
                _print_status(f"❌ Error starting interview {interview_id}: {e}", "error")
                await _log_to_db(
                    interview_id=interview_id,
                    event_type="scheduler_error",
                    status="failed",
                    message=str(e),
                )
        
        return started
        
    except Exception as e:
        _print_status(f"Scheduler error: {e}", "error")
        _log_event("error", "scheduler_check_failed", error=str(e))
        return []


async def start_interview_now(interview_id: str) -> bool:
    """
    Start an interview immediately.
    
    This imports and calls the start_interview function.
    """
    try:
        # Import here to avoid circular imports
        from start_interview import start_interview
        
        _print_status(f"Starting interview {interview_id}...", "pending")
        
        # Run the start_interview function
        success = await start_interview(interview_id)
        
        return success
        
    except Exception as e:
        _print_status(f"Error in start_interview: {e}", "error")
        
        # Update status to failed
        await update_interview(
            interview_id,
            interview_status=InterviewStatus.FAILED.value,
            error_message=f"Scheduler failed to start: {str(e)}",
        )
        
        return False


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
        after_time=past,
    )


async def run_scheduler_loop(interval_seconds: int = CHECK_INTERVAL):
    """
    Run the scheduler in a continuous loop.
    
    This is the main entry point for keeping interviews running automatically.
    """
    print()
    print("=" * 60)
    print("🚀 RCRUTR INTERVIEW SCHEDULER STARTED")
    print("=" * 60)
    print(f"Check interval: {interval_seconds} seconds")
    print(f"Start early: {START_EARLY_MINUTES} minute(s)")
    print(f"Max overdue: {MAX_OVERDUE_MINUTES} minutes")
    print()
    print("Watching for interviews to start...")
    print("Press Ctrl+C to stop")
    print()
    
    await _log_to_db(
        interview_id=None,
        event_type="scheduler_loop_started",
        status="success",
        message=f"Scheduler loop started with {interval_seconds}s interval",
    )
    
    while True:
        try:
            started = await check_and_start_interviews()
            
            if started:
                print(f"\n✅ Started {len(started)} interview(s): {', '.join(started)}\n")
                
        except KeyboardInterrupt:
            print("\n\nScheduler stopped by user.")
            break
        except Exception as e:
            _print_status(f"Loop error: {e}", "error")
        
        await asyncio.sleep(interval_seconds)


async def run_once():
    """Run the scheduler check once and exit."""
    print()
    print("=" * 60)
    print("🔍 RUNNING SCHEDULER CHECK (ONE-TIME)")
    print("=" * 60)
    print()
    
    started = await check_and_start_interviews()
    
    print()
    if started:
        print(f"✅ Started {len(started)} interview(s)")
        for iid in started:
            print(f"   - {iid}")
    else:
        print("No interviews were started.")
    print()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RCRUTR Interview Scheduler')
    parser.add_argument('--loop', action='store_true', 
                        help='Run continuously, checking every 30 seconds')
    parser.add_argument('--once', action='store_true',
                        help='Check once and exit')
    parser.add_argument('--interval', type=int, default=CHECK_INTERVAL,
                        help=f'Check interval in seconds (default: {CHECK_INTERVAL})')
    parser.add_argument('--upcoming', action='store_true',
                        help='Show upcoming interviews')
    
    args = parser.parse_args()
    
    if args.upcoming:
        async def show_upcoming():
            interviews = await get_upcoming_interviews(hours=24)
            print()
            print("=" * 60)
            print("📅 UPCOMING INTERVIEWS (next 24 hours)")
            print("=" * 60)
            
            if not interviews:
                print("No upcoming interviews.")
            else:
                for i in interviews:
                    scheduled = i.get('scheduled_time')
                    if scheduled:
                        time_str = scheduled.strftime("%Y-%m-%d %H:%M UTC")
                    else:
                        time_str = "Unknown"
                    
                    print(f"\n  {i['interview_id']}")
                    print(f"  Candidate: {i.get('candidate_name', 'Unknown')}")
                    print(f"  Scheduled: {time_str}")
                    print(f"  Status: {i.get('interview_status')}")
            print()
        
        asyncio.run(show_upcoming())
    
    elif args.loop:
        asyncio.run(run_scheduler_loop(interval_seconds=args.interval))
    
    elif args.once:
        asyncio.run(run_once())
    
    else:
        print()
        print("RCRUTR Interview Scheduler")
        print()
        print("Usage:")
        print("  python scheduler.py --loop      # Run continuously (recommended)")
        print("  python scheduler.py --once      # Check once and exit")
        print("  python scheduler.py --upcoming  # Show upcoming interviews")
        print()
        print("For interviews to auto-start, run: python scheduler.py --loop")
        print()


if __name__ == "__main__":
    main()