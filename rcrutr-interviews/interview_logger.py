"""
Interview Logger - Logs all events to database for debugging.

Usage:
    from interview_logger import log_event, get_recent_logs

    # Log an event
    await log_event(
        interview_id="int_xxx",
        event_type="avatar_joining",
        status="pending",
        component="bey_client",
        message="Sending avatar to Zoom meeting",
    )
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from config import DATABASE_URL


def _get_db_url() -> str:
    """Get database URL."""
    db_url = DATABASE_URL
    if not db_url:
        raise ValueError("DATABASE_URL not set")
    if db_url and not db_url.startswith("postgresql"):
        db_url = f"postgresql://backteam:Airecruiter1_@{db_url}:5432/recruiter_brain"
    return db_url


async def log_event(
    interview_id: Optional[str],
    event_type: str,
    status: str,
    component: str,
    message: str = None,
    data: Dict[str, Any] = None,
    error: str = None,
    error_stack: str = None,
) -> Optional[int]:
    """
    Log an interview event to the database.
    
    Args:
        interview_id: The interview ID (can be None for system events)
        event_type: Type of event (e.g., 'scheduled', 'avatar_joining', 'error')
        status: Status ('success', 'failed', 'pending', 'timeout')
        component: Component that generated the event
        message: Human-readable message
        data: Additional JSON data
        error: Error message if failed
        error_stack: Full stack trace if error
    
    Returns:
        Log entry ID or None if failed
    """
    db_url = _get_db_url()
    
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO interview_logs 
                    (interview_id, event_type, event_status, component, message, event_data, error_message, error_stack)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        interview_id,
                        event_type,
                        status,
                        component,
                        message,
                        Jsonb(data) if data else None,
                        error,
                        error_stack,
                    )
                )
                row = await cur.fetchone()
            await conn.commit()
        
        # Also print to console for immediate visibility
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_icon = "✅" if status == "success" else "❌" if status == "failed" else "⏳" if status == "pending" else "⚠️"
        print(f"[{timestamp}] {status_icon} [{component}] {event_type}: {message or ''}")
        
        return row[0] if row else None
        
    except Exception as e:
        # Fallback to console if DB fails
        print(f"[LOG ERROR] Failed to log to DB: {e}")
        print(f"[{component}] {event_type} ({status}): {message}")
        return None


async def log_error(
    interview_id: Optional[str],
    component: str,
    message: str,
    exception: Exception = None,
) -> Optional[int]:
    """Convenience function to log errors with stack trace."""
    error_stack = traceback.format_exc() if exception else None
    error_msg = str(exception) if exception else None
    
    return await log_event(
        interview_id=interview_id,
        event_type="error",
        status="failed",
        component=component,
        message=message,
        error=error_msg,
        error_stack=error_stack,
    )


async def get_recent_logs(
    interview_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get recent log entries."""
    db_url = _get_db_url()
    
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                if interview_id:
                    await cur.execute(
                        """
                        SELECT * FROM interview_logs 
                        WHERE interview_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (interview_id, limit)
                    )
                else:
                    await cur.execute(
                        """
                        SELECT * FROM interview_logs 
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (limit,)
                    )
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[LOG ERROR] Failed to get logs: {e}")
        return []


async def get_logs_for_interview(interview_id: str) -> List[Dict[str, Any]]:
    """Get all logs for a specific interview."""
    return await get_recent_logs(interview_id=interview_id, limit=200)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description='View interview logs')
    parser.add_argument('--interview', '-i', type=str, help='Interview ID to filter')
    parser.add_argument('--limit', '-n', type=int, default=20, help='Number of logs to show')
    parser.add_argument('--tail', '-f', action='store_true', help='Follow mode (refresh every 2s)')
    
    args = parser.parse_args()
    
    async def show_logs():
        logs = await get_recent_logs(interview_id=args.interview, limit=args.limit)
        
        if not logs:
            print("No logs found.")
            return
        
        print()
        print("=" * 80)
        print(f"{'TIME':<12} {'STATUS':<8} {'COMPONENT':<15} {'EVENT':<20} MESSAGE")
        print("=" * 80)
        
        for log in reversed(logs):  # Show oldest first
            time_str = log['created_at'].strftime("%H:%M:%S") if log['created_at'] else "??:??:??"
            status = log['event_status'][:7]
            component = (log['component'] or "")[:14]
            event = (log['event_type'] or "")[:19]
            message = (log['message'] or "")[:50]
            
            # Color based on status
            if log['event_status'] == 'success':
                status_icon = "✅"
            elif log['event_status'] == 'failed':
                status_icon = "❌"
            elif log['event_status'] == 'pending':
                status_icon = "⏳"
            else:
                status_icon = "•"
            
            print(f"{time_str:<12} {status_icon} {status:<6} {component:<15} {event:<20} {message}")
            
            if log.get('error_message'):
                print(f"             └── ERROR: {log['error_message'][:60]}")
        
        print("=" * 80)
        print(f"Showing {len(logs)} logs" + (f" for interview {args.interview}" if args.interview else ""))
    
    async def tail_logs():
        last_id = 0
        print("Tailing logs... (Ctrl+C to stop)")
        print()
        
        while True:
            logs = await get_recent_logs(interview_id=args.interview, limit=10)
            
            for log in reversed(logs):
                if log['id'] > last_id:
                    last_id = log['id']
                    time_str = log['created_at'].strftime("%H:%M:%S")
                    status_icon = "✅" if log['event_status'] == 'success' else "❌" if log['event_status'] == 'failed' else "⏳"
                    print(f"[{time_str}] {status_icon} [{log['component']}] {log['event_type']}: {log.get('message', '')}")
            
            await asyncio.sleep(2)
    
    if args.tail:
        asyncio.run(tail_logs())
    else:
        asyncio.run(show_logs())
