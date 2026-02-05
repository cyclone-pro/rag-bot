#!/usr/bin/env python3
"""
Debug script to check interview status and scheduler queries.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

from db import get_interview, get_pending_interviews, list_interviews


async def debug_interviews():
    print("=" * 60)
    print("DEBUGGING INTERVIEWS")
    print("=" * 60)
    
    now = datetime.now(tz=timezone.utc)
    print(f"\nCurrent time (UTC): {now.isoformat()}")
    
    # 1. List all recent interviews
    print("\n1. All recent interviews:")
    interviews, total = await list_interviews(limit=10)
    print(f"   Found {total} total interviews")
    
    for i in interviews:
        print(f"\n   Interview: {i['interview_id']}")
        print(f"   Status: {i['interview_status']}")
        print(f"   Scheduled: {i['scheduled_time']}")
        print(f"   Candidate: {i.get('candidate_name')}")
        
        # Check if scheduled_time is in the future
        scheduled = i['scheduled_time']
        if scheduled:
            if scheduled.tzinfo is None:
                # Assume UTC if no timezone
                scheduled = scheduled.replace(tzinfo=timezone.utc)
            
            if scheduled > now:
                diff = scheduled - now
                print(f"   ⏳ Starts in: {int(diff.total_seconds() // 60)} minutes")
            else:
                diff = now - scheduled
                print(f"   ⚠️ Was scheduled {int(diff.total_seconds() // 60)} minutes ago")
    
    # 2. Check pending interviews (what scheduler sees)
    print("\n" + "=" * 60)
    print("2. Pending interviews (scheduler view):")
    
    # What get_upcoming_interviews uses
    future = now + timedelta(hours=24)
    pending = await get_pending_interviews(before_time=future, after_time=now)
    
    print(f"   Query: status='scheduled' AND scheduled_time BETWEEN {now.isoformat()} AND {future.isoformat()}")
    print(f"   Found: {len(pending)} interviews")
    
    for p in pending:
        print(f"\n   - {p['interview_id']}: {p.get('candidate_name')}")
        print(f"     Status: {p['interview_status']}")
        print(f"     Scheduled: {p['scheduled_time']}")
    
    # 3. Check a specific interview
    print("\n" + "=" * 60)
    print("3. Check specific interview (enter ID or press Enter to skip):")
    
    # Check the most recent one
    if interviews:
        latest = interviews[0]
        print(f"\n   Latest interview: {latest['interview_id']}")
        print(f"   Status: {latest['interview_status']}")
        print(f"   Scheduled: {latest['scheduled_time']}")
        print(f"   Meeting URL: {latest.get('meeting_url', 'N/A')}")
        print(f"   Organization ID: {latest.get('organization_id', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(debug_interviews())