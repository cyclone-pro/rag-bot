#!/usr/bin/env python3
"""
Simple Zoom meeting test - just creates a meeting to verify credentials work.

Run: python test_zoom_only.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check credentials
ZOOM_ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID", "")
ZOOM_CLIENT_ID = os.getenv("ZOOM_CLIENT_ID", "")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET", "")

print("=" * 60)
print("ZOOM CREDENTIALS CHECK")
print("=" * 60)
print(f"ZOOM_ACCOUNT_ID: {'✅ Set' if ZOOM_ACCOUNT_ID else '❌ Missing'}")
print(f"ZOOM_CLIENT_ID: {'✅ Set' if ZOOM_CLIENT_ID else '❌ Missing'}")
print(f"ZOOM_CLIENT_SECRET: {'✅ Set' if ZOOM_CLIENT_SECRET else '❌ Missing'}")
print()

if not all([ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET]):
    print("❌ Missing Zoom credentials!")
    print()
    print("To get Zoom credentials:")
    print("1. Go to https://marketplace.zoom.us/")
    print("2. Click 'Develop' → 'Build App'")
    print("3. Choose 'Server-to-Server OAuth'")
    print("4. Create app, add scopes:")
    print("   - meeting:write:admin")
    print("   - meeting:read:admin")
    print("   - user:read:admin")
    print("5. Activate the app")
    print("6. Copy credentials to .env file")
    sys.exit(1)

# Import zoom client
from zoom_client import create_meeting, check_zoom_connection

print("=" * 60)
print("TESTING ZOOM API CONNECTION")
print("=" * 60)

ok, msg = check_zoom_connection()
print(f"Connection: {'✅ Success' if ok else '❌ Failed'}")
print(f"Message: {msg}")
print()

if not ok:
    print("❌ Cannot connect to Zoom API")
    print("   Check your credentials are correct")
    sys.exit(1)

print("=" * 60)
print("CREATING TEST MEETING")
print("=" * 60)

# Schedule for 5 minutes from now
scheduled_time = datetime.now(tz=timezone.utc) + timedelta(minutes=5)

meeting = create_meeting(
    topic="RCRUTR Test - Python Developer Interview",
    start_time=scheduled_time,
    duration_minutes=30,
    timezone="UTC",
    waiting_room=True,
)

if meeting:
    print("✅ Meeting created successfully!")
    print()
    print("=" * 60)
    print("MEETING DETAILS")
    print("=" * 60)
    print(f"Meeting ID:     {meeting.id}")
    print(f"Topic:          {meeting.topic}")
    print(f"Start Time:     {scheduled_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Duration:       {meeting.duration} minutes")
    print(f"Passcode:       {meeting.password}")
    print()
    print(f"JOIN URL (share with candidate):")
    print(f"  {meeting.join_url}")
    print()
    print(f"HOST URL (for avatar/host):")
    print(f"  {meeting.start_url[:80]}...")
    print()
else:
    print("❌ Failed to create meeting")
    sys.exit(1)
