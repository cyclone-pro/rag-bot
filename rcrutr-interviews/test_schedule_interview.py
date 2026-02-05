#!/usr/bin/env python3
"""
Test script to schedule a Python Developer interview on Zoom.

This script:
1. Creates test candidate data (or uses real candidate from Milvus)
2. Creates test job data (or uses real job from Milvus)
3. Schedules an interview via the API
4. Outputs the meeting link

Run: python test_schedule_interview.py
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta, timezone

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from config import (
    ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET,
    BEY_API_KEY, DATABASE_URL, MILVUS_HOST,
)


def check_credentials():
    """Check if all required credentials are set."""
    print("=" * 60)
    print("CREDENTIAL CHECK")
    print("=" * 60)
    
    checks = {
        "ZOOM_ACCOUNT_ID": bool(ZOOM_ACCOUNT_ID),
        "ZOOM_CLIENT_ID": bool(ZOOM_CLIENT_ID),
        "ZOOM_CLIENT_SECRET": bool(ZOOM_CLIENT_SECRET),
        "BEY_API_KEY": bool(BEY_API_KEY),
        "DATABASE_URL": bool(DATABASE_URL),
        "MILVUS_HOST": bool(MILVUS_HOST),
    }
    
    all_ok = True
    for key, ok in checks.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {key}")
        if not ok:
            all_ok = False
    
    print()
    return all_ok


def test_zoom_connection():
    """Test Zoom API connection."""
    print("=" * 60)
    print("TESTING ZOOM CONNECTION")
    print("=" * 60)
    
    try:
        from zoom_client import check_zoom_connection
        ok, msg = check_zoom_connection()
        
        if ok:
            print(f"  ✅ Zoom API connected: {msg}")
            return True
        else:
            print(f"  ❌ Zoom API failed: {msg}")
            return False
    except Exception as e:
        print(f"  ❌ Zoom test error: {e}")
        return False


def test_db_connection():
    """Test database connection."""
    print("=" * 60)
    print("TESTING DATABASE CONNECTION")
    print("=" * 60)
    
    try:
        from db import check_db_connection
        ok, msg = asyncio.run(check_db_connection())
        
        if ok:
            print(f"  ✅ Database connected: {msg}")
            return True
        else:
            print(f"  ❌ Database failed: {msg}")
            return False
    except Exception as e:
        print(f"  ❌ Database test error: {e}")
        return False


def test_milvus_connection():
    """Test Milvus connection."""
    print("=" * 60)
    print("TESTING MILVUS CONNECTION")
    print("=" * 60)
    
    try:
        from milvus_client import check_milvus_connection
        ok, msg = check_milvus_connection()
        
        if ok:
            print(f"  ✅ Milvus connected: {msg}")
            return True
        else:
            print(f"  ❌ Milvus failed: {msg}")
            return False
    except Exception as e:
        print(f"  ❌ Milvus test error: {e}")
        return False


def create_zoom_meeting():
    """Create a test Zoom meeting."""
    print("=" * 60)
    print("CREATING ZOOM MEETING")
    print("=" * 60)
    
    from zoom_client import create_meeting
    
    # Schedule for 10 minutes from now
    scheduled_time = datetime.now(tz=timezone.utc) + timedelta(minutes=10)
    
    meeting = create_meeting(
        topic="RCRUTR Test Interview - Python Developer",
        start_time=scheduled_time,
        duration_minutes=30,
        timezone="UTC",
        waiting_room=True,
    )
    
    if meeting:
        print(f"  ✅ Meeting created successfully!")
        print(f"  Meeting ID: {meeting.id}")
        print(f"  Join URL: {meeting.join_url}")
        print(f"  Passcode: {meeting.password}")
        print(f"  Start Time: {scheduled_time.isoformat()}")
        return meeting
    else:
        print(f"  ❌ Failed to create meeting")
        return None


async def schedule_full_interview():
    """Schedule a complete interview with candidate and job data."""
    print("=" * 60)
    print("SCHEDULING FULL INTERVIEW")
    print("=" * 60)
    
    from db import insert_interview, generate_interview_id
    from zoom_client import create_meeting
    from interview_prompt import generate_interview_questions, format_questions_for_db
    from models import CandidateData, JobData
    
    # Test candidate data (Python backend developer)
    candidate = CandidateData(
        candidate_id="test_cand_python_001",
        name="Alex Johnson",
        email="alex.johnson@example.com",
        phone="+1-555-0123",
        semantic_summary="""
        Mid-level Python developer with 4 years of experience in backend development.
        Strong expertise in FastAPI, Django, and Flask frameworks. Experienced with 
        vector databases including Milvus and Pinecone. Has worked extensively with 
        GCP services including Cloud Run, Cloud SQL, BigQuery, and Vertex AI.
        Built RAG systems and semantic search applications. Good understanding of
        microservices architecture and containerization with Docker and Kubernetes.
        """,
        current_tech_stack=["Python", "FastAPI", "PostgreSQL", "Milvus", "GCP", "Docker", "Redis"],
        top_5_skills_with_years="Python:4, FastAPI:3, PostgreSQL:4, GCP:3, Milvus:2",
        employment_history=[
            {
                "company": "TechCorp Inc",
                "title": "Backend Developer",
                "start_date": "2022-01",
                "end_date": "Present"
            },
            {
                "company": "StartupXYZ",
                "title": "Junior Developer",
                "start_date": "2020-06",
                "end_date": "2021-12"
            }
        ]
    )
    
    # Test job data (Python Developer position)
    job = JobData(
        job_id="test_job_python_001",
        title="Mid-Level Python Developer",
        company="Elite Solutions",
        department="Engineering",
        location="Remote / US",
        employment_type="Full-time",
        salary_range="$100,000 - $130,000",
        jd_text="""
Title: Mid-Level Python Developer
Seniority: Mid-Level (3-5 years)
Work Model: Remote
Location: US

About the Role:
We're looking for a talented Python Developer to join our backend team. 
You'll be working on building scalable APIs, implementing vector search 
capabilities, and deploying services on Google Cloud Platform.

Required Skills:
- Python (3+ years)
- FastAPI or Django
- PostgreSQL or similar relational database
- Vector databases (Milvus, Pinecone, or similar)
- GCP services (Cloud Run, Cloud SQL, BigQuery)
- Docker and containerization
- REST API design

Nice to Have:
- Experience with RAG systems and LLMs
- Kubernetes experience
- CI/CD pipelines
- Redis or caching systems

Responsibilities:
- Design and implement backend APIs using FastAPI
- Build and maintain vector search systems
- Deploy and manage services on GCP
- Write clean, tested, and documented code
- Collaborate with frontend and ML teams
- Participate in code reviews

Compensation: $100,000 - $130,000 + benefits
"""
    )
    
    # Schedule meeting for 10 minutes from now (for testing)
    scheduled_time = datetime.now(tz=timezone.utc) + timedelta(minutes=10)
    
    print(f"  Candidate: {candidate.name}")
    print(f"  Job: {job.title} at {job.company}")
    print(f"  Scheduled: {scheduled_time.isoformat()}")
    print()
    
    # Create Zoom meeting
    print("  Creating Zoom meeting...")
    meeting = create_meeting(
        topic=f"Interview: {candidate.name} - {job.title}",
        start_time=scheduled_time,
        duration_minutes=30,
        timezone="UTC",
        waiting_room=True,
    )
    
    if not meeting:
        print("  ❌ Failed to create Zoom meeting")
        return None
    
    print(f"  ✅ Zoom meeting created: {meeting.id}")
    
    # Generate interview questions
    print("  Generating interview questions...")
    questions = generate_interview_questions(candidate, job)
    questions_json = format_questions_for_db(questions)
    
    print(f"  ✅ Generated {len(questions_json)} questions")
    for q in questions_json:
        print(f"     Q{q['index']}: {q['question'][:60]}...")
    print()
    
    # Generate interview ID
    interview_id = generate_interview_id()
    
    # Insert into database
    print("  Inserting into database...")
    try:
        db_id, int_id = await insert_interview(
            interview_id=interview_id,
            candidate_id=candidate.candidate_id,
            job_id=job.job_id,
            scheduled_time=scheduled_time,
            timezone_str="UTC",
            avatar_key="zara",
            # Job data
            job_title=job.title,
            job_description=job.jd_text,
            job_company=job.company,
            job_location=job.location,
            # Candidate data
            candidate_name=candidate.name,
            candidate_email=candidate.email,
            candidate_phone=candidate.phone,
            candidate_skills={"top_skills": candidate.top_5_skills_with_years},
            candidate_summary=candidate.semantic_summary,
            candidate_tech_stack=candidate.current_tech_stack,
            candidate_employment_history=candidate.employment_history,
            # Meeting data
            meeting_id=meeting.id,
            meeting_url=meeting.join_url,
            meeting_passcode=meeting.password,
            meeting_host_url=meeting.start_url,
            # Questions
            questions=questions_json,
            total_questions=len(questions_json),
            # Notes
            notes="Test interview for Python Developer position",
        )
        print(f"  ✅ Interview saved to database (ID: {db_id})")
    except Exception as e:
        print(f"  ❌ Database insert failed: {e}")
        return None
    
    # Print summary
    print()
    print("=" * 60)
    print("INTERVIEW SCHEDULED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print(f"  Interview ID:    {interview_id}")
    print(f"  Candidate:       {candidate.name}")
    print(f"  Position:        {job.title}")
    print(f"  Scheduled Time:  {scheduled_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Avatar:          Zara")
    print()
    print("  ZOOM MEETING DETAILS:")
    print(f"  -------------------")
    print(f"  Meeting ID:      {meeting.id}")
    print(f"  Join URL:        {meeting.join_url}")
    print(f"  Passcode:        {meeting.password}")
    print()
    print("  NEXT STEPS:")
    print(f"  1. Share this link with candidate: {meeting.join_url}")
    print(f"  2. Start the interview: POST /api/interview/{interview_id}/start")
    print(f"  3. Or wait for scheduler to auto-start at scheduled time")
    print()
    
    return {
        "interview_id": interview_id,
        "meeting_url": meeting.join_url,
        "meeting_passcode": meeting.password,
        "scheduled_time": scheduled_time.isoformat(),
    }


def main():
    """Main test function."""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     RCRUTR INTERVIEWS - TEST INTERVIEW SCHEDULER           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Check credentials
    creds_ok = check_credentials()
    print()
    
    if not creds_ok:
        print("⚠️  Some credentials are missing. Please check your .env file.")
        print("   Zoom credentials are required to create meetings.")
        print()
        
        # Check if Zoom specifically is missing
        if not all([ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET]):
            print("   To get Zoom credentials:")
            print("   1. Go to https://marketplace.zoom.us/")
            print("   2. Build App → Server-to-Server OAuth")
            print("   3. Add scopes: meeting:write:admin, meeting:read:admin")
            print("   4. Copy Account ID, Client ID, Client Secret to .env")
            print()
            return
    
    # Test connections
    db_ok = test_db_connection()
    print()
    
    milvus_ok = test_milvus_connection()
    print()
    
    zoom_ok = test_zoom_connection()
    print()
    
    if not zoom_ok:
        print("❌ Zoom connection failed. Cannot create meeting.")
        print("   Please check your Zoom credentials in .env")
        return
    
    if not db_ok:
        print("❌ Database connection failed. Cannot save interview.")
        print("   Please check your DATABASE_URL in .env")
        return
    
    # Schedule the interview
    print()
    result = asyncio.run(schedule_full_interview())
    
    if result:
        print("=" * 60)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print(f"Meeting URL: {result['meeting_url']}")
        print(f"Passcode: {result['meeting_passcode']}")
        print()
    else:
        print("❌ Test failed. Check the errors above.")


if __name__ == "__main__":
    main()
