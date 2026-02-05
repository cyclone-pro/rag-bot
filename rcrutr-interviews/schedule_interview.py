#!/usr/bin/env python3
"""
Schedule Interview - Creates Zoom meeting but agent joins at scheduled time.

This is the proper production flow:
1. User schedules interview for future time
2. Zoom meeting is created immediately
3. Candidate receives meeting link via email
4. At scheduled time, scheduler triggers agent to join
5. Agent conducts interview

Usage:
  # Schedule for specific time
  python schedule_interview.py --candidate cand_123 --job job_456 --time "2026-02-03T14:00:00Z"
  
  # Schedule for 1 hour from now (for testing)
  python schedule_interview.py --candidate cand_123 --job job_456 --in-minutes 60
  
  # Use test data
  python schedule_interview.py --test --in-minutes 5
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

from db import insert_interview, generate_interview_id, get_interview
from zoom_client import create_meeting
from interview_prompt import generate_interview_questions, format_questions_for_db
from milvus_client import get_candidate, get_job
from models import CandidateData, JobData, InterviewStatus
from config import BEY_LLM_API_ID


def get_test_candidate() -> CandidateData:
    """Return test candidate data."""
    return CandidateData(
        candidate_id="test_cand_python_001",
        name="Alex Johnson",
        email="alex.johnson@example.com",
        phone="+1-555-0123",
        semantic_summary="""
        Mid-level Python developer with 4 years of experience in backend development.
        Strong expertise in FastAPI, Django, and Flask frameworks. Experienced with 
        vector databases including Milvus and Pinecone. Has worked extensively with 
        GCP services including Cloud Run, Cloud SQL, BigQuery, and Vertex AI.
        """,
        current_tech_stack=["Python", "FastAPI", "PostgreSQL", "Milvus", "GCP", "Docker"],
        top_5_skills_with_years="Python:4, FastAPI:3, PostgreSQL:4, GCP:3, Milvus:2",
        employment_history=[
            {"company": "TechCorp Inc", "title": "Backend Developer", "start_date": "2022-01", "end_date": "Present"},
        ]
    )


def get_test_job() -> JobData:
    """Return test job data."""
    return JobData(
        job_id="test_job_python_001",
        title="Mid-Level Python Developer",
        company="Elite Solutions",
        location="Remote / US",
        jd_text="""
Title: Mid-Level Python Developer
Required Skills: Python, FastAPI, PostgreSQL, Vector databases, GCP, Docker
Responsibilities: Backend APIs, Vector search systems, GCP deployment
Salary: $100,000 - $130,000
"""
    )


async def schedule_interview(
    candidate_id: str,
    job_id: str,
    scheduled_time: datetime,
    *,
    candidate_name: str = None,
    candidate_email: str = None,
    use_test_data: bool = False,
    avatar: str = "zara",
    notes: str = None,
    organization_id: str = None,  # UUID as string
) -> dict:
    """
    Schedule an interview for a future time.
    
    This creates the Zoom meeting immediately but the agent will only
    join at the scheduled time (triggered by the scheduler).
    
    Returns:
        dict with interview_id, meeting_url, scheduled_time, etc.
    """
    
    print("=" * 60)
    print("SCHEDULING INTERVIEW")
    print("=" * 60)
    
    # 1. Get candidate data
    print("\n1. Getting candidate data...")
    if use_test_data:
        candidate = get_test_candidate()
    else:
        candidate = get_candidate(candidate_id)
        if not candidate:
            print(f"   ❌ Candidate not found: {candidate_id}")
            return None
    
    # Override name/email if provided
    if candidate_name:
        candidate.name = candidate_name
    if candidate_email:
        candidate.email = candidate_email
    
    print(f"   Candidate: {candidate.name}")
    print(f"   Email: {candidate.email}")
    
    # 2. Get job data
    print("\n2. Getting job data...")
    if use_test_data:
        job = get_test_job()
    else:
        job = get_job(job_id)
        if not job:
            print(f"   ❌ Job not found: {job_id}")
            return None
    
    print(f"   Job: {job.title}")
    print(f"   Company: {job.company}")
    
    # 3. Create Zoom meeting
    print("\n3. Creating Zoom meeting...")
    print(f"   Scheduled for: {scheduled_time.isoformat()}")
    
    # Calculate time until interview
    now = datetime.now(tz=timezone.utc)
    time_until = scheduled_time - now
    hours, remainder = divmod(int(time_until.total_seconds()), 3600)
    minutes = remainder // 60
    print(f"   Time until interview: {hours}h {minutes}m")
    
    meeting = create_meeting(
        topic=f"Interview: {candidate.name} - {job.title}",
        start_time=scheduled_time,
        duration_minutes=30,
        waiting_room=False,  # Disabled for autonomous operation
        join_before_host=True,
    )
    
    if not meeting:
        print("   ❌ Failed to create Zoom meeting")
        return None
    
    print(f"   ✅ Meeting created: {meeting.id}")
    
    # 4. Generate questions
    print("\n4. Generating interview questions...")
    questions = generate_interview_questions(candidate, job)
    questions_json = format_questions_for_db(questions)
    print(f"   ✅ Generated {len(questions_json)} questions")
    
    # 5. Save to database
    print("\n5. Saving to database...")
    interview_id = generate_interview_id()
    
    await insert_interview(
        interview_id=interview_id,
        candidate_id=candidate.candidate_id,
        job_id=job.job_id,
        scheduled_time=scheduled_time,
        timezone_str="UTC",
        avatar_key=avatar,
        organization_id=organization_id,
        job_title=job.title,
        job_description=job.jd_text,
        job_company=job.company,
        job_location=job.location,
        candidate_name=candidate.name,
        candidate_email=candidate.email,
        candidate_phone=candidate.phone,
        candidate_skills={"top_skills": candidate.top_5_skills_with_years},
        candidate_summary=candidate.semantic_summary,
        candidate_tech_stack=candidate.current_tech_stack,
        candidate_employment_history=candidate.employment_history,
        meeting_id=meeting.id,
        meeting_url=meeting.join_url,
        meeting_passcode=meeting.password,
        meeting_host_url=meeting.start_url,
        questions=questions_json,
        total_questions=len(questions_json),
        notes=notes,
    )
    
    print(f"   ✅ Interview saved: {interview_id}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ INTERVIEW SCHEDULED!")
    print("=" * 60)
    print(f"""
Interview Details:
  ID:            {interview_id}
  Candidate:     {candidate.name}
  Position:      {job.title}
  Scheduled:     {scheduled_time.strftime('%Y-%m-%d %H:%M UTC')}
  Time Until:    {hours}h {minutes}m
  Avatar:        {avatar.title()}
  LLM:           {'GPT-4o-mini' if BEY_LLM_API_ID else 'Bey default'}

Meeting Details:
  URL:           {meeting.join_url}
  Passcode:      {meeting.password}

What Happens Next:
  1. ✅ Meeting created - candidate can join at scheduled time
  2. ⏳ At scheduled time, scheduler will send avatar to meeting
  3. 🤖 Avatar will greet and interview the candidate
  4. 📝 Results will be saved and synced to Milvus

To manually start (skip scheduler):
  python start_interview.py {interview_id}

To check upcoming interviews:
  curl http://localhost:8080/api/scheduler/upcoming
""")
    
    return {
        "interview_id": interview_id,
        "candidate_name": candidate.name,
        "job_title": job.title,
        "scheduled_time": scheduled_time.isoformat(),
        "meeting_url": meeting.join_url,
        "meeting_passcode": meeting.password,
    }


def main():
    parser = argparse.ArgumentParser(description='Schedule a future interview')
    parser.add_argument('--candidate', type=str, help='Candidate ID from Milvus')
    parser.add_argument('--job', type=str, help='Job ID from Milvus')
    parser.add_argument('--time', type=str, help='Scheduled time (ISO format, e.g., 2026-02-03T14:00:00Z)')
    parser.add_argument('--in-minutes', type=int, help='Schedule N minutes from now')
    parser.add_argument('--avatar', type=str, default='zara', choices=['zara', 'scott', 'sam'])
    parser.add_argument('--test', action='store_true', help='Use test candidate/job data')
    parser.add_argument('--name', type=str, help='Override candidate name')
    parser.add_argument('--email', type=str, help='Override candidate email')
    parser.add_argument('--notes', type=str, help='Notes for this interview')
    parser.add_argument('--org', type=str, help='Organization ID (UUID) for multi-tenant support')
    
    args = parser.parse_args()
    
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       RCRUTR INTERVIEW SCHEDULER                           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Determine scheduled time
    if args.time:
        scheduled_time = datetime.fromisoformat(args.time.replace('Z', '+00:00'))
    elif args.in_minutes:
        scheduled_time = datetime.now(tz=timezone.utc) + timedelta(minutes=args.in_minutes)
    else:
        print("❌ Please specify --time or --in-minutes")
        print()
        print("Examples:")
        print("  python schedule_interview.py --test --in-minutes 60")
        print("  python schedule_interview.py --candidate cand_123 --job job_456 --time '2026-02-03T14:00:00Z'")
        sys.exit(1)
    
    # Validate candidate/job
    if not args.test:
        if not args.candidate or not args.job:
            print("❌ Please specify --candidate and --job, or use --test")
            sys.exit(1)
    
    candidate_id = args.candidate or "test_cand_python_001"
    job_id = args.job or "test_job_python_001"
    
    # Run
    result = asyncio.run(schedule_interview(
        candidate_id=candidate_id,
        job_id=job_id,
        scheduled_time=scheduled_time,
        candidate_name=args.name,
        candidate_email=args.email,
        use_test_data=args.test,
        avatar=args.avatar,
        notes=args.notes,
        organization_id=args.org,
    ))
    
    if result:
        print("\n✅ Done!")
    else:
        print("\n❌ Failed to schedule interview")
        sys.exit(1)


if __name__ == "__main__":
    main()
