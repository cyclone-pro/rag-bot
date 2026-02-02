#!/usr/bin/env python3
"""
Start an interview - sends the AI avatar to the Zoom meeting.

This script:
1. Looks up the interview from database
2. Creates Bey agent with interview prompt
3. Creates call session
4. Sends avatar to Zoom meeting
5. Avatar joins and waits for candidate

Usage:
  python start_interview.py int_759d1598e5b0
  
Or to create and start immediately:
  python start_interview.py --new
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime, timedelta, timezone

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from config import BEY_API_KEY, AVATARS, DEFAULT_AVATAR
from db import get_interview, update_interview, update_interview_status
from models import InterviewStatus, CandidateData, JobData
from bey_client import create_agent, create_call, send_to_external_meeting, get_avatar_config
from interview_prompt import build_interview_prompt


async def start_interview(interview_id: str):
    """Start an interview by sending the avatar to Zoom."""
    
    print("=" * 60)
    print(f"STARTING INTERVIEW: {interview_id}")
    print("=" * 60)
    
    # 1. Get interview from database
    print("\n1. Fetching interview from database...")
    interview = await get_interview(interview_id)
    
    if not interview:
        print(f"   ❌ Interview not found: {interview_id}")
        return False
    
    print(f"   ✅ Found interview")
    print(f"   Candidate: {interview.get('candidate_name')}")
    print(f"   Job: {interview.get('job_title')}")
    print(f"   Meeting URL: {interview.get('meeting_url')}")
    print(f"   Status: {interview.get('interview_status')}")
    
    meeting_url = interview.get('meeting_url')
    if not meeting_url:
        print("   ❌ No meeting URL found!")
        return False
    
    # 2. Build candidate and job data for prompt
    print("\n2. Building interview prompt...")
    
    candidate = CandidateData(
        candidate_id=interview['candidate_id'],
        name=interview.get('candidate_name'),
        email=interview.get('candidate_email'),
        phone=interview.get('candidate_phone'),
        semantic_summary=interview.get('candidate_summary'),
        current_tech_stack=interview.get('candidate_tech_stack'),
        top_5_skills_with_years=interview.get('candidate_skills', {}).get('top_skills') if interview.get('candidate_skills') else None,
        employment_history=interview.get('candidate_employment_history'),
    )
    
    job = JobData(
        job_id=interview['job_id'],
        title=interview.get('job_title'),
        company=interview.get('job_company'),
        location=interview.get('job_location'),
        jd_text=interview.get('job_description'),
    )
    
    # Get avatar config
    avatar_key = interview.get('avatar_key', DEFAULT_AVATAR)
    avatar_config = get_avatar_config(avatar_key)
    avatar_name = avatar_config['name']
    avatar_id = avatar_config['id']
    
    print(f"   Avatar: {avatar_name} ({avatar_key})")
    
    # Build the prompt
    prompt_config = build_interview_prompt(
        candidate=candidate,
        job=job,
        avatar_name=avatar_name,
    )
    
    system_prompt = prompt_config['system_prompt']
    greeting = prompt_config['greeting']
    
    print(f"   ✅ Prompt built ({len(system_prompt)} chars)")
    print(f"   Greeting: {greeting[:60]}...")
    
    # 3. Create Bey agent
    print("\n3. Creating Bey agent...")
    
    agent_name = f"{avatar_name} - Interview {interview_id[:8]}"
    
    agent = create_agent(
        name=agent_name,
        system_prompt=system_prompt,
        greeting=greeting,
        avatar_id=avatar_id,
    )
    
    if not agent:
        print("   ❌ Failed to create Bey agent!")
        await update_interview_status(interview_id, InterviewStatus.FAILED, error_message="Failed to create Bey agent")
        return False
    
    print(f"   ✅ Agent created: {agent.id}")
    
    # 4. Create call session
    print("\n4. Creating call session...")
    
    call = create_call(
        agent_id=agent.id,
        username=interview.get('candidate_name', 'Candidate'),
    )
    
    if not call:
        print("   ❌ Failed to create call session!")
        await update_interview_status(interview_id, InterviewStatus.FAILED, error_message="Failed to create call session")
        return False
    
    print(f"   ✅ Call created: {call.id}")
    print(f"   LiveKit URL: {call.livekit_url[:50]}...")
    
    # 5. Send avatar to Zoom meeting
    print("\n5. Sending avatar to Zoom meeting...")
    print(f"   Meeting URL: {meeting_url}")
    
    bot_name = f"{avatar_name} - RCRUTR AI Interviewer"
    
    result = send_to_external_meeting(
        call_id=call.id,
        meeting_url=meeting_url,
        livekit_url=call.livekit_url,
        livekit_token=call.livekit_token,
        bot_name=bot_name,
    )
    
    if not result:
        print("   ❌ Failed to send avatar to meeting!")
        await update_interview_status(interview_id, InterviewStatus.FAILED, error_message="Failed to send avatar to meeting")
        return False
    
    print(f"   ✅ Avatar sent to meeting!")
    print(f"   Bot ID: {result.bot_id}")
    print(f"   Status: {result.status}")
    
    # 6. Update database
    print("\n6. Updating database...")
    
    await update_interview(
        interview_id,
        agent_id=agent.id,
        call_id=call.id,
        livekit_url=call.livekit_url,
        livekit_token=call.livekit_token,
        bot_id=result.bot_id,
        avatar_joined_at=datetime.now(tz=timezone.utc),
        interview_status=InterviewStatus.WAITING_FOR_CANDIDATE.value,
    )
    
    print(f"   ✅ Database updated")
    
    # Success!
    print("\n" + "=" * 60)
    print("✅ AVATAR SENT TO MEETING SUCCESSFULLY!")
    print("=" * 60)
    print(f"""
Interview ID: {interview_id}
Agent ID:     {agent.id}
Call ID:      {call.id}
Bot ID:       {result.bot_id}

The avatar ({avatar_name}) should now be joining the Zoom meeting.
It will greet the candidate when they join.

Meeting URL: {meeting_url}
""")
    
    return True


async def create_and_start_interview():
    """Create a new interview and immediately start it."""
    
    from db import insert_interview, generate_interview_id
    from zoom_client import create_meeting
    from interview_prompt import generate_interview_questions, format_questions_for_db
    
    print("=" * 60)
    print("CREATING AND STARTING NEW INTERVIEW")
    print("=" * 60)
    
    # Test candidate data
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
        Built RAG systems and semantic search applications.
        """,
        current_tech_stack=["Python", "FastAPI", "PostgreSQL", "Milvus", "GCP", "Docker", "Redis"],
        top_5_skills_with_years="Python:4, FastAPI:3, PostgreSQL:4, GCP:3, Milvus:2",
        employment_history=[
            {"company": "TechCorp Inc", "title": "Backend Developer", "start_date": "2022-01", "end_date": "Present"},
            {"company": "StartupXYZ", "title": "Junior Developer", "start_date": "2020-06", "end_date": "2021-12"}
        ]
    )
    
    job = JobData(
        job_id="test_job_python_001",
        title="Mid-Level Python Developer",
        company="Elite Solutions",
        location="Remote / US",
        jd_text="""
Title: Mid-Level Python Developer
Seniority: Mid-Level (3-5 years)
Work Model: Remote

Required Skills:
- Python (3+ years)
- FastAPI or Django
- PostgreSQL
- Vector databases (Milvus, Pinecone)
- GCP services (Cloud Run, Cloud SQL)
- Docker

Responsibilities:
- Design and implement backend APIs
- Build vector search systems
- Deploy services on GCP
"""
    )
    
    # Create Zoom meeting
    print("\n1. Creating Zoom meeting...")
    scheduled_time = datetime.now(tz=timezone.utc) + timedelta(minutes=2)
    
    meeting = create_meeting(
        topic=f"Interview: {candidate.name} - {job.title}",
        start_time=scheduled_time,
        duration_minutes=30,
        waiting_room=True,
    )
    
    if not meeting:
        print("   ❌ Failed to create Zoom meeting")
        return None
    
    print(f"   ✅ Meeting created: {meeting.id}")
    print(f"   Join URL: {meeting.join_url}")
    
    # Generate questions
    questions = generate_interview_questions(candidate, job)
    questions_json = format_questions_for_db(questions)
    
    # Save to database
    print("\n2. Saving to database...")
    interview_id = generate_interview_id()
    
    await insert_interview(
        interview_id=interview_id,
        candidate_id=candidate.candidate_id,
        job_id=job.job_id,
        scheduled_time=scheduled_time,
        timezone_str="UTC",
        avatar_key="zara",
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
    )
    
    print(f"   ✅ Interview saved: {interview_id}")
    
    # Now start it
    print("\n3. Starting interview (sending avatar)...")
    success = await start_interview(interview_id)
    
    if success:
        print("\n" + "=" * 60)
        print("INTERVIEW READY!")
        print("=" * 60)
        print(f"""
Join the meeting as the candidate:
  URL: {meeting.join_url}
  Passcode: {meeting.password}

The avatar (Zara) should join shortly and greet you!
""")
    
    return interview_id


def main():
    parser = argparse.ArgumentParser(description='Start an RCRUTR interview')
    parser.add_argument('interview_id', nargs='?', help='Interview ID to start')
    parser.add_argument('--new', action='store_true', help='Create a new interview and start it')
    
    args = parser.parse_args()
    
    if args.new:
        asyncio.run(create_and_start_interview())
    elif args.interview_id:
        success = asyncio.run(start_interview(args.interview_id))
        if not success:
            sys.exit(1)
    else:
        print("Usage:")
        print("  python start_interview.py <interview_id>  # Start existing interview")
        print("  python start_interview.py --new           # Create and start new interview")
        print()
        print("Example:")
        print("  python start_interview.py int_759d1598e5b0")
        sys.exit(1)


if __name__ == "__main__":
    main()