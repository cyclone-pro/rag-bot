#!/usr/bin/env python3
"""
Autonomous Interview Runner

This script runs a fully autonomous interview:
1. Creates Zoom meeting (avatar as host, candidates go to waiting room)
2. Sends avatar to meeting using HOST URL (bypasses waiting room)
3. Polls waiting room for candidate
4. Admits candidate when they join
5. Interview proceeds automatically

Usage:
  python run_autonomous_interview.py --new
  python run_autonomous_interview.py <interview_id>
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from config import AVATARS, DEFAULT_AVATAR, INTERVIEW_CANDIDATE_TIMEOUT_MINUTES, NAME_MATCH_THRESHOLD
from db import get_interview, update_interview, insert_interview, generate_interview_id
from models import InterviewStatus, CandidateData, JobData
from bey_client import create_agent, create_call, send_to_external_meeting, get_avatar_config
from interview_prompt import build_interview_prompt, generate_interview_questions, format_questions_for_db
from zoom_client import (
    create_meeting, 
    get_waiting_room_participants, 
    admit_participant,
    should_admit_participant,
    calculate_name_similarity,
)


async def create_test_interview() -> Optional[str]:
    """Create a test interview and return the interview_id."""
    
    print("=" * 60)
    print("CREATING TEST INTERVIEW")
    print("=" * 60)
    
    # Test candidate
    candidate = CandidateData(
        candidate_id="test_cand_python_001",
        name="Alex Johnson",  # This is the expected name
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
    
    job = JobData(
        job_id="test_job_python_001",
        title="Mid-Level Python Developer",
        company="Elite Solutions",
        location="Remote / US",
        jd_text="""
Title: Mid-Level Python Developer
Required Skills: Python, FastAPI, PostgreSQL, Vector databases, GCP, Docker
Responsibilities: Backend APIs, Vector search systems, GCP deployment
"""
    )
    
    # Create Zoom meeting
    print("\n1. Creating Zoom meeting...")
    scheduled_time = datetime.now(tz=timezone.utc) + timedelta(minutes=1)
    
    # IMPORTANT: For fully autonomous operation:
    # - waiting_room=False: Everyone joins directly (simpler)
    # - OR waiting_room=True + join_before_host=True: Avatar joins as host, admits candidates
    #
    # For now, we disable waiting room for simplicity
    meeting = create_meeting(
        topic=f"Interview: {candidate.name} - {job.title}",
        start_time=scheduled_time,
        duration_minutes=30,
        waiting_room=False,  # Disable waiting room for autonomous operation
        join_before_host=True,  # Allow candidate to join even if avatar delays
    )
    
    if not meeting:
        print("   ❌ Failed to create Zoom meeting")
        return None
    
    print(f"   ✅ Meeting created: {meeting.id}")
    print(f"   ⚠️  Waiting room: DISABLED (autonomous mode)")
    print(f"   Join URL (for candidate): {meeting.join_url}")
    print(f"   Passcode: {meeting.password}")
    
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
        meeting_url=meeting.join_url,  # For candidate
        meeting_passcode=meeting.password,
        meeting_host_url=meeting.start_url,  # For avatar (bypasses waiting room)
        questions=questions_json,
        total_questions=len(questions_json),
    )
    
    print(f"   ✅ Interview saved: {interview_id}")
    
    return interview_id


async def send_avatar_to_meeting(interview: dict) -> bool:
    """Send the avatar to the Zoom meeting using HOST URL."""
    
    interview_id = interview['interview_id']
    
    # Get avatar config
    avatar_key = interview.get('avatar_key', DEFAULT_AVATAR)
    avatar_config = get_avatar_config(avatar_key)
    avatar_name = avatar_config['name']
    avatar_id = avatar_config['id']
    
    # Build candidate/job for prompt
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
    
    # Build prompt
    prompt_config = build_interview_prompt(candidate=candidate, job=job, avatar_name=avatar_name)
    
    print(f"\n3. Creating Bey agent ({avatar_name})...")
    agent = create_agent(
        name=f"{avatar_name} - Interview {interview_id[:8]}",
        system_prompt=prompt_config['system_prompt'],
        greeting=prompt_config['greeting'],
        avatar_id=avatar_id,
    )
    
    if not agent:
        print("   ❌ Failed to create agent")
        return False
    print(f"   ✅ Agent created: {agent.id}")
    
    print("\n4. Creating call session...")
    call = create_call(agent_id=agent.id, username=interview.get('candidate_name', 'Candidate'))
    
    if not call:
        print("   ❌ Failed to create call")
        return False
    print(f"   ✅ Call created: {call.id}")
    
    # USE HOST URL - This bypasses the waiting room!
    meeting_host_url = interview.get('meeting_host_url')
    meeting_url = interview.get('meeting_url')
    
    # Prefer host URL, fall back to regular URL
    avatar_url = meeting_host_url if meeting_host_url else meeting_url
    
    print("\n5. Sending avatar to meeting...")
    print(f"   Using: {'HOST URL (bypasses waiting room)' if meeting_host_url else 'Regular URL'}")
    
    result = send_to_external_meeting(
        call_id=call.id,
        meeting_url=avatar_url,
        livekit_url=call.livekit_url,
        livekit_token=call.livekit_token,
        bot_name=f"{avatar_name} - RCRUTR AI Interviewer",
    )
    
    if not result:
        print("   ❌ Failed to send avatar")
        return False
    
    print(f"   ✅ Avatar sent! Bot ID: {result.bot_id}")
    
    # Update database
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
    
    return True


async def poll_and_admit_candidate(interview: dict) -> bool:
    """
    Poll the waiting room and admit the candidate when they join.
    
    This runs in a loop until:
    - Candidate is admitted, or
    - Timeout is reached
    """
    
    interview_id = interview['interview_id']
    meeting_id = interview.get('meeting_id')
    expected_name = interview.get('candidate_name', '')
    
    print("\n6. Waiting for candidate to join...")
    print(f"   Expected name: {expected_name}")
    print(f"   Timeout: {INTERVIEW_CANDIDATE_TIMEOUT_MINUTES} minutes")
    print(f"   Name match threshold: {NAME_MATCH_THRESHOLD}")
    print()
    
    timeout_seconds = INTERVIEW_CANDIDATE_TIMEOUT_MINUTES * 60
    poll_interval = 5  # Check every 5 seconds
    elapsed = 0
    
    while elapsed < timeout_seconds:
        # Poll waiting room
        participants = get_waiting_room_participants(meeting_id)
        
        if participants:
            print(f"   [{elapsed}s] Found {len(participants)} in waiting room:")
            
            for p in participants:
                print(f"      - {p.user_name} (ID: {p.id})")
                
                # Check name match
                action, similarity = should_admit_participant(p.user_name, expected_name)
                
                if action.value == "admit":
                    print(f"      ✅ Name matches! (similarity: {similarity:.2f})")
                    print(f"      Admitting {p.user_name}...")
                    
                    admitted = admit_participant(meeting_id, p.id)
                    
                    if admitted:
                        print(f"      ✅ Candidate admitted!")
                        
                        await update_interview(
                            interview_id,
                            candidate_joined_at=datetime.now(tz=timezone.utc),
                            interview_started_at=datetime.now(tz=timezone.utc),
                            interview_status=InterviewStatus.IN_PROGRESS.value,
                        )
                        return True
                    else:
                        print(f"      ⚠️ Failed to admit via API, candidate may need manual admission")
                else:
                    print(f"      ❌ Name doesn't match (similarity: {similarity:.2f})")
                    print(f"      Expected: '{expected_name}', Got: '{p.user_name}'")
                    # Don't reject - just wait for the right person
        else:
            mins = elapsed // 60
            secs = elapsed % 60
            print(f"   [{mins}m {secs}s] Waiting room empty, checking again...", end='\r')
        
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    
    print(f"\n   ❌ Timeout! Candidate didn't join within {INTERVIEW_CANDIDATE_TIMEOUT_MINUTES} minutes")
    return False


async def run_autonomous_interview(interview_id: str):
    """Run a fully autonomous interview."""
    
    print("=" * 60)
    print(f"STARTING AUTONOMOUS INTERVIEW")
    print("=" * 60)
    print(f"Interview ID: {interview_id}")
    
    # Get interview
    print("\n1. Fetching interview...")
    interview = await get_interview(interview_id)
    
    if not interview:
        print(f"   ❌ Interview not found: {interview_id}")
        return False
    
    print(f"   ✅ Found: {interview.get('candidate_name')} - {interview.get('job_title')}")
    
    # Print meeting info for candidate
    print("\n" + "=" * 60)
    print("📧 SHARE THIS WITH CANDIDATE:")
    print("=" * 60)
    print(f"Meeting URL: {interview.get('meeting_url')}")
    print(f"Passcode: {interview.get('meeting_passcode')}")
    print("=" * 60 + "\n")
    
    # Send avatar
    avatar_sent = await send_avatar_to_meeting(interview)
    if not avatar_sent:
        print("❌ Failed to send avatar to meeting")
        return False
    
    # Since waiting room is disabled, no need to poll/admit
    # The avatar and candidate both join directly
    
    print("\n" + "=" * 60)
    print("✅ AVATAR SENT TO MEETING!")
    print("=" * 60)
    print("""
The avatar has entered the lobby and is joining the call.

WHAT HAPPENS NEXT:
1. Avatar (Zara) joins the meeting
2. Candidate joins using the meeting link  
3. Avatar greets and interviews the candidate
4. When done, Bey sends webhook with transcript

Since waiting room is DISABLED:
- Both avatar and candidate join directly
- No manual admission needed!

You can join too using the meeting link to observe.
""")
    
    await update_interview(
        interview_id,
        interview_status=InterviewStatus.IN_PROGRESS.value,
        interview_started_at=datetime.now(tz=timezone.utc),
    )
    
    return True


async def main_new():
    """Create and run a new autonomous interview."""
    interview_id = await create_test_interview()
    if interview_id:
        # Small delay to ensure DB is ready
        await asyncio.sleep(1)
        await run_autonomous_interview(interview_id)


async def main_existing(interview_id: str):
    """Run an existing interview autonomously."""
    await run_autonomous_interview(interview_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run autonomous RCRUTR interview')
    parser.add_argument('interview_id', nargs='?', help='Interview ID to run')
    parser.add_argument('--new', action='store_true', help='Create new interview and run')
    parser.add_argument('--name', type=str, help='Expected candidate name (for new interview)')
    
    args = parser.parse_args()
    
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       RCRUTR AUTONOMOUS INTERVIEW RUNNER                   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    if args.new:
        asyncio.run(main_new())
    elif args.interview_id:
        asyncio.run(main_existing(args.interview_id))
    else:
        print("Usage:")
        print("  python run_autonomous_interview.py --new         # Create & run new")
        print("  python run_autonomous_interview.py <interview_id> # Run existing")
        print()
        sys.exit(1)
