#!/usr/bin/env python3
"""
Manually process a completed interview.

This script:
1. Fetches call data from Bey API (if available)
2. OR processes from provided transcript
3. Updates PostgreSQL with results
4. Syncs Q&A embeddings to Milvus

Usage:
  python process_interview.py int_759d1598e5b0
  python process_interview.py int_759d1598e5b0 --transcript transcript.json
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from db import get_interview, update_interview, update_interview_results
from models import InterviewStatus
from milvus_client import store_qa_embeddings
from bey_client import get_call
from webhook_handler import (
    build_transcript,
    extract_qa_pairs,
    parse_sentiment,
    extract_keywords,
    calculate_evaluation_score,
    generate_fit_assessment,
)


async def process_interview_manually(
    interview_id: str,
    messages: Optional[List[Dict]] = None,
    evaluation: Optional[Dict] = None,
    call_data: Optional[Dict] = None,
):
    """
    Manually process an interview with provided or fetched data.
    """
    
    print("=" * 60)
    print(f"PROCESSING INTERVIEW: {interview_id}")
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
    print(f"   Status: {interview.get('interview_status')}")
    print(f"   Call ID: {interview.get('call_id')}")
    
    # 2. Try to fetch from Bey if no messages provided
    if not messages and interview.get('call_id'):
        print("\n2. Trying to fetch call data from Bey...")
        call_info = get_call(interview['call_id'])
        if call_info:
            print(f"   ✅ Got call data from Bey")
            messages = call_info.get('messages', [])
            evaluation = call_info.get('evaluation', {})
            call_data = call_info.get('call_data', {})
        else:
            print(f"   ⚠️ Could not fetch from Bey API")
    
    if not messages:
        print("\n   ❌ No messages/transcript available!")
        print("   Please provide a transcript file with --transcript")
        return False
    
    print(f"\n3. Processing {len(messages)} messages...")
    
    # 3. Calculate duration
    duration_seconds = None
    if call_data:
        started_at = call_data.get('startedAt')
        ended_at = call_data.get('endedAt')
        if started_at and ended_at:
            try:
                start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
                duration_seconds = int((end - start).total_seconds())
                print(f"   Duration: {duration_seconds} seconds ({duration_seconds//60} min)")
            except:
                pass
    
    # 4. Build transcript
    full_transcript = build_transcript(messages)
    print(f"   Transcript length: {len(full_transcript)} chars")
    
    # 5. Extract Q&A pairs
    qa_pairs = extract_qa_pairs(messages)
    print(f"   Q&A pairs extracted: {len(qa_pairs)}")
    
    # 6. Calculate sentiment
    sentiment_score = parse_sentiment(evaluation or {})
    print(f"   Sentiment score: {sentiment_score}")
    
    # 7. Extract keywords
    keyword_matches = extract_keywords(
        full_transcript,
        interview.get('candidate_skills', {}),
        interview.get('job_description', ''),
    )
    print(f"   Keywords found: {keyword_matches.get('total_keywords', 0)}")
    
    # 8. Calculate evaluation score
    evaluation_score = calculate_evaluation_score(qa_pairs, keyword_matches, sentiment_score)
    print(f"   Evaluation score: {evaluation_score}")
    
    # 9. Generate fit assessment
    fit_assessment = generate_fit_assessment(qa_pairs, keyword_matches, sentiment_score, evaluation_score)
    print(f"   Fit assessment generated")
    
    # 10. Update database
    print("\n4. Updating database...")
    
    now = datetime.now(tz=timezone.utc)
    
    update_success = await update_interview(
        interview_id,
        conversation_log=messages,
        full_transcript=full_transcript,
        sentiment_score=sentiment_score,
        evaluation_score=evaluation_score,
        fit_assessment=fit_assessment,
        keyword_matches=keyword_matches,
        evaluation_raw=evaluation,
        questions_asked=len(qa_pairs),
        call_duration_seconds=duration_seconds,
        interview_status=InterviewStatus.COMPLETED.value,
        interview_ended_at=now,
        completed_at=now,
        # Also set these if missing
        candidate_joined_at=interview.get('candidate_joined_at') or interview.get('avatar_joined_at'),
        interview_started_at=interview.get('interview_started_at') or interview.get('avatar_joined_at'),
    )
    
    if update_success:
        print(f"   ✅ Database updated")
    else:
        print(f"   ❌ Database update failed")
        return False
    
    # 11. Sync to Milvus
    print("\n5. Syncing to Milvus...")
    
    if qa_pairs:
        milvus_success = store_qa_embeddings(
            interview_id=interview_id,
            candidate_id=interview['candidate_id'],
            job_id=interview['job_id'],
            job_title=interview.get('job_title', ''),
            job_description=interview.get('job_description', ''),
            qa_pairs=qa_pairs,
        )
        
        if milvus_success:
            await update_interview(
                interview_id,
                milvus_synced=True,
                milvus_synced_at=now,
            )
            print(f"   ✅ Synced {len(qa_pairs)} Q&A pairs to Milvus")
        else:
            print(f"   ❌ Milvus sync failed")
    else:
        print(f"   ⚠️ No Q&A pairs to sync")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ INTERVIEW PROCESSED SUCCESSFULLY!")
    print("=" * 60)
    print(f"""
Interview ID:      {interview_id}
Status:            COMPLETED
Duration:          {duration_seconds or 'N/A'} seconds
Questions Asked:   {len(qa_pairs)}
Sentiment Score:   {sentiment_score}
Evaluation Score:  {evaluation_score}/100
Milvus Synced:     {'Yes' if milvus_success else 'No'}

FIT ASSESSMENT:
{fit_assessment}
""")
    
    return True


async def process_from_webhook_file(interview_id: str, webhook_file: str):
    """Process interview from saved webhook payload file."""
    
    print(f"Loading webhook data from: {webhook_file}")
    
    with open(webhook_file, 'r') as f:
        payload = json.load(f)
    
    messages = payload.get('messages', [])
    evaluation = payload.get('evaluation', {})
    call_data = payload.get('call_data', {})
    
    print(f"Loaded {len(messages)} messages from file")
    
    return await process_interview_manually(
        interview_id,
        messages=messages,
        evaluation=evaluation,
        call_data=call_data,
    )


async def create_sample_transcript(interview_id: str):
    """Create a sample transcript for testing."""
    
    # Sample messages simulating an interview
    messages = [
        {"sender": "ai", "message": "Hi Alex Johnson! I'm Zara, and I'll be conducting your screening interview today for the Mid-Level Python Developer position. This should take about 15-20 minutes. Are you ready to get started?", "sent_at": "2026-02-02T02:16:00Z"},
        {"sender": "user", "message": "Hi Zara! Yes, I'm ready. Thanks for having me.", "sent_at": "2026-02-02T02:16:15Z"},
        {"sender": "ai", "message": "Great! Before we begin, could you please confirm your full name?", "sent_at": "2026-02-02T02:16:20Z"},
        {"sender": "user", "message": "Yes, my name is Alex Johnson.", "sent_at": "2026-02-02T02:16:30Z"},
        {"sender": "ai", "message": "Thank you, Alex. What is your current work authorization status in the US?", "sent_at": "2026-02-02T02:16:35Z"},
        {"sender": "user", "message": "I'm a US Citizen.", "sent_at": "2026-02-02T02:16:45Z"},
        {"sender": "ai", "message": "Where are you currently located? And would you be open to relocation if the role requires it?", "sent_at": "2026-02-02T02:16:50Z"},
        {"sender": "user", "message": "I'm based in Austin, Texas. The role is remote so relocation isn't needed, but I'm flexible if required.", "sent_at": "2026-02-02T02:17:05Z"},
        {"sender": "ai", "message": "Let me briefly describe the Mid-Level Python Developer role. You'll be working on building scalable APIs, implementing vector search, and deploying on GCP. Does this sound like something you'd be interested in?", "sent_at": "2026-02-02T02:17:15Z"},
        {"sender": "user", "message": "Absolutely! That's exactly what I've been doing at my current job. I've built several FastAPI services and worked with Milvus for vector search. Very excited about this opportunity.", "sent_at": "2026-02-02T02:17:40Z"},
        {"sender": "ai", "message": "I see you have Python experience. Can you tell me about a significant project where you used Python? What frameworks did you work with?", "sent_at": "2026-02-02T02:17:50Z"},
        {"sender": "user", "message": "Sure! At TechCorp, I built a RAG system using FastAPI for the backend. We used PostgreSQL for structured data and Milvus for vector embeddings. The system processes about 10,000 queries per day and has 99.9% uptime.", "sent_at": "2026-02-02T02:18:30Z"},
        {"sender": "ai", "message": "That sounds impressive! What cloud platforms have you worked with? Can you describe a project where you deployed and managed cloud infrastructure?", "sent_at": "2026-02-02T02:18:40Z"},
        {"sender": "user", "message": "I've worked primarily with GCP. For the RAG system I mentioned, we deployed on Cloud Run with Cloud SQL for the database. We also used BigQuery for analytics and Vertex AI for some ML features. I set up the entire CI/CD pipeline using Cloud Build.", "sent_at": "2026-02-02T02:19:20Z"},
        {"sender": "ai", "message": "Do you have experience with containerization? Tell me about how you've used Docker or Kubernetes in your projects.", "sent_at": "2026-02-02T02:19:30Z"},
        {"sender": "user", "message": "Yes, all our services are containerized with Docker. I write the Dockerfiles and we use multi-stage builds for smaller images. For Kubernetes, we have a GKE cluster for our more complex services. I've set up deployments, services, and configmaps.", "sent_at": "2026-02-02T02:20:10Z"},
        {"sender": "ai", "message": "Thank you for your time today, Alex. I really enjoyed learning about your experience with Python, FastAPI, and GCP. The team will review our conversation and reach out with next steps. Do you have any questions for me?", "sent_at": "2026-02-02T02:20:30Z"},
        {"sender": "user", "message": "Thank you, Zara! I'm curious about the team size and the main projects I'd be working on if selected.", "sent_at": "2026-02-02T02:20:50Z"},
        {"sender": "ai", "message": "The team details will be shared by the recruiter in the next round. Thank you again for your time, Alex. Have a wonderful day!", "sent_at": "2026-02-02T02:21:05Z"},
        {"sender": "user", "message": "Thank you! Looking forward to hearing back. Bye!", "sent_at": "2026-02-02T02:21:15Z"},
    ]
    
    evaluation = {
        "user_sentiment": "satisfied",
        "duration_minutes": "5.25",
        "messages_count": str(len(messages)),
        "topic": "Python developer screening interview",
    }
    
    call_data = {
        "startedAt": "2026-02-02T02:16:00+00:00",
        "endedAt": "2026-02-02T02:21:15+00:00",
    }
    
    return await process_interview_manually(
        interview_id,
        messages=messages,
        evaluation=evaluation,
        call_data=call_data,
    )


def main():
    parser = argparse.ArgumentParser(description='Process completed RCRUTR interview')
    parser.add_argument('interview_id', help='Interview ID to process')
    parser.add_argument('--transcript', type=str, help='Path to webhook payload JSON file')
    parser.add_argument('--sample', action='store_true', help='Use sample transcript for testing')
    
    args = parser.parse_args()
    
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       RCRUTR INTERVIEW PROCESSOR                           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    if args.sample:
        asyncio.run(create_sample_transcript(args.interview_id))
    elif args.transcript:
        asyncio.run(process_from_webhook_file(args.interview_id, args.transcript))
    else:
        asyncio.run(process_interview_manually(args.interview_id))


if __name__ == "__main__":
    main()
