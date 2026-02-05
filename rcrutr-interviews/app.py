"""
RCRUTR Interviews - AI-powered candidate interview system.

This service handles scheduling and conducting video interviews
with candidates using AI avatars via Zoom and Bey.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    SERVICE_NAME, SERVICE_VERSION, DEBUG, ADMIN_API_KEY,
    AVATARS, DEFAULT_AVATAR, INTERVIEW_CANDIDATE_TIMEOUT_MINUTES,
    INTERVIEW_AVATAR_JOIN_RETRY,
)
from models import (
    ScheduleInterviewRequest, ScheduleInterviewResponse,
    InterviewResponse, InterviewListResponse,
    InterviewStatus, BeyCallEndedPayload,
    CancelInterviewRequest,
)
from db import (
    check_db_connection,
    insert_interview, get_interview, list_interviews,
    update_interview, update_interview_status, update_interview_meeting,
    update_interview_bey, generate_interview_id,
    get_pending_interviews,
)
from milvus_client import get_candidate, get_job, check_milvus_connection
from zoom_client import (
    create_meeting, delete_meeting, check_zoom_connection,
    get_waiting_room_participants, should_admit_participant,
    admit_participant, remove_participant,
)
from bey_client import (
    create_agent, create_call, send_to_external_meeting,
    get_avatar_config, check_bey_connection, delete_agent,
)
from interview_prompt import (
    build_interview_prompt, generate_interview_questions,
    format_questions_for_db,
)
from webhook_handler import process_call_ended, process_incomplete_interview

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(message)s',
)
logger = logging.getLogger(SERVICE_NAME)


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    _log_event("info", "app_starting", service=SERVICE_NAME, version=SERVICE_VERSION)
    
    # Check database connection on startup
    db_ok, db_msg = await check_db_connection()
    if db_ok:
        _log_event("info", "db_connection_ok")
    else:
        _log_event("warning", "db_connection_failed", message=db_msg)
    
    yield
    
    _log_event("info", "app_shutting_down")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="RCRUTR Interviews",
    description="AI-powered candidate interview scheduling and management",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HEALTH & STATUS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/api/status")
async def get_status():
    """Get service status with dependency health checks."""
    db_ok, db_msg = await check_db_connection()
    milvus_ok, milvus_msg = check_milvus_connection()
    zoom_ok, zoom_msg = check_zoom_connection()
    bey_ok, bey_msg = check_bey_connection()
    
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "dependencies": {
            "database": {"ok": db_ok, "message": db_msg},
            "milvus": {"ok": milvus_ok, "message": milvus_msg},
            "zoom": {"ok": zoom_ok, "message": zoom_msg},
            "bey": {"ok": bey_ok, "message": bey_msg},
        },
        "avatars": list(AVATARS.keys()),
    }


# =============================================================================
# INTERVIEW SCHEDULING
# =============================================================================

@app.post("/api/schedule-interview", response_model=ScheduleInterviewResponse)
async def schedule_interview(
    request: ScheduleInterviewRequest,
    x_admin_key: Optional[str] = Header(None),
):
    """
    Schedule a new interview.
    
    This endpoint:
    1. Fetches candidate data from Milvus
    2. Fetches job data from Milvus
    3. Creates a Zoom meeting
    4. Generates interview questions
    5. Stores everything in the database
    
    Returns the interview details including the meeting URL.
    """
    _log_event("info", "schedule_interview_request",
               candidate_id=request.candidate_id,
               job_id=request.job_id,
               scheduled_time=request.scheduled_time.isoformat())
    
    # Validate admin key (optional, for extra security)
    if ADMIN_API_KEY and x_admin_key != ADMIN_API_KEY:
        _log_event("warning", "schedule_interview_unauthorized")
        # Continue anyway for now - can enforce later
    
    # 1. Fetch candidate data
    candidate = get_candidate(request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail=f"Candidate not found: {request.candidate_id}")
    
    # Override name/email if provided
    if request.candidate_name:
        candidate.name = request.candidate_name
    if request.candidate_email:
        candidate.email = request.candidate_email
    
    if not candidate.name:
        raise HTTPException(status_code=400, detail="Candidate name is required")
    
    # 2. Fetch job data
    job = get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {request.job_id}")
    
    # 3. Get avatar config
    avatar_config = get_avatar_config(request.avatar)
    avatar_name = avatar_config["name"]
    
    # 4. Create Zoom meeting
    meeting_topic = f"Interview: {candidate.name} - {job.title}"
    meeting = create_meeting(
        topic=meeting_topic,
        start_time=request.scheduled_time,
        duration_minutes=30,
        timezone=request.timezone,
        waiting_room=True,
    )
    
    if not meeting:
        raise HTTPException(status_code=500, detail="Failed to create Zoom meeting")
    
    # 5. Generate interview questions
    questions = generate_interview_questions(candidate, job)
    questions_json = format_questions_for_db(questions)
    
    # 6. Generate interview ID
    interview_id = generate_interview_id()
    
    # 7. Store in database
    try:
        await insert_interview(
            interview_id=interview_id,
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            scheduled_time=request.scheduled_time,
            timezone=request.timezone,
            avatar_key=request.avatar,
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
            notes=request.notes,
        )
    except Exception as e:
        # Cleanup: delete the Zoom meeting if DB insert fails
        delete_meeting(meeting.id)
        _log_event("error", "schedule_interview_db_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save interview: {str(e)}")
    
    _log_event("info", "schedule_interview_success",
               interview_id=interview_id,
               meeting_id=meeting.id)
    
    return ScheduleInterviewResponse(
        interview_id=interview_id,
        meeting_url=meeting.join_url,
        meeting_passcode=meeting.password,
        scheduled_time=request.scheduled_time,
        candidate_name=candidate.name,
        job_title=job.title,
        status=InterviewStatus.MEETING_CREATED,
    )


@app.get("/api/interview/{interview_id}", response_model=InterviewResponse)
async def get_interview_details(interview_id: str):
    """Get interview details by ID."""
    interview = await get_interview(interview_id)
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    return InterviewResponse(
        interview_id=interview["interview_id"],
        candidate_id=interview["candidate_id"],
        job_id=interview["job_id"],
        candidate_name=interview.get("candidate_name"),
        candidate_email=interview.get("candidate_email"),
        candidate_phone=interview.get("candidate_phone"),
        job_title=interview.get("job_title"),
        scheduled_time=interview.get("scheduled_time"),
        timezone=interview.get("timezone", "UTC"),
        meeting_id=interview.get("meeting_id"),
        meeting_url=interview.get("meeting_url"),
        meeting_passcode=interview.get("meeting_passcode"),
        status=InterviewStatus(interview.get("interview_status", "scheduled")),
        avatar_joined_at=interview.get("avatar_joined_at"),
        candidate_joined_at=interview.get("candidate_joined_at"),
        interview_started_at=interview.get("interview_started_at"),
        interview_ended_at=interview.get("interview_ended_at"),
        duration_seconds=interview.get("call_duration_seconds"),
        total_questions=interview.get("total_questions", 0),
        questions_asked=interview.get("questions_asked", 0),
        sentiment_score=float(interview["sentiment_score"]) if interview.get("sentiment_score") else None,
        evaluation_score=float(interview["evaluation_score"]) if interview.get("evaluation_score") else None,
        fit_assessment=interview.get("fit_assessment"),
        created_at=interview.get("created_at"),
        updated_at=interview.get("updated_at"),
    )


@app.get("/api/interviews", response_model=InterviewListResponse)
async def list_all_interviews(
    status: Optional[str] = None,
    candidate_id: Optional[str] = None,
    job_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """List interviews with optional filters."""
    offset = (page - 1) * page_size
    
    interviews, total = await list_interviews(
        status=status,
        candidate_id=candidate_id,
        job_id=job_id,
        limit=page_size,
        offset=offset,
    )
    
    return InterviewListResponse(
        interviews=[
            InterviewResponse(
                interview_id=i["interview_id"],
                candidate_id=i["candidate_id"],
                job_id=i["job_id"],
                candidate_name=i.get("candidate_name"),
                job_title=i.get("job_title"),
                scheduled_time=i.get("scheduled_time"),
                status=InterviewStatus(i.get("interview_status", "scheduled")),
                created_at=i.get("created_at"),
            )
            for i in interviews
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.post("/api/interview/{interview_id}/cancel")
async def cancel_interview(
    interview_id: str,
    request: Optional[CancelInterviewRequest] = None,
):
    """Cancel a scheduled interview."""
    interview = await get_interview(interview_id)
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    # Only allow cancelling scheduled or meeting_created interviews
    status = interview.get("interview_status")
    if status not in ("scheduled", "meeting_created"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel interview with status: {status}"
        )
    
    # Delete Zoom meeting
    meeting_id = interview.get("meeting_id")
    if meeting_id:
        delete_meeting(meeting_id)
    
    # Update status
    reason = request.reason if request else "Cancelled by user"
    await update_interview_status(
        interview_id,
        InterviewStatus.CANCELLED,
        error_message=reason,
    )
    
    _log_event("info", "interview_cancelled", interview_id=interview_id, reason=reason)
    
    return {"status": "cancelled", "interview_id": interview_id}


# =============================================================================
# INTERVIEW EXECUTION
# =============================================================================

@app.post("/api/interview/{interview_id}/start")
async def start_interview(
    interview_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Manually start an interview (send avatar to meeting).
    
    This is typically called by a scheduler, but can be triggered manually.
    """
    interview = await get_interview(interview_id)
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    status = interview.get("interview_status")
    if status not in ("scheduled", "meeting_created"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start interview with status: {status}"
        )
    
    # Start the interview in background
    background_tasks.add_task(execute_interview, interview_id)
    
    return {
        "status": "starting",
        "interview_id": interview_id,
        "message": "Avatar is joining the meeting",
    }


async def execute_interview(interview_id: str):
    """
    Execute the interview process:
    1. Create Bey agent
    2. Create call session
    3. Send avatar to Zoom meeting
    4. Wait for candidate
    5. Handle timeout if candidate doesn't join
    """
    _log_event("info", "execute_interview_start", interview_id=interview_id)
    
    interview = await get_interview(interview_id)
    if not interview:
        _log_event("error", "execute_interview_not_found", interview_id=interview_id)
        return
    
    try:
        # Update status
        await update_interview_status(interview_id, InterviewStatus.AVATAR_JOINING)
        
        # Get avatar config
        avatar_key = interview.get("avatar_key", DEFAULT_AVATAR)
        avatar_config = get_avatar_config(avatar_key)
        
        # Build interview prompt
        from models import CandidateData, JobData
        
        candidate = CandidateData(
            candidate_id=interview["candidate_id"],
            name=interview.get("candidate_name"),
            email=interview.get("candidate_email"),
            phone=interview.get("candidate_phone"),
            semantic_summary=interview.get("candidate_summary"),
            current_tech_stack=interview.get("candidate_tech_stack"),
            top_5_skills_with_years=interview.get("candidate_skills", {}).get("top_skills"),
            employment_history=interview.get("candidate_employment_history"),
        )
        
        job = JobData(
            job_id=interview["job_id"],
            title=interview.get("job_title"),
            company=interview.get("job_company"),
            location=interview.get("job_location"),
            jd_text=interview.get("job_description"),
        )
        
        prompt_config = build_interview_prompt(
            candidate=candidate,
            job=job,
            avatar_name=avatar_config["name"],
        )
        
        # Create Bey agent with retry
        agent = None
        for attempt in range(INTERVIEW_AVATAR_JOIN_RETRY + 1):
            agent = create_agent(
                name=f"{avatar_config['name']} - Interview {interview_id[:8]}",
                system_prompt=prompt_config["system_prompt"],
                greeting=prompt_config["greeting"],
                avatar_id=avatar_config["id"],
            )
            if agent:
                break
            _log_event("warning", "execute_interview_agent_retry",
                       interview_id=interview_id, attempt=attempt + 1)
            await asyncio.sleep(2)
        
        if not agent:
            await update_interview_status(
                interview_id,
                InterviewStatus.FAILED,
                error_message="Failed to create Bey agent after retries",
            )
            return
        
        # Create call session
        call = create_call(agent.id, username=interview.get("candidate_name", "Candidate"))
        if not call:
            delete_agent(agent.id)
            await update_interview_status(
                interview_id,
                InterviewStatus.FAILED,
                error_message="Failed to create call session",
            )
            return
        
        # Send avatar to Zoom meeting
        meeting_url = interview.get("meeting_url")
        result = send_to_external_meeting(
            call_id=call.id,
            meeting_url=meeting_url,
            livekit_url=call.livekit_url,
            livekit_token=call.livekit_token,
            bot_name=f"{avatar_config['name']} - RCRUTR AI",
        )
        
        if not result or result.status != "success":
            delete_agent(agent.id)
            await update_interview_status(
                interview_id,
                InterviewStatus.FAILED,
                error_message="Failed to send avatar to meeting",
            )
            return
        
        # Update interview with Bey details
        await update_interview_bey(
            interview_id,
            agent_id=agent.id,
            call_id=call.id,
            livekit_url=call.livekit_url,
            livekit_token=call.livekit_token,
            bot_id=result.bot_id,
        )
        
        _log_event("info", "execute_interview_avatar_joined",
                   interview_id=interview_id,
                   agent_id=agent.id,
                   call_id=call.id)
        
        # Start waiting for candidate (with timeout)
        await wait_for_candidate(interview_id, interview.get("meeting_id"))
        
    except Exception as e:
        _log_event("error", "execute_interview_failed",
                   interview_id=interview_id, error=str(e))
        await update_interview_status(
            interview_id,
            InterviewStatus.FAILED,
            error_message=str(e),
        )


async def wait_for_candidate(interview_id: str, meeting_id: str):
    """
    Wait for candidate to join the meeting and admit them.
    
    Times out after INTERVIEW_CANDIDATE_TIMEOUT_MINUTES.
    """
    interview = await get_interview(interview_id)
    if not interview:
        return
    
    expected_name = interview.get("candidate_name", "")
    timeout_minutes = INTERVIEW_CANDIDATE_TIMEOUT_MINUTES
    check_interval = 10  # seconds
    max_checks = (timeout_minutes * 60) // check_interval
    
    _log_event("info", "wait_for_candidate_start",
               interview_id=interview_id,
               expected_name=expected_name,
               timeout_minutes=timeout_minutes)
    
    for check in range(max_checks):
        await asyncio.sleep(check_interval)
        
        # Check waiting room
        participants = get_waiting_room_participants(meeting_id)
        
        for participant in participants:
            action, similarity = should_admit_participant(
                participant.user_name,
                expected_name,
            )
            
            if action.value == "admit":
                # Admit the candidate
                admitted = admit_participant(meeting_id, participant.id)
                
                if admitted:
                    await update_interview(
                        interview_id,
                        candidate_joined_at=datetime.now(tz=timezone.utc),
                        interview_started_at=datetime.now(tz=timezone.utc),
                        interview_status=InterviewStatus.IN_PROGRESS.value,
                    )
                    
                    _log_event("info", "candidate_admitted",
                               interview_id=interview_id,
                               participant_name=participant.user_name,
                               similarity=similarity)
                    return
            else:
                # Reject non-matching participant
                remove_participant(meeting_id, participant.id)
                _log_event("info", "participant_rejected",
                           interview_id=interview_id,
                           participant_name=participant.user_name,
                           similarity=similarity)
    
    # Timeout - candidate didn't join
    _log_event("warning", "candidate_timeout",
               interview_id=interview_id,
               timeout_minutes=timeout_minutes)
    
    await process_incomplete_interview(
        interview_id,
        reason=f"Candidate did not join within {timeout_minutes} minutes",
    )


# =============================================================================
# WEBHOOKS
# =============================================================================

@app.post("/webhook/bey")
async def bey_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle webhooks from Bey (call_ended, etc.)
    
    This endpoint receives all webhooks from Bey. It:
    1. Checks if the agent belongs to a specific organization
    2. Forwards to org's webhook if configured
    3. Otherwise processes locally
    """
    try:
        body = await request.json()
        event_type = body.get("event_type", "")
        call_data = body.get("call_data", {})
        agent_id = call_data.get("agentId", "")
        
        _log_event("info", "bey_webhook_received", 
                   event_type=event_type, agent_id=agent_id)
        
        # Check for multi-tenant routing
        from webhook_handler import route_webhook_to_org
        
        if agent_id:
            forward_result = await route_webhook_to_org(agent_id, body)
            if forward_result and forward_result.get("status") == "forwarded":
                return {"status": "forwarded", "organization_id": forward_result.get("organization_id")}
        
        # Process locally
        if event_type == "call_ended":
            payload = BeyCallEndedPayload(**body)
            background_tasks.add_task(process_call_ended, payload)
            return {"status": "processing", "event_type": event_type}
        
        return {"status": "ignored", "event_type": event_type}
        
    except Exception as e:
        _log_event("error", "bey_webhook_error", error=str(e))
        return JSONResponse(
            status_code=200,  # Always return 200 to avoid retries
            content={"status": "error", "message": str(e)},
        )


@app.post("/webhook/zoom")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle webhooks from Zoom (participant.joined, meeting.ended, etc.)
    """
    try:
        body = await request.json()
        event = body.get("event", "")
        
        _log_event("info", "zoom_webhook_received", event=event)
        
        # Handle Zoom URL validation challenge
        if event == "endpoint.url_validation":
            plain_token = body.get("payload", {}).get("plainToken", "")
            import hashlib
            import hmac
            from config import WEBHOOK_SECRET
            
            encrypted_token = hmac.new(
                WEBHOOK_SECRET.encode(),
                plain_token.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return {
                "plainToken": plain_token,
                "encryptedToken": encrypted_token,
            }
        
        # Other events - process as needed
        return {"status": "received", "event": event}
        
    except Exception as e:
        _log_event("error", "zoom_webhook_error", error=str(e))
        return JSONResponse(
            status_code=200,
            content={"status": "error", "message": str(e)},
        )


# =============================================================================
# SCHEDULER ENDPOINTS
# =============================================================================

@app.post("/api/scheduler/check")
async def scheduler_check(background_tasks: BackgroundTasks):
    """
    Trigger the scheduler to check and start due interviews.
    
    Call this endpoint via Cloud Scheduler every minute to ensure
    interviews start at their scheduled time.
    """
    from scheduler import check_and_start_interviews
    
    _log_event("info", "scheduler_check_triggered")
    
    started = await check_and_start_interviews()
    
    return {
        "status": "ok",
        "interviews_started": len(started),
        "interview_ids": started,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/scheduler/upcoming")
async def scheduler_upcoming(hours: int = 24):
    """
    Get interviews scheduled in the next N hours.
    
    Useful for monitoring what interviews are coming up.
    """
    from scheduler import get_upcoming_interviews
    
    interviews = await get_upcoming_interviews(hours=hours)
    
    return {
        "count": len(interviews),
        "hours": hours,
        "interviews": [
            {
                "interview_id": i["interview_id"],
                "candidate_name": i.get("candidate_name"),
                "job_title": i.get("job_title"),
                "scheduled_time": i.get("scheduled_time").isoformat() if i.get("scheduled_time") else None,
                "status": i.get("interview_status"),
            }
            for i in interviews
        ],
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
