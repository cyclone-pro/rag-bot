"""Database operations for RCRUTR Interviews."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from config import DATABASE_URL
from models import InterviewStatus

logger = logging.getLogger("rcrutr_interviews_db")


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


def _get_db_url() -> str:
    """Get database URL from environment."""
    db_url = DATABASE_URL
    if not db_url:
        raise ValueError("DATABASE_URL not set")
    # Fix if just IP address
    if db_url and not db_url.startswith("postgresql"):
        db_url = f"postgresql://backteam:Airecruiter1_@{db_url}:5432/recruiter_brain"
    return db_url


def generate_interview_id() -> str:
    """Generate unique interview ID."""
    return f"int_{uuid.uuid4().hex[:12]}"


# =============================================================================
# TABLE CREATION
# =============================================================================

CREATE_TABLE_SQL = """
-- candidate_interviews table for scheduled video interviews
CREATE TABLE IF NOT EXISTS candidate_interviews (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    interview_id VARCHAR(64) UNIQUE NOT NULL,
    
    -- References
    candidate_id VARCHAR(64) NOT NULL,
    job_id VARCHAR(64) NOT NULL,
    
    -- Job Snapshot
    job_title VARCHAR(256),
    job_description TEXT,
    job_company VARCHAR(256),
    job_location VARCHAR(256),
    
    -- Candidate Snapshot
    candidate_name VARCHAR(256),
    candidate_email VARCHAR(256),
    candidate_phone VARCHAR(64),
    candidate_skills JSONB,
    candidate_summary TEXT,
    candidate_tech_stack JSONB,
    candidate_employment_history JSONB,
    
    -- Scheduling
    scheduled_time TIMESTAMPTZ NOT NULL,
    timezone VARCHAR(64) DEFAULT 'UTC',
    
    -- Zoom Meeting Details
    meeting_id VARCHAR(256),
    meeting_url TEXT,
    meeting_join_url TEXT,
    meeting_passcode VARCHAR(64),
    meeting_host_url TEXT,
    meeting_created_at TIMESTAMPTZ,
    
    -- Bey/LiveKit Details
    avatar_key VARCHAR(32) DEFAULT 'zara',
    agent_id VARCHAR(64),
    call_id VARCHAR(64),
    livekit_url TEXT,
    livekit_token TEXT,
    livekit_room_name VARCHAR(256),
    bot_id VARCHAR(64),
    
    -- Interview Status
    interview_status VARCHAR(32) DEFAULT 'scheduled',
    
    -- Timing
    avatar_joined_at TIMESTAMPTZ,
    candidate_joined_at TIMESTAMPTZ,
    interview_started_at TIMESTAMPTZ,
    interview_ended_at TIMESTAMPTZ,
    call_duration_seconds INTEGER,
    
    -- Questions
    questions JSONB,
    current_question_index INTEGER DEFAULT 0,
    total_questions INTEGER DEFAULT 8,
    questions_asked INTEGER DEFAULT 0,
    
    -- Transcript & Results
    conversation_log JSONB,
    full_transcript TEXT,
    
    -- Evaluation
    sentiment_score DECIMAL(5,4),
    keyword_matches JSONB,
    evaluation_score DECIMAL(5,2),
    fit_assessment TEXT,
    evaluation_raw JSONB,
    
    -- Milvus Sync
    milvus_synced BOOLEAN DEFAULT FALSE,
    milvus_synced_at TIMESTAMPTZ,
    
    -- Processing
    worker_id VARCHAR(64),
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    
    -- Recruiter Notes
    notes TEXT,
    recruiter_id VARCHAR(64),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_candidate_interviews_interview_id ON candidate_interviews(interview_id);
CREATE INDEX IF NOT EXISTS idx_candidate_interviews_candidate_id ON candidate_interviews(candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_interviews_job_id ON candidate_interviews(job_id);
CREATE INDEX IF NOT EXISTS idx_candidate_interviews_status ON candidate_interviews(interview_status);
CREATE INDEX IF NOT EXISTS idx_candidate_interviews_scheduled ON candidate_interviews(scheduled_time);
CREATE INDEX IF NOT EXISTS idx_candidate_interviews_milvus ON candidate_interviews(milvus_synced);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_candidate_interviews_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_candidate_interviews_updated_at ON candidate_interviews;
CREATE TRIGGER trigger_update_candidate_interviews_updated_at
    BEFORE UPDATE ON candidate_interviews
    FOR EACH ROW
    EXECUTE FUNCTION update_candidate_interviews_updated_at();
"""


async def create_tables() -> bool:
    """Create the candidate_interviews table if not exists."""
    db_url = _get_db_url()
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(CREATE_TABLE_SQL)
            await conn.commit()
        _log_event("info", "db_tables_created")
        return True
    except Exception as e:
        _log_event("error", "db_create_tables_failed", error=str(e))
        raise


# =============================================================================
# CRUD OPERATIONS
# =============================================================================

async def insert_interview(
    interview_id: str,
    candidate_id: str,
    job_id: str,
    scheduled_time: datetime,
    *,
    timezone: str = "UTC",
    avatar_key: str = "zara",
    # Job data
    job_title: Optional[str] = None,
    job_description: Optional[str] = None,
    job_company: Optional[str] = None,
    job_location: Optional[str] = None,
    # Candidate data
    candidate_name: Optional[str] = None,
    candidate_email: Optional[str] = None,
    candidate_phone: Optional[str] = None,
    candidate_skills: Optional[Dict] = None,
    candidate_summary: Optional[str] = None,
    candidate_tech_stack: Optional[List] = None,
    candidate_employment_history: Optional[List] = None,
    # Meeting data
    meeting_id: Optional[str] = None,
    meeting_url: Optional[str] = None,
    meeting_passcode: Optional[str] = None,
    meeting_host_url: Optional[str] = None,
    # Questions
    questions: Optional[List[Dict]] = None,
    total_questions: int = 8,
    # Notes
    notes: Optional[str] = None,
    recruiter_id: Optional[str] = None,
) -> Tuple[int, str]:
    """Insert a new interview record. Returns (id, interview_id)."""
    
    db_url = _get_db_url()
    
    data = {
        "interview_id": interview_id,
        "candidate_id": candidate_id,
        "job_id": job_id,
        "scheduled_time": scheduled_time,
        "timezone": timezone,
        "avatar_key": avatar_key,
        "interview_status": InterviewStatus.SCHEDULED.value,
        "job_title": job_title,
        "job_description": job_description,
        "job_company": job_company,
        "job_location": job_location,
        "candidate_name": candidate_name,
        "candidate_email": candidate_email,
        "candidate_phone": candidate_phone,
        "candidate_skills": candidate_skills,
        "candidate_summary": candidate_summary,
        "candidate_tech_stack": candidate_tech_stack,
        "candidate_employment_history": candidate_employment_history,
        "meeting_id": meeting_id,
        "meeting_url": meeting_url,
        "meeting_passcode": meeting_passcode,
        "meeting_host_url": meeting_host_url,
        "questions": questions,
        "total_questions": total_questions,
        "notes": notes,
        "recruiter_id": recruiter_id,
        "created_at": datetime.now(tz=timezone_utc()),
    }
    
    # Filter out None values and build query
    json_columns = {"candidate_skills", "candidate_tech_stack", "candidate_employment_history", "questions"}
    cols, placeholders, vals = [], [], []
    
    for k, v in data.items():
        if v is None:
            continue
        cols.append(k)
        placeholders.append("%s")
        if k in json_columns:
            vals.append(Jsonb(v))
        else:
            vals.append(v)
    
    query = f"""
        INSERT INTO candidate_interviews ({', '.join(cols)})
        VALUES ({', '.join(placeholders)})
        RETURNING id, interview_id
    """
    
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, vals)
                row = await cur.fetchone()
            await conn.commit()
        
        _log_event("info", "db_insert_interview_ok", interview_id=interview_id, candidate_id=candidate_id)
        return (row["id"], row["interview_id"])
    except Exception as e:
        _log_event("error", "db_insert_interview_failed", interview_id=interview_id, error=str(e))
        raise


async def get_interview(interview_id: str) -> Optional[Dict[str, Any]]:
    """Get interview by interview_id."""
    db_url = _get_db_url()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM candidate_interviews WHERE interview_id = %s",
                    (interview_id,)
                )
                row = await cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        _log_event("error", "db_get_interview_failed", interview_id=interview_id, error=str(e))
        return None


async def get_interview_by_meeting(meeting_id: str) -> Optional[Dict[str, Any]]:
    """Get interview by Zoom meeting ID."""
    db_url = _get_db_url()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM candidate_interviews WHERE meeting_id = %s",
                    (meeting_id,)
                )
                row = await cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        _log_event("error", "db_get_interview_by_meeting_failed", meeting_id=meeting_id, error=str(e))
        return None


async def get_interview_by_call(call_id: str) -> Optional[Dict[str, Any]]:
    """Get interview by Bey call ID."""
    db_url = _get_db_url()
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM candidate_interviews WHERE call_id = %s",
                    (call_id,)
                )
                row = await cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        _log_event("error", "db_get_interview_by_call_failed", call_id=call_id, error=str(e))
        return None


async def list_interviews(
    *,
    status: Optional[str] = None,
    candidate_id: Optional[str] = None,
    job_id: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """List interviews with filters. Returns (interviews, total_count)."""
    db_url = _get_db_url()
    
    where_clauses = []
    params = []
    
    if status:
        where_clauses.append("interview_status = %s")
        params.append(status)
    if candidate_id:
        where_clauses.append("candidate_id = %s")
        params.append(candidate_id)
    if job_id:
        where_clauses.append("job_id = %s")
        params.append(job_id)
    if from_date:
        where_clauses.append("scheduled_time >= %s")
        params.append(from_date)
    if to_date:
        where_clauses.append("scheduled_time <= %s")
        params.append(to_date)
    
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                # Get total count
                await cur.execute(f"SELECT COUNT(*) as cnt FROM candidate_interviews WHERE {where_sql}", params)
                total = (await cur.fetchone())["cnt"]
                
                # Get paginated results
                query = f"""
                    SELECT * FROM candidate_interviews 
                    WHERE {where_sql}
                    ORDER BY scheduled_time DESC
                    LIMIT %s OFFSET %s
                """
                await cur.execute(query, params + [limit, offset])
                rows = await cur.fetchall()
                
        return ([dict(r) for r in rows], total)
    except Exception as e:
        _log_event("error", "db_list_interviews_failed", error=str(e))
        return ([], 0)


async def get_pending_interviews(
    before_time: datetime,
    after_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Get interviews that are scheduled and need to be started."""
    db_url = _get_db_url()
    
    try:
        async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                if after_time:
                    await cur.execute(
                        """
                        SELECT * FROM candidate_interviews 
                        WHERE interview_status = 'scheduled'
                          AND scheduled_time <= %s
                          AND scheduled_time >= %s
                        ORDER BY scheduled_time ASC
                        """,
                        (before_time, after_time)
                    )
                else:
                    await cur.execute(
                        """
                        SELECT * FROM candidate_interviews 
                        WHERE interview_status = 'scheduled'
                          AND scheduled_time <= %s
                        ORDER BY scheduled_time ASC
                        """,
                        (before_time,)
                    )
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        _log_event("error", "db_get_pending_interviews_failed", error=str(e))
        return []


async def update_interview(
    interview_id: str,
    **updates: Any,
) -> bool:
    """Update interview fields."""
    if not updates:
        return True
    
    db_url = _get_db_url()
    json_columns = {
        "candidate_skills", "candidate_tech_stack", "candidate_employment_history",
        "questions", "conversation_log", "keyword_matches", "evaluation_raw"
    }
    
    set_parts, vals = [], []
    for k, v in updates.items():
        if k in json_columns and v is not None:
            set_parts.append(f"{k} = %s")
            vals.append(Jsonb(v))
        else:
            set_parts.append(f"{k} = %s")
            vals.append(v)
    
    vals.append(interview_id)
    query = f"UPDATE candidate_interviews SET {', '.join(set_parts)} WHERE interview_id = %s"
    
    try:
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, vals)
            await conn.commit()
        _log_event("info", "db_update_interview_ok", interview_id=interview_id, fields=list(updates.keys()))
        return True
    except Exception as e:
        _log_event("error", "db_update_interview_failed", interview_id=interview_id, error=str(e))
        return False


async def update_interview_status(
    interview_id: str,
    status: InterviewStatus,
    *,
    error_message: Optional[str] = None,
) -> bool:
    """Update interview status."""
    updates = {"interview_status": status.value}
    if error_message:
        updates["error_message"] = error_message
    
    # Set completed_at if terminal status
    if status in (InterviewStatus.COMPLETED, InterviewStatus.INCOMPLETE, InterviewStatus.FAILED, InterviewStatus.CANCELLED):
        updates["completed_at"] = datetime.now(tz=timezone_utc())
    
    return await update_interview(interview_id, **updates)


async def update_interview_meeting(
    interview_id: str,
    meeting_id: str,
    meeting_url: str,
    meeting_passcode: Optional[str] = None,
    meeting_host_url: Optional[str] = None,
) -> bool:
    """Update interview with Zoom meeting details."""
    return await update_interview(
        interview_id,
        meeting_id=meeting_id,
        meeting_url=meeting_url,
        meeting_passcode=meeting_passcode,
        meeting_host_url=meeting_host_url,
        meeting_created_at=datetime.now(tz=timezone_utc()),
        interview_status=InterviewStatus.MEETING_CREATED.value,
    )


async def update_interview_bey(
    interview_id: str,
    agent_id: str,
    call_id: str,
    livekit_url: str,
    livekit_token: str,
    bot_id: Optional[str] = None,
) -> bool:
    """Update interview with Bey/LiveKit details."""
    return await update_interview(
        interview_id,
        agent_id=agent_id,
        call_id=call_id,
        livekit_url=livekit_url,
        livekit_token=livekit_token,
        bot_id=bot_id,
        avatar_joined_at=datetime.now(tz=timezone_utc()),
        interview_status=InterviewStatus.WAITING_FOR_CANDIDATE.value,
    )


async def update_interview_results(
    interview_id: str,
    *,
    conversation_log: Optional[List[Dict]] = None,
    full_transcript: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    evaluation_score: Optional[float] = None,
    fit_assessment: Optional[str] = None,
    keyword_matches: Optional[Dict] = None,
    evaluation_raw: Optional[Dict] = None,
    questions_asked: Optional[int] = None,
    call_duration_seconds: Optional[int] = None,
) -> bool:
    """Update interview with results after completion."""
    updates = {
        "interview_ended_at": datetime.now(tz=timezone_utc()),
        "interview_status": InterviewStatus.COMPLETED.value,
        "completed_at": datetime.now(tz=timezone_utc()),
    }
    
    if conversation_log is not None:
        updates["conversation_log"] = conversation_log
    if full_transcript is not None:
        updates["full_transcript"] = full_transcript
    if sentiment_score is not None:
        updates["sentiment_score"] = sentiment_score
    if evaluation_score is not None:
        updates["evaluation_score"] = evaluation_score
    if fit_assessment is not None:
        updates["fit_assessment"] = fit_assessment
    if keyword_matches is not None:
        updates["keyword_matches"] = keyword_matches
    if evaluation_raw is not None:
        updates["evaluation_raw"] = evaluation_raw
    if questions_asked is not None:
        updates["questions_asked"] = questions_asked
    if call_duration_seconds is not None:
        updates["call_duration_seconds"] = call_duration_seconds
    
    return await update_interview(interview_id, **updates)


# =============================================================================
# HEALTH CHECK
# =============================================================================

async def check_db_connection() -> Tuple[bool, str]:
    """Check database connection health."""
    try:
        db_url = _get_db_url()
        async with await AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        return (True, "ok")
    except Exception as e:
        _log_event("error", "db_health_failed", error=str(e))
        return (False, str(e))


def timezone_utc():
    """Return UTC timezone."""
    return timezone.utc
