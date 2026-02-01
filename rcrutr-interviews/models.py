"""Pydantic models for RCRUTR Interviews."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class InterviewStatus(str, Enum):
    """Interview status enum."""
    SCHEDULED = "scheduled"
    MEETING_CREATED = "meeting_created"
    AVATAR_JOINING = "avatar_joining"
    WAITING_FOR_CANDIDATE = "waiting_for_candidate"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"  # Candidate didn't join / timeout
    FAILED = "failed"  # Technical failure
    CANCELLED = "cancelled"


class ParticipantAction(str, Enum):
    """Action to take on participant."""
    ADMIT = "admit"
    REJECT = "reject"
    WAITING = "waiting"


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ScheduleInterviewRequest(BaseModel):
    """Request to schedule a new interview."""
    candidate_id: str = Field(..., description="Candidate ID from Milvus candidates_v3")
    job_id: str = Field(..., description="Job ID from Milvus job_postings")
    scheduled_time: datetime = Field(..., description="When the interview should start")
    timezone: str = Field(default="UTC", description="Timezone for the interview")
    avatar: str = Field(default="zara", description="Avatar to use (scott, sam, zara)")
    
    # Optional overrides
    candidate_name: Optional[str] = Field(None, description="Override candidate name")
    candidate_email: Optional[str] = Field(None, description="Override candidate email")
    notes: Optional[str] = Field(None, description="Recruiter notes")


class StartInterviewRequest(BaseModel):
    """Request to manually start an interview."""
    interview_id: str


class CancelInterviewRequest(BaseModel):
    """Request to cancel an interview."""
    reason: Optional[str] = None


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class InterviewResponse(BaseModel):
    """Interview details response."""
    interview_id: str
    candidate_id: str
    job_id: str
    
    # Candidate info
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    candidate_phone: Optional[str] = None
    
    # Job info
    job_title: Optional[str] = None
    
    # Scheduling
    scheduled_time: Optional[datetime] = None
    timezone: str = "UTC"
    
    # Meeting
    meeting_id: Optional[str] = None
    meeting_url: Optional[str] = None
    meeting_passcode: Optional[str] = None
    
    # Status
    status: InterviewStatus = InterviewStatus.SCHEDULED
    
    # Timing
    avatar_joined_at: Optional[datetime] = None
    candidate_joined_at: Optional[datetime] = None
    interview_started_at: Optional[datetime] = None
    interview_ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # Results (after completion)
    total_questions: int = 0
    questions_asked: int = 0
    sentiment_score: Optional[float] = None
    evaluation_score: Optional[float] = None
    fit_assessment: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ScheduleInterviewResponse(BaseModel):
    """Response after scheduling interview."""
    interview_id: str
    meeting_url: str
    meeting_passcode: Optional[str] = None
    scheduled_time: datetime
    candidate_name: str
    job_title: str
    status: InterviewStatus


class InterviewListResponse(BaseModel):
    """List of interviews."""
    interviews: List[InterviewResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# CANDIDATE & JOB MODELS
# =============================================================================

class CandidateData(BaseModel):
    """Candidate data from Milvus."""
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    semantic_summary: Optional[str] = None
    current_tech_stack: Optional[List[str]] = None
    top_5_skills_with_years: Optional[str] = None  # e.g., "Python:5, Java:3"
    employment_history: Optional[List[Dict[str, Any]]] = None


class JobData(BaseModel):
    """Job data from Milvus."""
    job_id: str
    title: Optional[str] = None
    company: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    salary_range: Optional[str] = None
    jd_text: Optional[str] = None


# =============================================================================
# INTERVIEW QUESTION MODELS
# =============================================================================

class InterviewQuestion(BaseModel):
    """Single interview question."""
    index: int
    question_type: str  # "basic" or "technical"
    question: str
    expected_keywords: Optional[List[str]] = None
    is_followup: bool = False
    parent_index: Optional[int] = None  # If followup, which question it follows


class InterviewQuestions(BaseModel):
    """Complete set of interview questions."""
    total: int
    basic_questions: List[InterviewQuestion]
    technical_questions: List[InterviewQuestion]
    
    def all_questions(self) -> List[InterviewQuestion]:
        """Get all questions in order."""
        return self.basic_questions + self.technical_questions


# =============================================================================
# WEBHOOK MODELS
# =============================================================================

class BeyCallEndedPayload(BaseModel):
    """Payload from Bey call_ended webhook."""
    event_type: str
    call_id: str
    call_data: Dict[str, Any]
    evaluation: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]]
    sentiment_disclaimer: Optional[str] = None


class ZoomParticipantJoinedPayload(BaseModel):
    """Payload from Zoom participant.joined webhook."""
    event: str
    payload: Dict[str, Any]


# =============================================================================
# ZOOM MODELS
# =============================================================================

class ZoomMeeting(BaseModel):
    """Zoom meeting details."""
    id: str
    join_url: str
    start_url: str
    password: Optional[str] = None
    topic: str
    start_time: datetime
    duration: int  # minutes
    timezone: str


class ZoomParticipant(BaseModel):
    """Zoom meeting participant."""
    id: str
    user_name: str
    email: Optional[str] = None
    join_time: Optional[datetime] = None
    status: str  # "waiting", "in_meeting"


# =============================================================================
# BEY MODELS
# =============================================================================

class BeyAgent(BaseModel):
    """Bey agent details."""
    id: str
    name: str


class BeyCall(BaseModel):
    """Bey call session details."""
    id: str
    agent_id: str
    livekit_url: str
    livekit_token: str


class BeySendToExternalResponse(BaseModel):
    """Response from Bey send-to-external."""
    bot_id: str
    status: str
