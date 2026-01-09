"""
Data Models
Pydantic models for request/response validation
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


# ==========================================
# Candidate Models
# ==========================================

class CandidateData(BaseModel):
    """Candidate information"""
    candidate_id: str = Field(..., description="Unique candidate identifier")
    name: str = Field(..., min_length=1, max_length=255)
    email: Optional[str] = Field(None, description="Candidate email")
    phone_number: str = Field(..., description="Phone number in E.164 format")
    
    # Background
    projects: List[str] = Field(default_factory=list, description="List of past projects")
    skills: List[str] = Field(default_factory=list, description="Technical skills")
    experience_years: Optional[int] = Field(None, ge=0, le=50)
    current_company: Optional[str] = Field(None, max_length=200, description="Current employer")
    resume_summary: Optional[str] = Field(None, max_length=1000, description="Brief resume summary")
    
    
    # Metadata
    timezone: str = Field(default="UTC", description="Candidate timezone")
    preferred_language: str = Field(default="en-US", description="Preferred language code")
    
    @validator("phone_number")
    def validate_phone(cls, v):
        """Ensure phone number is in E.164 format"""
        if not v.startswith("+"):
            raise ValueError("Phone number must be in E.164 format (e.g., +14155551234)")
        return v


# ==========================================
# Job Description Models
# ==========================================

class JobDescriptionData(BaseModel):
    """Job description information"""
    job_id: str = Field(..., description="Unique job description identifier")
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Full job description")
    
    # Requirements
    requirements: List[str] = Field(default_factory=list, description="Required skills/qualifications")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have skills")
    experience_required: Optional[int] = Field(None, ge=0, le=20, description="Years of experience")
    
    # Compensation (optional)
    salary_min: Optional[int] = Field(None, description="Minimum salary")
    salary_max: Optional[int] = Field(None, description="Maximum salary")
    
    # Location
    location: Optional[str] = Field(None, description="Job location")
    remote_ok: bool = Field(default=False, description="Remote work allowed")


# ==========================================
# Interview Request/Response Models
# ==========================================

class StartInterviewRequest(BaseModel):
    """Request to start a new interview"""
    candidate: CandidateData
    job_description: JobDescriptionData
    
    # Scheduling
    scheduled_time: Optional[datetime] = Field(None, description="When to conduct interview")
    
    # Customization
    custom_questions: Optional[List[str]] = Field(None, description="Custom questions to ask")
    interview_duration_minutes: int = Field(default=12, ge=5, le=30, description="Interview duration")


class InterviewResponse(BaseModel):
    """Response after starting interview"""
    interview_id: str
    status: str = Field(..., description="initiated, calling, in_progress, completed, failed")
    
    # LiveKit details
    livekit_room_name: str
    
    # Call details
    call_sid: Optional[str] = Field(None, description="Telnyx call SID")
    
    # Timestamps
    created_at: datetime
    scheduled_time: Optional[datetime] = None
    
    # Links
    interview_url: Optional[str] = Field(None, description="URL to view interview")


class InterviewStatus(BaseModel):
    """Current interview status"""
    interview_id: str
    status: str
    
    # Progress
    questions_asked: int = 0
    questions_completed: int = 0
    current_question_index: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # Results (if completed)
    evaluation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recording_url: Optional[str] = None


# ==========================================
# Evaluation Models
# ==========================================

class InterviewEvaluation(BaseModel):
    """Interview evaluation results"""
    interview_id: str
    
    # Scores
    evaluation_score: float = Field(..., ge=0.0, le=1.0, description="Overall score")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment analysis")
    
    # Analysis
    summary: str = Field(..., description="Interview summary")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    
    # Skills
    skills_discussed: List[str] = Field(default_factory=list)
    skills_coverage: Dict[str, Any] = Field(default_factory=dict)
    
    # Recommendations
    recommendation: str = Field(..., description="hire, reject, maybe")
    fit_assessment: str = Field(..., description="Detailed fit assessment")
    
    # Metadata
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)


# ==========================================
# Transcript Models
# ==========================================

class ConversationTurn(BaseModel):
    """Single conversation turn"""
    speaker: str = Field(..., description="agent or candidate")
    text: str
    timestamp: datetime
    
    # Metadata
    question_index: Optional[int] = None
    duration_seconds: Optional[int] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class InterviewTranscript(BaseModel):
    """Full interview transcript"""
    interview_id: str
    conversation: List[ConversationTurn]
    
    # Summary
    full_text: str = Field(..., description="Complete transcript as text")
    word_count: int = Field(..., ge=0)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ==========================================
# Search Models
# ==========================================

class InterviewSearchRequest(BaseModel):
    """Search for similar interviews"""
    query: str = Field(..., min_length=1, description="Search query")
    
    # Filters
    candidate_id: Optional[str] = None
    job_id: Optional[str] = None
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Date range
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    
    # Pagination
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class InterviewSearchResult(BaseModel):
    """Search result item"""
    interview_id: str
    candidate_id: str
    job_title: str
    
    # Relevance
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    evaluation_score: Optional[float] = None
    
    # Preview
    summary: str
    skills_discussed: List[str]
    
    # Metadata
    interview_date: datetime
    
    # Links
    interview_url: str


class InterviewSearchResponse(BaseModel):
    """Search results"""
    query: str
    results: List[InterviewSearchResult]
    
    # Pagination
    total_results: int
    limit: int
    offset: int
    
    # Performance
    search_time_ms: float
