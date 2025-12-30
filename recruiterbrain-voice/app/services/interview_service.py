"""
Interview Service - Batch-Optimized Business Logic
Handles interview lifecycle with minimal database writes
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from app.services.database import (
    create_interview_record,
    update_interview_status,
    save_interview_completion,
    insert_interview_to_milvus
)


logger = logging.getLogger(__name__)


class InterviewSession:
    """
    In-memory interview session
    Collects data during interview, writes in batch at end
    """
    
    def __init__(self, interview_id: str, candidate_data: Dict, jd_data: Dict):
        self.interview_id = interview_id
        self.candidate_data = candidate_data
        self.jd_data = jd_data
        
        # In-memory transcript (not written to DB until interview ends)
        self.transcript: List[Dict[str, Any]] = []
        
        # Metadata
        self.started_at: datetime = None
        self.completed_at: datetime = None
        self.current_question_index: int = 0
        self.questions_asked: int = 0
        self.questions_completed: int = 0
        
        # Skills tracking
        self.skills_discussed: List[str] = []
    
    def add_agent_message(self, text: str, question_index: int = None):
        """Add agent message to transcript (in memory)"""
        self.transcript.append({
            "speaker": "agent",
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
            "question_index": question_index
        })
        
        if question_index is not None:
            self.questions_asked += 1
            self.current_question_index = question_index
    
    def add_candidate_message(self, text: str, duration_seconds: int = None):
        """Add candidate message to transcript (in memory)"""
        self.transcript.append({
            "speaker": "candidate",
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
            "question_index": self.current_question_index,
            "duration_seconds": duration_seconds
        })
        
        self.questions_completed += 1
    
    def extract_full_text(self) -> str:
        """Extract full conversation as text for embedding"""
        parts = []
        for turn in self.transcript:
            speaker = turn["speaker"].title()
            text = turn["text"]
            parts.append(f"{speaker}: {text}")
        
        return "\n\n".join(parts)
    
    async def save_completion(self, evaluation: Dict[str, Any]):
        """
        Save interview completion - BATCH WRITE
        
        This is called ONCE at the end of interview
        Writes everything to PostgreSQL and Milvus in one go
        """
        try:
            self.completed_at = datetime.utcnow()
            duration = int((self.completed_at - self.started_at).total_seconds())
            
            # 1. Save to PostgreSQL (batch write)
            await save_interview_completion(
                interview_id=self.interview_id,
                transcript=self.transcript,
                evaluation={
                    "questions_asked": self.questions_asked,
                    "questions_completed": self.questions_completed,
                    "score": evaluation.get("score", 0.0),
                    "summary": evaluation.get("summary", ""),
                    "skills_discussed": self.skills_discussed,
                    "skills_coverage": evaluation.get("skills_coverage", {})
                },
                duration_seconds=duration
            )
            
            # 2. Insert to Milvus (async batch)
            transcript_text = self.extract_full_text()
            
            await insert_interview_to_milvus(
                interview_id=self.interview_id,
                candidate_id=self.candidate_data["candidate_id"],
                job_id=self.jd_data["job_id"],
                job_title=self.jd_data.get("title", ""),
                transcript_text=transcript_text,
                evaluation_summary=evaluation.get("summary", ""),
                skills_discussed=self.skills_discussed,
                evaluation_score=evaluation.get("score", 0.0),
                interview_date=self.started_at
            )
            
            logger.info(f"✅ Interview {self.interview_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving interview completion: {e}")
            return False


# Global session storage (in-memory)
# Key: interview_id, Value: InterviewSession
active_sessions: Dict[str, InterviewSession] = {}


async def start_interview_session(
    interview_id: str,
    candidate_data: Dict[str, Any],
    jd_data: Dict[str, Any],
    livekit_room: str
) -> InterviewSession:
    """
    Start interview session
    
    Creates minimal database record, keeps session in memory
    """
    # Create database record (minimal write)
    await create_interview_record(
        interview_id=interview_id,
        candidate_data=candidate_data,
        jd_data=jd_data,
        livekit_room=livekit_room
    )
    
    # Create in-memory session
    session = InterviewSession(interview_id, candidate_data, jd_data)
    session.started_at = datetime.utcnow()
    
    # Store in global dict
    active_sessions[interview_id] = session
    
    logger.info(f"Started interview session: {interview_id}")
    return session


async def complete_interview_session(
    interview_id: str,
    evaluation: Dict[str, Any]
) -> bool:
    """
    Complete interview session
    
    Batch writes everything to PostgreSQL + Milvus
    """
    session = active_sessions.get(interview_id)
    
    if not session:
        logger.error(f"Session {interview_id} not found")
        return False
    
    # Save everything in batch
    success = await session.save_completion(evaluation)
    
    # Remove from memory
    if success:
        del active_sessions[interview_id]
    
    return success


def get_session(interview_id: str) -> InterviewSession:
    """Get active interview session"""
    return active_sessions.get(interview_id)
