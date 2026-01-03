"""
Real-time conversation logger
Logs every turn during the interview
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Logs conversation turns in real-time"""
    
    def __init__(self, interview_id: str, candidate_data: Dict[str, Any], job_data: Dict[str, Any]):
        self.interview_id = interview_id
        self.candidate_data = candidate_data
        self.job_data = job_data
        self.turns: List[Dict[str, Any]] = []
        self.start_time = datetime.utcnow()
        self.questions: List[Dict[str, Any]] = []
        self.current_question = None
        
        logger.info(f"📝 ConversationLogger initialized for {interview_id}")
    
    def log_agent_turn(self, text: str, timestamp: datetime = None):
        """Log when agent speaks (question)"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        turn = {
            "speaker": "agent",
            "text": text,
            "timestamp": timestamp.isoformat(),
            "elapsed_seconds": (timestamp - self.start_time).total_seconds()
        }
        
        self.turns.append(turn)
        
        # Track as potential question
        self.current_question = {
            "text": text,
            "asked_at": timestamp
        }
        
        logger.debug(f"💬 [AGENT]: {text[:80]}...")
    
    def log_candidate_turn(self, text: str, timestamp: datetime = None):
        """Log when candidate speaks (answer)"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        turn = {
            "speaker": "candidate",
            "text": text,
            "timestamp": timestamp.isoformat(),
            "elapsed_seconds": (timestamp - self.start_time).total_seconds()
        }
        
        self.turns.append(turn)
        
        # Pair with last question if exists
        if self.current_question:
            answer_duration = (timestamp - self.current_question["asked_at"]).total_seconds()
            self.questions.append({
                "question": self.current_question["text"],
                "answer": text,
                "asked_at": self.current_question["asked_at"],
                "answered_at": timestamp,
                "duration_seconds": answer_duration
            })
            self.current_question = None
        
        logger.debug(f"💬 [CANDIDATE]: {text[:80]}...")
    
    def get_qa_pairs(self) -> List[Dict[str, Any]]:
        """Extract Q&A pairs from conversation"""
        return self.questions
    
    def get_full_conversation_json(self) -> Dict[str, Any]:
        """Get complete conversation as JSON"""
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        return {
            "interview_id": self.interview_id,
            "candidate": self.candidate_data,
            "job": self.job_data,
            "metadata": {
                "total_turns": len(self.turns),
                "total_questions": len(self.questions),
                "duration_seconds": duration,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "turns": self.turns,
            "qa_pairs": []  # Will be populated by processor
        }
    
    def get_full_transcript_text(self) -> str:
        """Get conversation as plain text"""
        lines = []
        for turn in self.turns:
            speaker = "Interviewer" if turn["speaker"] == "agent" else "Candidate"
            lines.append(f"{speaker}: {turn['text']}")
        return "\n\n".join(lines)