"""
Application constants and configuration values.
"""

from enum import Enum


# Interview States
class InterviewState(str, Enum):
    """Possible states during an interview."""
    IDLE = "idle"
    GREETING = "greeting"
    ASKING_QUESTION = "asking_question"
    LISTENING = "listening"
    PROCESSING_ANSWER = "processing_answer"
    RESPONDING = "responding"
    ASKING_FOLLOWUP = "asking_followup"
    CLOSING = "closing"
    ERROR_RECOVERY = "error_recovery"
    COMPLETED = "completed"
    FAILED = "failed"


# Speaker Types
class Speaker(str, Enum):
    """Who is speaking in the conversation."""
    ASSISTANT = "assistant"
    CANDIDATE = "candidate"


# Turn Types
class TurnType(str, Enum):
    """Type of conversational turn."""
    GREETING = "greeting"
    QUESTION = "question"
    ANSWER = "answer"
    FOLLOWUP = "followup"
    ACKNOWLEDGMENT = "acknowledgment"
    CANDIDATE_QUESTION = "candidate_question"
    ANSWER_TO_CANDIDATE = "answer_to_candidate"
    CLARIFICATION = "clarification"
    CLOSING = "closing"


# LLM Decision Actions
class LLMAction(str, Enum):
    """Possible actions LLM can decide to take."""
    ASK_FOLLOWUP = "ask_followup"
    NEXT_QUESTION = "next_question"
    ACKNOWLEDGE = "acknowledge"
    REPHRASE = "rephrase"
    ANSWER_CANDIDATE_QUESTION = "answer_candidate_question"
    SKIP_QUESTION = "skip_question"
    END_INTERVIEW = "end_interview"


# Question Priority Levels
class QuestionPriority(str, Enum):
    """Priority levels for interview questions."""
    CRITICAL = "critical"  # Must ask
    HIGH = "high"          # Important
    MEDIUM = "medium"      # Can skip if time-pressed
    LOW = "low"            # Optional


# Interview Status
class InterviewStatus(str, Enum):
    """Overall interview lifecycle status."""
    PENDING = "pending"
    CONSENT_SENT = "consent_sent"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Call Status
class CallStatus(str, Enum):
    """Telnyx call status."""
    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no-answer"
    CANCELLED = "cancelled"


# Sentiment Types
class Sentiment(str, Enum):
    """Answer sentiment classification."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    TECHNICAL = "technical"


# Fit Assessment
class FitAssessment(str, Enum):
    """Post-interview fit evaluation."""
    STRONG_FIT = "strong_fit"
    GOOD_FIT = "good_fit"
    WEAK_FIT = "weak_fit"
    NO_FIT = "no_fit"


# Audio Formats
AUDIO_FORMAT_MULAW = "mulaw"
AUDIO_FORMAT_PCM = "pcm16"
AUDIO_SAMPLE_RATE = 8000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE_MS = 20  # Telnyx sends 20ms chunks


# Silence Thresholds (seconds)
SILENCE_THRESHOLDS = {
    "thinking_pause": 3,      # Candidate thinking
    "needs_prompt": 7,        # Prompt "take your time"
    "skip_offer": 15,         # Offer to skip question
}

# Question-specific silence thresholds
QUESTION_SILENCE_THRESHOLDS = {
    "behavioral": {
        "thinking_pause": 5,
        "needs_prompt": 10,
        "skip_offer": 20,
    },
    "technical": {
        "thinking_pause": 3,
        "needs_prompt": 7,
        "skip_offer": 15,
    },
    "simple_factual": {
        "thinking_pause": 2,
        "needs_prompt": 5,
        "skip_offer": 10,
    }
}


# Time Budgets (seconds)
TIME_BUDGET = {
    "greeting": 15,
    "per_question": 90,      # Question (10s) + Answer (60s) + Ack (5s) + Gap (15s)
    "closing": 20,
    "buffer": 60,
    "total_target": 720,     # 12 minutes
}


# Answer Quality Thresholds
ANSWER_QUALITY_THRESHOLDS = {
    "excellent": 0.85,
    "good": 0.70,
    "acceptable": 0.50,
    "poor": 0.30,
}


# STT Confidence Thresholds
STT_CONFIDENCE_THRESHOLDS = {
    "high": 0.90,
    "medium": 0.70,
    "low": 0.50,
}


# Error Retry Limits
ERROR_RETRY_LIMITS = {
    "no_audio_detected": 2,
    "low_confidence": 1,
    "connection_lost": 2,
}


# Milvus Collections
MILVUS_COLLECTIONS = {
    "candidates": "candidates_v3",
    "interviews": "interview_transcripts_v1",
}


# Common TTS Phrases (for caching)
COMMON_TTS_PHRASES = [
    "Great!",
    "Interesting.",
    "Thank you.",
    "I see.",
    "That makes sense.",
    "Could you elaborate?",
    "Take your time.",
    "Are you still there?",
    "Let's move to the next question.",
]


# Voice Settings
VOICE_SETTINGS = {
    "female_professional": {
        "name": "en-US-Neural2-F",
        "speaking_rate": 1.0,
        "pitch": 0.0,
    },
    "female_warm": {
        "name": "en-US-Neural2-C",
        "speaking_rate": 0.95,
        "pitch": 2.0,
    },
    "male_professional": {
        "name": "en-US-Neural2-J",
        "speaking_rate": 1.0,
        "pitch": 0.0,
    },
}


# Default Voice
DEFAULT_VOICE = "female_professional"
