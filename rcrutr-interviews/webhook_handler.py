"""Webhook handler for processing Bey call_ended events."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from models import BeyCallEndedPayload, InterviewStatus
from db import (
    get_interview_by_call,
    update_interview_results,
    update_interview_status,
    update_interview,
)
from milvus_client import store_qa_embeddings

logger = logging.getLogger("rcrutr_interviews_webhook")


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


async def process_call_ended(payload: BeyCallEndedPayload) -> Dict[str, Any]:
    """
    Process a call_ended webhook from Bey.
    
    This function:
    1. Finds the interview by call_id
    2. Parses the transcript
    3. Extracts Q&A pairs
    4. Calculates sentiment and evaluation scores
    5. Stores results in PostgreSQL
    6. Stores Q&A embeddings in Milvus
    
    Returns:
        Dict with processing results
    """
    call_id = payload.call_id
    _log_event("info", "webhook_call_ended_received", call_id=call_id)
    
    # Find the interview
    interview = await get_interview_by_call(call_id)
    if not interview:
        _log_event("warning", "webhook_interview_not_found", call_id=call_id)
        return {"status": "error", "message": "Interview not found for call_id"}
    
    interview_id = interview["interview_id"]
    _log_event("info", "webhook_processing_interview", 
               interview_id=interview_id, call_id=call_id)
    
    try:
        # Parse messages from payload
        messages = payload.messages
        evaluation = payload.evaluation or {}
        call_data = payload.call_data
        
        # Calculate duration
        started_at = call_data.get("startedAt")
        ended_at = call_data.get("endedAt")
        duration_seconds = None
        
        if started_at and ended_at:
            try:
                start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
                duration_seconds = int((end - start).total_seconds())
            except Exception as e:
                _log_event("warning", "webhook_duration_parse_failed", error=str(e))
        
        # Build full transcript
        full_transcript = build_transcript(messages)
        
        # Extract Q&A pairs
        qa_pairs = extract_qa_pairs(messages)
        
        # Calculate sentiment score
        sentiment_score = parse_sentiment(evaluation)
        
        # Extract keyword matches from transcript
        keyword_matches = extract_keywords(
            full_transcript,
            interview.get("candidate_skills", {}),
            interview.get("job_description", ""),
        )
        
        # Calculate evaluation score (simple heuristic for now)
        evaluation_score = calculate_evaluation_score(
            qa_pairs,
            keyword_matches,
            sentiment_score,
        )
        
        # Generate fit assessment
        fit_assessment = generate_fit_assessment(
            qa_pairs,
            keyword_matches,
            sentiment_score,
            evaluation_score,
        )
        
        # Update interview in database
        await update_interview_results(
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
        )
        
        _log_event("info", "webhook_interview_updated", 
                   interview_id=interview_id,
                   questions_asked=len(qa_pairs),
                   duration_seconds=duration_seconds)
        
        # Store Q&A embeddings in Milvus
        if qa_pairs:
            milvus_success = store_qa_embeddings(
                interview_id=interview_id,
                candidate_id=interview["candidate_id"],
                job_id=interview["job_id"],
                job_title=interview.get("job_title", ""),
                job_description=interview.get("job_description", ""),
                qa_pairs=qa_pairs,
            )
            
            if milvus_success:
                await update_interview(interview_id, milvus_synced=True, milvus_synced_at=datetime.now(tz=timezone.utc))
                _log_event("info", "webhook_milvus_synced", interview_id=interview_id)
            else:
                _log_event("warning", "webhook_milvus_sync_failed", interview_id=interview_id)
        
        return {
            "status": "success",
            "interview_id": interview_id,
            "questions_asked": len(qa_pairs),
            "duration_seconds": duration_seconds,
            "sentiment_score": sentiment_score,
            "evaluation_score": evaluation_score,
        }
        
    except Exception as e:
        _log_event("error", "webhook_processing_failed", 
                   interview_id=interview_id, error=str(e))
        await update_interview_status(
            interview_id,
            InterviewStatus.FAILED,
            error_message=f"Webhook processing failed: {str(e)}",
        )
        return {"status": "error", "message": str(e)}


def build_transcript(messages: List[Dict[str, Any]]) -> str:
    """Build a readable transcript from messages."""
    lines = []
    for msg in messages:
        sender = msg.get("sender", "unknown")
        message = msg.get("message", "")
        
        # Map sender to role
        if sender == "ai":
            role = "Interviewer"
        elif sender == "user":
            role = "Candidate"
        else:
            role = sender.title()
        
        lines.append(f"{role}: {message}")
    
    return "\n".join(lines)


def extract_qa_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract question-answer pairs from messages.
    
    Groups AI questions with subsequent user responses.
    """
    qa_pairs = []
    current_question = None
    question_index = 0
    
    for msg in messages:
        sender = msg.get("sender", "")
        message = msg.get("message", "").strip()
        sent_at = msg.get("sent_at", "")
        
        if sender == "ai":
            # Check if this is a question
            if is_question(message):
                if current_question:
                    # Save previous Q&A if we have an answer
                    if current_question.get("answer"):
                        qa_pairs.append(current_question)
                
                question_index += 1
                current_question = {
                    "index": question_index,
                    "question": message,
                    "answer": "",
                    "timestamp": sent_at,
                    "question_time": sent_at,
                    "answer_time": None,
                }
        
        elif sender == "user" and current_question:
            # Append to current answer
            if current_question["answer"]:
                current_question["answer"] += " " + message
            else:
                current_question["answer"] = message
            current_question["answer_time"] = sent_at
    
    # Don't forget the last Q&A pair
    if current_question and current_question.get("answer"):
        qa_pairs.append(current_question)
    
    return qa_pairs


def is_question(text: str) -> bool:
    """Check if a message is a question."""
    text = text.strip()
    
    # Direct question mark
    if text.endswith("?"):
        return True
    
    # Question starters
    question_starters = [
        "can you", "could you", "would you", "do you", "did you",
        "have you", "are you", "were you", "what", "where", "when",
        "why", "how", "tell me", "describe", "explain", "walk me",
    ]
    
    text_lower = text.lower()
    return any(text_lower.startswith(starter) for starter in question_starters)


def parse_sentiment(evaluation: Dict[str, Any]) -> Optional[float]:
    """Parse sentiment from Bey evaluation."""
    sentiment = evaluation.get("user_sentiment", "")
    
    # Map sentiment labels to scores
    sentiment_map = {
        "very_satisfied": 0.9,
        "satisfied": 0.7,
        "neutral": 0.5,
        "dissatisfied": 0.3,
        "very_dissatisfied": 0.1,
        # Also handle simple labels
        "positive": 0.7,
        "negative": 0.3,
    }
    
    sentiment_lower = sentiment.lower().replace(" ", "_")
    return sentiment_map.get(sentiment_lower, 0.5)


def extract_keywords(
    transcript: str,
    candidate_skills: Dict[str, Any],
    job_description: str,
) -> Dict[str, Any]:
    """Extract mentioned keywords from transcript."""
    transcript_lower = transcript.lower()
    
    # Build list of keywords to look for
    keywords_to_check = set()
    
    # From candidate skills
    if isinstance(candidate_skills, dict):
        for skill in candidate_skills.keys():
            keywords_to_check.add(skill.lower())
    elif isinstance(candidate_skills, list):
        for skill in candidate_skills:
            if isinstance(skill, str):
                keywords_to_check.add(skill.lower())
    
    # Common tech keywords
    common_tech = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "nodejs", "django", "flask", "fastapi", "spring", "springboot",
        "aws", "azure", "gcp", "kubernetes", "docker", "jenkins", "ci/cd",
        "sql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "api", "rest", "graphql", "microservices", "agile", "scrum",
    ]
    keywords_to_check.update(common_tech)
    
    # Find matches
    matched = {}
    for keyword in keywords_to_check:
        # Use word boundary matching
        pattern = r'\b' + re.escape(keyword) + r'\b'
        count = len(re.findall(pattern, transcript_lower))
        if count > 0:
            matched[keyword] = count
    
    return {
        "keywords_found": matched,
        "total_keywords": len(matched),
        "top_keywords": sorted(matched.items(), key=lambda x: x[1], reverse=True)[:10],
    }


def calculate_evaluation_score(
    qa_pairs: List[Dict[str, Any]],
    keyword_matches: Dict[str, Any],
    sentiment_score: Optional[float],
) -> float:
    """
    Calculate an overall evaluation score.
    
    This is a simple heuristic - in production you might use an LLM for this.
    """
    score = 50.0  # Start at 50
    
    # Factor 1: Number of questions answered (up to +20)
    questions_answered = len(qa_pairs)
    score += min(questions_answered * 2.5, 20)
    
    # Factor 2: Keywords mentioned (up to +15)
    keywords_count = keyword_matches.get("total_keywords", 0)
    score += min(keywords_count * 1.5, 15)
    
    # Factor 3: Sentiment (up to +15)
    if sentiment_score:
        # Sentiment 0.5 = neutral = +7.5, 1.0 = very positive = +15
        score += sentiment_score * 15
    else:
        score += 7.5  # Neutral default
    
    # Factor 4: Answer length (engagement indicator, up to +10)
    if qa_pairs:
        avg_answer_length = sum(len(qa.get("answer", "")) for qa in qa_pairs) / len(qa_pairs)
        # 100+ chars average = good engagement
        score += min(avg_answer_length / 10, 10)
    
    # Cap at 100
    return min(round(score, 2), 100.0)


def generate_fit_assessment(
    qa_pairs: List[Dict[str, Any]],
    keyword_matches: Dict[str, Any],
    sentiment_score: Optional[float],
    evaluation_score: float,
) -> str:
    """Generate a brief fit assessment summary."""
    
    # Determine fit level
    if evaluation_score >= 80:
        fit_level = "Strong"
        recommendation = "Recommend proceeding to next round"
    elif evaluation_score >= 60:
        fit_level = "Good"
        recommendation = "Consider for next round with additional screening"
    elif evaluation_score >= 40:
        fit_level = "Moderate"
        recommendation = "May need additional evaluation"
    else:
        fit_level = "Weak"
        recommendation = "Not recommended for this role"
    
    # Get top keywords
    top_keywords = keyword_matches.get("top_keywords", [])
    skills_mentioned = ", ".join([kw[0] for kw in top_keywords[:5]]) if top_keywords else "None detected"
    
    # Sentiment description
    if sentiment_score and sentiment_score >= 0.7:
        sentiment_desc = "positive and engaged"
    elif sentiment_score and sentiment_score <= 0.3:
        sentiment_desc = "reserved or uncertain"
    else:
        sentiment_desc = "neutral"
    
    assessment = f"""
Fit Assessment: {fit_level}
Score: {evaluation_score}/100

Summary:
- Questions answered: {len(qa_pairs)}
- Technical skills discussed: {skills_mentioned}
- Candidate demeanor: {sentiment_desc}

Recommendation: {recommendation}
""".strip()
    
    return assessment


async def process_incomplete_interview(
    interview_id: str,
    reason: str = "Candidate did not join",
) -> bool:
    """Mark an interview as incomplete (e.g., candidate no-show)."""
    try:
        await update_interview_status(
            interview_id,
            InterviewStatus.INCOMPLETE,
            error_message=reason,
        )
        _log_event("info", "interview_marked_incomplete", 
                   interview_id=interview_id, reason=reason)
        return True
    except Exception as e:
        _log_event("error", "mark_incomplete_failed", 
                   interview_id=interview_id, error=str(e))
        return False
