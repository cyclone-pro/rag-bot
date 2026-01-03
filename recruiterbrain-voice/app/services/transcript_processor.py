"""
Post-interview processing
Processes conversation log and generates embeddings
"""

import logging
import json
from typing import Dict, Any, List
from datetime import datetime

from app.utils.sentiment_analyzer import analyze_sentiment, get_sentiment_label
from app.utils.keyword_extractor import extract_keywords
from app.services.embedding_service import get_embedding_service
from app.services.milvus_service import get_milvus_service
from app.services.database import get_db_session
from sqlalchemy import text

logger = logging.getLogger(__name__)


async def process_interview_transcript(
    interview_id: str,
    conversation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process interview transcript after call ends
    
    Steps:
    1. Extract Q&A pairs from turns
    2. Analyze sentiment for each answer
    3. Extract keywords from each answer
    4. Generate embeddings for each answer
    5. Save embeddings to Milvus
    6. Update conversation_log with milvus_ids
    7. Save to PostgreSQL
    8. Calculate overall metrics
    9. Refresh materialized view
    
    Args:
        interview_id: Interview ID
        conversation_data: Full conversation JSON from ConversationLogger
    
    Returns:
        Processed interview data
    """
    logger.info(f"🔄 Processing transcript for {interview_id}")
    
    try:
        # Extract Q&A pairs from turns
        qa_pairs = extract_qa_pairs_from_turns(conversation_data["turns"])
        
        if not qa_pairs:
            logger.warning(f"⚠️  No Q&A pairs found for {interview_id}")
            # Save what we have anyway
            await save_basic_transcript(interview_id, conversation_data)
            return conversation_data
        
        logger.info(f"📋 Extracted {len(qa_pairs)} Q&A pairs")
        
        # Get services
        embedding_service = get_embedding_service()
        milvus_service = get_milvus_service()
        
        # Process each Q&A pair
        processed_qa_pairs = []
        embeddings_data = []
        
        for idx, qa in enumerate(qa_pairs, 1):
            # Sentiment analysis
            sentiment = analyze_sentiment(qa["answer"])
            
            # Keyword extraction
            keywords = extract_keywords(qa["answer"])
            
            # Generate embedding
            embedding = embedding_service.generate_embedding(qa["answer"])
            
            # Create milvus ID
            milvus_id = f"milvus_{interview_id}_{idx}"
            
            # Prepare processed Q&A
            processed_qa = {
                "index": idx,
                "question": qa["question"],
                "answer": qa["answer"],
                "asked_at": qa["asked_at"],
                "answered_at": qa["answered_at"],
                "duration_seconds": qa.get("duration_seconds", 0),
                "sentiment": sentiment["score"],
                "sentiment_label": get_sentiment_label(sentiment["score"]),
                "keywords": keywords,
                "milvus_id": milvus_id
            }
            
            processed_qa_pairs.append(processed_qa)
            
            # Prepare for Milvus
            embeddings_data.append({
                "id": milvus_id,
                "interview_id": interview_id,
                "candidate_id": conversation_data["candidate"]["candidate_id"],
                "question_index": idx,
                "answer_snippet": qa["answer"][:500],  # First 500 chars
                "embedding": embedding
            })
        
        # Save to Milvus
        logger.info(f"💾 Saving {len(embeddings_data)} embeddings to Milvus...")
        milvus_service.insert_embeddings(embeddings_data)
        
        # Calculate overall metrics
        avg_sentiment = sum(qa["sentiment"] for qa in processed_qa_pairs) / len(processed_qa_pairs)
        all_keywords = []
        for qa in processed_qa_pairs:
            all_keywords.extend(qa["keywords"]["tech_keywords"])
        
        # Remove duplicates
        unique_keywords = list(set(all_keywords))
        
        # Update conversation_data
        conversation_data["qa_pairs"] = processed_qa_pairs
        conversation_data["metrics"] = {
            "average_sentiment": avg_sentiment,
            "total_questions": len(processed_qa_pairs),
            "keywords_mentioned": unique_keywords
        }
        
        # Save to PostgreSQL
        await save_to_database(interview_id, conversation_data, avg_sentiment, unique_keywords)
        
        # Refresh materialized view
        await refresh_materialized_view()
        
        logger.info(f"✅ Processed transcript for {interview_id}")
        
        return conversation_data
        
    except Exception as e:
        logger.error(f"❌ Error processing transcript: {e}", exc_info=True)
        # Try to save basic transcript anyway
        try:
            await save_basic_transcript(interview_id, conversation_data)
        except:
            pass
        raise


def extract_qa_pairs_from_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract Q&A pairs from conversation turns
    
    Logic: Agent turn = question, next candidate turn = answer
    """
    qa_pairs = []
    current_question = None
    
    for turn in turns:
        if turn["speaker"] == "agent":
            # New question
            current_question = {
                "question": turn["text"],
                "asked_at": turn["timestamp"]
            }
        elif turn["speaker"] == "candidate" and current_question:
            # Answer to previous question
            try:
                asked_dt = datetime.fromisoformat(current_question["asked_at"])
                answered_dt = datetime.fromisoformat(turn["timestamp"])
                duration = (answered_dt - asked_dt).total_seconds()
            except:
                duration = 0
            
            qa_pairs.append({
                "question": current_question["question"],
                "answer": turn["text"],
                "asked_at": current_question["asked_at"],
                "answered_at": turn["timestamp"],
                "duration_seconds": duration
            })
            current_question = None
    
    return qa_pairs


async def save_to_database(
    interview_id: str,
    conversation_data: Dict[str, Any],
    avg_sentiment: float,
    keywords: List[str]
):
    """Save processed data to PostgreSQL"""
    
    # Get full transcript as text
    turns = conversation_data.get("turns", [])
    full_text_lines = []
    for turn in turns:
        speaker = "Interviewer" if turn["speaker"] == "agent" else "Candidate"
        full_text_lines.append(f"{speaker}: {turn['text']}")
    full_transcript = "\n\n".join(full_text_lines)
    
    # Get duration
    duration = conversation_data.get("metadata", {}).get("duration_seconds", 0)
    
    async with get_db_session() as session:
        query = text("""
            UPDATE interviews
            SET 
                conversation_log = :conversation_log::jsonb,
                full_transcript = :full_transcript,
                interview_status = 'completed',
                sentiment_score = :sentiment_score,
                keyword_matches = :keyword_matches::jsonb,
                call_duration_seconds = :duration,
                completed_at = NOW(),
                updated_at = NOW()
            WHERE interview_id = :interview_id
        """)
        
        await session.execute(
            query,
            {
                "interview_id": interview_id,
                "conversation_log": json.dumps(conversation_data),
                "full_transcript": full_transcript,
                "sentiment_score": avg_sentiment,
                "keyword_matches": json.dumps({"keywords": keywords}),
                "duration": duration
            }
        )
        
        logger.info(f"💾 Saved to PostgreSQL: {interview_id}")


async def save_basic_transcript(
    interview_id: str,
    conversation_data: Dict[str, Any]
):
    """Save basic transcript without processing (fallback)"""
    
    # Get full transcript as text
    turns = conversation_data.get("turns", [])
    full_text_lines = []
    for turn in turns:
        speaker = "Interviewer" if turn["speaker"] == "agent" else "Candidate"
        full_text_lines.append(f"{speaker}: {turn['text']}")
    full_transcript = "\n\n".join(full_text_lines)
    
    # Get duration
    duration = conversation_data.get("metadata", {}).get("duration_seconds", 0)
    
    async with get_db_session() as session:
        query = text("""
            UPDATE interviews
            SET 
                conversation_log = :conversation_log::jsonb,
                full_transcript = :full_transcript,
                interview_status = 'completed',
                call_duration_seconds = :duration,
                completed_at = NOW(),
                updated_at = NOW()
            WHERE interview_id = :interview_id
        """)
        
        await session.execute(
            query,
            {
                "interview_id": interview_id,
                "conversation_log": json.dumps(conversation_data),
                "full_transcript": full_transcript,
                "duration": duration
            }
        )
        
        logger.info(f"💾 Saved basic transcript to PostgreSQL: {interview_id}")


async def refresh_materialized_view():
    """Refresh the interview_qa_flat materialized view"""
    try:
        async with get_db_session() as session:
            await session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY interview_qa_flat"))
            logger.info("✅ Refreshed materialized view")
    except Exception as e:
        logger.warning(f"⚠️  Failed to refresh materialized view: {e}")
        # Non-critical, continue anyway