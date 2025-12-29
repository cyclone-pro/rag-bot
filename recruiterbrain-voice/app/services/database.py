"""
Database Service - Adapted for Existing Schema
Maps to your existing 37-column interviews table
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from pymilvus import connections, Collection, utility
import torch
from sentence_transformers import SentenceTransformer

from app.config.settings import settings


logger = logging.getLogger(__name__)


# ============================================
# PostgreSQL Connection Pool (Same as before)
# ============================================

postgres_engine = create_async_engine(
    settings.postgres_url,
    pool_size=50,
    max_overflow=100,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_timeout=30,
    echo=False,
    connect_args={
        "server_settings": {
            "application_name": "recruiterbrain_voice_interview",
            "jit": "off"
        },
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0
    }
)

AsyncSessionLocal = async_sessionmaker(
    postgres_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)


@asynccontextmanager
async def get_db_session():
    """Get database session with automatic cleanup"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            await session.close()


# ============================================
# Milvus Connection Pool (Same as before)
# ============================================

class MilvusConnectionPool:
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.connections_created = False
        self.collection_name = "interview_transcripts_v2"
        
    def connect(self):
        if self.connections_created:
            return
        
        try:
            for i in range(self.pool_size):
                alias = f"milvus_conn_{i}"
                connections.connect(
                    alias=alias,
                    host=settings.milvus_host,
                    port=settings.milvus_port,
                    pool_size=5
                )
            
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
                pool_size=10
            )
            
            self.connections_created = True
            logger.info(f"✅ Milvus connection pool created ({self.pool_size} connections)")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def get_collection(self, alias: str = "default") -> Collection:
        if not self.connections_created:
            self.connect()
        return Collection(self.collection_name, using=alias)
    
    async def disconnect(self):
        try:
            for i in range(self.pool_size):
                connections.disconnect(alias=f"milvus_conn_{i}")
            connections.disconnect(alias="default")
            logger.info("Milvus connections closed")
        except Exception as e:
            logger.error(f"Error disconnecting Milvus: {e}")


milvus_pool = MilvusConnectionPool(pool_size=10)


# ============================================
# E5-Base-V2 Embeddings (Same as before)
# ============================================

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        if self.model is None:
            logger.info("Loading e5-base-v2 model...")
            self.model = SentenceTransformer(
                'intfloat/e5-base-v2',
                device=self.device
            )
            logger.info(f"✅ e5-base-v2 loaded on {self.device}")
    
    def generate_embedding(self, text: str) -> List[float]:
        self.load_model()
        prefixed_text = f"passage: {text}"
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def generate_query_embedding(self, query: str) -> List[float]:
        self.load_model()
        prefixed_query = f"query: {query}"
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embedding, text)


embedding_service = EmbeddingService()


# ============================================
# Interview Database Operations
# ADAPTED FOR YOUR EXISTING SCHEMA
# ============================================

async def create_interview_record(
    interview_id: str,
    candidate_data: Dict[str, Any],
    jd_data: Dict[str, Any],
    livekit_room: str,
    scheduled_time: datetime = None
) -> bool:
    """
    Create initial interview record
    Maps to your existing 37-column schema
    """
    try:
        async with get_db_session() as session:
            # Your table uses 'interview_status' not 'status'
            query = text("""
                INSERT INTO interviews (
                    interview_id,
                    candidate_id,
                    jd_id,
                    job_title,
                    job_description,
                    interview_status,
                    call_status,
                    livekit_room_name,
                    scheduled_time,
                    timezone,
                    created_at,
                    updated_at
                ) VALUES (
                    :interview_id,
                    :candidate_id,
                    :jd_id,
                    :job_title,
                    :job_description,
                    'initiated',
                    'pending',
                    :livekit_room_name,
                    :scheduled_time,
                    :timezone,
                    NOW(),
                    NOW()
                )
            """)
            
            await session.execute(query, {
                "interview_id": interview_id,
                "candidate_id": candidate_data.get("candidate_id"),
                "jd_id": jd_data.get("jd_id"),
                "job_title": jd_data.get("title", ""),
                "job_description": jd_data.get("description", ""),
                "livekit_room_name": livekit_room,
                "scheduled_time": scheduled_time or datetime.utcnow(),
                "timezone": candidate_data.get("timezone", "UTC")
            })
            
            logger.info(f"Created interview record: {interview_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error creating interview record: {e}")
        return False


async def update_interview_status(
    interview_id: str,
    interview_status: str = None,
    call_status: str = None,
    **kwargs
) -> bool:
    """
    Update interview status
    Your table has both 'interview_status' and 'call_status'
    """
    try:
        async with get_db_session() as session:
            set_clauses = ["updated_at = NOW()"]
            params = {"interview_id": interview_id}
            
            if interview_status:
                set_clauses.append("interview_status = :interview_status")
                params["interview_status"] = interview_status
            
            if call_status:
                set_clauses.append("call_status = :call_status")
                params["call_status"] = call_status
            
            for key, value in kwargs.items():
                set_clauses.append(f"{key} = :{key}")
                params[key] = value
            
            query = text(f"""
                UPDATE interviews
                SET {', '.join(set_clauses)}
                WHERE interview_id = :interview_id
            """)
            
            await session.execute(query, params)
            logger.info(f"Updated interview {interview_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error updating interview: {e}")
        return False


async def save_interview_completion(
    interview_id: str,
    conversation_log: List[Dict[str, Any]],
    full_transcript: str,
    evaluation: Dict[str, Any],
    call_duration_seconds: int,
    recording_url: str = None
) -> bool:
    """
    Save complete interview - BATCH WRITE
    Maps to your existing columns
    """
    try:
        async with get_db_session() as session:
            query = text("""
                UPDATE interviews
                SET 
                    interview_status = 'completed',
                    call_status = 'completed',
                    completed_at = NOW(),
                    call_ended_at = NOW(),
                    call_duration_seconds = :call_duration_seconds,
                    
                    -- Your conversation columns
                    conversation_log = :conversation_log::jsonb,
                    full_transcript = :full_transcript,
                    
                    -- Evaluation results
                    evaluation_score = :evaluation_score,
                    sentiment_score = :sentiment_score,
                    keyword_matches = :keyword_matches::jsonb,
                    fit_assessment = :fit_assessment,
                    
                    -- Recording
                    recording_url = :recording_url,
                    recording_duration_seconds = :recording_duration_seconds,
                    
                    -- Question tracking
                    current_question_index = :questions_completed,
                    
                    updated_at = NOW()
                WHERE interview_id = :interview_id
            """)
            
            await session.execute(query, {
                "interview_id": interview_id,
                "call_duration_seconds": call_duration_seconds,
                "conversation_log": str(conversation_log),
                "full_transcript": full_transcript,
                "evaluation_score": evaluation.get("score", 0.0),
                "sentiment_score": evaluation.get("sentiment_score", 0.0),
                "keyword_matches": str(evaluation.get("keyword_matches", {})),
                "fit_assessment": evaluation.get("fit_assessment", ""),
                "recording_url": recording_url,
                "recording_duration_seconds": call_duration_seconds,
                "questions_completed": evaluation.get("questions_completed", 0)
            })
            
            logger.info(f"✅ Saved complete interview: {interview_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error saving interview completion: {e}")
        return False


async def insert_interview_to_milvus(
    interview_id: str,
    candidate_id: str,
    jd_id: str,
    job_title: str,
    full_transcript: str,
    evaluation_summary: str,
    skills_discussed: List[str],
    evaluation_score: float,
    interview_date: datetime
) -> bool:
    """
    Insert interview to Milvus - BATCH operation
    Same as before
    """
    try:
        logger.info(f"Generating embedding for interview {interview_id}...")
        
        # Generate embedding
        embedding = await embedding_service.generate_embedding_async(full_transcript)
        
        # Prepare data
        data = [{
            "interview_id": interview_id,
            "candidate_id": candidate_id,
            "jd_id": jd_id,
            "interview_embedding": embedding,
            "job_title": job_title,
            "interview_date": int(interview_date.timestamp()),
            "evaluation_score": float(evaluation_score),
            "full_transcript_text": full_transcript[:8192],
            "skills_discussed": ", ".join(skills_discussed[:50]),
            "interview_summary": evaluation_summary[:1024]
        }]
        
        # Insert
        collection = milvus_pool.get_collection()
        result = collection.insert(data)
        
        logger.info(f"✅ Inserted to Milvus: {interview_id}")
        
        # Mark as synced
        await update_interview_status(
            interview_id,
            milvus_synced=True,
            milvus_sync_at=datetime.utcnow()
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error inserting to Milvus: {e}")
        return False


# ============================================
# Initialize/Cleanup (Same as before)
# ============================================

async def initialize_database_connections():
    try:
        async with get_db_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("✅ PostgreSQL connection pool ready")
        
        milvus_pool.connect()
        logger.info("✅ Milvus connection pool ready")
        
        embedding_service.load_model()
        logger.info("✅ E5-Base-V2 model loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


async def cleanup_database_connections():
    try:
        await postgres_engine.dispose()
        await milvus_pool.disconnect()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
