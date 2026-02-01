"""Milvus client for fetching candidate and job data."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from pymilvus import Collection, connections, utility

from config import (
    MILVUS_URI, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN,
    MILVUS_CANDIDATES_COLLECTION, MILVUS_JOBS_COLLECTION, MILVUS_QA_COLLECTION,
    HF_TOKEN, EMBEDDING_MODEL, EMBEDDING_DEVICE,
)
from models import CandidateData, JobData

logger = logging.getLogger("rcrutr_interviews_milvus")

# Global embedder
_EMBEDDER = None


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


def _get_embedder():
    """Get or create sentence transformer embedder."""
    global _EMBEDDER
    if _EMBEDDER is None:
        # Set HF token
        if HF_TOKEN:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            os.environ["HF_HUB_TOKEN"] = HF_TOKEN
        
        from sentence_transformers import SentenceTransformer
        _log_event("info", "loading_embedding_model", model=EMBEDDING_MODEL)
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    return _EMBEDDER


def _connect_milvus(alias: str = "default") -> None:
    """Connect to Milvus."""
    # Check if already connected
    try:
        if connections.has_connection(alias):
            return
    except Exception:
        pass
    
    kwargs: Dict[str, Any] = {"alias": alias}
    
    if MILVUS_URI:
        kwargs["uri"] = MILVUS_URI
        if MILVUS_TOKEN:
            kwargs["token"] = MILVUS_TOKEN
    elif MILVUS_HOST:
        kwargs["host"] = MILVUS_HOST
        kwargs["port"] = MILVUS_PORT
    else:
        raise ValueError("MILVUS_URI or MILVUS_HOST must be set")
    
    _log_event("info", "milvus_connecting", **{k: v for k, v in kwargs.items() if k != "token"})
    connections.connect(**kwargs)
    _log_event("info", "milvus_connected")


def _disconnect_milvus(alias: str = "default") -> None:
    """Disconnect from Milvus."""
    try:
        connections.disconnect(alias)
    except Exception:
        pass


# =============================================================================
# CANDIDATE OPERATIONS
# =============================================================================

def get_candidate(candidate_id: str) -> Optional[CandidateData]:
    """Fetch candidate data from Milvus candidates_v3 collection."""
    try:
        _connect_milvus()
        
        collection = Collection(MILVUS_CANDIDATES_COLLECTION)
        collection.load()
        
        # Query by candidate_id
        results = collection.query(
            expr=f'candidate_id == "{candidate_id}"',
            output_fields=[
                "candidate_id", "name", "email", "phone",
                "semantic_summary", "current_tech_stack",
                "top_5_skills_with_years", "employment_history"
            ],
            limit=1,
        )
        
        if not results:
            _log_event("warning", "candidate_not_found", candidate_id=candidate_id)
            return None
        
        row = results[0]
        _log_event("info", "candidate_fetched", candidate_id=candidate_id)
        
        # Parse fields
        tech_stack = row.get("current_tech_stack")
        if isinstance(tech_stack, str):
            try:
                tech_stack = json.loads(tech_stack)
            except:
                tech_stack = [tech_stack] if tech_stack else None
        
        employment_history = row.get("employment_history")
        if isinstance(employment_history, str):
            try:
                employment_history = json.loads(employment_history)
            except:
                employment_history = None
        
        return CandidateData(
            candidate_id=row.get("candidate_id", candidate_id),
            name=row.get("name"),
            email=row.get("email"),
            phone=row.get("phone"),
            semantic_summary=row.get("semantic_summary"),
            current_tech_stack=tech_stack,
            top_5_skills_with_years=row.get("top_5_skills_with_years"),
            employment_history=employment_history,
        )
        
    except Exception as e:
        _log_event("error", "get_candidate_failed", candidate_id=candidate_id, error=str(e))
        return None


def search_candidates(
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[CandidateData]:
    """Search candidates by semantic similarity."""
    try:
        _connect_milvus()
        
        # Generate embedding for query
        embedder = _get_embedder()
        query_embedding = embedder.encode(f"query: {query}").tolist()
        
        collection = Collection(MILVUS_CANDIDATES_COLLECTION)
        collection.load()
        
        # Build filter expression
        expr = None
        if filters:
            expr_parts = []
            for k, v in filters.items():
                if isinstance(v, str):
                    expr_parts.append(f'{k} == "{v}"')
                else:
                    expr_parts.append(f'{k} == {v}')
            expr = " && ".join(expr_parts) if expr_parts else None
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",  # Adjust based on your schema
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            expr=expr,
            output_fields=[
                "candidate_id", "name", "email", "phone",
                "semantic_summary", "current_tech_stack",
                "top_5_skills_with_years", "employment_history"
            ],
        )
        
        candidates = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                candidates.append(CandidateData(
                    candidate_id=entity.get("candidate_id"),
                    name=entity.get("name"),
                    email=entity.get("email"),
                    phone=entity.get("phone"),
                    semantic_summary=entity.get("semantic_summary"),
                    current_tech_stack=entity.get("current_tech_stack"),
                    top_5_skills_with_years=entity.get("top_5_skills_with_years"),
                    employment_history=entity.get("employment_history"),
                ))
        
        _log_event("info", "search_candidates_ok", query=query[:50], count=len(candidates))
        return candidates
        
    except Exception as e:
        _log_event("error", "search_candidates_failed", query=query[:50], error=str(e))
        return []


# =============================================================================
# JOB OPERATIONS
# =============================================================================

def get_job(job_id: str) -> Optional[JobData]:
    """Fetch job data from Milvus job_postings collection."""
    try:
        _connect_milvus()
        
        collection = Collection(MILVUS_JOBS_COLLECTION)
        collection.load()
        
        # Query by job_id
        results = collection.query(
            expr=f'job_id == "{job_id}"',
            output_fields=[
                "job_id", "title", "company", "department",
                "location", "employment_type", "salary_range", "jd_text"
            ],
            limit=1,
        )
        
        if not results:
            _log_event("warning", "job_not_found", job_id=job_id)
            return None
        
        row = results[0]
        _log_event("info", "job_fetched", job_id=job_id)
        
        return JobData(
            job_id=row.get("job_id", job_id),
            title=row.get("title"),
            company=row.get("company"),
            department=row.get("department"),
            location=row.get("location"),
            employment_type=row.get("employment_type"),
            salary_range=row.get("salary_range"),
            jd_text=row.get("jd_text"),
        )
        
    except Exception as e:
        _log_event("error", "get_job_failed", job_id=job_id, error=str(e))
        return None


# =============================================================================
# QA EMBEDDINGS (Store interview Q&A)
# =============================================================================

def store_qa_embeddings(
    interview_id: str,
    candidate_id: str,
    job_id: str,
    job_title: str,
    job_description: str,
    qa_pairs: List[Dict[str, Any]],
) -> bool:
    """Store interview Q&A pairs in Milvus for semantic search."""
    try:
        _connect_milvus()
        
        collection = Collection(MILVUS_QA_COLLECTION)
        collection.load()
        
        embedder = _get_embedder()
        
        # Prepare data for insertion
        entities = []
        for qa in qa_pairs:
            question_index = qa.get("index", 0)
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            
            # Create embedding from Q&A pair
            text = f"Question: {question}\nAnswer: {answer}"
            embedding = embedder.encode(f"passage: {text}").tolist()
            
            # Truncate answer for snippet
            answer_snippet = answer[:500] if answer else ""
            
            entities.append({
                "id": f"qa_{interview_id}_{question_index}",
                "interview_id": interview_id,
                "candidate_id": candidate_id,
                "job_id": job_id,
                "job_title": job_title[:256] if job_title else "",
                "job_description": job_description[:2048] if job_description else "",
                "question_index": question_index,
                "answer_snippet": answer_snippet,
                "interview_date": int(qa.get("timestamp", 0)),
                "embedding": embedding,
            })
        
        if not entities:
            _log_event("warning", "store_qa_no_entities", interview_id=interview_id)
            return False
        
        # Insert into Milvus
        collection.insert(entities)
        collection.flush()
        
        _log_event("info", "store_qa_embeddings_ok", interview_id=interview_id, count=len(entities))
        return True
        
    except Exception as e:
        _log_event("error", "store_qa_embeddings_failed", interview_id=interview_id, error=str(e))
        return False


# =============================================================================
# HEALTH CHECK
# =============================================================================

def check_milvus_connection() -> tuple[bool, str]:
    """Check Milvus connection health."""
    try:
        _connect_milvus()
        collections = utility.list_collections()
        
        # Check required collections exist
        required = [MILVUS_CANDIDATES_COLLECTION, MILVUS_JOBS_COLLECTION]
        missing = [c for c in required if c not in collections]
        
        if missing:
            return (False, f"Missing collections: {missing}")
        
        return (True, f"ok, collections: {collections}")
    except Exception as e:
        _log_event("error", "milvus_health_failed", error=str(e))
        return (False, str(e))
