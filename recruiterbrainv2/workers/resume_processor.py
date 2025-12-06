"""Background worker for processing resume uploads."""
import logging
from typing import Dict, Any

from ..jobs import get_job_store, JobStatus
from ..ingestion import (
    parse_file,
    scrub_pii,
    merge_pii,
    extract_resume_data,
    generate_embeddings,
    insert_candidate,
)

logger = logging.getLogger(__name__)


def process_resume_upload(job_id: str, filename: str, file_bytes: bytes, source_channel: str = "Upload"):
    """
    Process resume upload in background.
    
    This is the main background task that runs async.
    
    Steps:
    1. Update job status to PROCESSING
    2. Parse file
    3. Scrub PII
    4. LLM extraction
    5. Generate embeddings
    6. Insert to Milvus
    7. Update job status to COMPLETED
    """
    job_store = get_job_store()
    
    try:
        # Update status to PROCESSING
        job_store.update_job(job_id, JobStatus.PROCESSING)
        logger.info(f"[Job {job_id}] Started processing: {filename}")
        
        # Step 1: Parse file
        logger.info(f"[Job {job_id}] Step 1/5: Parsing file...")
        resume_text = parse_file(filename, file_bytes)
        
        if len(resume_text) < 100:
            raise ValueError("Resume text too short. Please upload a valid resume.")
        
        # Step 2: Scrub PII
        logger.info(f"[Job {job_id}] Step 2/5: Scrubbing PII...")
        sanitized_text, pii = scrub_pii(resume_text)
        
        # Step 3: LLM extraction
        logger.info(f"[Job {job_id}] Step 3/5: Extracting data with LLM...")
        extracted_data = extract_resume_data(sanitized_text)
        
        # Step 4: Merge PII back
        logger.info(f"[Job {job_id}] Step 4/5: Merging PII and generating embeddings...")
        complete_data = merge_pii(extracted_data, pii)
        
        # Generate embeddings
        embeddings = generate_embeddings(complete_data)
        
        # Step 5: Insert to Milvus
        logger.info(f"[Job {job_id}] Step 5/5: Inserting to Milvus...")
        candidate_id = insert_candidate(complete_data, embeddings, source_channel=source_channel)
        
        # Success!
        result = {
            "candidate_id": candidate_id,
            "name": complete_data.get("name", "Unknown"),
            "career_stage": complete_data.get("career_stage"),
            "skills_count": len(complete_data.get("skills_extracted", "").split(",")),
        }
        
        job_store.update_job(job_id, JobStatus.COMPLETED, result=result)
        logger.info(f"[Job {job_id}] ✅ Completed: {candidate_id}")
        
        # Invalidate search cache
        from ..retrieval_engine import invalidate_search_cache
        invalidate_search_cache()
        
    except Exception as e:
        # Failure
        logger.exception(f"[Job {job_id}] ❌ Failed: {e}")
        job_store.update_job(job_id, JobStatus.FAILED, error=str(e))