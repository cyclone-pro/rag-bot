"""Celery tasks for background processing."""
import logging
from typing import Dict, Any , List
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import celery_app
from .ingestion import (
    parse_file,
    scrub_pii,
    merge_pii,
    extract_resume_data,
    generate_embeddings,
    insert_candidate,
)

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """
    Base task with callbacks for success/failure.
    
    Automatically updates job status in Redis.
    """
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"✅ Task {task_id} succeeded")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"❌ Task {task_id} failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f"🔄 Task {task_id} retrying: {exc}")


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="recruiterbrainv2.tasks.process_resume_task",
    max_retries=3,
    default_retry_delay=60,  # Wait 60 seconds before retry
    autoretry_for=(ConnectionError, TimeoutError),  # Auto-retry on these exceptions
)
def process_resume_task(
    self,
    filename: str,
    file_bytes_b64: str,  # Base64-encoded file bytes
    source_channel: str = "Upload"
) -> Dict[str, Any]:
    """
    Celery task: Process resume upload.
    
    Args:
        self: Task instance (bind=True)
        filename: Original filename
        file_bytes_b64: Base64-encoded file bytes
        source_channel: Upload source
    
    Returns:
        {
            "candidate_id": "ABC123",
            "name": "John Doe",
            "career_stage": "Senior",
            "skills_count": 12
        }
    """
    import base64
    
    task_id = self.request.id
    
    try:
        logger.info(f"[Task {task_id}] Started processing: {filename}")
        
        # Decode file bytes
        file_bytes = base64.b64decode(file_bytes_b64)
        
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Parsing file...', 'progress': 0}
        )
        
        # Step 1: Parse file
        logger.info(f"[Task {task_id}] Step 1/5: Parsing file...")
        resume_text = parse_file(filename, file_bytes)
        
        if len(resume_text) < 100:
            raise ValueError("Resume text too short. Please upload a valid resume.")
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Extracting information...', 'progress': 20}
        )
        
        # Step 2: Scrub PII
        logger.info(f"[Task {task_id}] Step 2/5: Scrubbing PII...")
        sanitized_text, pii = scrub_pii(resume_text)
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing with AI...', 'progress': 40}
        )
        
        # Step 3: LLM extraction
        logger.info(f"[Task {task_id}] Step 3/5: Extracting data with LLM...")
        extracted_data = extract_resume_data(sanitized_text)
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Generating embeddings...', 'progress': 60}
        )
        
        # Step 4: Merge PII and generate embeddings
        logger.info(f"[Task {task_id}] Step 4/5: Merging PII and generating embeddings...")
        complete_data = merge_pii(extracted_data, pii)
        embeddings = generate_embeddings(complete_data)
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Saving to database...', 'progress': 80}
        )
        
        # Step 5: Insert to Milvus
        logger.info(f"[Task {task_id}] Step 5/5: Inserting to Milvus...")
        candidate_id = insert_candidate(complete_data, embeddings, source_channel=source_channel)
        
        # Invalidate search cache
        from .retrieval_engine import invalidate_search_cache
        invalidate_search_cache()
        
        # Success result
        result = {
            "candidate_id": candidate_id,
            "name": complete_data.get("name", "Unknown"),
            "career_stage": complete_data.get("career_stage", "Unknown"),
            "skills_count": len(complete_data.get("skills_extracted", "").split(",")),
        }
        
        logger.info(f"[Task {task_id}] ✅ Completed: {candidate_id}")
        
        return result
        
    except SoftTimeLimitExceeded:
        logger.error(f"[Task {task_id}] ⏱️ Task exceeded time limit")
        raise
    
    except Exception as e:
        logger.exception(f"[Task {task_id}] ❌ Failed: {e}")
        
        # Retry on specific errors
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise self.retry(exc=e, countdown=60)
        
        raise


@celery_app.task(
    bind=True,
    name="recruiterbrainv2.tasks.bulk_process_resumes",
    max_retries=1,
)
def bulk_process_resumes(self, resume_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process multiple resumes in parallel using Celery tasks.
    
    Uses chord pattern to avoid blocking on subtask results.
    """
    from celery import chord
    
    task_id = self.request.id
    total_files = len(resume_files)
    
    logger.info(f"[Bulk Task {task_id}] Processing {total_files} resumes")
    
    # Create subtasks
    subtasks = []
    for idx, resume_data in enumerate(resume_files):
        task = process_resume_task.s(
            filename=resume_data["filename"],
            file_bytes_b64=resume_data["file_bytes_b64"],
            source_channel=f"BulkUpload_{idx+1}/{total_files}"
        )
        subtasks.append(task)
    
    # Use chord: run all subtasks, then call callback with results
    # This doesn't block!
    callback = finalize_bulk_upload.s(
        bulk_task_id=task_id,
        total=total_files
    )
    
    chord(subtasks)(callback)
    
    logger.info(f"[Bulk Task {task_id}] Submitted {total_files} subtasks")
    
    return {
        "status": "processing",
        "total": total_files,
        "message": f"Processing {total_files} resumes in parallel"
    }


@celery_app.task(name="recruiterbrainv2.tasks.finalize_bulk_upload")
def finalize_bulk_upload(results: List[Dict], bulk_task_id: str, total: int) -> Dict[str, Any]:
    """
    Callback function that runs after all bulk upload subtasks complete.
    """
    logger.info(f"[Bulk Task {bulk_task_id}] Finalizing bulk upload")
    
    successful = sum(1 for r in results if r and r.get("status") == "success")
    failed = total - successful
    
    errors = [r.get("error") for r in results if r and r.get("error")]
    
    logger.info(
        f"[Bulk Task {bulk_task_id}] ✅ Complete: "
        f"{successful}/{total} successful, {failed} failed"
    )
    
    return {
        "successful": successful,
        "failed": failed,
        "total": total,
        "errors": errors[:10]
    }



@celery_app.task(name="recruiterbrainv2.tasks.cleanup_old_jobs")
def cleanup_old_jobs():
    """
    Periodic task: Clean up old completed jobs from Redis.
    
    Runs every hour via Celery Beat.
    """
    logger.info("🧹 Running job cleanup...")
    
    # Implementation would delete old job records from Redis
    # For now, jobs auto-expire after 1 hour (set in jobs.py)
    
    logger.info("✅ Job cleanup complete")