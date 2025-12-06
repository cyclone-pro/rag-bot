"""FastAPI entrypoint for RecruiterBrain v2."""
from __future__ import annotations
import os
import sys
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import logging
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import UploadFile, File
from fastapi import BackgroundTasks
from .jobs import get_job_store, JobStatus
from .workers import process_resume_upload

from .formatter import format_for_chat, format_for_insight
from .retrieval_engine import search_candidates_v2
from .ingestion import (
    parse_file,
    scrub_pii,
    merge_pii,
    extract_resume_data,
    generate_embeddings,
    insert_candidate,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RecruiterBrain v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to console
        logging.FileHandler('recruiterbrainv2.log')  # Also save to file
    ]
)

# Set specific loggers
logging.getLogger('recruiterbrainv2').setLevel(logging.INFO)
logging.getLogger('pymilvus').setLevel(logging.WARNING)  # Reduce Milvus noise
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)  # Reduce embedding noise

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    logger.debug("Serving home page")
    return templates.TemplateResponse("index.html", {"request": request})

# ---- Simple in-process rate limiter ----
_RATE_WINDOW_SECONDS = 60.0
_RATE_MAX_REQUESTS = 60
_rate_state: Dict[str, tuple[float, int]] = {}


def _rate_key(request: Request) -> str:
    return request.client.host if request and request.client else "unknown"


def rate_limiter(request: Request) -> None:
    now = time.time()
    key = _rate_key(request)
    window_start, count = _rate_state.get(key, (now, 0))

    if now - window_start > _RATE_WINDOW_SECONDS:
        _rate_state[key] = (now, 1)
        return
    if count >= _RATE_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again soon.")
    _rate_state[key] = (window_start, count + 1)


# ---- Schemas ----
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    filters: Optional[Dict[str, Any]] = None
    show_contacts: bool = False


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    error: Optional[str] = None


class InsightRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    filters: Optional[Dict[str, Any]] = None
    show_contacts: bool = False


class InsightResponse(BaseModel):
    text: Optional[str] = None
    rows: Optional[Any] = None
    total_matched: Optional[int] = None
    scarcity_message: Optional[str] = None
    data_quality_banner: Optional[str] = None
    error: Optional[str] = None

class ResumeUploadResponse(BaseModel):
    candidate_id: str
    name: str
    status: str
    message: str
# ====================  ASYNC UPLOAD ENDPOINTS ====================

class AsyncResumeUploadResponse(BaseModel):
    """Response for async resume upload."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ---- Endpoints ----
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "RecruiterBrain v2 API. Try POST /v2/chat or /v2/insight."}


@app.post("/v2/chat", response_model=ChatResponse)
def chat_v2_endpoint(chat_input: ChatRequest, _: None = Depends(rate_limiter)) -> Dict[str, Optional[str]]:
    question = chat_input.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    filters = chat_input.filters or {}
    top_k = int(filters.get("top_k", 10))
    career_stage = filters.get("career_stage")
    industry = filters.get("industry")

    logger.info("V2 Chat request: len=%d", len(question))

    try:
        results = search_candidates_v2(
            query=question,
            top_k=top_k,
            career_stage=career_stage,
            industry=industry,
        )
        answer = format_for_chat(results, show_contacts=chat_input.show_contacts)
        return ChatResponse(answer=answer).model_dump()
    except Exception as exc:  # pragma: no cover - depends on Milvus env
        logger.exception("V2 chat error")
        return ChatResponse(error=str(exc)).model_dump()


@app.post("/v2/insight", response_model=InsightResponse)
def insight_v2_endpoint(payload: InsightRequest, _: None = Depends(rate_limiter)) -> Dict[str, Any]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    filters = payload.filters or {}
    top_k = int(filters.get("top_k", 20))
    career_stage = filters.get("career_stage")
    industry = filters.get("industry")

    logger.info("V2 Insight request: len=%d", len(question))

    try:
        results = search_candidates_v2(
            query=question,
            top_k=top_k,
            career_stage=career_stage,
            industry=industry,
        )
        formatted = format_for_insight(results)
        return InsightResponse(
            text="V2 Insight results ready",
            rows=formatted.get("rows"),
            total_matched=formatted.get("total_matched"),
            scarcity_message=formatted.get("scarcity_message"),
            data_quality_banner=formatted.get("data_quality_banner"),
        ).model_dump()
    except Exception as exc:  # pragma: no cover - depends on Milvus env
        logger.exception("V2 insight error")
        return InsightResponse(error=str(exc)).model_dump()

@app.post("/v2/upload_resume", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    _: None = Depends(rate_limiter),
) -> Dict[str, Any]:
    """
    Upload and ingest a resume (PDF/DOCX/ZIP).
    
    Pipeline:
    1. Parse file → extract text
    2. Scrub PII → remove sensitive data
    3. LLM extraction → structured data
    4. Merge PII back
    5. Generate 3 embeddings
    6. Insert into Milvus
    """
    filename = file.filename or "unknown.pdf"
    
    logger.info("="*60)
    logger.info(f"📤 RESUME UPLOAD STARTED: {filename}")
    logger.info("="*60)
    
    try:
        # Step 0: Read file bytes
        logger.info("Step 0: Reading file bytes...")
        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        logger.info(f"✅ File read complete: {file_size_mb:.2f} MB")
        
        # Step 1: Parse file
        logger.info(f"Step 1: Parsing file ({filename})...")
        resume_text = parse_file(filename, file_bytes)
        logger.info(f"✅ Text extracted: {len(resume_text)} characters")
        
        if len(resume_text) < 100:
            logger.error(f"❌ Resume text too short: {len(resume_text)} chars")
            raise ValueError("Resume text too short. Please upload a valid resume.")
        
        # Step 2: Scrub PII
        logger.info("Step 2: Scrubbing PII from resume text...")
        sanitized_text, pii = scrub_pii(resume_text)
        logger.info(f"✅ PII scrubbed: {len(pii)} fields removed ({list(pii.keys())})")
        logger.info(f"   Sanitized text length: {len(sanitized_text)} chars")
        
        # Step 3: Extract data with LLM
        logger.info("Step 3: Sending sanitized text to LLM for extraction...")
        logger.info(f"   Text preview: {sanitized_text[:200]}...")
        extracted_data = extract_resume_data(sanitized_text)
        logger.info(f"✅ LLM extraction complete")
        logger.info(f"   Candidate name: {extracted_data.get('name', 'Unknown')}")
        logger.info(f"   Career stage: {extracted_data.get('career_stage', 'Unknown')}")
        logger.info(f"   Skills found: {len(extracted_data.get('skills_extracted', '').split(','))}")
        
        # Step 4: Merge PII back
        logger.info("Step 4: Merging PII back into extracted data...")
        complete_data = merge_pii(extracted_data, pii)
        logger.info(f"✅ PII merged back")
        logger.info(f"   Email: {complete_data.get('email', 'None')}")
        logger.info(f"   Phone: {complete_data.get('phone', 'None')}")
        
        # Step 5: Generate embeddings
        logger.info("Step 5: Generating 3 embeddings (summary, tech, role)...")
        embeddings = generate_embeddings(complete_data)
        logger.info(f"✅ Embeddings generated:")
        logger.info(f"   summary_embedding: {len(embeddings['summary_embedding'])} dimensions")
        logger.info(f"   tech_embedding: {len(embeddings['tech_embedding'])} dimensions")
        logger.info(f"   role_embedding: {len(embeddings['role_embedding'])} dimensions")
        
        # Step 6: Insert into Milvus
        logger.info("Step 6: Inserting candidate into Milvus...")
        candidate_id = insert_candidate(complete_data, embeddings, source_channel="Upload")
        logger.info(f"✅ Candidate inserted: {candidate_id}")
        
        logger.info("="*60)
        logger.info(f"✅ UPLOAD COMPLETE: {complete_data.get('name')} ({candidate_id})")
        logger.info("="*60)

        # After successful insert:
        candidate_id = insert_candidate(complete_data, embeddings, source_channel="Upload")
        # Invalidate search cache
        from .retrieval_engine import invalidate_search_cache
        invalidate_search_cache()
        
        return ResumeUploadResponse(
            candidate_id=candidate_id,
            name=complete_data.get("name", "Unknown"),
            status="success",
            message=f"Resume successfully ingested for {complete_data.get('name')}"
        ).model_dump()
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={"detail": str(e)}
        )
    
    except Exception as e:
        logger.exception(f"❌ UPLOAD FAILED: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Upload failed: {str(e)}"}
        )

@app.post("/v2/ingest_resume", response_model=ResumeUploadResponse)
async def ingest_resume(
    file: UploadFile = File(...),
    _: None = Depends(rate_limiter),
) -> Dict[str, Any]:
    return await upload_resume(file, _)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("recruiterbrainv2.__main__:app", host="0.0.0.0", port=8000, reload=True)

@app.post("/v2/upload_resume_async", response_model=AsyncResumeUploadResponse)
async def upload_resume_async(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: None = Depends(rate_limiter),
) -> Dict[str, Any]:
    """
    Upload resume asynchronously (immediate response, background processing).
    
    Returns job_id immediately. Use GET /v2/jobs/{job_id} to check status.
    
    Flow:
    1. Validate file
    2. Create job
    3. Return job_id immediately
    4. Process in background
    """
    filename = file.filename or "unknown.pdf"
    
    logger.info("="*60)
    logger.info(f"📤 ASYNC RESUME UPLOAD: {filename}")
    logger.info("="*60)
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        
        # Validate file size (max 10MB)
        if file_size_mb > 10:
            raise ValueError(f"File too large: {file_size_mb:.2f} MB (max 10 MB)")
        
        logger.info(f"✅ File validated: {file_size_mb:.2f} MB")
        
        # Create job
        job_store = get_job_store()
        job_id = job_store.create_job(
            job_type="resume_upload",
            params={
                "filename": filename,
                "file_size_mb": file_size_mb,
            }
        )
        
        # Schedule background task
        background_tasks.add_task(
            process_resume_upload,
            job_id=job_id,
            filename=filename,
            file_bytes=file_bytes,
            source_channel="Upload"
        )
        
        logger.info(f"✅ Job {job_id} created and queued")
        
        return AsyncResumeUploadResponse(
            job_id=job_id,
            status="queued",
            message=f"Resume upload queued for processing. Check status at /v2/jobs/{job_id}"
        ).model_dump()
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/v2/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get job status and result.
    
    Status values:
    - "queued": Job is waiting to be processed
    - "processing": Job is currently being processed
    - "completed": Job finished successfully (result available)
    - "failed": Job failed (error message available)
    """
    job_store = get_job_store()
    job_data = job_store.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatusResponse(**job_data).model_dump()


@app.delete("/v2/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a job (if still queued).
    
    Note: Already-processing jobs cannot be cancelled.
    """
    job_store = get_job_store()
    job_data = job_store.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job_data["status"] == JobStatus.QUEUED:
        job_store.update_job(job_id, JobStatus.FAILED, error="Cancelled by user")
        return {"message": f"Job {job_id} cancelled"}
    else:
        return {"message": f"Job {job_id} is already {job_data['status']}, cannot cancel"}

