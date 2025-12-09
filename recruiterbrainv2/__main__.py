"""FastAPI entrypoint for RecruiterBrain v2."""
from __future__ import annotations
import hashlib
import os
import sys
from celery import result
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse  
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import logging
import time
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import base64
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import UploadFile, File
from fastapi import BackgroundTasks
from fastapi import Response
import zipfile
import io

from pymilvus import client

from recruiterbrainv2.config import COLLECTION, ENABLE_CACHE, SEARCH_OUTPUT_FIELDS, get_milvus_client
from .jobs import get_job_store, JobStatus
from .workers import process_resume_upload
from .celery_app import celery_app
from .cache import get_cache, generate_cache_key

from .tasks import process_resume_task, bulk_process_resumes
from .formatter import format_for_chat, format_for_insight
from .retrieval_engine import search_candidates_v2, invalidate_search_cache
from .rate_limiter import RateLimitExceeded, get_rate_limiter
from .ingestion import (
    parse_file,
    scrub_pii,
    merge_pii,
    extract_resume_data,
    generate_embeddings,
    insert_candidate,
)
from .audio_transcription import transcribe_audio_whisper
from .fit_analyzer import analyze_candidate_fit
from .skill_extractor import extract_requirements
# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('recruiterbrainv2.log')
    ]
)

logging.getLogger('recruiterbrainv2').setLevel(logging.INFO)
logging.getLogger('pymilvus').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ==================== FASTAPI APP ====================
app = FastAPI(title="RecruiterBrain v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==================== STATIC FILES ====================
BASE_DIR = os.path.dirname(__file__)
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"✅ Static files mounted from: {static_dir}")
else:
    logger.warning(f"⚠️  Static directory not found at: {static_dir}")
# ==================== TEMPLATES ====================
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ==================== RATE LIMITING ====================

# Rate limit configurations (requests, seconds)
RATE_LIMITS = {
    "chat": (20, 60),           # 20/minute
    "insight": (10, 60),        # 10/minute
    "upload": (5, 60),          # 5/minute
    "bulk_upload": (1, 3600),   # 1/hour
    "job_status": (60, 60),     # 60/minute
}

@app.get("/styles.css")
async def serve_styles():
    """Serve CSS file at root level."""
    css_path = os.path.join(BASE_DIR, "static", "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/script.js")
async def serve_script():
    """Serve JS file at root level."""
    js_path = os.path.join(BASE_DIR, "static", "script.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")
def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    # Check X-Forwarded-For header (for proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"


async def check_rate_limit(request: Request, endpoint: str):
    """
    Check rate limit for endpoint.
    
    Raises HTTPException if limit exceeded.
    """
    if endpoint not in RATE_LIMITS:
        return  # No limit for this endpoint
    
    limit, window = RATE_LIMITS[endpoint]
    client_ip = get_client_ip(request)
    rate_key = f"{endpoint}:{client_ip}"
    
    limiter = get_rate_limiter()
    is_allowed, retry_after = limiter.check_limit(rate_key, limit, window)
    
    if not is_allowed:
        logger.warning(
            f"Rate limit exceeded: endpoint={endpoint}, ip={client_ip}, "
            f"limit={limit}/{window}s, retry_after={retry_after}s"
        )
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests to {endpoint}. Please slow down.",
                "limit": limit,
                "window": window,
                "retry_after_seconds": retry_after,
                "retry_at": (datetime.utcnow() + timedelta(seconds=retry_after)).isoformat() + "Z"
            },
            headers={"Retry-After": str(retry_after)}
        )

@app.get("/v2/pool/stats")
async def get_pool_stats():
    """
    Get Milvus connection pool statistics.
    
    Useful for monitoring and debugging.
    """
    from .config import get_milvus_pool, ENABLE_CONNECTION_POOL
    
    if not ENABLE_CONNECTION_POOL:
        return {
            "pooling_enabled": False,
            "message": "Connection pooling is disabled"
        }
    
    pool = get_milvus_pool()
    stats = pool.get_stats()
    
    return {
        "pooling_enabled": True,
        "stats": stats,
        "health": {
            "total_connections": stats["active_connections"],
            "available": stats["available"],
            "in_use": stats["in_use"],
            "utilization_percent": round((stats["in_use"] / stats["pool_size"]) * 100, 1)
        }
    }


@app.post("/v2/pool/health_check")
async def pool_health_check():
    """Run health check on all pool connections."""
    from .config import get_milvus_pool, ENABLE_CONNECTION_POOL
    
    if not ENABLE_CONNECTION_POOL:
        return {"error": "Connection pooling is disabled"}
    
    pool = get_milvus_pool()
    healthy_count = pool.health_check_all()
    stats = pool.get_stats()
    
    return {
        "healthy_connections": healthy_count,
        "total_connections": stats["active_connections"],
        "health_percentage": round((healthy_count / stats["active_connections"]) * 100, 1)
    }
# ==================== CUSTOM ERROR HANDLERS ====================

@app.exception_handler(429)
async def rate_limit_exception_handler(request: Request, exc: HTTPException):
    """Custom handler for 429 Rate Limit errors."""
    
    retry_after = exc.headers.get("Retry-After", "60")
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate Limit Exceeded",
            "message": "You've made too many requests. Please slow down.",
            "retry_after_seconds": int(retry_after),
            "retry_at": (datetime.utcnow() + timedelta(seconds=int(retry_after))).isoformat() + "Z",
            "details": exc.detail if hasattr(exc, 'detail') else None
        },
        headers={"Retry-After": retry_after}
    )


# ==================== PYDANTIC SCHEMAS ====================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    filters: Optional[Dict[str, Any]] = None
    show_contacts: bool = False


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    error: Optional[str] = None

class AnalyzeFitRequest(BaseModel):
    job_description: str = Field(..., min_length=10, max_length=10000, description="Original job query/description")
    candidate_id: str = Field(..., min_length=1, max_length=64, description="Candidate ID to analyze")

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


class AsyncResumeUploadResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==================== ROUTES ====================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the main HTML interface."""
    logger.debug("Serving home page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
def root() -> Dict[str, str]:
    """API root endpoint."""
    return {
        "message": "RecruiterBrain v2 API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "POST /v2/chat",
            "insight": "POST /v2/insight",
            "upload": "POST /v2/upload_resume_celery",
            "job_status": "GET /v2/jobs/celery/{task_id}"
        }
    }


@app.post("/v2/chat", response_model=ChatResponse)
async def chat_v2_endpoint(request: Request, chat_input: ChatRequest,response: Response) -> Dict[str, Optional[str]]:
    """
    Chat endpoint - conversational search.
    
    Rate limit: 20 requests per minute per IP.
    """
    # Check rate limit
    await check_rate_limit(request, "chat")
    # Get current limit status
    limiter = get_rate_limiter()
    client_ip = request.client.host
    
    # Add headers
    response.headers["X-RateLimit-Limit"] = "20"
    response.headers["X-RateLimit-Window"] = "60"
    question = chat_input.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    filters = chat_input.filters or {}
    top_k = int(filters.get("top_k", 10))
    career_stage = filters.get("career_stage")
    industry = filters.get("industry")

    logger.info(f"V2 Chat request: len={len(question)}, ip={get_client_ip(request)}")

    try:
        results = search_candidates_v2(
            query=question,
            top_k=top_k,
            career_stage=career_stage,
            industry=industry,
        )
        answer = format_for_chat(results, show_contacts=chat_input.show_contacts)
        return ChatResponse(answer=answer).model_dump()
    
    except Exception as exc:
        logger.exception("V2 chat error")
        return ChatResponse(error=str(exc)).model_dump()


@app.post("/v2/insight", response_model=InsightResponse)
async def insight_v2_endpoint(request: Request, payload: InsightRequest) -> Dict[str, Any]:
    """
    Insight endpoint - detailed candidate ranking.
    
    Rate limit: 10 requests per minute per IP (more expensive operation).
    """
    # Check rate limit
    await check_rate_limit(request, "insight")
    
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    filters = payload.filters or {}
    top_k = int(filters.get("top_k", 20))
    career_stage = filters.get("career_stage")
    industry = filters.get("industry")

    logger.info(f"V2 Insight request: len={len(question)}, ip={get_client_ip(request)}")

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
    
    except Exception as exc:
        logger.exception("V2 insight error")
        return InsightResponse(error=str(exc)).model_dump()


@app.post("/v2/upload_resume", response_model=ResumeUploadResponse)
async def upload_resume(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Synchronous resume upload (blocking).
    
    Rate limit: 5 uploads per minute per IP.
    
    Note: For production, use /v2/upload_resume_celery instead.
    """
    # Check rate limit
    await check_rate_limit(request, "upload")
    
    filename = file.filename or "unknown.pdf"
    
    logger.info("="*60)
    logger.info(f"📤 RESUME UPLOAD STARTED: {filename}")
    logger.info("="*60)
    
    try:
        # Read file bytes
        logger.info("Step 0: Reading file bytes...")
        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        logger.info(f"✅ File read complete: {file_size_mb:.2f} MB")
        
        # Parse file
        logger.info(f"Step 1: Parsing file ({filename})...")
        resume_text = parse_file(filename, file_bytes)
        logger.info(f"✅ Text extracted: {len(resume_text)} characters")
        
        if len(resume_text) < 100:
            raise ValueError("Resume text too short. Please upload a valid resume.")
        
        # Scrub PII
        logger.info("Step 2: Scrubbing PII...")
        sanitized_text, pii = scrub_pii(resume_text)
        logger.info(f"✅ PII scrubbed: {len(pii)} fields")
        
        # LLM extraction
        logger.info("Step 3: LLM extraction...")
        extracted_data = extract_resume_data(sanitized_text)
        logger.info(f"✅ Extracted: {extracted_data.get('name', 'Unknown')}")
        
        # Merge PII
        logger.info("Step 4: Merging PII...")
        complete_data = merge_pii(extracted_data, pii)
        logger.info("✅ PII merged")
        
        # Generate embeddings
        logger.info("Step 5: Generating embeddings...")
        embeddings = generate_embeddings(complete_data)
        logger.info("✅ Embeddings generated")
        
        # Insert to Milvus
        logger.info("Step 6: Inserting to Milvus...")
        candidate_id = insert_candidate(complete_data, embeddings, source_channel="Upload")
        logger.info(f"✅ Candidate inserted: {candidate_id}")
        
        # Invalidate search cache
        invalidate_search_cache()
        
        logger.info("="*60)
        logger.info(f"✅ UPLOAD COMPLETE: {complete_data.get('name')} ({candidate_id})")
        logger.info("="*60)
        
        return ResumeUploadResponse(
            candidate_id=candidate_id,
            name=complete_data.get("name", "Unknown"),
            status="success",
            message=f"Resume successfully ingested for {complete_data.get('name')}"
        ).model_dump()
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception(f"❌ UPLOAD FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/v2/upload_resume_async", response_model=AsyncResumeUploadResponse)
async def upload_resume_async(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """
    Asynchronous resume upload using FastAPI BackgroundTasks.
    
    Rate limit: 5 uploads per minute per IP.
    
    Returns immediately with job_id. Poll GET /v2/jobs/{job_id} for status.
    """
    # Check rate limit
    await check_rate_limit(request, "upload")
    
    filename = file.filename or "unknown.pdf"
    logger.info(f"📤 ASYNC RESUME UPLOAD: {filename}")
    
    try:
        # Read and validate file
        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        
        if file_size_mb > 10:
            raise ValueError(f"File too large: {file_size_mb:.2f} MB (max 10 MB)")
        
        logger.info(f"✅ File validated: {file_size_mb:.2f} MB")
        
        # Create job
        job_store = get_job_store()
        job_id = job_store.create_job(
            job_type="resume_upload",
            params={"filename": filename, "file_size_mb": file_size_mb}
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
            message=f"Resume queued. Check status at /v2/jobs/{job_id}"
        ).model_dump()
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/v2/upload_resume_celery", response_model=AsyncResumeUploadResponse)
async def upload_resume_celery(
    request: Request,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Production-grade async upload using Celery.
    
    Rate limit: 5 uploads per minute per IP.
    
    Advantages:
    - Distributed workers
    - Automatic retries
    - Persistent queue
    - Monitoring with Flower
    """
    # Check rate limit
    await check_rate_limit(request, "upload")
    
    filename = file.filename or "unknown.pdf"
    logger.info(f"📤 CELERY RESUME UPLOAD: {filename}")
    
    try:
        # Read and validate file
        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        
        if file_size_mb > 10:
            raise ValueError(f"File too large: {file_size_mb:.2f} MB (max 10 MB)")
        
        logger.info(f"✅ File validated: {file_size_mb:.2f} MB")
        
        # Encode to base64 (Celery serialization)
        file_bytes_b64 = base64.b64encode(file_bytes).decode('utf-8')
        
        # Submit to Celery
        task = process_resume_task.apply_async(
            kwargs={
                "filename": filename,
                "file_bytes_b64": file_bytes_b64,
                "source_channel": "Upload"
            },
            queue="resume_processing"
        )
        
        logger.info(f"✅ Celery task submitted: {task.id}")
        
        return AsyncResumeUploadResponse(
            job_id=task.id,
            status="queued",
            message=f"Resume queued. Check status at /v2/jobs/celery/{task.id}"
        ).model_dump()
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/v2/bulk_upload_resumes")
async def bulk_upload_resumes(
    request: Request,
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """Bulk upload multiple resumes (with optional Celery)."""
    # await check_rate_limit(request, "bulk_upload")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch")
    
    logger.info(f"📤 BULK UPLOAD: {len(files)} files")
    
    resume_data = []
    
    for file in files:
        file_bytes = await file.read()
        filename = file.filename or "unknown.pdf"
        
        # Check if ZIP file
        if filename.endswith('.zip'):
            logger.info(f"📦 Extracting ZIP file: {filename}")
            
            import zipfile
            import io
            
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zip_file:
                    for member in zip_file.namelist():
                        if member.endswith('/') or member.startswith('__MACOSX'):
                            continue
                        
                        if member.endswith(('.docx', '.pdf', '.doc', '.txt')):
                            member_bytes = zip_file.read(member)
                            member_bytes_b64 = base64.b64encode(member_bytes).decode('utf-8')
                            
                            resume_data.append({
                                "filename": member,
                                "file_bytes_b64": member_bytes_b64,
                            })
                            
                logger.info(f"✅ Extracted {len(resume_data)} resumes from ZIP")
            
            except Exception as e:
                logger.error(f"❌ Failed to extract ZIP: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {e}")
        
        else:
            file_bytes_b64 = base64.b64encode(file_bytes).decode('utf-8')
            resume_data.append({
                "filename": filename,
                "file_bytes_b64": file_bytes_b64,
            })
    
    if not resume_data:
        raise HTTPException(status_code=400, detail="No valid resume files found")
    
    logger.info(f"📋 Total resumes to process: {len(resume_data)}")
    
    # ====== USE CELERY OR BACKGROUNDTASKS ======
    from .config import USE_CELERY
    
    if USE_CELERY:
        # Use Celery (parallel processing)
        logger.info("Using Celery for parallel processing")
        
        task = bulk_process_resumes.apply_async(
            args=[resume_data],
            queue="bulk_processing"
        )
        
        return {
            "job_id": task.id,
            "status": "processing",
            "total_files": len(resume_data),
            "message": f"Processing {len(resume_data)} resumes in parallel (Celery)"
        }
    
    else:
        # Use BackgroundTasks (sequential processing)
        logger.info("Using BackgroundTasks for sequential processing")
        
        job_store = get_job_store()
        job_id = job_store.create_job(
            job_type="bulk_upload",
            params={"total_files": len(resume_data)}
        )
        
        # Process each resume sequentially in background
        for idx, resume in enumerate(resume_data):
            background_tasks.add_task(
                process_resume_upload,
                job_id=f"{job_id}_file_{idx}",
                filename=resume["filename"],
                file_bytes=base64.b64decode(resume["file_bytes_b64"]),
                source_channel=f"BulkUpload_{idx+1}/{len(resume_data)}"
            )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "total_files": len(resume_data),
            "message": f"Processing {len(resume_data)} resumes sequentially (BackgroundTasks)"
        }

@app.get("/v2/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(request: Request, job_id: str) -> Dict[str, Any]:
    """
    Get job status (FastAPI BackgroundTasks).
    
    Rate limit: 60 requests per minute per IP.
    """
    # Check rate limit
    await check_rate_limit(request, "job_status")
    
    job_store = get_job_store()
    job_data = job_store.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatusResponse(**job_data).model_dump()


@app.get("/v2/jobs/celery/{task_id}")
async def get_celery_job_status(request: Request, task_id: str) -> Dict[str, Any]:
    """
    Get Celery task status.
    
    Rate limit: 60 requests per minute per IP.
    """
    # Check rate limit
    await check_rate_limit(request, "job_status")
    
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id, app=celery_app)
    
    response = {
        "job_id": task_id,
        "status": task.state,
        "result": None,
        "error": None,
        "progress": None,
    }
    
    if task.state == 'PENDING':
        response["status"] = "queued"
    
    elif task.state == 'PROCESSING':
        if task.info:
            response["progress"] = task.info.get("progress", 0)
            response["status_message"] = task.info.get("status", "Processing...")
    
    elif task.state == 'SUCCESS':
        response["status"] = "completed"
        response["result"] = task.result
    
    elif task.state == 'FAILURE':
        response["status"] = "failed"
        response["error"] = str(task.info)
    
    elif task.state == 'RETRY':
        response["status"] = "retrying"
        response["error"] = str(task.info)
    
    return response


@app.delete("/v2/jobs/{job_id}")
async def cancel_job(request: Request, job_id: str):
    """Cancel a queued job."""
    job_store = get_job_store()
    job_data = job_store.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job_data["status"] == JobStatus.QUEUED:
        job_store.update_job(job_id, JobStatus.FAILED, error="Cancelled by user")
        return {"message": f"Job {job_id} cancelled"}
    else:
        return {"message": f"Job {job_id} is {job_data['status']}, cannot cancel"}

@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Custom 429 response with helpful message."""
    
    retry_after = exc.headers.get("Retry-After", "60")
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate Limit Exceeded",
            "message": "You've made too many requests. Please slow down.",
            "retry_after_seconds": int(retry_after),
            "retry_at": (datetime.utcnow() + timedelta(seconds=int(retry_after))).isoformat() + "Z",
            "documentation": "https://your-docs.com/rate-limits"
        },
        headers={"Retry-After": retry_after}
    )
@app.post("/v2/transcribe")
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Transcribe audio to text using Whisper API.
    
    Used for voice input in UI.
    
    Rate limit: 20 requests per minute (same as chat).
    """
    # Check rate limit
    await check_rate_limit(request, "chat")  # Reuse chat limit
    
    filename = audio.filename or "audio.webm"
    
    logger.info(f"🎤 Audio transcription request: {filename}")
    
    try:
        # Read audio bytes
        audio_bytes = await audio.read()
        file_size_kb = len(audio_bytes) / 1024
        
        # Validate size (max 25MB for Whisper)
        if file_size_kb > 25 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too large: {file_size_kb:.1f}KB (max 25MB)"
            )
        
        logger.info(f"Audio size: {file_size_kb:.1f}KB")
        
        # Transcribe
        transcription = transcribe_audio_whisper(audio_bytes, filename)
        
        if not transcription:
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please try again."
            )
        
        logger.info(f"✅ Transcription: {transcription}")
        
        return {
            "text": transcription,
            "filename": filename,
            "size_kb": round(file_size_kb, 1)
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.exception(f"Transcription error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@app.post("/v2/analyze_fit")
async def analyze_candidate_fit_endpoint(
    request: Request,
    payload: AnalyzeFitRequest  # Use the Pydantic model here
) -> Dict[str, Any]:
    """
    Deep candidate fit analysis (Phase 2).
    
    Called when user clicks "Why?" button in insight mode.
    
    Rate limit: 20 requests per minute (same as chat).
    
    Request body:
    {
        "job_description": "Find Oracle Fusion ERP developers...",
        "candidate_id": "D6DA7C"
    }
    
    Response:
    {
        "candidate_id": "D6DA7C",
        "candidate_name": "Athreya Sunki",
        "fit_level": "not_fit",
        "score": 20,
        "fit_badge": "❌ NOT A FIT",
        "explanation": "Detailed explanation...",
        "strengths": [...],
        "weaknesses": [...],
        "recommendation": "DO NOT PROCEED...",
        "critical_mismatch": {...},
        "onboarding_estimate": {...}
    }
    """
    # Check rate limit (reuse chat limit)
    await check_rate_limit(request, "chat")
    
    # Extract from payload
    job_description = payload.job_description
    candidate_id = payload.candidate_id
    
    logger.info(f"🎯 Fit analysis requested: candidate={candidate_id}, query_len={len(job_description)}")
    
    try:
        # Get Milvus client
        client = get_milvus_client()
        
        # Check cache with version awareness
        cache = get_cache()
        cache_key = generate_cache_key(
            "fit_analysis",
            candidate_id=candidate_id,
            query_hash=hashlib.md5(job_description.encode()).hexdigest()[:8]
        )
        
        # Try to get from cache
        if ENABLE_CACHE:
            cached = cache.get(cache_key)
            
            if cached:
                # Check if candidate was updated after cache
                cached_version = cached.get("candidate_version", "")
                
                # Fetch current candidate version
                try:
                    current_candidate = client.query(
                        collection_name=COLLECTION,
                        filter=f'candidate_id == "{candidate_id}"',
                        output_fields=["last_updated"],
                        limit=1
                    )
                    
                    if current_candidate:
                        current_version = current_candidate[0].get("last_updated", "")
                        
                        # Compare versions
                        if current_version == cached_version:
                            logger.info(f"✅ Cache HIT (version match): {candidate_id}")
                            return cached
                        else:
                            logger.info(f"⚠️  Cache STALE (version mismatch): {candidate_id}")
                            cache.delete(cache_key)  # Invalidate stale cache
                except Exception as cache_check_error:
                    logger.warning(f"Cache version check failed: {cache_check_error}")
                    # Continue with fresh analysis
        
        # Step 1: Extract requirements from job description
        requirements = extract_requirements(job_description)
        
        # CHECK FOR EXTRACTION ERRORS
        if requirements.get("error"):
            return {
                "candidate_id": candidate_id,
                "candidate_name": "Unknown",
                "fit_level": "unknown",
                "score": 0,
                "fit_badge": "⚠️ UNCLEAR REQUIREMENTS",
                "explanation": f"Cannot analyze fit: {requirements['error']}",
                "strengths": [],
                "weaknesses": [],
                "recommendation": "Please provide a more detailed job description with specific skills and requirements.",
                "error": requirements["error"],
                "matched_skills": [],
                "missing_skills": [],
                "critical_mismatch": None,
                "onboarding_estimate": None,
                "candidate_version": "",
                "analyzed_at": datetime.utcnow().isoformat() + "Z"
            }
        
        # If no skills extracted at all
        if not requirements.get("must_have_skills") and not requirements.get("nice_to_have_skills"):
            return {
                "candidate_id": candidate_id,
                "candidate_name": "Unknown",
                "fit_level": "unknown",
                "score": 0,
                "fit_badge": "⚠️ NO REQUIREMENTS",
                "explanation": "No technical skills or requirements were detected in the job description. Unable to perform meaningful fit analysis.",
                "strengths": ["Cannot determine without requirements"],
                "weaknesses": ["Cannot determine without requirements"],
                "recommendation": "Please provide a job description that includes:\n- Required technical skills\n- Preferred technologies\n- Experience level\n- Domain expertise needed",
                "error": "No requirements extracted",
                "matched_skills": [],
                "missing_skills": [],
                "critical_mismatch": None,
                "onboarding_estimate": None,
                "candidate_version": "",
                "analyzed_at": datetime.utcnow().isoformat() + "Z"
            }
        
        # Add original query text (needed for module detection)
        requirements["query_text"] = job_description
        
        logger.info(f"   Requirements: {len(requirements.get('must_have_skills', []))} skills, seniority={requirements.get('seniority_level')}")
        
        # Step 2: Fetch candidate from Milvus
        filter_expr = f'candidate_id == "{candidate_id}"'
        
        results = client.query(
            collection_name=COLLECTION,
            filter=filter_expr,
            output_fields=SEARCH_OUTPUT_FIELDS,
            limit=1
        )
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Candidate {candidate_id} not found"
            )
        
        candidate = results[0]
        candidate_version = candidate.get("last_updated", "")
        
        logger.info(f"   Found candidate: {candidate.get('name')}")
        
        # Step 3: Compute quick match details (Phase 1)
        from .ranker import compute_match_details_enhanced
        
        quick_match = compute_match_details_enhanced(
            candidate=candidate,
            required_skills=requirements.get("must_have_skills", []),
            requirements=requirements
        )
        
        logger.info(f"   Quick match: {quick_match['fit_level']} ({quick_match['match_percentage']}%)")
        
        # Step 4: Generate detailed analysis (Phase 2)
        from .fit_analyzer import analyze_candidate_fit
        
        analysis = analyze_candidate_fit(
            candidate=candidate,
            requirements=requirements,
            quick_match=quick_match
        )
        
        logger.info(f"   ✅ Analysis complete: fit={analysis['fit_level']}")
        
        # Step 5: Prepare result
        result = {
            "candidate_id": candidate_id,
            "candidate_name": candidate.get("name", "Unknown"),
            "fit_level": analysis["fit_level"],
            "score": analysis["score"],
            "fit_badge": analysis["fit_badge"],
            "explanation": analysis["explanation"],
            "strengths": analysis["strengths"],
            "weaknesses": analysis["weaknesses"],
            "recommendation": analysis["recommendation"],
            "critical_mismatch": analysis.get("critical_mismatch"),
            "onboarding_estimate": analysis.get("onboarding_estimate"),
            "matched_skills": quick_match.get("matched_skills", []),
            "missing_skills": quick_match.get("missing_skills", []),
            "candidate_version": candidate_version,
            "analyzed_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Cache with version
        if ENABLE_CACHE:
            cache.set(cache_key, result, ttl=3600)  # 1 hour TTL
            logger.info(f"   Cached result for {candidate_id}")
        
        return result
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.exception(f"❌ Fit analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fit analysis failed: {str(e)}"
        )
# ==================== HEALTH CHECK ====================

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "recruiterbrainv2.__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )