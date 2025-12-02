"""Console + API entry points for recruiterbrain."""
from __future__ import annotations

import argparse
from datetime import timedelta
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time
import json
import asyncio
import io
import zipfile


from fastapi import Body, FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import storage
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File, Form
from recruiterbrain.resume_ingestion import ingest_resume_upload,ingest_resume_bytes
from recruiterbrain.env_loader import load_env
from recruiterbrain.logging_config import configure_logging
from recruiterbrain.shared_config import get_encoder, get_milvus_client


load_env()
configure_logging()
logger = logging.getLogger(__name__)

if __package__ in (None, ""):
    # Allow running via `python path/to/__main__.py` by adding repo root to sys.path.
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from recruiterbrain.app_workflow import (  # type: ignore
        answer_question,
        get_last_insight_result,
        llm_plan,
        run_cli,
    )
    from recruiterbrain.shared_config import (  # type: ignore
        INSIGHT_DEFAULT_K,
        ROUTER_CONFIDENCE_THRESHOLD,
    )
else:
    from .app_workflow import answer_question, get_last_insight_result, llm_plan, run_cli
    from .shared_config import INSIGHT_DEFAULT_K, ROUTER_CONFIDENCE_THRESHOLD


BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="Recruiter Brain UI + API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    _ = get_milvus_client()
    _ = get_encoder()
    logger.info("Warmup: Milvus client and encoder loaded")
except Exception as exc:
    logger.warning("Warmup failed (will lazy-load on demand): %s", exc)

# ---- Tiny in-memory cache for identical questions ----
_CHAT_CACHE: Dict[str, ChatResponse] = {}
_CHAT_CACHE_MAX = 200  # cap to avoid unbounded growth
def _chat_cache_key(chat_input: ChatRequest) -> str:
    return json.dumps(
        {
            "q": chat_input.question.strip(),
            "tools": chat_input.required_tools or [],
            "filters": chat_input.filters or {},
            "show_contacts": chat_input.show_contacts,
        },
        sort_keys=True,
    )

# ---- Simple in-process rate limiting ----
# For prod you’d back this with Redis / Memorystore.

_RATE_LIMIT_STATE: Dict[str, Tuple[float, int]] = {}
_RATE_WINDOW_SECONDS = 60.0          # 1-minute window
_RATE_MAX_REQUESTS = 30              # 30 requests / minute / IP


def _rate_key(request: Request) -> str:
    # For now, use client IP; later you can swap this for user_id / API key
    return request.client.host or "unknown"


def rate_limiter(request: Request) -> None:
    now = time.time()
    key = _rate_key(request)
    window_start, count = _RATE_LIMIT_STATE.get(key, (now, 0))

    # New window
    if now - window_start > _RATE_WINDOW_SECONDS:
        _RATE_LIMIT_STATE[key] = (now, 1)
        return

    # Existing window
    if count >= _RATE_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again soon.")

    _RATE_LIMIT_STATE[key] = (window_start, count + 1)



class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=12000)
    required_tools: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    show_contacts: bool = False


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    error: Optional[str] = None


class InsightRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=12000)
    required_tools: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    show_contacts: bool = False


class InsightResponse(BaseModel):
    text: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    scarcity_message: Optional[str] = None
    data_quality_banner: Optional[str] = None
    total_matched: Optional[int] = None
    clarify: Optional[str] = None
    error: Optional[str] = None


class MarkdownExport(BaseModel):
    markdown: Optional[str] = None
    error: Optional[str] = None


class JSONExport(BaseModel):
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ResumeIngestResponse(BaseModel):
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    status: str
class BulkIngestItem(BaseModel):
    filename: str
    candidate_id: str | None = None
    name: str | None = None
    email: str | None = None
    status: str
    error: str | None = None


class BulkIngestResponse(BaseModel):
    successes: List[BulkIngestItem]
    failures: List[BulkIngestItem]


def _normalize_required_tools(tools: Optional[List[str]]) -> List[str]:
    if not tools:
        return []
    normalized = []
    for tool in tools:
        text = str(tool).strip().lower()
        if text and text not in normalized:
            normalized.append(text)

    MAX_CORE_TOOLS = 25
    return normalized[:MAX_CORE_TOOLS]


def _build_insight_response(text: str) -> InsightResponse:
    latest = get_last_insight_result() or {}
    return InsightResponse(
        text=text,
        rows=latest.get("rows"),
        scarcity_message=latest.get("scarcity_message"),
        data_quality_banner=latest.get("data_quality_banner"),
        total_matched=latest.get("total_matched"),
    )

async def _ingest_one_resume_concurrent(
    filename: str,
    data: bytes,
    source_channel: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict | Exception]:
    """
    Wrap ingest_resume_bytes() in a background thread with concurrency control.

    Returns (filename, result_or_exception).
    """
    async with semaphore:
        try:
            result = await asyncio.to_thread(
                ingest_resume_bytes,
                filename,
                data,
                source_channel,
            )
            return filename, result
        except Exception as exc:
            return filename, exc

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    logger.debug("Serving home page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat_input: ChatRequest,_: None = Depends(rate_limiter),) -> Dict[str, Optional[str]] | JSONResponse:
    key = _chat_cache_key(chat_input)
    if key in _CHAT_CACHE:
         logger.debug("Chat cache hit")
         return _CHAT_CACHE[key].model_dump()
    question = chat_input.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    if len(question) > 4000:
        raise HTTPException(status_code=413, detail="Question too long.")

    logger.info(
        "Chat request received (len=%d, show_contacts=%s)",
        len(question),
        chat_input.show_contacts,
    )

    plan_override: Optional[Dict[str, Any]] = None
    question_text = question
    if chat_input.show_contacts:
        question_text = f"{question_text} give their info"

    if chat_input.required_tools or chat_input.filters:
        plan_override = llm_plan(question_text)
        plan_override["question"] = question_text
        if chat_input.required_tools:
            plan_override["required_tools"] = [
                str(tool).strip().lower() for tool in chat_input.required_tools if tool
            ]
        if chat_input.filters:
            plan_override["filters"] = chat_input.filters

    try:
        answer = answer_question(question_text, plan_override=plan_override)
    except RuntimeError as exc:
        logger.error("Setup issue while answering question: %s", exc)
        return ChatResponse(error=f"Setup issue: {exc}").model_dump()
    except Exception as exc:  # pragma: no cover - surface friendly error to UI
        logger.exception("Unhandled error in chat endpoint")
        return JSONResponse(
            status_code=500,
            content=ChatResponse(error=str(exc)).model_dump(),
        )
    resp = ChatResponse(answer=answer)
    if len(_CHAT_CACHE) >= _CHAT_CACHE_MAX:
        # naive eviction: clear everything when full
        _CHAT_CACHE.clear()
    _CHAT_CACHE[key] = resp
    return ChatResponse(answer=answer).model_dump()

@app.get("/candidates/{candidate_id}/resume_url")
def get_resume_url(candidate_id: str, ttl_minutes: int = 15) -> Dict[str, str]:
    """
    Given a candidate_id (e.g. 'HJTRD4'), generate a fresh signed URL to their raw resume in GCS.

    - Does NOT touch Milvus; uses deterministic GCS path: <prefix>/<candidate_id>.
    - ttl_minutes: how long the URL should be valid for (default: 15 minutes).
    """

    bucket_name = os.getenv("GCS_RESUME_BUCKET")
    prefix = os.getenv("GCS_RESUME_PREFIX", "raw-resumes")

    if not bucket_name:
        raise HTTPException(status_code=500, detail="GCS_RESUME_BUCKET not configured")

    if ttl_minutes <= 0 or ttl_minutes > 24 * 60:
        # Safety bounds: 1 minute to 24 hours
        ttl_minutes = 15

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    object_name = f"{prefix}/{candidate_id}"
    blob = bucket.blob(object_name)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="Resume not found for this candidate_id")

    url = blob.generate_signed_url(
        expiration=timedelta(minutes=ttl_minutes),
        method="GET",
    )

    return {
        "candidate_id": candidate_id,
        "resume_url": url,
        "expires_in_minutes": ttl_minutes,
    }

@app.post("/ingest_resumes_bulk", response_model=BulkIngestResponse)
async def ingest_resumes_bulk(
    request: Request,
    files: List[UploadFile] = File(...),
    source_channel: str = Form("BulkUpload"),
    _: None = Depends(rate_limiter),
) -> Dict[str, Any]:
    """
    Bulk ingest multiple resumes in a single request (with basic concurrency + ZIP support).

    - Accepts multiple files.
    - If a file is a .zip, expands it and ingests each valid entry.
    - Reuses the same pipeline as /ingest_resume (GCS + Milvus).
    """

    MAX_FILES = 200  # guardrail: total logical resumes per request
    MAX_CONCURRENCY = 3  # avoid hammering OpenAI + CPU

    # 1) Collect (filename, data) pairs for all files & zip entries
    logical_files: list[tuple[str, bytes]] = []

    for upload in files:
        filename = upload.filename or "unknown"
        raw_data = await upload.read()

        if filename.lower().endswith(".zip"):
            # Expand zip
            try:
                with zipfile.ZipFile(io.BytesIO(raw_data)) as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
                        inner_name = member.filename

                        # Only process supported extensions
                        lower = inner_name.lower()
                        if not (
                            lower.endswith(".pdf")
                            or lower.endswith(".doc")
                            or lower.endswith(".docx")
                            or lower.endswith(".txt")
                        ):
                            logger.info(
                                "Skipping non-resume file in ZIP: %s", inner_name
                            )
                            continue

                        file_bytes = zf.read(member)
                        logical_files.append((inner_name, file_bytes))
            except Exception as exc:
                logger.exception("Failed to process ZIP file %s", filename)
                # register the whole zip as a failure
                logical_files.append(
                    (
                        filename,
                        b"",  # will fail later in ingestion
                    )
                )
        else:
            # Regular single resume file
            logical_files.append((filename, raw_data))

    if len(logical_files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many resumes in bulk request; max {MAX_FILES}. Found {len(logical_files)}.",
        )

    # 2) Launch concurrent ingestion tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks: list[asyncio.Task] = []

    for (fname, data) in logical_files:
        task = asyncio.create_task(
            _ingest_one_resume_concurrent(
                filename=fname,
                data=data,
                source_channel=source_channel,
                semaphore=semaphore,
            )
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=False)

    # 3) Build successes / failures
    successes: List[BulkIngestItem] = []
    failures: List[BulkIngestItem] = []

    for (filename, outcome) in results:
        if isinstance(outcome, Exception):
            logger.exception("Bulk ingest failed for %s", filename, exc_info=outcome)
            failures.append(
                BulkIngestItem(
                    filename=filename,
                    candidate_id=None,
                    name=None,
                    email=None,
                    status="error",
                    error=str(outcome),
                )
            )
        else:
            # outcome is the dict returned by ingest_resume_bytes()
            successes.append(
                BulkIngestItem(
                    filename=filename,
                    candidate_id=outcome.get("candidate_id"),
                    name=outcome.get("name"),
                    email=outcome.get("email"),
                    status=outcome.get("status", "inserted"),
                    error=None,
                )
            )

    return BulkIngestResponse(successes=successes, failures=failures).model_dump()


@app.post("/insight", response_model=InsightResponse)
def insight_endpoint(payload: InsightRequest,_: None = Depends(rate_limiter),) -> Dict[str, Any]:
    question = payload.question.strip()
    logger.info("Insight request received (filters=%s, tools=%s)", payload.filters is not None, payload.required_tools)
    plan = llm_plan(question)
    router_conf = plan.get("_router_confidence", 1.0)
    if router_conf < ROUTER_CONFIDENCE_THRESHOLD:
        clarify = plan.get("clarify") or "Did you want an insight ranking on tools Milvus, dbt, AWS, Vertex AI?"
        logger.info("Router asked for clarification before insight run")
        return InsightResponse(clarify=clarify).model_dump()

    req_tools = _normalize_required_tools(payload.required_tools)
    if req_tools:
        plan["required_tools"] = req_tools
    plan["intent"] = "insight"
    plan["k"] = int(plan.get("k") or INSIGHT_DEFAULT_K)
    if payload.filters:
        plan["filters"] = payload.filters

    ask_text = question
    if payload.show_contacts:
        ask_text = (ask_text + " give their info").strip()
    plan["question"] = ask_text

    text = answer_question(ask_text, plan_override=plan)
    logger.info("Insight response ready")
    return _build_insight_response(text).model_dump()


@app.post("/filter", response_model=InsightResponse)
def filter_endpoint(payload: InsightRequest) -> Dict[str, Any]:
    question = payload.question.strip() or "Filter update"
    logger.debug("Filter endpoint invoked; normalized question to '%s'", question)
    updated_payload = payload.model_copy(update={"question": question})
    return insight_endpoint(updated_payload)


@app.get("/export/md", response_model=MarkdownExport)
def export_md() -> Dict[str, Any]:
    latest = get_last_insight_result()
    if not latest:
        return MarkdownExport(error="No insight result available yet.").model_dump()
    logger.debug("Exporting last insight result as Markdown")
    lines = ["# Insight Ranking", f"Total matched: {latest.get('total_matched', 0)}"]
    scarcity = latest.get("scarcity_message")
    if scarcity:
        lines.append(f"*{scarcity}*")
    dq = latest.get("data_quality_banner")
    if dq:
        lines.append(f"_{dq}_")
    rows = latest.get("rows") or []
    for row in rows:
        lines.append(f"- **{row.get('tier', 'Tier')}** - {row.get('title_line')}")
        lines.append(f"  - {row.get('match_chip')}")
        lines.append(f"  - Why: {row.get('why')}")
        lines.append(f"  - Notes: {row.get('notes')}")
    markdown = "\n".join(lines)
    return MarkdownExport(markdown=markdown).model_dump()


@app.get("/export/json", response_model=JSONExport)
def export_json() -> Dict[str, Any]:
    latest = get_last_insight_result()
    if not latest:
        return JSONExport(error="No insight result available yet.").model_dump()
    logger.debug("Exporting last insight result as JSON")
    return JSONExport(payload=latest).model_dump()


@app.post("/ingest_resume", response_model=ResumeIngestResponse)
async def ingest_resume_endpoint(
    request: Request,
    file: UploadFile = File(...),
    source_channel: str = Form("Upload"),
    _: None = Depends(rate_limiter),
) -> Dict[str, Any]:
    """
    Upload a single resume (PDF/DOCX/TXT), parse it with the LLM,
    embed it, and insert into Milvus (new_candidate_pool).
    """
    logger.info("Received resume upload: filename=%s source_channel=%s", file.filename, source_channel)
    try:
        result = await ingest_resume_upload(file, source_channel=source_channel)
        return ResumeIngestResponse(
            candidate_id=result["candidate_id"],
            name=result.get("name"),
            email=result.get("email"),
            status=result["status"],
        ).model_dump()
    except Exception as exc:
        logger.exception("Error ingesting resume")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/help")
def help_menu() -> Dict[str, Any]:
    return {
        "examples": [
            "Compare candidates for Milvus, dbt, AWS, Vertex AI (insight)",
            "List top 10 with dbt only, min 3 yrs exp, healthcare industry",
            "Show only Perfect & Good and explain who's missing Milvus",
            "Allow Milvus equivalents and re-rank",
            "Filter to 5-10 YOE, Finance industry, must-have Vertex AI",
            "Why is Candidate X ranked above Candidate Y?",
            "Relax to 2/4 and show differences vs strict 4/4",
            "Show evidence for top-3 only",
            "What skill is the scarcest in this pool?",
            "Which candidates used these tools most recently?",
        ]
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recruiterbrain",
        description="Ask recruiting questions over the Milvus candidate pool.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="If supplied, answer a single question and exit. Otherwise start the interactive shell.",
    )
    args = parser.parse_args(argv)

    if args.question:
        try:
            print(answer_question(args.question))
        except Exception as exc:  # pragma: no cover - CLI feedback only
            parser.exit(status=1, message=f"Error: {exc}\n")
        return 0

    run_cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
