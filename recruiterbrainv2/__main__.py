"""FastAPI entrypoint for RecruiterBrain v2."""
from __future__ import annotations
import os
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import logging
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .formatter import format_for_chat, format_for_insight
from .retrieval_engine import search_candidates_v2

logger = logging.getLogger(__name__)

app = FastAPI(title="RecruiterBrain v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    question: str = Field(..., min_length=1, max_length=5000)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("recruiterbrainv2.__main__:app", host="0.0.0.0", port=8000, reload=True)

