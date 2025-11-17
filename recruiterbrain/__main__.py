"""Console + API entry points for recruiterbrain."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates

from recruiterbrain.logging_config import configure_logging

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


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    required_tools: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    show_contacts: bool = False


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    error: Optional[str] = None


class InsightRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
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


def _normalize_required_tools(tools: Optional[List[str]]) -> List[str]:
    if not tools:
        return []
    normalized = []
    for tool in tools:
        text = str(tool).strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _build_insight_response(text: str) -> InsightResponse:
    latest = get_last_insight_result() or {}
    return InsightResponse(
        text=text,
        rows=latest.get("rows"),
        scarcity_message=latest.get("scarcity_message"),
        data_quality_banner=latest.get("data_quality_banner"),
        total_matched=latest.get("total_matched"),
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    logger.debug("Serving home page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat_input: ChatRequest) -> Dict[str, Optional[str]] | JSONResponse:
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
    return ChatResponse(answer=answer).model_dump()


@app.post("/insight", response_model=InsightResponse)
def insight_endpoint(payload: InsightRequest) -> Dict[str, Any]:
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
