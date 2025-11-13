"""User workflow + LLM planning glue for recruiter brain."""
from __future__ import annotations

import json
from typing import Any, Dict

from core_retrieval import ann_search
from shared_config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RETURN_TOP,
    TOP_K,
    VECTOR_FIELD_DEFAULT,
    get_openai_client,
)
from shared_utils import render_candidate


def _default_plan(question: str) -> Dict[str, Any]:
    return {
        "intent": "count",
        "vector_field": VECTOR_FIELD_DEFAULT,
        "must_have_keywords": [],
        "industry_equals": None,
        "require_domains": [],
        "require_career_stage": "Any",
        "networking_required": False,
        "top_k": TOP_K,
        "return_top": RETURN_TOP,
        "question": question,
    }


def llm_plan(question: str) -> Dict[str, Any]:
    client = get_openai_client()
    if not client:
        return _default_plan(question)

    system_prompt = (
        "You parse recruiting analytics questions into a JSON plan for Milvus retrieval over a resume collection. "
        "Never add fields not in schema. Output ONLY JSON. Keys:\n"
        "{\n"
        '  "intent": "count|list|why",\n'
        '  "vector_field": "summary_embedding|skills_embedding",\n'
        '  "must_have_keywords": ["keyword", ...],\n'
        '  "industry_equals": "string or null",\n'
        '  "require_domains": ["Healthcare IT","Construction","CAD","NLP","GenAI", ...],\n'
        '  "require_career_stage": "Entry|Mid|Senior|Lead/Manager|Director+|Any",\n'
        '  "networking_required": true|false,\n'
        '  "top_k": 1000,\n'
        '  "return_top": 20\n'
        "}\n"
        "If the user asks for 'total how many', set intent='count'. If they want names, 'list'. "
        "If they want a brief justification, 'why'. Use 'summary_embedding' by default."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            messages=messages,
        )
        raw = resp.choices[0].message.content.strip()
        plan = json.loads(raw)
    except Exception:
        return _default_plan(question)

    plan = plan or {}
    plan.setdefault("vector_field", VECTOR_FIELD_DEFAULT)
    plan.setdefault("top_k", TOP_K)
    plan.setdefault("return_top", RETURN_TOP)
    plan.setdefault("intent", "count")
    plan.setdefault("must_have_keywords", [])
    plan.setdefault("require_domains", [])
    plan.setdefault("require_career_stage", "Any")
    plan.setdefault("networking_required", False)
    plan["question"] = question
    return plan


def answer_question(question: str) -> str:
    plan = llm_plan(question)
    paired_hits, total_matches = ann_search(plan)
    intent = (plan.get("intent") or "count").lower()
    return_top = int(plan.get("return_top") or RETURN_TOP)

    top_hits = paired_hits[:return_top]

    if intent == "count":
        return f"Total matched candidates: {total_matches}"

    if intent == "list":
        lines = [
            f"{idx}. {render_candidate(entity, sim, detailed=False)}"
            for idx, (entity, sim) in enumerate(top_hits, start=1)
        ]
        body = "\n".join(lines)
        return f"Total matched: {total_matches}\n\n{body}" if body else f"Total matched: {total_matches}"

    # intent == "why"
    blocks = [render_candidate(entity, sim, detailed=True) for entity, sim in top_hits]
    if not blocks:
        return f"Total matched: {total_matches}"
    return f"Total matched: {total_matches}\n\n" + "\n\n".join(blocks)


def print_help() -> None:
    print(
        """Commands:
  /help                     Show this help
  /exit | exit | quit | q   Quit
Type natural language questions, e.g.:
  "total how many candidates have experience in construction, know CAD and Python"
  "list top 10 names for Django + HIPAA + networking"
  "why these 5 fit FHIR + GenAI + NLP"
"""
    )


def run_cli() -> None:
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set; default heuristic planner will be used.")
    print("LLM+Milvus assistant ready. Ask questions (type /help).")
    while True:
        try:
            line = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        low = line.lower()
        if low in {"exit", "/exit", "quit", "q"}:
            print("Bye.")
            break
        if low == "/help":
            print_help()
            continue
        try:
            answer = answer_question(line)
            print(answer)
        except Exception as exc:  # pragma: no cover - best-effort logging on CLI
            print(f"Error: {exc}")


__all__ = ["answer_question", "llm_plan", "print_help", "run_cli"]
