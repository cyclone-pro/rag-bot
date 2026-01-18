"""Interview analysis utilities for qa_embeddings."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import QA_COLLECTION, get_milvus_client
from .utils.keyword_extractor import extract_keywords
from .utils.sentiment_analyzer import analyze_sentiment, get_sentiment_label

logger = logging.getLogger(__name__)

QA_BASE_FIELDS = [
    "id",
    "interview_id",
    "candidate_id",
    "question_index",
    "answer_snippet",
]


def _safe_filter_value(value: str) -> str:
    return value.replace('"', '""').replace("'", "''").strip()


def _with_milvus_client(fn):
    client_or_pool = get_milvus_client()
    if hasattr(client_or_pool, "connection"):
        with client_or_pool.connection() as client:
            return fn(client)
    return fn(client_or_pool)


def _get_collection_fields(collection_name: str) -> List[str]:
    def _fetch(client):
        try:
            desc = client.describe_collection(collection_name)
        except Exception as exc:
            logger.warning("Describe collection failed: %s", exc)
            return []

        fields: List[Dict[str, Any]] = []
        if isinstance(desc, dict):
            if isinstance(desc.get("fields"), list):
                fields = desc["fields"]
            elif isinstance(desc.get("schema"), dict):
                fields = desc["schema"].get("fields", []) or []

        names: List[str] = []
        for field in fields:
            if not isinstance(field, dict):
                continue
            name = field.get("name") or field.get("field_name")
            if name:
                names.append(name)
        return names

    return _with_milvus_client(_fetch)


def _query_records(
    filter_expr: str,
    output_fields: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    def _run(client):
        return client.query(
            collection_name=QA_COLLECTION,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit,
        )

    return _with_milvus_client(_run)


def _build_filter(
    candidate_id: Optional[str],
    interview_id: Optional[str],
    job_id: Optional[str],
) -> str:
    if interview_id:
        return f'interview_id == "{_safe_filter_value(interview_id)}"'
    if candidate_id:
        return f'candidate_id == "{_safe_filter_value(candidate_id)}"'
    if job_id:
        return f'job_id == "{_safe_filter_value(job_id)}"'
    raise ValueError("Provide candidate_id, interview_id, or job_id.")


def _compute_overall_score(
    avg_sentiment: float,
    tech_keyword_count: int,
    answer_count: int,
) -> Tuple[int, Dict[str, float]]:
    sentiment_norm = max(0.0, min(1.0, (avg_sentiment + 1.0) / 2.0))
    tech_norm = min(1.0, tech_keyword_count / 8.0)
    depth_norm = min(1.0, answer_count / 10.0)
    score = 100 * (0.6 * sentiment_norm + 0.25 * tech_norm + 0.15 * depth_norm)
    return int(round(score)), {
        "sentiment": round(sentiment_norm, 3),
        "keywords": round(tech_norm, 3),
        "depth": round(depth_norm, 3),
    }


def _parse_question_index(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _summarize_answers(
    records: List[Dict[str, Any]],
    include_answers: bool,
) -> Dict[str, Any]:
    answers: List[Dict[str, Any]] = []
    sentiment_scores: List[float] = []
    subjectivities: List[float] = []
    answer_texts: List[str] = []
    positive = neutral = negative = 0

    for rec in sorted(records, key=lambda r: _parse_question_index(r.get("question_index"))):
        snippet = (rec.get("answer_snippet") or "").strip()
        answer_texts.append(snippet)
        sentiment = analyze_sentiment(snippet) if snippet else {"score": 0.0, "polarity": 0.0, "subjectivity": 0.0}
        label = get_sentiment_label(sentiment["score"])

        sentiment_scores.append(sentiment["score"])
        subjectivities.append(sentiment["subjectivity"])

        if sentiment["score"] >= 0.1:
            positive += 1
        elif sentiment["score"] <= -0.1:
            negative += 1
        else:
            neutral += 1

        if include_answers:
            answers.append(
                {
                    "question_index": _parse_question_index(rec.get("question_index")),
                    "answer_snippet": snippet,
                    "sentiment": sentiment,
                    "sentiment_label": label,
                    "keywords": extract_keywords(snippet) if snippet else {
                        "all_keywords": [],
                        "tech_keywords": [],
                        "important_phrases": [],
                    },
                }
            )

    combined_text = " ".join([t for t in answer_texts if t])
    combined_keywords = extract_keywords(combined_text) if combined_text else {
        "all_keywords": [],
        "tech_keywords": [],
        "important_phrases": [],
    }

    answer_count = len(records)
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    avg_subjectivity = sum(subjectivities) / len(subjectivities) if subjectivities else 0.0
    overall_score, score_components = _compute_overall_score(
        avg_sentiment,
        len(combined_keywords.get("tech_keywords", [])),
        answer_count,
    )

    summary = {
        "answer_count": answer_count,
        "avg_sentiment": round(avg_sentiment, 3),
        "sentiment_label": get_sentiment_label(avg_sentiment),
        "avg_subjectivity": round(avg_subjectivity, 3),
        "sentiment_distribution": {
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
        },
        "keywords": combined_keywords,
        "overall_score": overall_score,
        "score_components": score_components,
    }

    if include_answers:
        summary["answers"] = answers

    return summary


def analyze_interviews(
    mode: str,
    candidate_id: Optional[str] = None,
    interview_id: Optional[str] = None,
    job_id: Optional[str] = None,
    latest_only: bool = False,
    limit: int = 500,
) -> Dict[str, Any]:
    fields = _get_collection_fields(QA_COLLECTION)
    fields_set = set(fields)

    output_fields = list(QA_BASE_FIELDS)
    if "job_id" in fields_set:
        output_fields.append("job_id")
    if "job_title" in fields_set:
        output_fields.append("job_title")
    if "job_description" in fields_set:
        output_fields.append("job_description")
    if "interview_date" in fields_set:
        output_fields.append("interview_date")

    if mode == "job" and "job_id" not in fields_set:
        return {
            "error": "job_id field not found in qa_embeddings. Add job_id or migrate to a v2 collection."
        }

    filter_expr = _build_filter(candidate_id, interview_id, job_id)
    records = _query_records(filter_expr, output_fields, limit)

    if not records:
        return {
            "mode": mode,
            "candidate_id": candidate_id,
            "interview_id": interview_id,
            "job_id": job_id,
            "error": "No interview records found for the given filter.",
        }

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        iid = rec.get("interview_id") or "unknown"
        grouped.setdefault(iid, []).append(rec)

    interview_items: List[Dict[str, Any]] = []
    for iid, recs in grouped.items():
        summary = _summarize_answers(recs, include_answers=mode in {"candidate", "interview"})
        interview_items.append(
            {
                "interview_id": iid,
                "candidate_id": recs[0].get("candidate_id"),
                "job_id": recs[0].get("job_id"),
                "job_title": recs[0].get("job_title"),
                "job_description": recs[0].get("job_description"),
                "interview_date": recs[0].get("interview_date"),
                **summary,
            }
        )

    latest_basis = "interview_id"
    if "interview_date" in fields_set:
        latest_basis = "interview_date"
        interview_items.sort(
            key=lambda x: x.get("interview_date") or 0,
            reverse=True,
        )
    else:
        interview_items.sort(key=lambda x: x.get("interview_id") or "", reverse=True)

    if mode == "interview":
        interview_items = interview_items[:1]

    if latest_only and mode == "candidate":
        interview_items = interview_items[:1]

    if latest_only and mode == "job":
        latest_by_candidate: Dict[str, Dict[str, Any]] = {}
        for item in interview_items:
            cid = item.get("candidate_id") or "unknown"
            existing = latest_by_candidate.get(cid)
            if not existing:
                latest_by_candidate[cid] = item
                continue
            if latest_basis == "interview_date":
                if (item.get("interview_date") or 0) > (existing.get("interview_date") or 0):
                    latest_by_candidate[cid] = item
            else:
                if (item.get("interview_id") or "") > (existing.get("interview_id") or ""):
                    latest_by_candidate[cid] = item
        interview_items = list(latest_by_candidate.values())

    if mode == "job":
        candidates: Dict[str, Dict[str, Any]] = {}
        for item in interview_items:
            cid = item.get("candidate_id") or "unknown"
            entry = candidates.setdefault(
                cid,
                {
                    "candidate_id": cid,
                    "interviews": 0,
                    "answer_count": 0,
                    "avg_sentiment": 0.0,
                    "avg_subjectivity": 0.0,
                    "tech_keywords": set(),
                    "top_keywords": set(),
                    "important_phrases": set(),
                    "overall_score": 0.0,
                    "latest_interview_id": item.get("interview_id"),
                },
            )
            entry["interviews"] += 1
            entry["answer_count"] += item.get("answer_count", 0)
            entry["avg_sentiment"] += item.get("avg_sentiment", 0.0)
            entry["avg_subjectivity"] += item.get("avg_subjectivity", 0.0)
            entry["overall_score"] += item.get("overall_score", 0.0)
            keywords = item.get("keywords", {})
            entry["tech_keywords"].update(keywords.get("tech_keywords", []))
            entry["top_keywords"].update(keywords.get("all_keywords", []))
            entry["important_phrases"].update(keywords.get("important_phrases", []))

        ranked: List[Dict[str, Any]] = []
        for entry in candidates.values():
            interviews = max(1, entry["interviews"])
            avg_sentiment = entry["avg_sentiment"] / interviews
            overall_score, score_components = _compute_overall_score(
                avg_sentiment,
                len(entry["tech_keywords"]),
                entry["answer_count"],
            )
            ranked.append(
                {
                    "candidate_id": entry["candidate_id"],
                    "interview_count": entry["interviews"],
                    "answer_count": entry["answer_count"],
                    "avg_sentiment": round(avg_sentiment, 3),
                    "avg_subjectivity": round(entry["avg_subjectivity"] / interviews, 3),
                    "sentiment_label": get_sentiment_label(avg_sentiment),
                    "overall_score": overall_score,
                    "score_components": score_components,
                    "tech_keywords": sorted(entry["tech_keywords"]),
                    "top_keywords": sorted(entry["top_keywords"])[:10],
                    "important_phrases": sorted(entry["important_phrases"])[:5],
                    "latest_interview_id": entry["latest_interview_id"],
                }
            )

        ranked.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        job_title = None
        job_description = None
        if interview_items:
            job_title = interview_items[0].get("job_title")
            job_description = interview_items[0].get("job_description")
        return {
            "mode": mode,
            "job_id": job_id,
            "job_title": job_title,
            "job_description": job_description,
            "latest_basis": latest_basis,
            "total_candidates": len(ranked),
            "candidates": ranked,
        }

    avg_sentiment = sum(i.get("avg_sentiment", 0.0) for i in interview_items) / max(len(interview_items), 1)
    summary_score, summary_components = _compute_overall_score(
        avg_sentiment,
        len({kw for item in interview_items for kw in item.get("keywords", {}).get("tech_keywords", [])}),
        sum(i.get("answer_count", 0) for i in interview_items),
    )

    return {
        "mode": mode,
        "candidate_id": candidate_id,
        "interview_id": interview_id,
        "latest_basis": latest_basis,
        "interviews": interview_items,
        "summary": {
            "interview_count": len(interview_items),
            "avg_sentiment": round(avg_sentiment, 3),
            "sentiment_label": get_sentiment_label(avg_sentiment),
            "overall_score": summary_score,
            "score_components": summary_components,
        },
    }
