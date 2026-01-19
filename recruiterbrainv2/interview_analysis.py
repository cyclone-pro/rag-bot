"""Interview analysis utilities for qa_embeddings."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import COLLECTION, QA_COLLECTION, get_milvus_client
from .skill_extractor import COMMON_TECH_SKILLS, SKILL_ALIASES, extract_requirements
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

CANDIDATE_OUTPUT_FIELDS = [
    "candidate_id",
    "name",
    "location_city",
    "location_state",
    "location_country",
    "total_experience_years",
    "career_stage",
    "skills_extracted",
    "tools_and_technologies",
    "programming_languages",
    "tech_stack_primary",
    "current_tech_stack",
    "domain_expertise",
    "semantic_summary",
    "top_5_skills_with_years",
]

EVIDENCE_ACTION_TERMS = {
    "built",
    "implemented",
    "designed",
    "led",
    "owned",
    "optimized",
    "deployed",
    "shipped",
    "scaled",
    "migrated",
    "automated",
    "improved",
    "reduced",
}

EVIDENCE_METRIC_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

_ALIAS_LOOKUP: Dict[str, List[str]] = {}
for skill in COMMON_TECH_SKILLS:
    _ALIAS_LOOKUP.setdefault(skill, set()).add(skill)
for alias, canonical in SKILL_ALIASES.items():
    _ALIAS_LOOKUP.setdefault(canonical, set()).add(alias)
_ALIAS_LOOKUP = {k: sorted(v) for k, v in _ALIAS_LOOKUP.items()}


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


def _normalize_skill(skill: str) -> str:
    normalized = skill.strip().lower()
    if not normalized:
        return ""
    return SKILL_ALIASES.get(normalized, normalized)


def _normalize_skill_list(skills: List[str]) -> List[str]:
    normalized = {_normalize_skill(s) for s in skills if isinstance(s, str)}
    return sorted(s for s in normalized if s)


def _extract_text_skills(text: str) -> List[str]:
    text_lower = text.lower()
    found = [skill for skill in COMMON_TECH_SKILLS if skill in text_lower]
    return _normalize_skill_list(found)


def _extract_jd_requirements(jd_text: str) -> Dict[str, Any]:
    data = extract_requirements(jd_text)
    must = _normalize_skill_list(data.get("must_have_skills") or [])
    nice = _normalize_skill_list(data.get("nice_to_have_skills") or [])

    if not must and not nice:
        must = _extract_text_skills(jd_text)

    if nice:
        nice = [s for s in nice if s not in must]

    domain_keywords: List[str] = []
    industry = data.get("industry")
    role_type = data.get("role_type")
    if isinstance(industry, str) and industry.strip():
        domain_keywords.append(industry.strip())
    if isinstance(role_type, str) and role_type.strip():
        domain_keywords.append(role_type.strip())

    return {
        "must_have_skills": must,
        "nice_to_have_skills": nice,
        "domain_keywords": domain_keywords,
        "seniority_level": data.get("seniority_level") or "Any",
        "role_type": role_type,
        "industry": industry,
    }


def _score_evidence(snippet: str) -> float:
    if not snippet:
        return 0.0
    text = snippet.lower()
    word_count = len(text.split())
    length_score = min(1.0, word_count / 25.0)
    action_hits = sum(1 for term in EVIDENCE_ACTION_TERMS if term in text)
    action_score = min(1.0, action_hits / 2.0)
    metric_score = 1.0 if EVIDENCE_METRIC_RE.search(text) else 0.0
    return round(0.5 * length_score + 0.3 * action_score + 0.2 * metric_score, 3)


def _build_skill_terms(skills: List[str]) -> Dict[str, List[str]]:
    terms: Dict[str, List[str]] = {}
    for skill in skills:
        if not skill:
            continue
        aliases = _ALIAS_LOOKUP.get(skill, [skill])
        terms[skill] = aliases
    return terms


def _collect_skill_evidence(
    answers: List[Dict[str, Any]],
    skill_terms: Dict[str, List[str]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    evidence: Dict[str, List[Dict[str, Any]]] = {skill: [] for skill in skill_terms}
    counts: Dict[str, int] = {skill: 0 for skill in skill_terms}

    for answer in answers:
        snippet = (answer.get("answer_snippet") or "").strip()
        if not snippet:
            continue
        snippet_lower = snippet.lower()
        for skill, terms in skill_terms.items():
            if any(term in snippet_lower for term in terms):
                counts[skill] += 1
                evidence[skill].append(
                    {
                        "snippet": snippet,
                        "interview_id": answer.get("interview_id"),
                        "question_index": answer.get("question_index"),
                        "quality_score": _score_evidence(snippet),
                    }
                )

    for skill in evidence:
        evidence[skill].sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        evidence[skill] = evidence[skill][:3]

    return evidence, counts


def _extract_resume_skills(candidate: Dict[str, Any]) -> List[str]:
    fields = [
        candidate.get("skills_extracted"),
        candidate.get("tools_and_technologies"),
        candidate.get("programming_languages"),
        candidate.get("tech_stack_primary"),
        candidate.get("current_tech_stack"),
        candidate.get("top_5_skills_with_years"),
        candidate.get("domain_expertise"),
    ]
    tokens: List[str] = []
    for field in fields:
        if not field:
            continue
        if isinstance(field, list):
            tokens.extend([str(item) for item in field])
        else:
            split_items = re.split(r"[,\n;/|]", str(field))
            tokens.extend(split_items)
    return _normalize_skill_list(tokens)


def _calculate_recency_score(latest_interview_date: Optional[int]) -> float:
    if not latest_interview_date:
        return 0.5
    try:
        now = datetime.now(timezone.utc).timestamp()
        days_old = max(0, (now - float(latest_interview_date)) / 86400)
    except Exception:
        return 0.5
    if days_old <= 30:
        return 1.0
    if days_old <= 90:
        return 0.7
    if days_old <= 180:
        return 0.4
    return 0.2


def _fetch_candidate_profiles(candidate_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not candidate_ids:
        return {}

    profiles: Dict[str, Dict[str, Any]] = {}
    chunk_size = 50

    def _run(client):
        for i in range(0, len(candidate_ids), chunk_size):
            chunk = candidate_ids[i : i + chunk_size]
            safe_ids = [f'"{_safe_filter_value(cid)}"' for cid in chunk]
            expr = f"candidate_id in [{', '.join(safe_ids)}]"
            results = client.query(
                collection_name=COLLECTION,
                filter=expr,
                output_fields=CANDIDATE_OUTPUT_FIELDS,
                limit=len(chunk),
            )
            for row in results:
                cid = row.get("candidate_id")
                if cid:
                    profiles[cid] = row

    _with_milvus_client(_run)
    return profiles


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
    jd_text: Optional[str] = None,
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

    if mode == "jd":
        if not jd_text or not jd_text.strip():
            return {"error": "jd_text is required for mode=jd"}
        if candidate_id or job_id or interview_id:
            filter_expr = _build_filter(candidate_id, interview_id, job_id)
        else:
            filter_expr = 'interview_id != ""'
    else:
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
        summary = _summarize_answers(recs, include_answers=mode in {"candidate", "interview", "jd"})
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

    if mode == "jd":
        jd_requirements = _extract_jd_requirements(jd_text or "")
        must_skills = jd_requirements.get("must_have_skills", [])
        nice_skills = jd_requirements.get("nice_to_have_skills", [])
        all_skills = _normalize_skill_list(must_skills + nice_skills)
        skill_terms = _build_skill_terms(all_skills)

        answers_by_candidate: Dict[str, List[Dict[str, Any]]] = {}
        latest_date_by_candidate: Dict[str, int] = {}
        latest_interview_by_candidate: Dict[str, str] = {}

        for rec in records:
            cid = rec.get("candidate_id") or "unknown"
            interview_date = rec.get("interview_date") or 0
            answers_by_candidate.setdefault(cid, []).append(
                {
                    "answer_snippet": rec.get("answer_snippet"),
                    "interview_id": rec.get("interview_id"),
                    "question_index": rec.get("question_index"),
                    "interview_date": interview_date,
                }
            )
            current_latest = latest_date_by_candidate.get(cid, 0)
            if interview_date and interview_date >= current_latest:
                latest_date_by_candidate[cid] = interview_date
                latest_interview_by_candidate[cid] = rec.get("interview_id") or latest_interview_by_candidate.get(cid, "")

        candidate_profiles = _fetch_candidate_profiles(list(answers_by_candidate.keys()))

        ranked: List[Dict[str, Any]] = []
        for cid, answers in answers_by_candidate.items():
            evidence, counts = _collect_skill_evidence(answers, skill_terms)
            matched_must = [s for s in must_skills if counts.get(s, 0) > 0]
            matched_nice = [s for s in nice_skills if counts.get(s, 0) > 0]
            missing_must = [s for s in must_skills if s not in matched_must]

            coverage = len(matched_must) / max(1, len(must_skills))
            depth = min(1.0, sum(counts.values()) / max(1, len(must_skills) * 2))
            recency = _calculate_recency_score(latest_date_by_candidate.get(cid))
            overall_score = int(round(100 * (coverage * 0.6 + depth * 0.3 + recency * 0.1)))

            candidate_profile = candidate_profiles.get(cid, {})
            resume_skills = _extract_resume_skills(candidate_profile)
            resume_matches = [s for s in must_skills if s in resume_skills]
            resume_only = [s for s in resume_matches if s not in matched_must]

            evidence_filtered = {
                skill: evidence.get(skill, [])
                for skill in matched_must + matched_nice
                if evidence.get(skill)
            }

            ranked.append(
                {
                    "candidate_id": cid,
                    "candidate_profile": candidate_profile,
                    "coverage_ratio": round(coverage, 3),
                    "depth_score": round(depth, 3),
                    "recency_score": round(recency, 3),
                    "overall_score": overall_score,
                    "matched_skills": matched_must,
                    "nice_to_have_matched": matched_nice,
                    "missing_skills": missing_must,
                    "resume_matches": resume_matches,
                    "resume_only_skills": resume_only,
                    "evidence": evidence_filtered,
                    "latest_interview_id": latest_interview_by_candidate.get(cid),
                    "interview_count": len({a.get("interview_id") for a in answers if a.get("interview_id")}),
                }
            )

        ranked.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        return {
            "mode": mode,
            "jd_summary": {
                "must_have_skills": must_skills,
                "nice_to_have_skills": nice_skills,
                "domain_keywords": jd_requirements.get("domain_keywords", []),
                "seniority_level": jd_requirements.get("seniority_level", "Any"),
                "role_type": jd_requirements.get("role_type"),
                "industry": jd_requirements.get("industry"),
            },
            "latest_basis": latest_basis,
            "total_candidates": len(ranked),
            "candidates": ranked,
            "limit": limit,
        }

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
        candidate_profiles = _fetch_candidate_profiles([entry["candidate_id"] for entry in ranked])
        for entry in ranked:
            entry["candidate_profile"] = candidate_profiles.get(entry["candidate_id"], {})
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
