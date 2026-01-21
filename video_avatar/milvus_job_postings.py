import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer


MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_JOB_COLLECTION", "job_postings")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

FIELD_LIMITS = {
    "job_id": 64,
    "title": 256,
    "company": 256,
    "department": 128,
    "location": 256,
    "employment_type": 32,
    "tax_term": 16,
    "salary_range": 128,
    "req_id": 64,
    "status": 64,
    "jd_text": 65535,
}

_EMBEDDER: Optional[SentenceTransformer] = None
_COLLECTION: Optional[Collection] = None


def _truncate(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit]


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value)


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    return _EMBEDDER


def _get_collection() -> Collection:
    global _COLLECTION
    if _COLLECTION is None:
        connections.connect(alias="job_postings", host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(MILVUS_COLLECTION, using="job_postings"):
            raise RuntimeError(f"Milvus collection not found: {MILVUS_COLLECTION}")
        _COLLECTION = Collection(MILVUS_COLLECTION, using="job_postings")
    return _COLLECTION


def _build_location(role: Dict[str, Any]) -> Optional[str]:
    cities = role.get("location_cities") or []
    states = role.get("location_states") or []
    if isinstance(cities, list) and isinstance(states, list) and cities and states:
        pairs = []
        for city, state in zip(cities, states):
            city_str = _coerce_str(city)
            state_str = _coerce_str(state)
            if city_str and state_str:
                pairs.append(f"{city_str}, {state_str}")
        if pairs:
            return " | ".join(pairs)
    work_model = _coerce_str(role.get("work_model"))
    if work_model == "remote":
        return "Remote"
    return _coerce_str(role.get("location_country"))


def _build_salary_range(role: Dict[str, Any]) -> Optional[str]:
    salary_min = role.get("salary_min")
    salary_max = role.get("salary_max")
    salary_currency = _coerce_str(role.get("salary_currency"))
    pay_min = role.get("pay_rate_min")
    pay_max = role.get("pay_rate_max")
    pay_currency = _coerce_str(role.get("pay_rate_currency"))
    pay_unit = _coerce_str(role.get("pay_rate_unit"))

    if salary_min is not None or salary_max is not None:
        low = salary_min if salary_min is not None else salary_max
        high = salary_max if salary_max is not None else salary_min
        if low == high:
            base = f"{low}"
        else:
            base = f"{low}-{high}"
        return f"{base} {salary_currency}".strip()

    if pay_min is not None or pay_max is not None:
        low = pay_min if pay_min is not None else pay_max
        high = pay_max if pay_max is not None else pay_min
        if low == high:
            base = f"{low}"
        else:
            base = f"{low}-{high}"
        suffix = ""
        if pay_currency:
            suffix += f" {pay_currency}"
        if pay_unit and pay_unit != "unspecified":
            suffix += f"/{pay_unit}"
        return f"{base}{suffix}".strip()

    return None


def _build_jd_text(role: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = _coerce_str(role.get("job_title"))
    if title:
        parts.append(f"Title: {title}")
    seniority = _coerce_str(role.get("seniority_level"))
    if seniority and seniority != "unspecified":
        parts.append(f"Seniority: {seniority}")
    work_model = _coerce_str(role.get("work_model"))
    if work_model and work_model != "unspecified":
        parts.append(f"Work model: {work_model}")
    location = _build_location(role)
    if location:
        parts.append(f"Location: {location}")

    must = role.get("must_have_skills") or []
    if must:
        parts.append("Must-have skills: " + ", ".join([str(x) for x in must]))
    nice = role.get("nice_to_have_skills") or []
    if nice:
        parts.append("Nice-to-have skills: " + ", ".join([str(x) for x in nice]))
    primary = role.get("primary_technologies") or []
    if primary:
        parts.append("Primary technologies: " + ", ".join([str(x) for x in primary]))

    responsibilities = role.get("responsibilities") or []
    if responsibilities:
        parts.append("Responsibilities: " + " | ".join([str(x) for x in responsibilities]))
    day_to_day = role.get("day_to_day") or []
    if day_to_day:
        parts.append("Day-to-day: " + " | ".join([str(x) for x in day_to_day]))
    other = role.get("other_constraints") or []
    if other:
        parts.append("Constraints: " + " | ".join([str(x) for x in other]))

    salary_range = _build_salary_range(role)
    if salary_range:
        parts.append(f"Compensation: {salary_range}")

    jd_text = "\n".join(parts).strip()
    if not jd_text:
        jd_text = title or "Job posting"
    return jd_text


def _prepare_posting(role: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    job_id = _truncate(_coerce_str(role.get("job_id")), FIELD_LIMITS["job_id"])
    title = _truncate(_coerce_str(role.get("job_title")), FIELD_LIMITS["title"])
    company = _coerce_str(role.get("end_client_name")) or _coerce_str(role.get("client_name")) or _coerce_str(role.get("vendor_name"))
    department = _coerce_str(role.get("industry"))
    location = _build_location(role)
    employment_type = _coerce_str(role.get("job_type")) or _coerce_str(role.get("employment_type"))
    tax_term = _coerce_str(role.get("employment_type"))
    salary_range = _build_salary_range(role)
    req_id = _coerce_str(role.get("external_requisition_id")) or job_id
    status = _coerce_str(role.get("status")) or "active"
    posted_ts = int(time.time())
    jd_text = _build_jd_text(role)

    posting = {
        "job_id": job_id,
        "title": title,
        "company": _truncate(company, FIELD_LIMITS["company"]),
        "department": _truncate(department, FIELD_LIMITS["department"]),
        "location": _truncate(location, FIELD_LIMITS["location"]),
        "employment_type": _truncate(employment_type, FIELD_LIMITS["employment_type"]),
        "tax_term": _truncate(tax_term, FIELD_LIMITS["tax_term"]),
        "salary_range": _truncate(salary_range, FIELD_LIMITS["salary_range"]),
        "req_id": _truncate(req_id, FIELD_LIMITS["req_id"]),
        "status": _truncate(status, FIELD_LIMITS["status"]),
        "posted_ts": posted_ts,
        "jd_text": _truncate(jd_text, FIELD_LIMITS["jd_text"]),
    }

    return posting, posting["jd_text"] or ""


def insert_job_postings(roles: Iterable[Dict[str, Any]]) -> int:
    if not MILVUS_HOST:
        raise RuntimeError("MILVUS_HOST is not configured")

    role_list = [r for r in roles if isinstance(r, dict)]
    if not role_list:
        return 0

    postings: List[Dict[str, Any]] = []
    texts: List[str] = []
    for role in role_list:
        posting, jd_text = _prepare_posting(role)
        if not posting.get("job_id"):
            continue
        postings.append(posting)
        texts.append(jd_text or "")

    if not postings:
        return 0

    embedder = _get_embedder()
    inputs = [f"passage: {text}" for text in texts]
    vectors = embedder.encode(inputs, normalize_embeddings=True)

    if hasattr(vectors, "tolist"):
        vectors_list = vectors.tolist()
    else:
        vectors_list = list(vectors)

    for posting, vector in zip(postings, vectors_list):
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension mismatch: {len(vector)} != {EMBEDDING_DIM}")
        posting["jd_embedding"] = vector

    collection = _get_collection()
    collection.insert(postings)
    return len(postings)


def check_milvus_connection() -> Tuple[bool, str]:
    if not MILVUS_HOST:
        return False, "MILVUS_HOST not configured"
    try:
        connections.connect(alias="job_postings_health", host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(MILVUS_COLLECTION, using="job_postings_health"):
            return False, f"collection not found: {MILVUS_COLLECTION}"
    except Exception as exc:
        return False, str(exc)
    return True, "ok"
