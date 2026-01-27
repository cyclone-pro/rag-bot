import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pymilvus import Collection, DataType, connections, utility
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).with_name(".env"))

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION") or os.getenv("MILVUS_JOB_COLLECTION", "job_postings")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Similarity threshold for detecting similar jobs
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.93"))
MAX_SIMILAR_JOBS = int(os.getenv("MAX_SIMILAR_JOBS", "3"))

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
_COMPANY_ARRAY: Optional[bool] = None

logger = logging.getLogger("bey_milvus")


def _log_event(level: str, message: str, **fields: Any) -> None:
    payload = {"message": message, **fields}
    record = json.dumps(payload, ensure_ascii=True)
    if level == "warning":
        logger.warning(record)
    elif level == "error":
        logger.error(record)
    else:
        logger.info(record)


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


def _truncate_or_default(value: Optional[str], limit: int, default: str = "") -> str:
    if value is None or value == "":
        return default
    truncated = _truncate(value, limit)
    return truncated if truncated is not None else default


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _log_event("info", "loading_embedding_model", model=EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    return _EMBEDDER


def _connect_milvus(alias: str, timeout: Optional[int] = None) -> None:
    kwargs: Dict[str, Any] = {"alias": alias}
    _log_event(
        "info",
        "milvus_connect_start",
        alias=alias,
        using_uri=bool(MILVUS_URI),
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        timeout=timeout,
    )
    if MILVUS_URI:
        kwargs["uri"] = MILVUS_URI
        if MILVUS_TOKEN:
            kwargs["token"] = MILVUS_TOKEN
    else:
        kwargs["host"] = MILVUS_HOST
        kwargs["port"] = MILVUS_PORT
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        connections.connect(**kwargs)
        _log_event("info", "milvus_connect_ok", alias=alias)
    except TypeError:
        kwargs.pop("timeout", None)
        connections.connect(**kwargs)
        _log_event("info", "milvus_connect_ok", alias=alias, retried_without_timeout=True)


def _get_collection() -> Collection:
    global _COLLECTION
    if _COLLECTION is None:
        _log_event("info", "milvus_collection_load_start", collection=MILVUS_COLLECTION)
        _connect_milvus("job_postings")
        if not utility.has_collection(MILVUS_COLLECTION, using="job_postings"):
            raise RuntimeError(f"Milvus collection not found: {MILVUS_COLLECTION}")
        _COLLECTION = Collection(MILVUS_COLLECTION, using="job_postings")
        _COLLECTION.load()
        _log_event("info", "milvus_collection_load_ok", collection=MILVUS_COLLECTION)
    return _COLLECTION


def _company_is_array(collection: Collection) -> bool:
    global _COMPANY_ARRAY
    if _COMPANY_ARRAY is not None:
        return _COMPANY_ARRAY
    for field in collection.schema.fields:
        if field.name == "company":
            _COMPANY_ARRAY = field.dtype == DataType.ARRAY
            return _COMPANY_ARRAY
    _COMPANY_ARRAY = False
    return _COMPANY_ARRAY


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
        parts.append("Required skills: " + ", ".join([str(x) for x in must]))
    nice = role.get("nice_to_have_skills") or []
    if nice:
        parts.append("Nice-to-have skills: " + ", ".join([str(x) for x in nice]))
    primary = role.get("primary_technologies") or []
    if primary:
        parts.append("Primary technologies: " + ", ".join([str(x) for x in primary]))

    cert_req = role.get("certifications_required") or []
    if cert_req:
        parts.append("Certifications required: " + ", ".join([str(x) for x in cert_req]))
    cert_pref = role.get("certifications_preferred") or []
    if cert_pref:
        parts.append("Certifications preferred: " + ", ".join([str(x) for x in cert_pref]))
    domains = role.get("domains") or []
    if domains:
        parts.append("Domains: " + ", ".join([str(x) for x in domains]))

    overall_years = role.get("overall_min_years")
    if overall_years is not None:
        parts.append(f"Minimum experience: {overall_years} years")
    primary_years = role.get("primary_role_min_years")
    if primary_years is not None:
        parts.append(f"Primary role experience: {primary_years} years")

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
    req_id = _coerce_str(role.get("external_requisition_id")) or ""
    status = _coerce_str(role.get("status")) or "active"
    posted_ts = int(time.time())
    jd_text = _build_jd_text(role)

    posting = {
        "job_id": job_id,
        "title": _truncate_or_default(title, FIELD_LIMITS["title"]),
        "company": _truncate_or_default(company, FIELD_LIMITS["company"], default="Unknown"),
        "department": _truncate_or_default(department, FIELD_LIMITS["department"]),
        "location": _truncate_or_default(location, FIELD_LIMITS["location"]),
        "employment_type": _truncate_or_default(employment_type, FIELD_LIMITS["employment_type"]),
        "tax_term": _truncate_or_default(tax_term, FIELD_LIMITS["tax_term"]),
        "salary_range": _truncate_or_default(salary_range, FIELD_LIMITS["salary_range"]),
        "req_id": _truncate_or_default(req_id, FIELD_LIMITS["req_id"]),
        "status": _truncate_or_default(status, FIELD_LIMITS["status"], default="active"),
        "posted_ts": posted_ts,
        "jd_text": _truncate_or_default(jd_text, FIELD_LIMITS["jd_text"], default="Job posting"),
    }

    return posting, posting["jd_text"] or ""


# ---------------------------
# Public API: Embedding Generation
# ---------------------------

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text using e5-base-v2.
    
    The text is prefixed with 'passage: ' as required by e5 models.
    Returns a normalized embedding vector.
    """
    _log_event("info", "embedding_generate_start", text_len=len(text))
    embedder = _get_embedder()
    input_text = f"passage: {text}"
    vector = embedder.encode(input_text, normalize_embeddings=True)
    
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts.
    
    Each text is prefixed with 'passage: ' as required by e5 models.
    Returns a list of normalized embedding vectors.
    """
    if not texts:
        return []
    
    _log_event("info", "embedding_generate_batch_start", count=len(texts))
    embedder = _get_embedder()
    inputs = [f"passage: {text}" for text in texts]
    vectors = embedder.encode(inputs, normalize_embeddings=True)
    
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return [list(v) for v in vectors]


# ---------------------------
# Public API: Similarity Search
# ---------------------------

def search_similar_jobs(
    embedding: List[float],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = MAX_SIMILAR_JOBS,
) -> List[Dict[str, Any]]:
    """Search Milvus for jobs similar to the given embedding.
    
    Args:
        embedding: The embedding vector to search with
        threshold: Minimum similarity score (0-1, default 0.93)
        top_k: Maximum number of results to return (default 3)
    
    Returns:
        List of dicts with keys: job_id, score
        Sorted by score descending, only includes results above threshold.
    """
    _log_event("info", "milvus_similarity_search_start", top_k=top_k, threshold=threshold)
    if not MILVUS_URI and not MILVUS_HOST:
        _log_event("warning", "milvus_not_configured_for_similarity_search")
        return []

    if len(embedding) != EMBEDDING_DIM:
        _log_event("error", "embedding_dimension_mismatch", expected=EMBEDDING_DIM, got=len(embedding))
        return []

    try:
        collection = _get_collection()
        
        # Search with top_k * 2 to have buffer for filtering
        search_params = {
            "metric_type": "IP",  # Inner product for normalized vectors = cosine similarity
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=[embedding],
            anns_field="jd_embedding",
            param=search_params,
            limit=top_k * 2,
            output_fields=["job_id"],
        )
        
        similar_jobs: List[Dict[str, Any]] = []
        
        if results and len(results) > 0:
            for hit in results[0]:
                score = float(hit.score)
                if score >= threshold:
                    similar_jobs.append({
                        "job_id": hit.entity.get("job_id"),
                        "score": round(score, 4),
                    })
        
        # Sort by score descending and limit to top_k
        similar_jobs.sort(key=lambda x: x["score"], reverse=True)
        similar_jobs = similar_jobs[:top_k]
        
        _log_event(
            "info",
            "milvus_similarity_search_complete",
            threshold=threshold,
            results_found=len(similar_jobs),
        )
        
        return similar_jobs
        
    except Exception as exc:
        _log_event("error", "milvus_similarity_search_failed", error=str(exc))
        return []


# ---------------------------
# Public API: Insert Job Postings
# ---------------------------

def insert_job_posting(
    role: Dict[str, Any],
    *,
    embedding: Optional[List[float]] = None,
) -> bool:
    """Insert a single job posting to Milvus.
    
    Args:
        role: The job role data (must include job_id)
        embedding: Pre-computed embedding. If None, will be generated.
    
    Returns:
        True if inserted successfully, False otherwise.
    """
    if not MILVUS_URI and not MILVUS_HOST:
        raise RuntimeError("MILVUS_URI or MILVUS_HOST is not configured")

    if not isinstance(role, dict):
        return False

    posting, jd_text = _prepare_posting(role)
    
    if not posting.get("job_id"):
        _log_event("warning", "milvus_insert_skipped_no_job_id")
        return False

    # Use pre-computed embedding or generate new one
    if embedding is None:
        embedding = generate_embedding(jd_text)
    
    if len(embedding) != EMBEDDING_DIM:
        _log_event("error", "embedding_dimension_mismatch", expected=EMBEDDING_DIM, got=len(embedding))
        return False

    posting["jd_embedding"] = embedding

    try:
        collection = _get_collection()
        
        if _company_is_array(collection):
            company_value = posting.get("company") or "Unknown"
            posting["company"] = [company_value]
        
        collection.insert([posting])
        
        _log_event("info", "milvus_insert_single_ok", job_id=posting.get("job_id"))
        return True
        
    except Exception as exc:
        _log_event("error", "milvus_insert_single_failed", job_id=posting.get("job_id"), error=str(exc))
        return False


def insert_job_postings(
    roles: List[Dict[str, Any]],
    *,
    embeddings: Optional[List[List[float]]] = None,
) -> int:
    """Insert multiple job postings to Milvus.
    
    Args:
        roles: List of job role data (each must include job_id)
        embeddings: Pre-computed embeddings matching roles order. If None, will be generated.
    
    Returns:
        Number of successfully inserted postings.
    """
    if not MILVUS_URI and not MILVUS_HOST:
        raise RuntimeError("MILVUS_URI or MILVUS_HOST is not configured")

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

    sample_jobs = [
        {"job_id": posting.get("job_id"), "title": posting.get("title")}
        for posting in postings[:10]
    ]
    _log_event(
        "info",
        "milvus_insert_start",
        count=len(postings),
        sample_jobs=sample_jobs,
        sample_truncated=len(postings) > 10,
    )

    # Use pre-computed embeddings or generate new ones
    if embeddings is not None and len(embeddings) == len(postings):
        vectors_list = embeddings
    else:
        vectors_list = generate_embeddings(texts)

    for posting, vector in zip(postings, vectors_list):
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension mismatch: {len(vector)} != {EMBEDDING_DIM}")
        posting["jd_embedding"] = vector

    try:
        collection = _get_collection()
        
        if _company_is_array(collection):
            for posting in postings:
                company_value = posting.get("company") or "Unknown"
                posting["company"] = [company_value]
        
        collection.insert(postings)
        _log_event("info", "milvus_insert_ok", count=len(postings))
        return len(postings)
        
    except Exception as exc:
        _log_event("error", "milvus_insert_failed", error=str(exc))
        raise

def sync_jobs_to_milvus(jobs: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Sync multiple jobs to Milvus. Returns (success_count, fail_count)."""
    success, fail = 0, 0
    _log_event("info", "milvus_sync_start", total=len(jobs))
    for job in jobs:
        try:
            if insert_job_posting(job):
                success += 1
            else:
                fail += 1
        except Exception as exc:
            _log_event("error", "milvus_sync_item_failed", job_id=job.get("job_id"), error=str(exc))
            fail += 1
    _log_event("info", "milvus_sync_complete", success=success, fail=fail)
    return success, fail
# ---------------------------
# Public API: Health Check
# ---------------------------

def check_milvus_connection(timeout: int = 5) -> Tuple[bool, str]:
    if not MILVUS_URI and not MILVUS_HOST:
        _log_event("warning", "milvus_health_not_configured")
        return False, "MILVUS_URI or MILVUS_HOST not configured"
    try:
        _log_event("info", "milvus_health_check_start", timeout=timeout)
        _connect_milvus("job_postings_health", timeout=timeout)
        if not utility.has_collection(MILVUS_COLLECTION, using="job_postings_health"):
            _log_event("warning", "milvus_health_collection_missing", collection=MILVUS_COLLECTION)
            return False, f"collection not found: {MILVUS_COLLECTION}"
    except Exception as exc:
        _log_event("error", "milvus_health_check_failed", error=str(exc))
        return False, str(exc)
    _log_event("info", "milvus_health_check_ok")
    return True, "ok"
