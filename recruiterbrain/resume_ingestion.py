from __future__ import annotations
import io
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Tuple
from fastapi import UploadFile
from pymilvus import MilvusException
import re
import hashlib
import base64
from recruiterbrain.shared_config import (
    COLLECTION,
    EMBED_MODEL,
    get_encoder,
    get_milvus_client,
    get_openai_client,
)
from recruiterbrain.resume_schema import (
    RESUME_JSON_SCHEMA,
    RESUME_EXTRACTION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

try:
    import docx  
    _DOCX_AVAILABLE = True
except ImportError:
    docx = None
    _DOCX_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "python-docx not installed; .docx extraction disabled."
    )

def short_id_from(text: str) -> str:
    """
    Generate a stable 6-character ID based on input text.
    Example output: 'A5F2C9'.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    code = base64.urlsafe_b64encode(digest)[0:6].decode("utf-8")
    return re.sub(r"[^A-Za-z0-9]", "", code).upper()


def _ensure_candidate_id(candidate: Dict[str, Any]) -> None:
    """Create 6-char deterministic ID based on name + email."""
    name  = (candidate.get("name")  or "UNKNOWN").strip().upper()
    email = (candidate.get("email") or "UNKNOWN").strip().upper()

    raw = f"{name}|{email}"

    candidate["candidate_id"] = short_id_from(raw)

# ---------- 1. File → raw text ---------- #

def extract_text_from_file_bytes(filename: str, data: bytes) -> str:
    """
    Simple, dependency-light extraction.
    You can swap in pypdf / python-docx if you prefer richer extraction.
    """
    name_lower = filename.lower()

    if name_lower.endswith(".txt"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")

    if name_lower.endswith(".pdf"):
        # Minimal PDF extraction using pypdf (if installed)
        try:
            import pypdf  # type: ignore
        except ImportError:
            logger.warning("pypdf not installed; treating PDF as binary bytes.")
            return ""
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if name_lower.endswith(".docx"):
        try:
            import docx  # type: ignore
        except ImportError:
            logger.warning("python-docx not installed; .docx extraction disabled.")
            return ""
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    # Fallback: try decoding as text
    return data.decode("utf-8", errors="ignore")


# ---------- 2. PII extraction (cheap regex) ---------- #


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{6,}\d)")

def extract_pii(text: str) -> Tuple[str, str]:
    """Return (email, phone) best guesses."""
    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)
    email = email_match.group(0) if email_match else "Unknown"
    phone = phone_match.group(0) if phone_match else "Unknown"
    return email, phone


# ---------- 3. LLM call to produce JSON ---------- #

def _call_llm_structured_resume(resume_text: str) -> Dict[str, Any]:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not configured; cannot run resume extraction.")

    # Use Chat Completions with JSON mode instead of `responses.create`
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": RESUME_EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": resume_text,
            },
        ],
    )

    # Depending on client version, `message.content` might be a string or a list of parts.
    content = resp.choices[0].message.content

    if isinstance(content, str):
        raw_json = content
    else:
        # Newer SDKs: content is a list of {"type": "text", "text": "..."} parts
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        raw_json = "".join(parts)

    try:
        return json.loads(raw_json)
    except Exception as e:
        logger.error("Could not parse JSON from LLM output: %r", raw_json[:500])
        raise RuntimeError(f"Failed to parse LLM resume JSON: {e}") from e


# ---------- 4. Build Milvus row & embeddings ---------- #




def _apply_pii_overrides(candidate: Dict[str, Any], email: str, phone: str) -> None:
    # Prefer regex-extracted email/phone; they’re often more reliable than LLM guesses
    if email != "Unknown":
        candidate["email"] = email
    if phone != "Unknown":
        candidate["phone"] = phone

def normalize_candidate_for_milvus(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize LLM JSON into shapes that match the Milvus new_candidate_pool schema.
    - Some fields are VARCHAR in Milvus but lists in JSON → join as a single string.
    - Score fields → float.
    - clouds → list of strings.
    """

    # 1) Fields that are VARCHAR in Milvus but conceptually lists in the JSON
    list_to_str_fields = {
        "degrees",
        "institutions",
        "languages_spoken",
        "primary_industry",
        "sub_industries",
        "skills_extracted",
        "tools_and_technologies",
        "certifications",
        "top_titles_mentioned",
        "domains_of_expertise",
        "employment_history",
        "semantic_summary",
        "keywords_summary",
        "evidence_skills",
        "evidence_domains",
        "evidence_certifications",
        "evidence_tools",
    }

    for field in list_to_str_fields:
        val = candidate.get(field)
        if isinstance(val, list):
            # join with comma or semicolon – whatever matches your synthetic CSV style
            candidate[field] = "; ".join(str(x) for x in val if x is not None)
        elif val is None:
            candidate[field] = ""

    # 2) Score fields – everything ending in "_score" should be float
    for key, val in list(candidate.items()):
        if key.endswith("_score"):
            try:
                candidate[key] = float(val)
            except (TypeError, ValueError):
                candidate[key] = 0.0

    # 3) total_experience_years also a float
    if "total_experience_years" in candidate:
        try:
            candidate["total_experience_years"] = float(
                candidate.get("total_experience_years") or 0.0
            )
        except (TypeError, ValueError):
            candidate["total_experience_years"] = 0.0

    # 4) clouds is an Array<VarChar> in Milvus → must be list[str]
    clouds = candidate.get("clouds")
    if isinstance(clouds, str):
        # Allow "AWS, GCP" style strings
        clouds_list = [c.strip() for c in clouds.split(",") if c.strip()]
        candidate["clouds"] = clouds_list
    elif clouds is None:
        candidate["clouds"] = []
    # if it's already a list, we keep it as-is

    return candidate

# --- old ---
# def _add_embeddings(candidate: Dict[str, Any]) -> None:

def _add_embeddings(candidate: Dict[str, Any], full_resume_text: str) -> None:
    encoder = get_encoder()

    # 1. Use the entire resume text for the main embedding.
    #    If for some reason it's empty, fall back to semantic_summary or keywords_summary.
    text_for_resume_embedding = (
        (full_resume_text or "").strip()
        or candidate.get("semantic_summary", "")
        or candidate.get("keywords_summary", "")
        or ""
    )

    # (Optional) clip extremely long text to keep encoder happy
    if len(text_for_resume_embedding) > 20000:
        text_for_resume_embedding = text_for_resume_embedding[:20000]

    # summary_embedding = representation of whole resume
    summary_vec = encoder.encode(text_for_resume_embedding)

    # 2. Skills embedding – focused on skills/keywords signals.
    skills_list = candidate.get("skills_extracted") or []
    skills_text = ", ".join(skills_list) or candidate.get("keywords_summary", "")
    if len(skills_text) > 20000:
        skills_text = skills_text[:20000]
    skills_vec = encoder.encode(skills_text or text_for_resume_embedding)

    candidate["summary_embedding"] = list(map(float, summary_vec))
    candidate["skills_embedding"] = list(map(float, skills_vec))
    candidate["resume_embedding_version"] = EMBED_MODEL

    from datetime import datetime
    candidate["last_updated"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"


def build_milvus_row(candidate: Dict[str, Any], full_resume_text: str) -> Dict[str, Any]:
    _ensure_candidate_id(candidate)
    _add_embeddings(candidate, full_resume_text=full_resume_text)
    return candidate


def insert_candidate_into_milvus(row: Dict[str, Any]) -> str:
    client = get_milvus_client()
    try:
        result = client.insert(
            collection_name=COLLECTION,
            data=[row],   # single row insert
        )
        # result["insert_count"], result["ids"] etc – depends on pymilvus version
        logger.info("Inserted candidate into %s (candidate_id=%s)", COLLECTION, row["candidate_id"])
        return row["candidate_id"]
    except MilvusException as exc:
        logger.exception("Milvus insert failed")
        raise RuntimeError(f"Milvus insert failed: {exc}") from exc


# ---------- 5. High-level entrypoint used by FastAPI ---------- #

async def ingest_resume_upload(file: UploadFile, source_channel: str = "Upload") -> Dict[str, Any]:
    data = await file.read()
    text = extract_text_from_file_bytes(file.filename, data)

    if not text.strip():
        raise RuntimeError("Could not extract text from resume file.")

    email, phone = extract_pii(text)
    logger.info("Extracted PII: email=%s phone=%s", email, phone)

    candidate = _call_llm_structured_resume(text)
    candidate = normalize_candidate_for_milvus(candidate)

    candidate.setdefault("source_channel", source_channel)
    candidate.setdefault("hiring_manager_notes", "Unknown")
    candidate.setdefault("interview_feedback", "Unknown")
    candidate.setdefault("offer_status", "Unknown")
    candidate.setdefault("assigned_recruiter", "Unknown")

    _apply_pii_overrides(candidate, email, phone)

    logger.info(
        "Normalized candidate for insert: candidate_id=%s name=%s email=%s",
        candidate.get("candidate_id"),
        candidate.get("name"),
        candidate.get("email"),
    )

   


    # ⬇️ pass the full resume text so we embed the whole thing
    row = build_milvus_row(candidate, full_resume_text=text)
    candidate_id = insert_candidate_into_milvus(row)

    return {
        "candidate_id": candidate_id,
        "email": row.get("email"),
        "name": row.get("name"),
        "status": "inserted",
    }

