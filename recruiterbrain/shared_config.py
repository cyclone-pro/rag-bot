"""Global configuration, env defaults, and shared clients for recruiter brain."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Final, List, Optional

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI  # lightweight import; instantiation is lazy
except Exception:  # pragma: no cover - OpenAI is optional during local dev
    OpenAI = None  # type: ignore

# ------------------------- Environment + constants ------------------------- #
MILVUS_URI: Optional[str] = os.environ.get("MILVUS_URI")
MILVUS_TOKEN: str = os.environ.get("MILVUS_TOKEN", "")
COLLECTION: str = os.environ.get("MILVUS_COLLECTION", "candidate_pool")
VECTOR_FIELD_DEFAULT: str = os.environ.get("RB_VECTOR_FIELD", "summary_embedding")
TOP_K: int = int(os.environ.get("RB_TOP_K", "1000"))
RETURN_TOP: int = int(os.environ.get("RB_RETURN_TOP", "20"))
PREFILTER_LIMIT: int = int(os.environ.get("RB_PREFILTER_LIMIT", "5000"))
METRIC: str = os.environ.get("RB_VECTOR_METRIC", "COSINE")
EF_SEARCH: int = int(os.environ.get("RB_EF_SEARCH", "128"))
EMBED_MODEL: str = os.environ.get("RB_EMBED_MODEL", os.environ.get("EMBED_MODEL_NAME", "intfloat/e5-base-v2"))
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")

# Fields we need whenever we hydrate a candidate row for display.
FIELDS: Final[List[str]] = [
    "candidate_id",
    "name",
    "career_stage",
    "primary_industry",
    "skills_extracted",
    "tools_and_technologies",
    "domains_of_expertise",
    "semantic_summary",
    "keywords_summary",
]

# ------------------------- Client helpers (lazy) --------------------------- #

@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    if not MILVUS_URI:
        raise RuntimeError("MILVUS_URI not configured; export it before running the app.")
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


__all__ = [
    "COLLECTION",
    "EF_SEARCH",
    "EMBED_MODEL",
    "FIELDS",
    "METRIC",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "PREFILTER_LIMIT",
    "RETURN_TOP",
    "TOP_K",
    "VECTOR_FIELD_DEFAULT",
    "get_encoder",
    "get_milvus_client",
    "get_openai_client",
]
