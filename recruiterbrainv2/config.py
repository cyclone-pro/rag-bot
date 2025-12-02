"""V2 Configuration - Simplified and production-ready."""
import os
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ========================
# Environment
# ========================
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION = os.getenv("MILVUS_COLLECTION", "new_candidate_pool")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ========================
# Search Parameters
# ========================
VECTOR_TOP_K = 100        # ANN candidates
KEYWORD_TOP_K = 50        # Keyword-matched candidates
FINAL_RETURN = 20         # Results to user

# Vector search
METRIC = "COSINE"
EF_SEARCH = 128

# Hybrid weights
VECTOR_WEIGHT = 0.6       # Semantic
KEYWORD_WEIGHT = 0.4      # Exact matching

# ========================
# Career Stages (ordered by seniority)
# ========================
CAREER_STAGES = [
    "Entry",
    "Mid",
    "Senior",
    "Lead/Manager",
    "Director+"
]

# ========================
# Milvus Schema Fields
# ========================
# Based on your screenshots
SEARCH_OUTPUT_FIELDS = [
    "candidate_id",
    "name",
    "email",
    "phone",
    "linkedin_url",
    "location_city",
    "location_state",
    "location_country",
    "total_experience_years",
    "education_level",
    "primary_industry",
    "sub_industries",
    "skills_extracted",
    "tools_and_technologies",
    "certifications",
    "domains_of_expertise",
    "top_titles_mentioned",
    "employment_history",
    "semantic_summary",
    "keywords_summary",
    "career_stage",
]

# ========================
# Lazy Clients
# ========================
@lru_cache(maxsize=1)
def get_milvus_client():
    """Get Milvus client (singleton)."""
    if not MILVUS_URI:
        raise RuntimeError("MILVUS_URI not configured")
    from pymilvus import MilvusClient
    logger.info("Initializing Milvus client for %s", COLLECTION)
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)


@lru_cache(maxsize=1)
def get_encoder():
    """Get embedding model (singleton)."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_openai_client() -> Optional[object]:
    """Get OpenAI client if available."""
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set")
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        logger.warning("openai package not installed")
        return None