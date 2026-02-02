"""Configuration and environment variables for RCRUTR Interviews."""

import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env next to this file (if present).
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    except Exception:
        # Keep runtime resilient if python-dotenv isn't installed.
        pass

# =============================================================================
# ZOOM API
# =============================================================================
ZOOM_ACCOUNT_ID: str = os.getenv("ZOOM_ACCOUNT_ID", "")
ZOOM_CLIENT_ID: str = os.getenv("ZOOM_CLIENT_ID", "")
ZOOM_CLIENT_SECRET: str = os.getenv("ZOOM_CLIENT_SECRET", "")
ZOOM_API_BASE: str = "https://api.zoom.us/v2"
ZOOM_OAUTH_URL: str = "https://zoom.us/oauth/token"

# =============================================================================
# BEY API
# =============================================================================
BEY_API_KEY: str = os.getenv("BEY_API_KEY", "")
BEY_API_URL: str = os.getenv("BEY_API_URL", "https://api.bey.dev/v1")

# Avatar IDs (same as video_avatar)
AVATARS = {
    "scott": {
        "id": "b63ba4e6-d346-45d0-ad28-5ddffaac0bd0_v2",
        "name": "Scott",
        "voice": "professional_male",
    },
    "sam": {
        "id": "1c7a7291-ee28-4800-8f34-acfbfc2d07c0",
        "name": "Sam",
        "voice": "professional_male",
    },
    "zara": {
        "id": "694c83e2-8895-4a98-bd16-56332ca3f449",
        "name": "Zara",
        "voice": "professional_female",
    },
}
DEFAULT_AVATAR = "zara"

# =============================================================================
# DATABASE (PostgreSQL)
# =============================================================================
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# =============================================================================
# MILVUS
# =============================================================================
MILVUS_URI: str = os.getenv("MILVUS_URI", "")
MILVUS_HOST: str = os.getenv("MILVUS_HOST", "")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
MILVUS_TOKEN: Optional[str] = os.getenv("MILVUS_TOKEN")

# Collections
MILVUS_CANDIDATES_COLLECTION: str = os.getenv("MILVUS_CANDIDATES_COLLECTION", "candidates_v3")
MILVUS_JOBS_COLLECTION: str = os.getenv("MILVUS_JOBS_COLLECTION", "job_postings")
MILVUS_QA_COLLECTION: str = os.getenv("MILVUS_QA_COLLECTION", "qa_embeddings_v2")

# =============================================================================
# OPENAI
# =============================================================================
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# =============================================================================
# HUGGINGFACE (for embeddings)
# =============================================================================
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

# =============================================================================
# INTERVIEW SETTINGS
# =============================================================================
INTERVIEW_TOTAL_QUESTIONS: int = 8
INTERVIEW_MAX_FOLLOWUPS: int = 2
INTERVIEW_CANDIDATE_TIMEOUT_MINUTES: int = 5
INTERVIEW_AVATAR_JOIN_RETRY: int = 2

# Name matching threshold (0.0 - 1.0, higher = stricter)
NAME_MATCH_THRESHOLD: float = float(os.getenv("NAME_MATCH_THRESHOLD", "0.7"))

# =============================================================================
# WEBHOOK
# =============================================================================
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")
ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "")

# =============================================================================
# GCS (for future recording storage)
# =============================================================================
GCS_BUCKET_RECORDINGS: str = os.getenv("GCS_BUCKET_RECORDINGS", "rcrutr-interview-recordings")

# =============================================================================
# SERVICE
# =============================================================================
SERVICE_NAME: str = "rcrutr-interviews"
SERVICE_VERSION: str = "0.1.0"
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
