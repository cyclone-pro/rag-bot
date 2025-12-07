"""V2 Configuration - Optimized for candidates_v3."""
import os
from pathlib import Path
import logging
from functools import lru_cache
from typing import Optional
from redis import Redis
from concurrent.futures import ThreadPoolExecutor
from .rate_limiter import RateLimitExceeded
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ========================
# LOAD LOCAL .ENV FILE
# ========================
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info(f"✅ Loaded .env from: {env_path}")
else:
    logger.warning(f"⚠️  .env not found at: {env_path}, using root .env")
    load_dotenv()

# ========================
# Environment
# ========================
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION = os.getenv("MILVUS_COLLECTION", "candidates_v3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ========================
# Cache Configuration 
# ========================
REDIS_URL = os.getenv("REDIS_URL", "")  # e.g., redis://localhost:6379/0
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))      # 1 hour
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "300"))  # 5 minutes
# ========================
# Connection Pool Config
# ========================
ENABLE_CONNECTION_POOL = os.getenv("ENABLE_CONNECTION_POOL", "False").lower() == "false"
MILVUS_POOL_SIZE = int(os.getenv("MILVUS_POOL_SIZE", "10"))
MILVUS_POOL_MAX_AGE = int(os.getenv("MILVUS_POOL_MAX_AGE", "3600"))  # 1 hour
MILVUS_POOL_MAX_USES = int(os.getenv("MILVUS_POOL_MAX_USES", "1000"))
MILVUS_POOL_HEALTH_CHECK_INTERVAL = int(os.getenv("MILVUS_POOL_HEALTH_CHECK_INTERVAL", "60"))  # 1 minute
# ========================
# Search Parameters
# ========================
VECTOR_TOP_K = 100        # ANN candidates
KEYWORD_TOP_K = 50        # Keyword-matched candidates
FINAL_RETURN = 20         # Results to user
_milvus_pool = None
USE_CELERY = os.getenv("USE_CELERY", "false").lower() == "true"

# Vector search
METRIC = "COSINE"
EF_SEARCH = 128

# Hybrid weights
VECTOR_WEIGHT = 0.6       # Semantic
KEYWORD_WEIGHT = 0.4      # Exact matching
# Parallel Search Config
# ========================
ENABLE_PARALLEL_SEARCH = os.getenv("ENABLE_PARALLEL_SEARCH", "true").lower() == "true"
SEARCH_THREAD_POOL_SIZE = int(os.getenv("SEARCH_THREAD_POOL_SIZE", "4"))

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
# Milvus Schema Fields for candidates_v3
# ========================
SEARCH_OUTPUT_FIELDS = [
    # Identity
    "candidate_id",
    "name",
    "email",
    "phone",
    "linkedin_url",
    "portfolio_url",
    "github_url",
    
    # Location
    "location_city",
    "location_state",
    "location_country",
    "relocation_willingness",
    "remote_preference",
    "availability_status",
    
    # Experience & Education
    "total_experience_years",
    "education_level",
    "degrees",
    "institutions",
    "languages_spoken",
    "management_experience_years",
    
    # Career & Role
    "career_stage",
    "years_in_current_role",
    "top_3_titles",
    "role_type",
    
    # Industry & Domain
    "industries_worked",
    "domain_expertise",
    "verticals_experience",
    
    # Technical Skills
    "skills_extracted",
    "tools_and_technologies",
    "certifications",
    "tech_stack_primary",
    "programming_languages",
    
    # Evidence & Summaries
    "employment_history",
    "semantic_summary",
    "keywords_summary",
    "evidence_skills",
    "evidence_projects",
    "evidence_leadership",
    
    # Recency & Depth
    "current_tech_stack",
    "years_since_last_update",
    "top_5_skills_with_years",
    
    # Metadata
    "source_channel",
    "last_updated",
]
# ========================
# Thread Pool for Parallel Search
# ========================
@lru_cache(maxsize=1)
def get_search_thread_pool() -> ThreadPoolExecutor:
    """Get thread pool for parallel search operations."""
    pool = ThreadPoolExecutor(
        max_workers=SEARCH_THREAD_POOL_SIZE,
        thread_name_prefix="search_worker"
    )
    logger.info(f"✅ Search thread pool initialized with {SEARCH_THREAD_POOL_SIZE} workers")
    return pool

# Milvus pool
@lru_cache(maxsize=1)
def get_milvus_pool():
    """
    Get Milvus connection pool (singleton).
    
    Returns pool for high-concurrency scenarios.
    """
    global _milvus_pool
    
    if _milvus_pool is None:
        from .milvus_pool import MilvusConnectionPool
        
        if not MILVUS_URI:
            raise RuntimeError("MILVUS_URI not configured")
        
        # Determine if Zilliz Cloud
        is_zilliz = MILVUS_TOKEN or "zillizcloud.com" in MILVUS_URI
        
        if is_zilliz:
            # Zilliz Cloud
            uri = MILVUS_URI.replace("http://", "").replace("https://", "")
            if not uri.startswith("https://"):
                uri = f"https://{uri}"
            
            _milvus_pool = MilvusConnectionPool(
                uri=uri,
                token=MILVUS_TOKEN,
                secure=True,
                pool_size=MILVUS_POOL_SIZE,
                max_age_seconds=MILVUS_POOL_MAX_AGE,
                max_uses=MILVUS_POOL_MAX_USES
            )
        else:
            # Local Milvus
            _milvus_pool = MilvusConnectionPool(
                uri=MILVUS_URI,
                pool_size=MILVUS_POOL_SIZE,
                max_age_seconds=MILVUS_POOL_MAX_AGE,
                max_uses=MILVUS_POOL_MAX_USES
            )
    
    return _milvus_pool
# ========================
# Lazy Clients
# ========================
@lru_cache(maxsize=1)
def get_milvus_client():
    """
    Get Milvus client.
    
    If connection pooling enabled: Returns pool context manager
    If disabled: Returns single client (legacy)
    """
    if ENABLE_CONNECTION_POOL:
        # Return pool for context manager usage
        return get_milvus_pool()
    else:
        # Legacy: single connection
        from pymilvus import MilvusClient
        
        if not MILVUS_URI:
            raise RuntimeError("MILVUS_URI not configured")
        
        logger.info(f"Initializing single Milvus client for {COLLECTION}")
        
        if MILVUS_TOKEN or "zillizcloud.com" in MILVUS_URI:
            uri = MILVUS_URI.replace("http://", "").replace("https://", "")
            if not uri.startswith("https://"):
                uri = f"https://{uri}"
            
            return MilvusClient(uri=uri, token=MILVUS_TOKEN, secure=True)
        else:
            return MilvusClient(uri=MILVUS_URI)


@lru_cache(maxsize=1)
def get_encoder():
    """Get embedding model (singleton)."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
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


@lru_cache(maxsize=1)
def get_search_thread_pool() -> ThreadPoolExecutor:
    """Get thread pool for parallel search operations."""
    pool = ThreadPoolExecutor(
        max_workers=SEARCH_THREAD_POOL_SIZE,
        thread_name_prefix="search_worker"
    )
    logger.info(f"✅ Search thread pool initialized with {SEARCH_THREAD_POOL_SIZE} workers")
    return pool