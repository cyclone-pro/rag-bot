"""Job management for async background tasks."""
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Try Redis (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Using in-memory job storage.")


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStore:
    """Abstract job storage interface."""
    
    def create_job(self, job_type: str, params: Dict[str, Any]) -> str:
        """Create a new job and return job_id."""
        raise NotImplementedError
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result."""
        raise NotImplementedError
    
    def update_job(self, job_id: str, status: JobStatus, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Update job status and result."""
        raise NotImplementedError


class InMemoryJobStore(JobStore):
    """In-memory job storage (simple, no persistence)."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ In-memory job store initialized")
    
    def create_job(self, job_type: str, params: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        
        self._jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JobStatus.QUEUED,
            "params": params,
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        
        logger.info(f"Created job {job_id} (type: {job_type})")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)
    
    def update_job(self, job_id: str, status: JobStatus, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            if result:
                self._jobs[job_id]["result"] = result
            
            if error:
                self._jobs[job_id]["error"] = error
            
            logger.info(f"Updated job {job_id}: status={status}")


class RedisJobStore(JobStore):
    """Redis-based job storage (persistent, distributed)."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # Jobs expire after 1 hour
        logger.info("✅ Redis job store initialized")
    
    def create_job(self, job_type: str, params: Dict[str, Any]) -> str:
        import json
        
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JobStatus.QUEUED,
            "params": params,
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        
        self.redis.setex(
            f"job:{job_id}",
            self.ttl,
            json.dumps(job_data)
        )
        
        logger.info(f"Created job {job_id} (type: {job_type}) in Redis")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        import json
        
        data = self.redis.get(f"job:{job_id}")
        if data:
            return json.loads(data)
        return None
    
    def update_job(self, job_id: str, status: JobStatus, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        import json
        
        job_data = self.get_job(job_id)
        if job_data:
            job_data["status"] = status
            job_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            if result:
                job_data["result"] = result
            
            if error:
                job_data["error"] = error
            
            self.redis.setex(
                f"job:{job_id}",
                self.ttl,
                json.dumps(job_data)
            )
            
            logger.info(f"Updated job {job_id}: status={status}")


# Singleton job store
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get job store singleton (Redis or in-memory)."""
    global _job_store
    
    if _job_store is None:
        # Try Redis first
        if REDIS_AVAILABLE:
            try:
                from .cache import get_cache
                cache = get_cache()
                
                # Check if cache is Redis-based
                if hasattr(cache, 'client'):
                    _job_store = RedisJobStore(cache.client)
                    logger.info("Using Redis job store")
                else:
                    _job_store = InMemoryJobStore()
                    logger.info("Using in-memory job store (cache is not Redis)")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis job store: {e}")
                _job_store = InMemoryJobStore()
        else:
            _job_store = InMemoryJobStore()
    
    return _job_store