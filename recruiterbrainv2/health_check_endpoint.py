"Adding health checks"
from fastapi import HTTPException
from typing import Dict, Any
import redis
from pymilvus import MilvusClient
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


async def check_redis() -> Dict[str, Any]:
    """Check if Redis is accessible."""
    try:
        from recruiterbrainv2.config import REDIS_URL
        
        # Parse Redis URL
        if REDIS_URL.startswith("redis://"):
            parts = REDIS_URL.replace("redis://", "").split(":")
            host = parts[0]
            port_db = parts[1].split("/") if "/" in parts[1] else [parts[1], "0"]
            port = int(port_db[0])
            db = int(port_db[1]) if len(port_db) > 1 else 0
        else:
            host, port, db = "localhost", 6379, 0
        
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            socket_connect_timeout=2
        )
        
        # Ping Redis
        client.ping()
        
        # Get basic info
        info = client.info()
        
        return {
            "status": "healthy",
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "uptime_days": info.get("uptime_in_days", 0)
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_milvus() -> Dict[str, Any]:
    """Check if Milvus is accessible."""
    try:
        from recruiterbrainv2.config import MILVUS_URI, MILVUS_COLLECTION, MILVUS_TOKEN
        
        # Create client
        if MILVUS_TOKEN:
            client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN, secure=True)
        else:
            client = MilvusClient(uri=MILVUS_URI)
        
        # List collections
        collections = client.list_collections()
        
        # Check if target collection exists
        collection_exists = MILVUS_COLLECTION in collections
        
        # Get collection stats if it exists
        stats = {}
        if collection_exists:
            try:
                # This is a simple check - adjust based on your Milvus version
                stats = client.get_collection_stats(MILVUS_COLLECTION)
            except:
                stats = {"note": "Collection exists but stats unavailable"}
        
        client.close()
        
        return {
            "status": "healthy",
            "collections": collections,
            "target_collection": MILVUS_COLLECTION,
            "target_exists": collection_exists,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Milvus health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_celery() -> Dict[str, Any]:
    """Check if Celery workers are running."""
    try:
        from recruiterbrainv2.celery_app import celery_app
        
        # Get worker stats
        stats = celery_app.control.inspect().stats()
        active = celery_app.control.inspect().active()
        
        if stats is None:
            return {
                "status": "unhealthy",
                "error": "No Celery workers running",
                "suggestion": "Start worker with: celery -A recruiterbrainv2.celery_app worker --loglevel=info"
            }
        
        worker_count = len(stats.keys())
        active_tasks = sum(len(tasks) for tasks in (active or {}).values())
        
        return {
            "status": "healthy",
            "workers": worker_count,
            "active_tasks": active_tasks,
            "worker_names": list(stats.keys())
        }
    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_openai() -> Dict[str, Any]:
    """Check if OpenAI API key is configured."""
    try:
        from recruiterbrainv2.config import OPENAI_API_KEY
        
        if not OPENAI_API_KEY:
            return {
                "status": "unhealthy",
                "error": "OPENAI_API_KEY not configured",
                "suggestion": "Add OPENAI_API_KEY to .env file"
            }
        
        if "your-key-here" in OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
            return {
                "status": "unhealthy",
                "error": "OPENAI_API_KEY appears to be placeholder",
                "suggestion": "Replace with actual API key from https://platform.openai.com/api-keys"
            }
        
        return {
            "status": "healthy",
            "key_prefix": OPENAI_API_KEY[:10] + "...",
            "configured": True
        }
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

