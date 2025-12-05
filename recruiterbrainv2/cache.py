"""Unified caching layer - supports Redis + in-memory fallback."""
import hashlib
import json
import logging
import os
from typing import Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import Redis (optional dependency)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Using in-memory cache only. Install: pip install redis")


class CacheBackend:
    """Abstract cache interface."""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        raise NotImplementedError
    
    def delete(self, key: str):
        raise NotImplementedError
    
    def clear(self):
        raise NotImplementedError


class RedisCache(CacheBackend):
    """Redis-backed cache (distributed, persistent)."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # Test connection
            self.client.ping()
            logger.info(f"✅ Redis cache connected: {host}:{port}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key[:50]}...")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key[:50]}...")
            return None
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        try:
            self.client.setex(key, ttl, json.dumps(value))
            logger.debug(f"Cache SET: {key[:50]}... (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Redis SET error: {e}")
    
    def delete(self, key: str):
        try:
            self.client.delete(key)
        except Exception as e:
            logger.warning(f"Redis DELETE error: {e}")
    
    def clear(self):
        try:
            self.client.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.warning(f"Redis CLEAR error: {e}")


class InMemoryCache(CacheBackend):
    """In-memory cache with LRU eviction (single-server only)."""
    
    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._max_size = max_size
        logger.info(f"✅ In-memory cache initialized (max_size={max_size})")
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            logger.debug(f"Cache HIT: {key[:50]}...")
            return self._cache[key]
        logger.debug(f"Cache MISS: {key[:50]}...")
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        # Simple LRU: if full, remove oldest
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
        logger.debug(f"Cache SET: {key[:50]}... (size: {len(self._cache)})")
        
        # Note: TTL not enforced in simple in-memory cache
        # For production, use Redis or implement TTL with background cleanup
    
    def delete(self, key: str):
        self._cache.pop(key, None)
    
    def clear(self):
        self._cache.clear()
        logger.info("In-memory cache cleared")


@lru_cache(maxsize=1)
def get_cache() -> CacheBackend:
    """
    Get cache singleton (auto-selects Redis or in-memory).
    
    Priority:
    1. Redis (if REDIS_URL set and redis package installed)
    2. In-memory fallback
    """
    redis_url = os.getenv("REDIS_URL")
    
    if REDIS_AVAILABLE and redis_url:
        # Parse Redis URL (e.g., redis://localhost:6379/0 or redis://:password@host:6379/0)
        try:
            if redis_url.startswith("redis://"):
                # Simple parsing (for production, use urllib.parse)
                parts = redis_url.replace("redis://", "").split(":")
                if "@" in parts[0]:
                    # Has password
                    password, host = parts[0].split("@")
                    port_db = parts[1].split("/")
                    port = int(port_db[0])
                    db = int(port_db[1]) if len(port_db) > 1 else 0
                else:
                    # No password
                    host = parts[0]
                    port_db = parts[1].split("/")
                    port = int(port_db[0])
                    db = int(port_db[1]) if len(port_db) > 1 else 0
                    password = None
                
                return RedisCache(host=host, port=port, db=db, password=password)
            else:
                logger.warning(f"Invalid REDIS_URL format: {redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
    
    # Fallback to in-memory
    return InMemoryCache(max_size=1000)


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a stable cache key from arguments.
    
    Example:
        generate_cache_key("llm", query="Find Python developers")
        -> "llm:hash_of_query"
    """
    # Combine all args into a single string
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    
    combined = "|".join(key_parts)
    
    # Hash for fixed-length key
    hash_obj = hashlib.md5(combined.encode())
    hash_str = hash_obj.hexdigest()[:16]  # First 16 chars
    
    return f"{prefix}:{hash_str}"


# ==================== CACHE DECORATORS ====================

def cached(prefix: str, ttl: int = 3600):
    """
    Decorator to cache function results.
    
    Usage:
        @cached("llm", ttl=3600)
        def extract_skills(query: str):
            # expensive LLM call
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.info(f"✅ Cache HIT: {func.__name__}")
                return cached_value
            
            # Cache miss - compute value
            logger.info(f"⚠️  Cache MISS: {func.__name__} - computing...")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator