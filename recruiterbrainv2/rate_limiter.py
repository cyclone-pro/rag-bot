"""Advanced rate limiting with Redis support and fallback."""
import logging
import time
from typing import Optional, Dict, Tuple
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Using in-memory rate limiting.")


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window: int, retry_after: int):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window} seconds. "
            f"Retry after {retry_after} seconds."
        )


class RateLimiter:
    """Abstract rate limiter interface."""
    
    def check_limit(self, key: str, limit: int, window: int) -> Tuple[bool, int]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Unique identifier (e.g., IP address, user_id)
            limit: Maximum requests allowed
            window: Time window in seconds
        
        Returns:
            (is_allowed, retry_after_seconds)
        """
        raise NotImplementedError


class InMemoryRateLimiter(RateLimiter):
    """
    In-memory rate limiter using sliding window.
    
    Good for single-server deployments.
    """
    
    def __init__(self):
        # key -> list of timestamps
        self._requests: Dict[str, list] = {}
        logger.info("✅ In-memory rate limiter initialized")
    
    def check_limit(self, key: str, limit: int, window: int) -> Tuple[bool, int]:
        now = time.time()
        
        # Get request history for this key
        if key not in self._requests:
            self._requests[key] = []
        
        # Remove old requests outside the window
        cutoff = now - window
        self._requests[key] = [ts for ts in self._requests[key] if ts > cutoff]
        
        # Check if limit exceeded
        if len(self._requests[key]) >= limit:
            # Calculate retry_after
            oldest_in_window = min(self._requests[key])
            retry_after = int(oldest_in_window + window - now) + 1
            return False, retry_after
        
        # Add current request
        self._requests[key].append(now)
        
        return True, 0
    
    def cleanup(self):
        """Remove expired keys to prevent memory leaks."""
        now = time.time()
        keys_to_remove = []
        
        for key, timestamps in self._requests.items():
            # If all timestamps are old (>1 hour), remove key
            if timestamps and max(timestamps) < now - 3600:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._requests[key]
        
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} expired rate limit keys")


class RedisRateLimiter(RateLimiter):
    """
    Redis-based rate limiter using sliding window.
    
    Works across multiple servers, persistent.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        logger.info("✅ Redis rate limiter initialized")
    
    def check_limit(self, key: str, limit: int, window: int) -> Tuple[bool, int]:
        """
        Sliding window rate limiting using Redis sorted sets.
        
        Uses ZREMRANGEBYSCORE + ZADD + ZCARD for atomic operations.
        """
        now = time.time()
        redis_key = f"ratelimit:{key}"
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Remove old entries outside window
            cutoff = now - window
            pipe.zremrangebyscore(redis_key, 0, cutoff)
            
            # Count current requests in window
            pipe.zcard(redis_key)
            
            # Execute pipeline
            results = pipe.execute()
            current_count = results[1]
            
            # Check if limit exceeded
            if current_count >= limit:
                # Get oldest timestamp in window
                oldest = self.redis.zrange(redis_key, 0, 0, withscores=True)
                if oldest:
                    oldest_ts = oldest[0][1]
                    retry_after = int(oldest_ts + window - now) + 1
                else:
                    retry_after = window
                
                return False, retry_after
            
            # Add current request
            self.redis.zadd(redis_key, {str(now): now})
            
            # Set expiry on the key (cleanup)
            self.redis.expire(redis_key, window + 60)
            
            return True, 0
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open (allow request) on Redis errors
            return True, 0


# Singleton rate limiter
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get rate limiter singleton (Redis or in-memory)."""
    global _rate_limiter
    
    if _rate_limiter is None:
        # Try Redis first
        if REDIS_AVAILABLE:
            try:
                from .cache import get_cache
                cache = get_cache()
                
                if hasattr(cache, 'client'):
                    _rate_limiter = RedisRateLimiter(cache.client)
                else:
                    _rate_limiter = InMemoryRateLimiter()
            except Exception as e:
                logger.warning(f"Failed to initialize Redis rate limiter: {e}")
                _rate_limiter = InMemoryRateLimiter()
        else:
            _rate_limiter = InMemoryRateLimiter()
    
    return _rate_limiter


# ==================== RATE LIMIT CONFIGURATIONS ====================

class RateLimitConfig:
    """Rate limit configurations for different endpoints."""
    
    # Per-IP limits (requests per minute)
    CHAT = (20, 60)           # 20 requests per 60 seconds
    INSIGHT = (10, 60)        # 10 requests per 60 seconds
    UPLOAD = (5, 60)          # 5 uploads per 60 seconds
    BULK_UPLOAD = (1, 3600)   # 1 bulk upload per hour
    JOB_STATUS = (60, 60)     # 60 status checks per minute
    
    # Global system limits
    OPENAI_GLOBAL = (100, 60)      # 100 OpenAI calls per minute (across all users)
    MILVUS_GLOBAL = (200, 1)       # 200 Milvus queries per second
    EMBEDDING_GLOBAL = (50, 60)    # 50 embedding generations per minute


def rate_limit(limit: int, window: int, key_prefix: str = "endpoint"):
    """
    Decorator for rate limiting FastAPI endpoints.
    
    Args:
        limit: Max requests allowed
        window: Time window in seconds
        key_prefix: Prefix for rate limit key
    
    Example:
        @app.get("/search")
        @rate_limit(limit=20, window=60, key_prefix="search")
        async def search(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if hasattr(arg, 'client'):
                    request = arg
                    break
            
            if not request:
                for value in kwargs.values():
                    if hasattr(value, 'client'):
                        request = value
                        break
            
            if not request:
                # No request object, skip rate limiting
                logger.warning("No request object found, skipping rate limiting")
                return await func(*args, **kwargs)
            
            # Get client IP
            client_ip = request.client.host
            
            # Check forwarded headers (for proxies)
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()
            
            # Build rate limit key
            rate_key = f"{key_prefix}:{client_ip}"
            
            # Check rate limit
            limiter = get_rate_limiter()
            is_allowed, retry_after = limiter.check_limit(rate_key, limit, window)
            
            if not is_allowed:
                logger.warning(
                    f"Rate limit exceeded for {client_ip} on {key_prefix}: "
                    f"{limit}/{window}s, retry after {retry_after}s"
                )
                
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": limit,
                        "window": window,
                        "retry_after": retry_after
                    },
                    headers={"Retry-After": str(retry_after)}
                )
            
            # Request allowed
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class GlobalRateLimiter:
    """
    Global rate limiter for system-wide resources.
    
    Use this to limit OpenAI API calls, Milvus queries, etc.
    across ALL users.
    """
    
    def __init__(self):
        self.limiter = get_rate_limiter()
    
    def check_openai_limit(self) -> bool:
        """Check if OpenAI API call is allowed."""
        limit, window = RateLimitConfig.OPENAI_GLOBAL
        is_allowed, _ = self.limiter.check_limit("global:openai", limit, window)
        
        if not is_allowed:
            logger.warning(f"Global OpenAI rate limit exceeded: {limit}/{window}s")
        
        return is_allowed
    
    def check_milvus_limit(self) -> bool:
        """Check if Milvus query is allowed."""
        limit, window = RateLimitConfig.MILVUS_GLOBAL
        is_allowed, _ = self.limiter.check_limit("global:milvus", limit, window)
        
        if not is_allowed:
            logger.warning(f"Global Milvus rate limit exceeded: {limit}/{window}s")
        
        return is_allowed
    
    def check_embedding_limit(self) -> bool:
        """Check if embedding generation is allowed."""
        limit, window = RateLimitConfig.EMBEDDING_GLOBAL
        is_allowed, _ = self.limiter.check_limit("global:embedding", limit, window)
        
        if not is_allowed:
            logger.warning(f"Global embedding rate limit exceeded: {limit}/{window}s")
        
        return is_allowed


# Global limiter singleton
global_limiter = GlobalRateLimiter()