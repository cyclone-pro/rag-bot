"""Rate limit monitoring and analytics."""
import logging
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class RateLimitMonitor:
    """Monitor rate limit hits for analytics."""
    
    def __init__(self):
        self.violations = defaultdict(int)
        self.last_reset = datetime.utcnow()
    
    def record_violation(self, endpoint: str, ip: str):
        """Record a rate limit violation."""
        key = f"{endpoint}:{ip}"
        self.violations[key] += 1
        
        logger.warning(
            f"Rate limit violation: endpoint={endpoint}, ip={ip}, "
            f"total_violations={self.violations[key]}"
        )
    
    def get_top_violators(self, limit: int = 10):
        """Get top rate limit violators."""
        return sorted(
            self.violations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def reset_stats(self):
        """Reset statistics (call hourly)."""
        self.violations.clear()
        self.last_reset = datetime.utcnow()
        logger.info("Rate limit stats reset")


monitor = RateLimitMonitor()