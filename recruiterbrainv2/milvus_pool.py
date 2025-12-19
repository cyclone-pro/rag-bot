"""Milvus connection pool for high-concurrency scenarios."""
import logging
import time
import threading
from queue import Queue, Empty, Full
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pymilvus import MilvusClient, MilvusException

logger = logging.getLogger(__name__)


class PooledMilvusConnection:
    """Wrapper for a Milvus connection with health tracking."""
    
    def __init__(self, uri: str, token: Optional[str] = None, secure: bool = False):
        self.uri = uri
        self.token = token
        self.secure = secure
        self.client: Optional[MilvusClient] = None
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.is_healthy = True
        self.lock = threading.Lock()
        
        # Create connection
        self._connect()
    
    def _connect(self):
        """Establish connection to Milvus."""
        try:
            logger.debug(f"Creating Milvus connection to {self.uri}")
            
            if self.token or self.secure:
                self.client = MilvusClient(
                    uri=self.uri,
                    token=self.token,
                    secure=self.secure
                )
            else:
                self.client = MilvusClient(uri=self.uri)
            
            self.is_healthy = True
            logger.debug("✅ Milvus connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to create Milvus connection: {e}")
            self.is_healthy = False
            raise
    
    def health_check(self) -> bool:
        """
        Check if connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.client:
            self.is_healthy = False
            return False
        
        try:
            # Try to list collections as health check
            self.client.list_collections()
            self.is_healthy = True
            return True
        
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.is_healthy = False
            return False
    
    def reconnect(self):
        """Attempt to reconnect if unhealthy."""
        if self.client:
            try:
                # Try to close existing connection
                del self.client
            except:
                pass
        
        self._connect()
    
    def mark_used(self):
        """Mark connection as used (for tracking)."""
        with self.lock:
            self.last_used = time.time()
            self.use_count += 1
    
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at
    
    def idle_time(self) -> float:
        """Get time since last use in seconds."""
        return time.time() - self.last_used


class MilvusConnectionPool:
    """
    Thread-safe connection pool for Milvus.
    
    Features:
    - Pool of reusable connections
    - Automatic health checking
    - Connection recycling (max age/uses)
    - Thread-safe borrowing/returning
    - Graceful degradation on errors
    """
    
    def __init__(
        self,
        uri: str,
        token: Optional[str] = None,
        secure: bool = False,
        pool_size: int = 10,
        max_age_seconds: int = 3600,
        max_uses: int = 1000,
        health_check_interval: int = 60
    ):
        """
        Initialize connection pool.
        
        Args:
            uri: Milvus URI
            token: Auth token (for Zilliz Cloud)
            secure: Use TLS
            pool_size: Number of connections in pool
            max_age_seconds: Max connection age before recycling
            max_uses: Max uses before recycling connection
            health_check_interval: Health check frequency
        """
        self.uri = uri
        self.token = token
        self.secure = secure
        self.pool_size = pool_size
        self.max_age_seconds = max_age_seconds
        self.max_uses = max_uses
        self.health_check_interval = health_check_interval
        
        # Connection pool (FIFO queue)
        self.pool: Queue = Queue(maxsize=pool_size)
        
        # Track all connections (for health monitoring)
        self.all_connections = []
        self.lock = threading.Lock()
        
        # Stats
        self.stats = {
            "total_created": 0,
            "total_borrowed": 0,
            "total_returned": 0,
            "total_recycled": 0,
            "active_connections": 0,
        }
        
        # Initialize pool
        self._initialize_pool()
        
        logger.info(
            f"✅ Milvus connection pool initialized: "
            f"size={pool_size}, uri={uri}"
        )
    
    def _initialize_pool(self):
        """Create initial connections."""
        for i in range(self.pool_size):
            try:
                conn = PooledMilvusConnection(
                    uri=self.uri,
                    token=self.token,
                    secure=self.secure
                )
                self.pool.put(conn)
                self.all_connections.append(conn)
                self.stats["total_created"] += 1
                self.stats["active_connections"] += 1
                
            except Exception as e:
                logger.error(f"Failed to create connection {i+1}/{self.pool_size}: {e}")
        
        logger.info(f"Created {self.stats['active_connections']}/{self.pool_size} connections")
    
    def get_connection(self, timeout: float = 5.0) -> PooledMilvusConnection:
        """
        Borrow a connection from the pool.
        
        Args:
            timeout: Max seconds to wait for available connection
        
        Returns:
            Pooled connection
        
        Raises:
            TimeoutError: If no connection available within timeout
        """
        try:
            # Try to get connection from pool
            conn = self.pool.get(timeout=timeout)
            
            # Check if connection needs recycling
            if self._should_recycle(conn):
                logger.info("Recycling connection (age/use limit)")
                self._recycle_connection(conn)
                # Create new connection
                conn = PooledMilvusConnection(
                    uri=self.uri,
                    token=self.token,
                    secure=self.secure
                )
                with self.lock:
                    self.all_connections.append(conn)
                    self.stats["total_created"] += 1
                    self.stats["total_recycled"] += 1
            
            # Health check
            if not conn.is_healthy:
                logger.warning("Connection unhealthy, attempting reconnect...")
                try:
                    conn.reconnect()
                except Exception as e:
                    logger.error(f"Reconnect failed: {e}")
                    # Create new connection
                    conn = PooledMilvusConnection(
                        uri=self.uri,
                        token=self.token,
                        secure=self.secure
                    )
            
            conn.mark_used()
            self.stats["total_borrowed"] += 1
            
            return conn
            
        except Empty:
            logger.error(f"Connection pool exhausted (timeout={timeout}s)")
            raise TimeoutError(
                f"No available Milvus connections in pool after {timeout}s. "
                f"Pool size: {self.pool_size}, Active: {self.stats['active_connections']}"
            )
    
    def return_connection(self, conn: PooledMilvusConnection):
        """
        Return a connection to the pool.
        
        Args:
            conn: Connection to return
        """
        try:
            self.pool.put(conn, timeout=1.0)
            self.stats["total_returned"] += 1
        
        except Full:
            logger.warning("Pool full, discarding connection")
            with self.lock:
                if conn in self.all_connections:
                    self.all_connections.remove(conn)
                self.stats["active_connections"] -= 1
    
    @contextmanager
    def connection(self):
        """
        Context manager for safe connection borrowing.
        
        Usage:
            with pool.connection() as conn:
                conn.client.search(...)
        """
        conn = None
        try:
            conn = self.get_connection()
            yield conn.client
        finally:
            if conn:
                self.return_connection(conn)

    def list_collections(self):
        """Expose list_collections for callers that expect a Milvus client."""
        with self.connection() as client:
            return client.list_collections()
    
    def _should_recycle(self, conn: PooledMilvusConnection) -> bool:
        """Check if connection should be recycled."""
        # Age-based recycling
        if conn.age() > self.max_age_seconds:
            return True
        
        # Use-based recycling
        if conn.use_count > self.max_uses:
            return True
        
        return False
    
    def _recycle_connection(self, conn: PooledMilvusConnection):
        """Recycle (close and remove) a connection."""
        try:
            if conn.client:
                del conn.client
        except:
            pass
        
        with self.lock:
            if conn in self.all_connections:
                self.all_connections.remove(conn)
            self.stats["active_connections"] -= 1
    
    def health_check_all(self):
        """Run health check on all connections."""
        unhealthy_count = 0
        
        with self.lock:
            for conn in self.all_connections:
                if not conn.health_check():
                    unhealthy_count += 1
        
        if unhealthy_count > 0:
            logger.warning(f"{unhealthy_count}/{len(self.all_connections)} connections unhealthy")
        
        return len(self.all_connections) - unhealthy_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self.stats,
            "pool_size": self.pool_size,
            "available": self.pool.qsize(),
            "in_use": self.stats["active_connections"] - self.pool.qsize(),
        }
    
    def close_all(self):
        """Close all connections (cleanup)."""
        logger.info("Closing all Milvus connections...")
        
        with self.lock:
            for conn in self.all_connections:
                try:
                    if conn.client:
                        del conn.client
                except:
                    pass
            
            self.all_connections.clear()
            self.stats["active_connections"] = 0
        
        # Clear queue
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
            except Empty:
                break
        
        logger.info("✅ All connections closed")
