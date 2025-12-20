import signal
import sys
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Handles graceful shutdown of the application.
    Ensures all connections are closed and pending tasks are finished.
    """
    
    def __init__(self):
        self.shutdown_initiated = False
        self.connections_to_close = []
        
    def add_connection(self, connection, name: str):
        """Register a connection that needs to be closed on shutdown."""
        self.connections_to_close.append((connection, name))
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self.shutdown_initiated:
            logger.warning("Force shutdown initiated!")
            sys.exit(1)
            
        self.shutdown_initiated = True
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        
        # Close all registered connections
        for connection, name in self.connections_to_close:
            try:
                logger.info(f"Closing {name}...")
                if hasattr(connection, 'close'):
                    connection.close()
                elif hasattr(connection, 'disconnect'):
                    connection.disconnect()
                logger.info(f"✓ {name} closed")
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")
        
        logger.info("Shutdown complete")
        sys.exit(0)
        
    def register_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # On Unix systems, also handle SIGHUP
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self.signal_handler)
        
        logger.info("Graceful shutdown handlers registered")

