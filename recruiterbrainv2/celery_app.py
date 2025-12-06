"""Celery application configuration for distributed task processing."""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from celery import Celery

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path, override=True)

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Create Celery app
celery_app = Celery(
    "recruiterbrainv2",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["recruiterbrainv2.tasks"]  # Auto-discover tasks
)

# Celery Configuration
celery_app.conf.update(
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,  # Store more metadata
    
    # Task routing and queues
    task_routes={
        "recruiterbrainv2.tasks.process_resume_task": {"queue": "resume_processing"},
        "recruiterbrainv2.tasks.bulk_process_resumes": {"queue": "bulk_processing"},
    },
    
    # Worker settings
    worker_prefetch_multiplier=4,  # How many tasks to prefetch per worker
    worker_max_tasks_per_child=1000,  # Restart worker after N tasks (prevent memory leaks)
    
    # Task execution limits
    task_time_limit=300,  # Hard limit: 5 minutes
    task_soft_time_limit=240,  # Soft limit: 4 minutes (raises exception)
    
    # Retry settings
    task_acks_late=True,  # Acknowledge task after completion (not before)
    task_reject_on_worker_lost=True,  # Requeue if worker crashes
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule (for periodic tasks - optional)
    beat_schedule={
        # Example: Clean up old jobs every hour
        "cleanup-old-jobs": {
            "task": "recruiterbrainv2.tasks.cleanup_old_jobs",
            "schedule": 3600.0,  # Every hour
        },
    },
)

logger.info(f"✅ Celery configured with broker: {CELERY_BROKER_URL}")