#!/bin/bash
# Start Celery worker for resume processing

echo "🚀 Starting Celery worker..."
echo "========================================"

# Activate virtual environment
source rag3/bin/activate

# Start worker with config
celery -A recruiterbrainv2.celery_app worker \
  --loglevel=info \
  --concurrency=4 \
  --queues=resume_processing,bulk_processing \
  --hostname=worker@%h \
  --max-tasks-per-child=1000

# Flags explained:
# --concurrency=4: Run 4 worker processes in parallel
# --queues: Listen to specific queues
# --max-tasks-per-child: Restart worker after 1000 tasks (prevent memory leaks)