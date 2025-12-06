#!/bin/bash
# Start Celery Beat for scheduled tasks

echo "⏰ Starting Celery Beat scheduler..."

source rag3/bin/activate

celery -A recruiterbrainv2.celery_app beat \
  --loglevel=info \
  --scheduler=celery.beat:PersistentScheduler