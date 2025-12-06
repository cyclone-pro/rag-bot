#!/bin/bash
# Start Flower monitoring dashboard

echo "🌸 Starting Flower monitoring..."
echo "Dashboard will be available at: http://localhost:5555"

source rag3/bin/activate

celery -A recruiterbrainv2.celery_app flower \
  --port=5555 \
  --basic_auth=admin:password123