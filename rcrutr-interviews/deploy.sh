#!/bin/bash
# Deploy RCRUTR Interviews to Google Cloud Run

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-taqforce}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="rcrutr-interviews"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=============================================="
echo "Deploying ${SERVICE_NAME} to Cloud Run"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# Build the image
echo "Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} --project ${PROJECT_ID}

# Deploy to Cloud Run
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 10 \
    --set-secrets "\
DATABASE_URL=bey-webhook-db-url:latest,\
OPENAI_API_KEY=OPENAI_API_KEY:latest,\
MILVUS_HOST=milvus-host:latest,\
BEY_API_KEY=bey-api-key:latest,\
ADMIN_API_KEY=admin-api-key:latest,\
HF_TOKEN=hf-token:latest,\
ZOOM_ACCOUNT_ID=zoom-account-id:latest,\
ZOOM_CLIENT_ID=zoom-client-id:latest,\
ZOOM_CLIENT_SECRET=zoom-client-secret:latest"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo ""
echo "=============================================="
echo "Deployment complete!"
echo "=============================================="
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test endpoints:"
echo "  Health:     curl ${SERVICE_URL}/health"
echo "  Status:     curl ${SERVICE_URL}/api/status"
echo ""
echo "Schedule interview:"
echo '  curl -X POST "'${SERVICE_URL}'/api/schedule-interview" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"candidate_id": "xxx", "job_id": "yyy", "scheduled_time": "2026-02-01T14:00:00Z"}'"'"
echo ""
