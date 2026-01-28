#!/bin/bash
# deploy.sh - Deploy Bey Webhook to Cloud Run using Cloud Build

set -e

PROJECT_ID="taqforce"
REGION="us-central1"
SERVICE_NAME="bey-webhook"
CLOUD_SQL_INSTANCE="taqforce:us-central1:recruiter-pg"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
GCS_BUCKET="rcrtrai-call-history"
SERVICE_ACCOUNT="467925804888-compute@developer.gserviceaccount.com"

echo "🔧 Configuration:"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Service: ${SERVICE_NAME}"
echo "   GCS Bucket: ${GCS_BUCKET}"
echo ""

# Create GCS bucket if it doesn't exist
echo "📦 Checking GCS bucket..."
if ! gsutil ls -b gs://${GCS_BUCKET} &>/dev/null; then
    echo "   Creating bucket gs://${GCS_BUCKET}..."
    gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${GCS_BUCKET}
    echo "   ✅ Bucket created"
else
    echo "   ✅ Bucket already exists"
fi

# Grant Cloud Run service account access to GCS bucket
echo "🔑 Granting GCS permissions to service account..."
gsutil iam ch serviceAccount:${SERVICE_ACCOUNT}:objectAdmin gs://${GCS_BUCKET}

# Build using Cloud Build (no local Docker needed)
echo "☁️  Building with Cloud Build..."
gcloud builds submit --tag ${IMAGE}:latest .

# Deploy
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE}:latest \
    --region ${REGION} \
    --platform managed \
    --allow-unauthenticated \
    --add-cloudsql-instances ${CLOUD_SQL_INSTANCE} \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "PYTHONUNBUFFERED=1,MILVUS_PORT=19530,MILVUS_COLLECTION=job_postings,SIMILARITY_THRESHOLD=0.90,MAX_SIMILAR_JOBS=3,BEY_API_URL=https://api.bey.dev/v1,GCS_BUCKET=${GCS_BUCKET}" \
    --set-secrets "DATABASE_URL=bey-webhook-db-url:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,MILVUS_HOST=milvus-host:latest,BEY_API_KEY=bey-api-key:latest,ADMIN_API_KEY=admin-api-key:latest,HF_TOKEN=hf-token:latest"

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
echo ""
echo "✅ Deployment complete!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo "📋 Webhook URL: ${SERVICE_URL}/webhook"
echo "🎭 Avatar Page: ${SERVICE_URL}/"
echo "📞 Call Page: ${SERVICE_URL}/call"
echo ""
echo "🔍 Test endpoints:"
echo "   curl ${SERVICE_URL}/health"
echo "   curl ${SERVICE_URL}/health/db"
echo "   curl ${SERVICE_URL}/health/milvus"
echo "   curl ${SERVICE_URL}/api/stats"
echo "   curl -X POST ${SERVICE_URL}/api/create-call -H 'Content-Type: application/json' -d '{\"username\":\"Test\"}'"