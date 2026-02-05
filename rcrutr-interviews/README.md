# RCRUTR Interviews - Setup & Run Instructions

## 1. Prerequisites

Make sure you have:
- Python 3.10+
- PostgreSQL access (you already have this)
- Milvus access (you already have this)
- Zoom account with API access (see below)

## 2. Get Zoom API Credentials

### Step-by-Step:

1. Go to https://marketplace.zoom.us/
2. Sign in with your Zoom account (or create one)
3. Click **"Develop"** dropdown → **"Build App"**
4. Choose **"Server-to-Server OAuth"** app type
5. Fill in:
   - App Name: `RCRUTR Interview Bot`
   - Company Name: `Elite Solutions`
   - Description: `AI-powered candidate interviews`
6. Click **Create**
7. Go to **"Scopes"** tab and add:
   - `meeting:write:admin` (Create meetings)
   - `meeting:read:admin` (Read meeting info)
   - `user:read:admin` (Get user info)
8. Click **"Activate your app"**
9. Go to **"App Credentials"** tab and copy:
   - Account ID
   - Client ID
   - Client Secret

### Add to .env file:
```
ZOOM_ACCOUNT_ID=your_account_id_here
ZOOM_CLIENT_ID=your_client_id_here
ZOOM_CLIENT_SECRET=your_client_secret_here
```

## 3. Create Database Table

Run this SQL in your PostgreSQL database (ONE TIME ONLY):

```bash
psql "postgresql://backteam:Airecruiter1_@34.60.92.122:5432/recruiter_brain" -f create_table.sql
```

Or use your PostgreSQL client (pgAdmin, DBeaver, etc.) to run `create_table.sql`.

## 4. Install Dependencies

```bash
cd rcrutr-interviews
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Run Locally

```bash
# Make sure .env is filled in
source .venv/bin/activate
python -m uvicorn app:app --reload --port 8080
```

Test it:
```bash
# Health check
curl http://localhost:8080/health

# Status check (shows all dependencies)
curl http://localhost:8080/api/status
```

## 6. Test Schedule Interview (without Zoom)

First, let's test the non-Zoom parts work. Temporarily comment out Zoom:

```bash
# Test endpoint
curl -X POST "http://localhost:8080/api/schedule-interview" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "test_candidate_123",
    "job_id": "test_job_456",
    "scheduled_time": "2026-02-02T14:00:00Z",
    "avatar": "zara"
  }'
```

## 7. Add Google Cloud Secrets (for Cloud Run deployment)

```bash
# Zoom credentials
echo -n "YOUR_ZOOM_ACCOUNT_ID" | gcloud secrets create zoom-account-id --data-file=-
echo -n "YOUR_ZOOM_CLIENT_ID" | gcloud secrets create zoom-client-id --data-file=-
echo -n "YOUR_ZOOM_CLIENT_SECRET" | gcloud secrets create zoom-client-secret --data-file=-
```

## 8. Deploy to Cloud Run

```bash
chmod +x deploy.sh
./deploy.sh
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /api/status | Status with dependency checks |
| POST | /api/schedule-interview | Schedule new interview |
| GET | /api/interview/{id} | Get interview details |
| GET | /api/interviews | List all interviews |
| POST | /api/interview/{id}/start | Start interview manually |
| POST | /api/interview/{id}/cancel | Cancel interview |
| POST | /webhook/bey | Bey webhook (call_ended) |
| POST | /webhook/zoom | Zoom webhook |

## Example: Schedule Interview

```bash
curl -X POST "https://your-service.run.app/api/schedule-interview" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "cand_abc123",
    "job_id": "job_xyz789",
    "scheduled_time": "2026-02-01T14:00:00Z",
    "timezone": "America/New_York",
    "avatar": "zara",
    "notes": "Follow up from recruiter call"
  }'
```

Response:
```json
{
  "interview_id": "int_abc123def456",
  "meeting_url": "https://zoom.us/j/123456789?pwd=xxx",
  "meeting_passcode": "123456",
  "scheduled_time": "2026-02-01T14:00:00Z",
  "candidate_name": "John Smith",
  "job_title": "Senior Python Developer",
  "status": "meeting_created"
}
```

## Troubleshooting

### "Candidate not found"
- Check that `candidate_id` exists in Milvus `candidates_v3` collection

### "Job not found"
- Check that `job_id` exists in Milvus `job_postings` collection

### "Failed to create Zoom meeting"
- Check Zoom credentials in .env
- Run: `curl http://localhost:8080/api/status` to see Zoom connection status

### Database connection issues
- Check DATABASE_URL is correct
- Make sure the table was created with create_table.sql
