# 📦 Complete Project Structure

```
recruiterbrain-voice-complete/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── interview.py           # Interview endpoints
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                # Configuration (Pydantic)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── interview.py               # Data models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database.py                # PostgreSQL + Milvus + Embeddings
│   │   └── interview_service.py       # Interview business logic
│   │
│   └── livekit_agent/
│       ├── __init__.py
│       └── worker.py                  # Voice agent (Deepgram, Google TTS, OpenAI)
│
├── migrations/
│   ├── 002_add_missing_columns.sql   # PostgreSQL migration
│   └── setup_milvus_v2.py            # Milvus collection setup
│
├── tests/
│   └── test_system.py                # System tests
│
├── .env.example                      # Environment template
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Automated setup script
├── README.md                         # Complete documentation
└── QUICKSTART.md                     # 5-minute setup guide
```

---

## 🎯 What Each File Does

### Application Core

**app/api/main.py**
- FastAPI application entry point
- Lifespan events (startup/shutdown)
- Database connection initialization
- CORS middleware
- Error handlers

**app/api/routes/interview.py**
- POST `/start` - Start new interview
- GET `/status/{id}` - Get interview status
- POST `/search` - Semantic search
- POST `/cancel/{id}` - Cancel interview

### Configuration

**app/config/settings.py**
- Loads all config from .env
- Pydantic validation
- Connection pool settings
- API keys management

### Data Models

**app/models/interview.py**
- CandidateData
- JobDescriptionData
- StartInterviewRequest/Response
- InterviewStatus
- InterviewEvaluation
- Search models

### Services

**app/services/database.py**
- PostgreSQL async connection pool (150 connections)
- Milvus connection pool (10 connections)
- E5-Base-V2 embedding service
- Batch insert operations
- Interview CRUD operations

**app/services/interview_service.py**
- In-memory interview sessions
- Batch save at end of interview
- Session management

### Voice Agent

**app/livekit_agent/worker.py**
- LiveKit agent worker
- Deepgram STT (Nova-2)
- Google Cloud TTS (Neural2/WaveNet)
- OpenAI GPT-4o-mini (LLM)
- Silero VAD
- Interview orchestration

### Database

**migrations/002_add_missing_columns.sql**
- Adds 4 columns to existing interviews table:
  - milvus_synced
  - milvus_sync_at
  - worker_id
  - livekit_room_name

**migrations/setup_milvus_v2.py**
- Creates interview_transcripts_v2 collection
- 768-dimensional vectors (e5-base-v2)
- HNSW index for vector search
- Scalar indexes for filtering

---

## 🚀 Deployment Workflows

### Development (Local)

```bash
# 1. Setup
./setup.sh

# 2. Start services
# Terminal 1:
python app/livekit_agent/worker.py

# Terminal 2:
python app/api/main.py

# 3. Test
python tests/test_system.py
```

### Production (Docker)

```dockerfile
# Dockerfile (to be created)
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY migrations/ migrations/

# Run migrations
RUN python migrations/setup_milvus_v2.py

# Start services
CMD ["python", "app/api/main.py"]
```

### Production (Kubernetes)

```yaml
# deployment.yaml (to be created)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recruiterbrain-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: recruiterbrain-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: POSTGRES_HOST
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: host
```

---

## 📊 Monitoring

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/api/v1/interview/status/{id}
```

### Database Monitoring

```sql
-- PostgreSQL connections
SELECT count(*) FROM pg_stat_activity WHERE datname = 'recruiterbrain';

-- Active interviews
SELECT status, count(*) FROM interviews GROUP BY status;

-- Milvus sync status
SELECT count(*) FROM interviews WHERE milvus_synced = false;
```

### Milvus Monitoring

```python
from pymilvus import Collection
collection = Collection("interview_transcripts_v2")
print(f"Entities: {collection.num_entities}")
print(f"Loaded: {collection.is_loaded}")
```

---

## 🔧 Configuration Reference

### Database Connection Pools

```env
# PostgreSQL
POSTGRES_POOL_SIZE=50         # Base connections
POSTGRES_MAX_OVERFLOW=100     # Additional when needed
POSTGRES_POOL_TIMEOUT=30      # Connection timeout (seconds)
POSTGRES_POOL_RECYCLE=3600    # Recycle connections (1 hour)

# Milvus
MILVUS_POOL_SIZE=10           # Concurrent Milvus operations
```

### Voice Settings

```env
# Deepgram STT
DEEPGRAM_MODEL=nova-2         # Best accuracy
DEEPGRAM_LANGUAGE=en-US

# Google TTS
GOOGLE_TTS_VOICE_NAME=en-US-Neural2-J  # Natural male voice
GOOGLE_TTS_SPEAKING_RATE=1.0           # Speed (0.25-4.0)
GOOGLE_TTS_PITCH=0.0                   # Pitch (-20 to 20)

# OpenAI LLM
OPENAI_MODEL=gpt-4o-mini      # Fast and cost-effective
OPENAI_TEMPERATURE=0.7        # Creativity (0-2)
OPENAI_MAX_TOKENS=1000        # Response length
```

### Interview Settings

```env
INTERVIEW_MAX_DURATION_SECONDS=720  # 12 minutes
INTERVIEW_QUESTIONS_COUNT=6         # Number of questions
INTERVIEW_TIMEOUT_SECONDS=30        # Response timeout
```

---

## 🎯 Scaling Guide

### For 100 Concurrent Interviews

**Infrastructure needed:**

```yaml
API Servers:
  - Count: 3-5 replicas
  - CPU: 2 cores each
  - RAM: 4GB each

LiveKit Workers:
  - Count: 10 workers
  - CPU: 2 cores each
  - RAM: 2GB each

PostgreSQL:
  - CPU: 4 cores
  - RAM: 8GB
  - Storage: 100GB SSD
  - Connections: 200 max

Milvus:
  - CPU: 4 cores
  - RAM: 16GB
  - Storage: 50GB SSD
```

**Cost estimate (cloud):**
- Servers: $300/month
- PostgreSQL: $100/month
- Milvus: $150/month
- LiveKit: $0.096 × 30,000 = $2,880/month
- Other services: $200/month
- **Total: ~$3,630/month for 30,000 interviews**
- **Per interview: $0.12 infrastructure + $0.28 usage = $0.40 total**

---

## 🐛 Debugging Tips

### Enable Debug Mode

```env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Check Logs

```bash
# API logs
tail -f /var/log/recruiterbrain/api.log

# Agent logs
tail -f /var/log/recruiterbrain/agent.log

# Database logs
tail -f /var/log/postgresql/postgresql-14-main.log
```

### Test Individual Components

```python
# Test PostgreSQL
from app.services.database import get_db_session
async with get_db_session() as session:
    result = await session.execute("SELECT 1")
    print(result.fetchone())

# Test Milvus
from pymilvus import connections, Collection
connections.connect()
collection = Collection("interview_transcripts_v2")
print(collection.num_entities)

# Test embeddings
from app.services.database import embedding_service
emb = embedding_service.generate_embedding("test")
print(len(emb))  # Should be 768
```

---

## ✅ Production Checklist

Before deploying to production:

### Security
- [ ] All API keys in environment variables (not hardcoded)
- [ ] Strong SECRET_KEY and JWT_SECRET_KEY
- [ ] PostgreSQL uses SSL
- [ ] API rate limiting enabled
- [ ] CORS properly configured
- [ ] Firewall rules set

### Performance
- [ ] Connection pooling configured
- [ ] Database indexes created
- [ ] Milvus indexes built
- [ ] Load tested with 100 concurrent
- [ ] Response times < 200ms
- [ ] Memory usage stable

### Reliability
- [ ] PostgreSQL backups automated
- [ ] Error handling comprehensive
- [ ] Logging configured
- [ ] Monitoring alerts set up
- [ ] Health checks working
- [ ] Auto-restart on crash

### Compliance
- [ ] GDPR audit logs
- [ ] PII encryption
- [ ] Data retention policy
- [ ] Candidate consent tracking
- [ ] Recording disclosure

---

**You're ready for production! 🚀**
