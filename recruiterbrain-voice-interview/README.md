# RecruiterBrain AI Voice Interview System

## Overview
AI-powered voice interview system that conducts real-time technical interviews with candidates using Telnyx, Deepgram, Google TTS, and OpenAI.

## Features
- ✅ Real-time conversational AI interviews
- ✅ Multi-turn dialogue with dynamic follow-ups
- ✅ Intelligent silence handling
- ✅ Interruption detection
- ✅ Candidate question handling
- ✅ Skills extraction & semantic search
- ✅ Post-interview evaluation
- ✅ Vector embeddings in Milvus

## Tech Stack
- **Telephony**: Telnyx (Voice + SMS)
- **Speech-to-Text**: Deepgram Nova-2
- **Text-to-Speech**: Google Cloud TTS (WaveNet)
- **LLM**: OpenAI GPT-4o-mini
- **Database**: PostgreSQL + Milvus
- **Framework**: FastAPI
- **Task Queue**: Celery + Redis

## Cost per Interview (~12 mins)
- Telnyx: $0.10
- Deepgram: $0.05
- Google TTS: $0.07
- OpenAI: <$0.01
- **Total: ~$0.23/interview**

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Milvus 2.3+
- Redis 7+
- Telnyx account with phone number
- Deepgram API key
- Google Cloud account (TTS enabled)
- OpenAI API key

### Installation
```bash
# Clone repository
git clone <repository-url>
cd recruiterbrain-voice-interview

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Setup database
python scripts/setup_database.py
python scripts/setup_milvus.py

# Run application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Setup
```bash
docker-compose up -d
```

## Usage

### 1. Create Interview
```bash
POST /api/v1/interviews/create
{
  "candidate_name": "John Smith",
  "candidate_phone": "+14155551234",
  "candidate_email": "john@example.com",
  "semantic_summary": "Senior backend engineer with 8 years Python...",
  "evidence_projects": ["Built microservices platform...", "..."],
  "jd_summary": "Looking for Senior Backend Engineer with Python, FastAPI, PostgreSQL...",
  "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"]
}
```

### 2. System Schedules Call
- Sends SMS with Calendly link
- Candidate books time slot
- Sends consent form
- On consent: Schedules Telnyx call

### 3. Interview Conducted
- AI calls candidate at scheduled time
- 6 personalized questions
- Multi-turn conversation
- Real-time transcription

### 4. Results Available
```bash
GET /api/v1/interviews/{interview_id}
# Returns conversation_log (JSONB) + evaluation
```

## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## API Documentation
See [docs/API.md](docs/API.md)

## State Machine
See [docs/STATE_MACHINE.md](docs/STATE_MACHINE.md)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md)

## License
MIT
