# Implementation Roadmap

## Project Created! ✅

Your RecruiterBrain AI Voice Interview project structure is ready.

## What's Been Created

### ✅ Complete Directory Structure
- All folders created with proper hierarchy
- All `__init__.py` files in place

### ✅ Configuration Files
- `.env.example` - Environment variables template
- `config/settings.py` - Typed settings management
- `config/constants.py` - All constants (states, enums, thresholds)
- `docker-compose.yml` - Full stack (Postgres, Redis, Milvus)
- `Dockerfile` - Application container
- `requirements.txt` - All dependencies

### ✅ Core Application Files
- `app/main.py` - FastAPI entry point with lifespan
- `app/api/routes/health.py` - Health check endpoints
- `app/api/routes/interview.py` - Interview CRUD (placeholders)
- `app/api/routes/webhooks.py` - Telnyx/Calendly webhooks (placeholders)

### ✅ Existing Services (Copied from your uploads)
- `app/services/telnyx_service.py` - ✅ Ready to use
- `app/services/deepgram_service.py` - ✅ Ready to use
- `app/services/google_tts_service.py` - ✅ Ready to use
- `app/database/models.py` - ✅ SQLAlchemy models
- `app/database/milvus_client.py` - ✅ Milvus schema
- `scripts/setup_milvus.py` - ✅ Collection creation

### 📝 Documentation
- `README.md` - Full project overview
- `PROJECT_STRUCTURE.md` - Complete file listing with status
- `IMPLEMENTATION_ROADMAP.md` - This file

## Implementation Order

### Week 1: Core Infrastructure

#### Day 1-2: Database Setup
- [ ] Implement `app/database/postgres.py`
  - AsyncPG connection pool
  - SQLAlchemy async session
- [ ] Implement `app/database/repositories/interview_repository.py`
  - CRUD operations for interviews table
- [ ] Implement `app/database/repositories/transcript_repository.py`
  - CRUD for interview_transcripts table
- [ ] Test: `python scripts/setup_database.py`

#### Day 3-4: State Machine & Audio
- [ ] Implement `app/core/state_machine.py`
  - InterviewState enum management
  - State transitions
  - State validation
- [ ] Implement `app/core/audio_processor.py`
  - Audio buffering (20ms chunks → 100ms)
  - Format conversion (mulaw → PCM)
  - VAD (Voice Activity Detection) prep
- [ ] Implement `app/websocket/audio_buffer.py`
  - Ring buffer for audio
  - Thread-safe operations

#### Day 5-7: WebSocket Handler
- [ ] Implement `app/websocket/telnyx_media_handler.py`
  - Accept WebSocket connection
  - Receive base64 audio from Telnyx
  - Send audio back to Telnyx
  - Handle connection lifecycle
- [ ] Implement `app/websocket/connection_manager.py`
  - Manage multiple concurrent calls
  - Call state tracking
- [ ] Test: Mock WebSocket connection

### Week 2: LLM & Decision Making

#### Day 8-9: LLM Service
- [ ] Implement `app/services/llm_service.py`
  - OpenAI GPT-4o-mini client
  - Context management
  - Token counting
  - Retry logic with exponential backoff
- [ ] Implement `prompts/system_prompts.py`
  - Main assistant system prompt
  - Decision-making prompt
- [ ] Test: LLM responses with sample contexts

#### Day 10-11: Question Generation
- [ ] Implement `app/ai/question_generator.py`
  - Parse JD + candidate profile
  - Generate 6 personalized questions
  - Prioritize questions (critical/high/medium)
- [ ] Implement `prompts/question_generation.py`
  - Question generation templates
- [ ] Test: Generate questions from sample JD

#### Day 12-14: Decision Engine
- [ ] Implement `app/ai/decision_engine.py`
  - Analyze candidate utterance
  - Decide: followup/next/acknowledge/skip
  - Return JSON decision
- [ ] Implement `prompts/decision_prompts.py`
  - Decision templates
  - Candidate question routing
- [ ] Test: Decision making with sample Q&A

### Week 3: Interview Orchestration

#### Day 15-17: Conversation Manager
- [ ] Implement `app/core/conversation_manager.py`
  - Manage conversation flow
  - Track turn history
  - Apply silence thresholds
  - Handle interruptions
- [ ] Implement `app/ai/answer_analyzer.py`
  - Quality scoring (0-1)
  - Completeness assessment
  - Off-topic detection
- [ ] Test: Conversation flow simulation

#### Day 18-21: Interview Orchestrator
- [ ] Implement `app/core/interview_orchestrator.py`
  - **THE MAIN COORDINATOR**
  - Initialize interview session
  - Coordinate all services:
    - Telnyx (call control)
    - WebSocket (audio stream)
    - Deepgram (STT)
    - LLM (decisions)
    - TTS (responses)
  - Execute state machine
  - Handle errors
- [ ] Implement error recovery logic
- [ ] Test: End-to-end interview simulation (text mode)

### Week 4: Skills & Evaluation

#### Day 22-23: Skills Extraction
- [ ] Implement `app/ai/skills_extractor.py`
  - Extract technical skills from text
  - Match against JD requirements
  - Confidence scoring
- [ ] Test: Skills extraction accuracy

#### Day 24-25: Evaluation Engine
- [ ] Implement `app/ai/evaluation_engine.py`
  - Analyze full transcript
  - Skills coverage
  - Technical depth
  - Fit assessment (strong/good/weak)
- [ ] Implement `prompts/evaluation_prompts.py`
- [ ] Test: Evaluation on sample transcripts

#### Day 26-28: Embedding & Milvus
- [ ] Implement `app/services/embedding_service.py`
  - Load e5-base-v2 model
  - Generate embeddings
  - Batch processing
- [ ] Implement `app/tasks/embedding_tasks.py`
  - Celery task for post-interview
  - Embed Q&A pairs
  - Insert to Milvus
- [ ] Implement `app/tasks/celery_app.py`
  - Celery configuration
- [ ] Test: Embedding pipeline

### Week 5: API & Integration

#### Day 29-31: Schemas & API Routes
- [ ] Implement `app/schemas/interview.py`
  - CreateInterviewRequest
  - InterviewResponse
  - InterviewStatusResponse
- [ ] Implement `app/schemas/conversation.py`
  - ConversationTurn
  - QAPair
- [ ] Implement `app/schemas/candidate.py`
  - CandidateInput
- [ ] Implement `app/schemas/webhooks.py`
  - TelnyxWebhookPayload
- [ ] Complete `app/api/routes/interview.py`
  - create_interview
  - get_interview
  - get_status
- [ ] Complete `app/api/routes/webhooks.py`
  - Handle call events
  - Handle SMS responses
  - Handle Calendly bookings

#### Day 32-33: Utilities
- [ ] Implement `app/utils/logger.py`
  - Structured logging
  - Log levels
  - File rotation
- [ ] Implement `app/utils/audio_utils.py`
  - Format conversions
  - Audio validation
- [ ] Implement `app/utils/time_utils.py`
  - Time tracking
  - Duration formatting
- [ ] Implement `app/utils/validators.py`
  - Phone number validation
  - Email validation

#### Day 34-35: Background Tasks
- [ ] Implement `app/tasks/post_interview.py`
  - Process completed interview
  - Generate evaluation
  - Trigger embeddings
  - Notify recruiter
- [ ] Test: Full background pipeline

### Week 6: Testing & Deployment

#### Day 36-38: Testing
- [ ] Write unit tests
  - `tests/unit/test_question_generator.py`
  - `tests/unit/test_answer_analyzer.py`
  - `tests/unit/test_decision_engine.py`
  - `tests/unit/test_state_machine.py`
- [ ] Write integration tests
  - `tests/integration/test_interview_flow.py`
  - `tests/integration/test_telnyx_webhooks.py`
  - `tests/integration/test_websocket_handler.py`
- [ ] End-to-end test with real Telnyx call

#### Day 39-40: Documentation
- [ ] Write `docs/API.md`
- [ ] Write `docs/ARCHITECTURE.md`
- [ ] Write `docs/STATE_MACHINE.md`
- [ ] Write `docs/WEBSOCKET_FLOW.md`
- [ ] Write `docs/DEPLOYMENT.md`

#### Day 41-42: Deployment
- [ ] Set up production environment
- [ ] Configure secrets
- [ ] Deploy to cloud
- [ ] Set up monitoring
- [ ] Load testing

## Quick Start Commands

### Development Setup
```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup databases
python scripts/setup_database.py
python scripts/setup_milvus.py

# Run application
uvicorn app.main:app --reload
```

### Docker Setup
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Testing
```bash
# Run all tests
pytest

# Run specific test
pytest tests/unit/test_question_generator.py

# With coverage
pytest --cov=app tests/
```

## Key Files to Implement First

### Priority 1 (Week 1)
1. `app/database/postgres.py`
2. `app/core/state_machine.py`
3. `app/websocket/telnyx_media_handler.py`

### Priority 2 (Week 2)
4. `app/services/llm_service.py`
5. `app/ai/question_generator.py`
6. `app/ai/decision_engine.py`

### Priority 3 (Week 3)
7. `app/core/interview_orchestrator.py`
8. `app/core/conversation_manager.py`

## Cost Tracking

At 250 interviews/month (12 mins each = 3,000 mins):
- Telnyx: $25.50
- Deepgram: $12.90
- Google TTS: $18
- OpenAI: ~$0.50
- **Total: ~$57/month**

Compare to Bland AI: $270/month

**Savings: $213/month or $2,556/year**

## Success Metrics

- ✅ Interview completion rate > 90%
- ✅ Average latency < 2 seconds
- ✅ STT confidence > 0.85
- ✅ Question quality score > 0.80
- ✅ Evaluation accuracy > 85%
- ✅ System uptime > 99%

## Next Steps

1. **Set up your environment**
   ```bash
   cd recruiterbrain-voice-interview
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start with database setup**
   - Implement `app/database/postgres.py`
   - Run `scripts/setup_database.py`

3. **Test existing services**
   ```bash
   python -c "from app.services.telnyx_service import TelnyxService; print('✅ Telnyx imported')"
   python -c "from app.services.deepgram_service import DeepgramSTTService; print('✅ Deepgram imported')"
   python -c "from app.services.google_tts_service import GoogleTTSService; print('✅ Google TTS imported')"
   ```

4. **Follow the week-by-week roadmap above**

## Questions?

Refer to:
- `PROJECT_STRUCTURE.md` - Complete file listing
- `README.md` - Project overview
- Your original uploaded files in `/mnt/user-data/uploads/`

---

**You're all set! Start with Week 1, Day 1. Good luck! 🚀**
