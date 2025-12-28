# Project Structure - RecruiterBrain Voice Interview

## Overview
Complete file structure for the AI Voice Interview system.

## Directory Tree

```
recruiterbrain-voice-interview/
│
├── README.md                          # Project overview and quick start
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── requirements.txt                    # Python dependencies
├── docker-compose.yml                  # Docker services configuration
├── Dockerfile                          # Application container
│
├── config/                             # Configuration modules
│   ├── __init__.py
│   ├── settings.py                    # ✅ CREATED - Environment settings
│   └── constants.py                   # ✅ CREATED - Static constants
│
├── app/                                # Main application
│   ├── __init__.py
│   ├── main.py                        # ✅ CREATED - FastAPI entry point
│   │
│   ├── api/                            # API layer
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── interview.py           # ✅ CREATED - Interview CRUD endpoints
│   │   │   ├── webhooks.py            # ✅ CREATED - Telnyx & Calendly webhooks
│   │   │   └── health.py              # ✅ CREATED - Health check
│   │   │
│   │   └── dependencies.py            # TODO - FastAPI dependencies
│   │
│   ├── core/                           # Core business logic
│   │   ├── __init__.py
│   │   ├── interview_orchestrator.py  # TODO - Main interview coordinator
│   │   ├── conversation_manager.py    # TODO - Conversation flow management
│   │   ├── state_machine.py           # TODO - Interview state management
│   │   └── audio_processor.py         # TODO - Audio buffering/processing
│   │
│   ├── services/                       # External service integrations
│   │   ├── __init__.py
│   │   ├── telnyx_service.py          # 📁 COPY FROM UPLOAD - Telnyx integration
│   │   ├── deepgram_service.py        # 📁 COPY FROM UPLOAD - Deepgram STT
│   │   ├── google_tts_service.py      # 📁 COPY FROM UPLOAD - Google TTS
│   │   ├── llm_service.py             # TODO - OpenAI/LLM service
│   │   ├── embedding_service.py       # TODO - e5-base-v2 embeddings
│   │   └── audio_service.py           # TODO - Audio conversion utilities
│   │
│   ├── ai/                             # AI/ML logic
│   │   ├── __init__.py
│   │   ├── question_generator.py      # TODO - Generate questions from JD
│   │   ├── answer_analyzer.py         # TODO - Analyze answer quality
│   │   ├── skills_extractor.py        # TODO - Extract skills from text
│   │   ├── decision_engine.py         # TODO - LLM decision making
│   │   └── evaluation_engine.py       # TODO - Post-interview evaluation
│   │
│   ├── database/                       # Database layer
│   │   ├── __init__.py
│   │   ├── postgres.py                # TODO - PostgreSQL connection
│   │   ├── milvus_client.py           # TODO - Milvus operations
│   │   ├── models.py                  # 📁 COPY FROM UPLOAD - SQLAlchemy models
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── interview_repository.py     # TODO - Interview DB operations
│   │       ├── transcript_repository.py    # TODO - Transcript DB operations
│   │       └── consent_repository.py       # TODO - Consent DB operations
│   │
│   ├── websocket/                      # WebSocket handlers
│   │   ├── __init__.py
│   │   ├── telnyx_media_handler.py    # TODO - Handle Telnyx media stream
│   │   ├── connection_manager.py      # TODO - WebSocket connections
│   │   └── audio_buffer.py            # TODO - Audio buffering logic
│   │
│   ├── schemas/                        # Pydantic models
│   │   ├── __init__.py
│   │   ├── interview.py               # TODO - Interview request/response schemas
│   │   ├── conversation.py            # TODO - Conversation turn schemas
│   │   ├── candidate.py               # TODO - Candidate input schema
│   │   └── webhooks.py                # TODO - Webhook payload schemas
│   │
│   ├── utils/                          # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py                  # TODO - Logging configuration
│   │   ├── audio_utils.py             # TODO - Audio format conversion
│   │   ├── time_utils.py              # TODO - Time tracking utilities
│   │   └── validators.py              # TODO - Input validation
│   │
│   └── tasks/                          # Background tasks
│       ├── __init__.py
│       ├── celery_app.py              # TODO - Celery configuration
│       ├── post_interview.py          # TODO - Post-interview processing
│       └── embedding_tasks.py         # TODO - Milvus embedding tasks
│
├── scripts/                            # Utility scripts
│   ├── setup_database.py              # TODO - Initialize PostgreSQL
│   ├── setup_milvus.py                # 📁 ADAPT FROM UPLOAD - Create Milvus collection
│   ├── test_telnyx.py                 # TODO - Test Telnyx integration
│   ├── test_call_flow.py              # TODO - Simulate interview
│   └── migrate_data.py                # TODO - Data migrations
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # TODO - Pytest fixtures
│   ├── unit/
│   │   ├── test_llm_service.py
│   │   ├── test_question_generator.py
│   │   ├── test_answer_analyzer.py
│   │   └── test_state_machine.py
│   │
│   └── integration/
│       ├── test_interview_flow.py
│       ├── test_telnyx_webhooks.py
│       └── test_websocket_handler.py
│
├── prompts/                            # LLM prompts
│   ├── system_prompts.py              # TODO - System prompts for AI assistant
│   ├── question_generation.py         # TODO - Question generation templates
│   ├── decision_prompts.py            # TODO - Decision-making prompts
│   └── evaluation_prompts.py          # TODO - Evaluation prompts
│
├── static/                             # Static files
│   └── audio/
│       ├── greeting.wav               # TODO - Pre-generated greetings
│       ├── acknowledgments/
│       │   ├── great.wav              # TODO - Common acknowledgments
│       │   ├── interesting.wav
│       │   └── thank_you.wav
│       └── prompts/
│           ├── take_your_time.wav
│           └── are_you_there.wav
│
└── docs/                               # Documentation
    ├── API.md                          # TODO - API documentation
    ├── ARCHITECTURE.md                 # TODO - System architecture
    ├── DEPLOYMENT.md                   # TODO - Deployment guide
    ├── WEBSOCKET_FLOW.md               # TODO - WebSocket handling
    └── STATE_MACHINE.md                # TODO - State machine docs
```

## File Status Legend
- ✅ CREATED - File has been created with structure
- 📁 COPY FROM UPLOAD - Copy from your uploaded files
- 📁 ADAPT FROM UPLOAD - Adapt from your uploaded files
- TODO - Needs to be implemented

## Next Steps

### Phase 1: Copy Existing Files
1. Copy `telnyx_service.py` from uploads
2. Copy `deepgram_service.py` from uploads
3. Copy `google_tts_service.py` from uploads
4. Copy `interview_models.py` → `app/database/models.py`
5. Adapt `interview_milvus_schema.py` → `scripts/setup_milvus.py`
6. Adapt `create_interview_collection.py` → include in setup_milvus.py

### Phase 2: Core Implementation (Week 1-2)
1. Implement `interview_orchestrator.py` - Main coordinator
2. Implement `state_machine.py` - Interview states
3. Implement `telnyx_media_handler.py` - WebSocket audio
4. Implement `audio_processor.py` - Audio buffering
5. Implement `conversation_manager.py` - Flow control

### Phase 3: AI Layer (Week 2-3)
1. Implement `question_generator.py` - From JD → questions
2. Implement `llm_service.py` - OpenAI integration
3. Implement `decision_engine.py` - LLM decisions
4. Implement `answer_analyzer.py` - Quality assessment
5. Implement `skills_extractor.py` - Skill extraction

### Phase 4: Data Layer (Week 3-4)
1. Implement `postgres.py` - DB connection
2. Implement `milvus_client.py` - Vector DB
3. Implement repositories (interview, transcript, consent)
4. Implement `embedding_service.py` - e5-base-v2
5. Implement `embedding_tasks.py` - Background jobs

### Phase 5: API & Integration (Week 4-5)
1. Implement all schemas (Pydantic models)
2. Complete API routes
3. Complete webhook handlers
4. Implement Celery tasks
5. Add utilities (logger, audio_utils, etc.)

### Phase 6: Testing & Polish (Week 5-6)
1. Unit tests
2. Integration tests
3. End-to-end interview simulation
4. Documentation
5. Deployment scripts

## Key Integration Points

### Input Flow
```
User API Call
  → create_interview(candidate_data, jd_summary)
  → question_generator.generate_questions()
  → interview_repository.create()
  → telnyx_service.send_consent_sms()
```

### Interview Flow
```
Telnyx Webhook: call.answered
  → interview_orchestrator.start_interview()
  → state_machine: GREETING → ASKING_QUESTION → LISTENING
  → telnyx_media_handler receives audio
  → deepgram_service.transcribe()
  → decision_engine.decide_next_action()
  → google_tts_service.synthesize()
  → telnyx_service.play_audio()
  → Loop until complete
```

### Post-Interview Flow
```
Telnyx Webhook: call.ended
  → celery: post_interview.process()
  → embedding_service.embed_qa_pairs()
  → milvus_client.insert()
  → evaluation_engine.evaluate()
  → interview_repository.update()
```

## Dependencies Between Modules

### Critical Path (must implement in order):
1. Database models & connections
2. Telnyx/Deepgram/TTS services (already have)
3. State machine
4. Audio processor + WebSocket handler
5. LLM decision engine
6. Interview orchestrator (ties everything together)

### Can be implemented in parallel:
- Question generator
- Answer analyzer
- Skills extractor
- Evaluation engine
- Embedding tasks
- API routes (mock orchestrator first)
