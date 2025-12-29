"""
LiveKit Voice Agent Worker
Handles phone interviews with:
- Deepgram for Speech-to-Text (Nova-2)
- Google Cloud TTS for Text-to-Speech (Neural2/WaveNet)
- OpenAI GPT-4o-mini for conversation logic
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, google, openai, silero

from app.config.settings import settings
from app.services.interview_service import (
    get_session,
    complete_interview_session
)


logger = logging.getLogger(__name__)


# ==========================================
# Agent Entry Point
# ==========================================

async def entrypoint(ctx: JobContext):
    """
    Main agent entry point
    Called when LiveKit dispatches a job to this worker
    """
    
    logger.info(f"Agent started for room: {ctx.room.name}")
    
    # Extract interview metadata from room metadata
    metadata = ctx.room.metadata
    interview_id = metadata.get("interview_id")
    
    if not interview_id:
        logger.error("No interview_id in room metadata!")
        return
    
    # Get interview session from memory
    session = get_session(interview_id)
    
    if not session:
        logger.error(f"Interview session {interview_id} not found!")
        return
    
    logger.info(f"Starting interview: {interview_id}")
    
    # Mark interview as started
    session.started_at = datetime.utcnow()
    
    # ==========================================
    # Initialize Voice Pipeline
    # ==========================================
    
    # Speech-to-Text: Deepgram Nova-2
    stt = deepgram.STT(
        api_key=settings.deepgram_api_key,
        model=settings.deepgram_model,
        language=settings.deepgram_language,
        smart_format=True,
        punctuate=True,
        keywords=[  # Boost technical terms
            "python", "javascript", "react", "fastapi", "postgresql",
            "kubernetes", "docker", "aws", "microservices", "api"
        ]
    )
    
    # Text-to-Speech: Google Cloud Neural2/WaveNet
    tts = google.TTS(
        credentials_file=settings.google_application_credentials,
        voice=settings.google_tts_voice_name,
        language=settings.google_tts_language_code,
        speaking_rate=settings.google_tts_speaking_rate,
        pitch=settings.google_tts_pitch
    )
    
    # LLM: OpenAI GPT-4o-mini
    llm = openai.LLM(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.openai_temperature
    )
    
    # VAD: Silero for voice activity detection
    vad = silero.VAD.load()
    
    # ==========================================
    # Create Voice Assistant
    # ==========================================
    
    # Build system prompt with interview context
    system_prompt = build_interview_prompt(
        candidate_data=session.candidate_data,
        jd_data=session.jd_data
    )
    
    assistant = agents.VoiceAssistant(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        
        # Conversation settings
        chat_ctx=agents.ChatContext(
            instructions=system_prompt,
            messages=[]
        ),
        
        # Interrupt settings
        allow_interruptions=True,
        interrupt_speech_duration=0.6,
        interrupt_min_words=3,
        
        # Response settings
        min_endpointing_delay=0.5,
        
        # Callbacks
        before_llm_cb=before_llm_callback,
        before_tts_cb=before_tts_callback,
    )
    
    # ==========================================
    # Start Assistant
    # ==========================================
    
    assistant.start(ctx.room)
    
    logger.info(f"Voice assistant started for interview {interview_id}")
    
    # Greet the candidate
    await asyncio.sleep(1.0)  # Let connection stabilize
    
    greeting = (
        f"Hello {session.candidate_data.get('name', 'there')}! "
        f"Thank you for taking the time to interview for the {session.jd_data.get('title', 'position')}. "
        f"This will be a {settings.interview_questions_count}-question technical interview "
        f"and should take about {settings.interview_max_duration_seconds // 60} minutes. "
        f"I'll be asking you questions about your experience and skills. "
        f"Feel free to take your time with your answers. Are you ready to begin?"
    )
    
    await assistant.say(greeting, allow_interruptions=True)
    
    # ==========================================
    # Interview Loop
    # ==========================================
    
    try:
        # Wait for interview completion or timeout
        await asyncio.wait_for(
            assistant.aclose(),
            timeout=settings.interview_max_duration_seconds
        )
    except asyncio.TimeoutError:
        logger.warning(f"Interview {interview_id} timed out")
        await assistant.say("Thank you for your time. We've reached the end of our interview. Goodbye!")
    
    # ==========================================
    # Save Interview Results
    # ==========================================
    
    # Get conversation history
    conversation_log = []
    for msg in assistant.chat_ctx.messages:
        conversation_log.append({
            "role": msg.role,
            "content": msg.content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Extract full transcript text
    full_transcript = "\n\n".join([
        f"{msg.role.upper()}: {msg.content}"
        for msg in assistant.chat_ctx.messages
    ])
    
    # Compute evaluation (you can enhance this with more sophisticated analysis)
    evaluation = await compute_evaluation(
        conversation_log=conversation_log,
        full_transcript=full_transcript,
        session=session
    )
    
    # Save to database (batch write)
    call_duration = int((datetime.utcnow() - session.started_at).total_seconds())
    
    await complete_interview_session(
        interview_id=interview_id,
        evaluation=evaluation
    )
    
    logger.info(f"Interview {interview_id} completed and saved")


# ==========================================
# Helper Functions
# ==========================================

def build_interview_prompt(
    candidate_data: Dict[str, Any],
    jd_data: Dict[str, Any]
) -> str:
    """Build system prompt for interview agent"""
    
    candidate_name = candidate_data.get("name", "the candidate")
    job_title = jd_data.get("title", "this position")
    requirements = jd_data.get("requirements", [])
    candidate_skills = candidate_data.get("skills", [])
    candidate_projects = candidate_data.get("projects", [])
    
    prompt = f"""You are Ava, an AI technical recruiter conducting a phone interview for the {job_title} position.

CANDIDATE INFORMATION:
- Name: {candidate_name}
- Skills: {', '.join(candidate_skills) if candidate_skills else 'Not specified'}
- Past Projects: {' | '.join(candidate_projects[:3]) if candidate_projects else 'Not specified'}

JOB REQUIREMENTS:
{chr(10).join([f'- {req}' for req in requirements[:5]])}

YOUR ROLE:
You are conducting a {settings.interview_questions_count}-question technical interview. Your goal is to:
1. Assess the candidate's technical skills and experience
2. Evaluate cultural fit and communication skills
3. Determine if they meet the job requirements

INTERVIEW STRUCTURE:
Ask {settings.interview_questions_count} questions total, covering:
1. Technical background and experience
2. Problem-solving abilities
3. Specific skills mentioned in the JD
4. Past project details
5. Scenario-based questions
6. Final thoughts and questions

CONVERSATION STYLE:
- Be professional yet warm and encouraging
- Listen actively and ask follow-up questions when needed
- Keep responses concise (2-3 sentences max)
- Use natural acknowledgments: "I see", "That's interesting", "Tell me more"
- If the candidate gives a short answer, probe deeper with "Can you elaborate on that?"
- Track time: Aim for ~2 minutes per question

IMPORTANT RULES:
- ONE question at a time
- Wait for complete answers before moving on
- Don't rush the candidate
- Don't interrupt with new questions
- If they're struggling, offer to move to the next question
- After {settings.interview_questions_count} questions, wrap up warmly

Begin with a brief introduction, then start with question 1."""
    
    return prompt


async def before_llm_callback(assistant: agents.VoiceAssistant, chat_ctx: agents.ChatContext):
    """
    Called before LLM generates response
    Use this to track conversation progress
    """
    # Count questions asked
    messages = chat_ctx.messages
    
    # Log for debugging
    logger.debug(f"Messages so far: {len(messages)}")


async def before_tts_callback(assistant: agents.VoiceAssistant, text: str):
    """
    Called before TTS converts text to speech
    Use this to modify or log what the agent will say
    """
    logger.debug(f"Agent will say: {text[:100]}...")


async def compute_evaluation(
    conversation_log: list,
    full_transcript: str,
    session: Any
) -> Dict[str, Any]:
    """
    Compute interview evaluation
    
    This is a simple version - you can enhance with:
    - LLM-based analysis
    - Skills extraction
    - Sentiment analysis
    - Answer quality scoring
    """
    
    # Count questions
    questions_asked = sum(1 for msg in conversation_log if msg["role"] == "assistant" and "?" in msg["content"])
    answers_given = sum(1 for msg in conversation_log if msg["role"] == "user")
    
    # Simple scoring (enhance this!)
    base_score = min(1.0, answers_given / settings.interview_questions_count)
    
    # Extract skills mentioned (simple keyword matching)
    skills_discussed = []
    for skill in session.candidate_data.get("skills", []):
        if skill.lower() in full_transcript.lower():
            skills_discussed.append(skill)
    
    evaluation = {
        "score": base_score,
        "sentiment_score": 0.5,  # Neutral (enhance with actual sentiment analysis)
        "questions_asked": questions_asked,
        "questions_completed": answers_given,
        "summary": f"Candidate completed {answers_given} out of {settings.interview_questions_count} questions.",
        "skills_discussed": skills_discussed,
        "skills_coverage": {
            "required": session.jd_data.get("requirements", []),
            "demonstrated": skills_discussed,
            "missing": list(set(session.jd_data.get("requirements", [])) - set(skills_discussed))
        },
        "fit_assessment": "Pending detailed review",
        "keyword_matches": {}
    }
    
    return evaluation


# ==========================================
# Worker Entry Point
# ==========================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run worker
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        )
    )
