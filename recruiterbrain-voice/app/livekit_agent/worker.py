"""
LiveKit Voice Agent Worker - LiveKit 0.12.0 Compatible
Handles phone interviews with AI voice assistant
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents import voice_assistant
from livekit.plugins import deepgram, openai, silero

from app.config.settings import settings


logger = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext):
    """Main agent entry point - called when LiveKit dispatches a job"""
    
    # CRITICAL DEBUG LOGGING
    print(f"🔥 ENTRYPOINT CALLED! Room: {ctx.room.name}")
    logger.info(f"🔥 ENTRYPOINT CALLED! Room: {ctx.room.name}")
    
    try:
        logger.info(f"🎤 Agent started for room: {ctx.room.name}")
        print(f"🎤 Agent started for room: {ctx.room.name}")
        
        # Extract interview metadata
        import json
        logger.info("📦 Extracting metadata...")
        print("📦 Extracting metadata...")
        
        metadata = ctx.job.room.metadata
        if isinstance(metadata, str) and metadata.strip():
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON metadata: {metadata}")
                metadata = {}
        elif not metadata:
            metadata = {}
        
        interview_id = metadata.get("interview_id", "unknown")
        logger.info(f"📋 Interview ID: {interview_id}")
        print(f"📋 Interview ID: {interview_id}")
        
        # Use default interview data (database loading disabled)
        candidate_name = "Candidate"
        job_title = "Software Engineer"
        
        # Build system prompt
        system_prompt = f"""You are Ava, an AI technical recruiter conducting a phone interview for a {job_title} position.

Your role:
- Conduct a friendly, professional technical interview
- Ask thoughtful questions about the candidate's experience
- Listen actively and ask follow-up questions
- Keep responses concise and natural

Interview structure:
1. Welcome the candidate warmly
2. Ask about their background and experience
3. Discuss specific technical skills
4. Ask about past projects
5. Answer any questions they have
6. Thank them for their time

Conversation style:
- Be warm, professional, and encouraging
- Keep responses to 2-3 sentences
- Use natural language, avoid being robotic
- Show genuine interest in their answers

Start by greeting {candidate_name} and asking them to tell you about themselves."""

        # Initialize components
        logger.info("🔧 Initializing voice pipeline components...")
        print("🔧 Initializing voice pipeline components...")
        
        # Speech-to-Text: Deepgram
        print("📝 Creating Deepgram STT...")
        stt = deepgram.STT(
            model="nova-2",
            language="en-US"
        )
        print("✅ Deepgram STT created")
        
        # Text-to-Speech: OpenAI
        print("🔊 Creating OpenAI TTS...")
        tts = openai.TTS(
            voice="alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
        )
        print("✅ OpenAI TTS created")
        
        # LLM: OpenAI
        print("🧠 Creating OpenAI LLM...")
        assistant_llm = openai.LLM(
            model="gpt-4o-mini"
        )
        print("✅ OpenAI LLM created")
        
        # VAD: Silero
        print("🎙️ Loading Silero VAD...")
        vad = silero.VAD.load()
        print("✅ Silero VAD loaded")
        
        logger.info("✅ Components initialized successfully")
        print("✅ Components initialized successfully")
        
        # Create initial chat context
        print("💬 Creating chat context...")
        initial_ctx = llm.ChatContext()
        initial_ctx.messages.append(
            llm.ChatMessage(
                role="system",
                content=system_prompt
            )
        )
        
        logger.info("💬 Chat context created")
        print("💬 Chat context created")
        
        # Create voice assistant
        print("🤖 Creating voice assistant...")
        assistant = voice_assistant.VoiceAssistant(
            vad=vad,
            stt=stt,
            llm=assistant_llm,
            tts=tts,
            chat_ctx=initial_ctx
        )
        
        logger.info("🤖 Voice assistant created")
        print("🤖 Voice assistant created")
        
        # Connect to room first
        print("🔗 Connecting to room...")
        await ctx.connect()
        logger.info("🔗 Connected to room")
        print("🔗 Connected to room")
        
        # Start the assistant
        print("🚀 Starting voice assistant...")
        assistant.start(ctx.room)
        
        logger.info("🚀 Voice assistant started successfully!")
        print("🚀 Voice assistant started successfully!")
        logger.info(f"📞 Ready for interview {interview_id}")
        
        # Wait for SIP participant to join
        print("⏳ Waiting for SIP participant to join...")
        
        @ctx.room.on("participant_connected")
        def on_participant_connected(participant):
            print(f"👤 Participant connected: {participant.identity}")
            logger.info(f"👤 Participant connected: {participant.identity}")
        
        # Greeting - wait a bit for connection to stabilize
        await asyncio.sleep(2.0)
        
        greeting = (
            f"Hello! Thank you for taking the time to interview with us today. "
            f"I'm Ava, and I'll be conducting your interview for the {job_title} position. "
            f"This should take about 10 to 15 minutes. "
            f"To start, could you please tell me a bit about yourself and your background?"
        )
        
        print("👋 Sending greeting...")
        await assistant.say(greeting, allow_interruptions=True)
        
        logger.info("👋 Greeting sent to candidate")
        
        # Wait for interview to complete
        await asyncio.sleep(900)  # 15 minutes max
        
        # Wrap up
        logger.info(f"✅ Interview {interview_id} completed")
        
    except Exception as e:
        logger.error(f"❌ Error in interview: {e}", exc_info=True)
        raise


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