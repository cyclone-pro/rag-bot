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
from livekit.protocol import agent

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
        candidate_name = "Atif"
        job_title = "Software Engineer"
        
        # Build system prompt
        system_prompt = f"""You are Ava, an AI technical recruiter conducting a phone interview for a {job_title} position.

Your role:
- Conduct a friendly, professional technical interview
- Ask thoughtful questions about the candidate's experience
- Listen actively and ask follow-up questions
- Keep responses concise and natural
- Naturally conclude the interview when you have enough information

Interview structure:
1. Welcome the candidate warmly
2. Ask about their background and experience (2-3 questions)
3. Discuss specific technical skills (2-3 questions)
4. Ask about past projects (1-2 questions)
5. Answer any questions they have
6. Thank them for their time and conclude

Conversation style:
- Be warm, professional, and encouraging
- Keep responses to 2-3 sentences maximum
- Use natural language, avoid being robotic
- Show genuine interest in their answers
- After covering the main topics (typically 8-12 exchanges), naturally wrap up the interview

Ending the interview:
When you've gathered enough information about their background, skills, and experience:
- Thank them sincerely for their time
- Let them know you appreciate them speaking with you
- Mention that the team will be in touch
- Wish them well

Example closing: "Thank you so much for your time today, {candidate_name}. We really appreciate you speaking with us. The team will review your responses and we'll be in touch soon. Have a great day!"

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
            voice="nova",
            model="tts-1",
            speed=0.9  # Options: alloy, echo, fable, onyx, nova, shimmer
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
        
        # Track conversation state
        turns_count = 0
        interview_ending = False
        goodbye_detected_time = None
        
        @assistant.on("agent_speech_committed")
        def on_agent_speech(msg):
            nonlocal turns_count, interview_ending, goodbye_detected_time
            turns_count += 1
            print(f"💬 Agent turn {turns_count}: {msg.text[:50]}...")
            
            # Check for goodbye phrases that indicate interview is ending
            goodbye_phrases = [
                "thank you for your time",
                "we'll be in touch",
                "have a great day",
                "thanks for speaking with us",
                "appreciate you speaking with us",
                "we'll reach out",
                "hear back from us",
                "that concludes",
                "end of our interview",
                "best of luck"
            ]
            
            msg_lower = msg.text.lower()
            if any(phrase in msg_lower for phrase in goodbye_phrases):
                print(f"👋 AI said goodbye: '{msg.text[:100]}...'")
                interview_ending = True
                goodbye_detected_time = asyncio.get_event_loop().time()
        
        # Monitor for completion
        start_time = asyncio.get_event_loop().time()
        
        while True:
            await asyncio.sleep(1)
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - start_time
            
            # PRIMARY: AI said goodbye - end after 5 seconds
            if interview_ending and goodbye_detected_time:
                time_since_goodbye = current_time - goodbye_detected_time
                if time_since_goodbye >= 5:
                    print(f"✅ Interview ended naturally (AI said goodbye)")
                    break
            
            # SAFETY: 15 minutes absolute max
            if elapsed > 900:
                print("⏰ Max time reached (15 min) - forcing end")
                await assistant.say("Thank you so much for your time today. We really appreciate you speaking with us. Have a great day!")
                await asyncio.sleep(3)
                break
            
            # SAFETY: Participant left
            if len(ctx.room.remote_participants) == 0:
                print("📞 Participant disconnected")
                await asyncio.sleep(2)
                break
        
        # End the call properly
        logger.info(f"🔚 Ending interview {interview_id}")
        print("🔚 Disconnecting call...")
        
        # Disconnect all participants
        await ctx.room.disconnect()
        
        logger.info(f"✅ Interview {interview_id} completed and call ended")
        
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
        ),
    )