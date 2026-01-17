"""
Enhanced LiveKit Voice Agent Worker
- Real-time conversation logging
- Personalized prompts from candidate data  
- Automatic call termination
- Raw conversation save (processing done in background)
"""

import logging
import asyncio
import json
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from livekit import api 

load_dotenv()

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents import voice_assistant
from livekit.plugins import deepgram, openai, silero

from app.config.settings import settings
from app.services.conversation_logger import ConversationLogger

logger = logging.getLogger(__name__)


def build_personalized_prompt(
    candidate_name: str,
    candidate_company: str,
    resume_summary: str,
    job_title: str,
    job_description: str,
    required_skills: list
) -> str:
    """Build dynamic system prompt based on candidate data"""
    
    skills_str = ", ".join(required_skills[:5]) if required_skills else "technical skills"
    
    prompt = f"""You are Ava, an AI technical recruiter conducting a phone interview for the {job_title} position.

CANDIDATE INFORMATION:
- Name: {candidate_name}
- Current Company: {candidate_company}
- Background: {resume_summary}

JOB DESCRIPTION:
{job_description}

KEY SKILLS TO ASSESS:
{skills_str}

Your role:
- Conduct a friendly, professional technical interview
- Ask targeted questions based on the candidate's background
- Assess their fit for the {job_title} role
- Explore their experience with: {skills_str}
- Keep responses concise and natural

Interview structure:
1. Welcome {candidate_name} warmly and mention you reviewed their background at {candidate_company}
2. Ask 2-3 questions about their current work and relevant experience
3. Deep-dive into 2-3 technical areas from: {skills_str}
4. Ask about 1-2 specific projects that demonstrate their skills
5. Answer any questions they have about the role
6. Naturally conclude when you've gathered sufficient information

Conversation style:
- Be warm, professional, and encouraging
- Keep responses to 2-3 sentences maximum
- Use natural language, avoid being robotic
- Reference their background to show you've reviewed their profile
- Ask follow-up questions based on their answers

Example opening: "Hi {candidate_name}! Thanks for taking the time to speak with me today. I'm Ava, and I'll be conducting your interview for our {job_title} position. I saw you're currently at {candidate_company} - that's great! Before we dive into the technical questions, could you tell me a bit about what you're working on there?"

Ending the interview:
When you've covered their background, technical skills, and project experience:
- Thank them sincerely
- Mention the team will review and be in touch
- Wish them well

Example closing: "Thank you so much for your time today, {candidate_name}. I really enjoyed learning about your work at {candidate_company} and your experience with {skills_str}. The team will review our conversation and we'll be in touch soon. Have a great day!"

Start by greeting {candidate_name} and asking about their current role at {candidate_company}."""

    return prompt


async def entrypoint(ctx: JobContext):
    """Main agent entry point"""
    
    print(f"🔥 ENTRYPOINT CALLED! Room: {ctx.room.name}")
    logger.info(f"🔥 Agent started for room: {ctx.room.name}")
    
    try:
        # Extract metadata
        metadata = ctx.job.room.metadata
        if isinstance(metadata, str) and metadata.strip():
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON")
                metadata = {}
        else:
            metadata = {}
        
        interview_id = metadata.get("interview_id", "unknown")
        candidate_data = metadata.get("candidate", {})
        job_data = metadata.get("job", {})
        
        # Extract data for prompt
        candidate_name = candidate_data.get("name", "Candidate")
        candidate_company = candidate_data.get("current_company", "your current company")
        resume_summary = candidate_data.get("resume_summary", "your background")
        job_title = job_data.get("title", "Software Engineer")
        job_description = job_data.get("description", "this position")
        required_skills = job_data.get("requirements", [])
        
        print(f"📋 Interview: {interview_id}")
        print(f"👤 Candidate: {candidate_name} from {candidate_company}")
        print(f"💼 Position: {job_title}")
        
        # Initialize conversation logger
        conv_logger = ConversationLogger(interview_id, candidate_data, job_data)
        print("📝 Conversation logger initialized")
        
        # Build personalized system prompt
        system_prompt = build_personalized_prompt(
            candidate_name=candidate_name,
            candidate_company=candidate_company,
            resume_summary=resume_summary,
            job_title=job_title,
            job_description=job_description,
            required_skills=required_skills
        )
        
        print("✅ Personalized prompt generated")
        
        # Initialize voice pipeline components
        print("🔧 Initializing components...")
        
        stt = deepgram.STT(model="nova-2", language="en-US")
        
        tts = openai.TTS(
            voice="nova",  # Warm, professional female voice
            model="tts-1",  
            speed=1.0  
        )
        
        assistant_llm = openai.LLM(model="gpt-4o-mini")
        
        vad = silero.VAD.load()
        
        print("✅ Components initialized")
        
        # Create chat context
        initial_ctx = llm.ChatContext()
        initial_ctx.messages.append(
            llm.ChatMessage(role="system", content=system_prompt)
        )
        
        # Create voice assistant
        assistant = voice_assistant.VoiceAssistant(
            vad=vad,
            stt=stt,
            llm=assistant_llm,
            tts=tts,
            chat_ctx=initial_ctx
        )
        
        # Connect and start
        await ctx.connect()
        assistant.start(ctx.room)
        
        print("🚀 Voice assistant started!")
        
        # Event handlers for conversation logging
        @assistant.on("user_speech_committed")
        def on_user_speech(msg):
            """Log candidate responses"""
            conv_logger.log_candidate_turn(msg.content)
        
        # Track interview state
        turns_count = 0
        interview_ending = False
        goodbye_detected_time = None
        
        @assistant.on("agent_speech_committed")
        def on_agent_speech(msg):
            """Log agent questions and detect goodbye"""
            nonlocal turns_count, interview_ending, goodbye_detected_time
            
            # Log the turn
            conv_logger.log_agent_turn(msg.content)
            turns_count += 1
            
            print(f"💬 Turn {turns_count}: {msg.content[:60]}...")
            
            # STRICT goodbye detection - requires ALL 3 conditions
            content_lower = msg.content.lower()
            
            # Condition 1: Has SPECIFIC goodbye phrase (not generic)
            specific_goodbye = any(phrase in content_lower for phrase in [
                "thank you so much for your time today",
                "thanks for your time today",
                "thank you for speaking with me today",
                "i really enjoyed learning about your work",
                "the team will review our conversation",
                "we'll be in touch soon"
            ])
            
            # Condition 2: Has final greeting
            final_greeting = any(phrase in content_lower for phrase in [
                "have a great day",
                "take care",
                "goodbye",
               
            ])
            
            # Condition 3: Enough turns (at least 8 complete exchanges)
            enough_turns = turns_count >= 8
            
            # ALL THREE conditions must be met
            if specific_goodbye and final_greeting and enough_turns:
                print(f"👋 Detected goodbye (turn {turns_count}, duration: {asyncio.get_event_loop().time() - start_time:.1f}s)")
                interview_ending = True
                goodbye_detected_time = asyncio.get_event_loop().time()
            elif final_greeting and not specific_goodbye:
                print(f"   ℹ️  Casual greeting at turn {turns_count} (not ending)")
        # Wait for SIP participant to join
        print("📞 Waiting for SIP participant to join room...")
        max_wait = 60
        waited = 0

        while len(ctx.room.remote_participants) == 0 and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5

        if len(ctx.room.remote_participants) == 0:
            print("⏰ No participant joined - call failed")
            await ctx.room.disconnect()
            return
        
        # Wait for audio track to be published (indicates call was actually answered)
        print("📞 Participant joined. Waiting for call to be answered...")
        participant = list(ctx.room.remote_participants.values())[0]
        
        max_wait_audio = 60
        waited_audio = 0
        
        while len(participant.track_publications) == 0 and waited_audio < max_wait_audio:
            await asyncio.sleep(0.5)
            waited_audio += 0.5
        
        if len(participant.track_publications) == 0:
            print("⏰ No audio track - call not answered")
            await ctx.room.disconnect()
            return
        
        # Candidate answered! Short wait for audio to stabilize
        print("✅ Candidate answered!")
        print("⏳ Waiting 2 seconds for call stabilization...")
        await asyncio.sleep(1.5)

        # Send personalized greeting
        greeting = (
            f"Hi {candidate_name}! Thanks for taking the time to speak with me today. "
            f"I'm Ava, and I'll be conducting your interview for our {job_title} position. "
            f"I saw you're currently at {candidate_company} - that's great! "
            f"Before we dive in, could you tell me a bit about what you're working on there?"
        )
        
        print("👋 Sending personalized greeting...")
        await assistant.say(greeting, allow_interruptions=True)
        
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
                    print(f"✅ Interview ended naturally (AI goodbye)")
                    break
            
            # SAFETY: 15 minutes max
            if elapsed > 900:
                print("⏰ Max time reached (15 min)")
                await assistant.say(
                    f"Thank you so much for your time today, {candidate_name}. "
                    f"We really appreciate you speaking with us. Have a great day!"
                )
                await asyncio.sleep(3)
                break
            
            # SAFETY: Participant left
            if len(ctx.room.remote_participants) == 0:
                print("📞 Participant disconnected")
                await asyncio.sleep(2)
                break
        
        # Get full conversation data
        conversation_data = conv_logger.get_full_conversation_json()
        
        print(f"📊 Interview complete:")
        print(f"   Total turns: {len(conversation_data['turns'])}")
        print(f"   Duration: {conversation_data['metadata']['duration_seconds']:.1f}s")
        
        # Hang up the SIP call first
        print("📞 Hanging up call...")
        try:
            # Get SIP participant
            sip_participants = [p for p in ctx.room.remote_participants.values() 
                               if p.identity.startswith("candidate-")]
            
            if sip_participants:
                sip_participant = sip_participants[0]
                # Remove participant from room (hangs up the call)
                livekit_api = api.LiveKitAPI(
                    settings.livekit_url,
                    settings.livekit_api_key,
                    settings.livekit_api_secret
                )
                await livekit_api.room.remove_participant(
                    api.RoomParticipantIdentity(
                        room=ctx.room.name,
                        identity=sip_participant.identity
                    )
                )
                print("✅ Call hung up")
        except Exception as e:
            print(f"⚠️  Failed to hang up: {e}")

        # Disconnect room
        print("🔚 Disconnecting room...")
        await asyncio.sleep(1)  # Give time for hangup to process
        await ctx.room.disconnect()
        
        # Save raw conversation data to database
        print("💾 Saving raw conversation data...")
        try:
            import asyncpg
            
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                database=settings.postgres_db,
                user="backteam",
                password=settings.postgres_password
            )
            
            try:
                await conn.execute(
                    """
                    UPDATE interviews
                    SET 
                        conversation_log = $1::jsonb,
                        interview_status = 'processing',
                        updated_at = NOW()
                    WHERE interview_id = $2
                    """,
                    json.dumps(conversation_data),
                    interview_id
                )
                print(f"✅ Raw conversation saved for {interview_id}")
                print("   Background processing will handle embeddings & analytics")
            finally:
                await conn.close()
                
        except Exception as e:
            print(f"⚠️  Failed to save: {e}")
            logger.error(f"Save error: {e}", exc_info=True)
        
        logger.info(f"✅ Interview {interview_id} completed")
        
    except Exception as e:
        logger.error(f"❌ Error in interview: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        ),
    )