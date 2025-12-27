"""
AI Call Agent - Basic Conversational Agent

Orchestrates:
- Telnyx (voice calls)
- Deepgram Nova-3 (streaming STT)
- Google Cloud TTS (WaveNet)
- OpenAI GPT-4o-mini (LLM)

This is a simple agent that has a basic conversation.
"""

import os
import logging
import asyncio
from typing import Optional, List, Dict
from datetime import datetime
import openai
from google_tts_service import GoogleTTSService, RECOMMENDED_VOICES
from deepgram_service import DeepgramSTTService
from telnyx_service import TelnyxService
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


class AICallAgent:
    """
    Simple AI call agent for conversational interactions.
    
    Flow:
    1. Candidate calls in (or we call them)
    2. Agent greets them
    3. Listens to their response (Deepgram streaming)
    4. Generates response (GPT-4o-mini)
    5. Speaks response (Google TTS)
    6. Repeat 3-5 until conversation ends
    """
    
    def __init__(
        self,
        agent_name: str = "AI Assistant",
        voice: str = "female_professional",
        model: str = "gpt-4o-mini"
    ):
        self.agent_name = agent_name
        self.voice = RECOMMENDED_VOICES.get(voice, RECOMMENDED_VOICES["female_professional"])
        self.model = model
        
        # Services
        self.telnyx = TelnyxService()
        self.tts = GoogleTTSService()
        self.stt = None  # Created per-call
        
        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.current_transcript = ""
        self.is_speaking = False
        
        # System prompt for LLM
        self.system_prompt = """You are a friendly AI assistant having a phone conversation.

Keep responses:
- Natural and conversational
- Brief (1-2 sentences max)
- Clear and easy to understand over the phone
- Appropriate for voice conversation (no markdown, lists, or formatting)

You're having a casual conversation. Be helpful and friendly."""
        
        logger.info(f"✅ AI Call Agent initialized: {agent_name}")
    
    async def start_call(
        self,
        phone_number: str,
        greeting: str = None
    ) -> str:
        """
        Initiate outbound call to phone number.
        
        Args:
            phone_number: E.164 format (+14155551234)
            greeting: Optional custom greeting
            
        Returns:
            call_control_id
        """
        # Default greeting
        if not greeting:
            greeting = f"Hello! This is {self.agent_name}. How can I help you today?"
        
        logger.info(f"📞 Starting call to {phone_number[:8]}...")
        
        # Initiate call via Telnyx
        call_result = self.telnyx.initiate_call(
            to_phone=phone_number,
            webhook_url=f"https://your-domain.com/webhooks/call",  # Your webhook URL
            metadata={"agent": self.agent_name}
        )
        
        call_id = call_result["call_control_id"]
        
        # Store initial greeting
        self.conversation_history.append({
            "role": "assistant",
            "content": greeting
        })
        
        logger.info(f"✅ Call initiated: {call_id}")
        
        return call_id
    
    async def handle_call_answered(self, call_id: str):
        """
        Handle call answered event.
        
        This is called when the candidate answers the phone.
        """
        logger.info(f"✅ Call answered: {call_id}")
        
        # Start recording
        self.telnyx.start_recording(call_id)
        
        # Greet the caller
        greeting = self.conversation_history[0]["content"] if self.conversation_history else "Hello!"
        await self.speak(call_id, greeting)
        
        # Start listening
        await self.start_listening(call_id)
    
    async def speak(self, call_id: str, text: str):
        """
        Speak text to caller using Google TTS.
        
        Args:
            call_id: Telnyx call control ID
            text: Text to speak
        """
        logger.info(f"🗣️  Agent says: {text}")
        
        self.is_speaking = True
        
        try:
            # Generate audio with Google TTS
            audio_bytes = self.tts.synthesize_speech(
                text=text,
                voice_name=self.voice
            )
            
            # Save to temp file (Telnyx needs a URL)
            # In production, upload to GCP Storage and get public URL
            temp_file = f"/tmp/tts_{call_id}_{datetime.utcnow().timestamp()}.wav"
            self.tts.save_audio_to_file(audio_bytes, temp_file)
            
            # For now, we need to upload this to a public URL
            # TODO: Upload to GCP Storage bucket
            audio_url = await self.upload_to_gcp(temp_file)
            
            # Play audio to caller
            self.telnyx.play_audio(call_id, audio_url)
            
            # Wait for audio to finish
            # Duration = len(audio) / sample_rate
            # For mulaw 8kHz: duration ≈ len(audio_bytes) / 8000
            duration = len(audio_bytes) / 8000
            await asyncio.sleep(duration)
            
        except Exception as e:
            logger.error(f"❌ Speech failed: {e}")
        finally:
            self.is_speaking = False
    
    async def start_listening(self, call_id: str):
        """
        Start listening to caller via Deepgram streaming.
        
        Args:
            call_id: Call control ID
        """
        logger.info("👂 Starting to listen...")
        
        # Create Deepgram STT service
        self.stt = DeepgramSTTService()
        
        # Callback for transcripts
        async def on_transcript(text: str, is_final: bool):
            if is_final:
                logger.info(f"👤 Caller said: {text}")
                
                # Process the response
                await self.process_user_input(call_id, text)
        
        # Start streaming
        await self.stt.start_streaming(on_transcript=on_transcript)
        
        # Start streaming call audio to Deepgram
        # Note: This requires Telnyx Media Streaming setup
        # For now, this is a placeholder
        await self.stream_call_audio(call_id)
    
    async def stream_call_audio(self, call_id: str):
        """
        Stream call audio to Deepgram.
        
        In production, this connects to Telnyx Media Streaming WebSocket
        and forwards audio chunks to Deepgram.
        
        For this basic example, this is simplified.
        """
        # Start Telnyx audio streaming
        stream_url = "wss://your-websocket-endpoint.com/audio-stream"
        self.telnyx.start_streaming(call_id, stream_url)
        
        # In your WebSocket handler, you would:
        # 1. Receive audio from Telnyx
        # 2. Forward to Deepgram: await self.stt.send_audio(audio_chunk)
        
        logger.info("Audio streaming started (WebSocket required for full implementation)")
    
    async def process_user_input(self, call_id: str, user_text: str):
        """
        Process user's speech and generate response.
        
        Args:
            call_id: Call control ID
            user_text: Transcribed text from user
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        # Generate LLM response
        response_text = await self.generate_response(user_text)
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        # Speak response
        await self.speak(call_id, response_text)
    
    async def generate_response(self, user_text: str) -> str:
        """
        Generate AI response using GPT-4o-mini.
        
        Args:
            user_text: User's input text
            
        Returns:
            AI response text
        """
        logger.info(f"🤖 Generating response to: {user_text[:50]}...")
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history
            
            # Call OpenAI
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=100  # Keep responses short for phone
            )
            
            response_text = response.choices[0].message.content.strip()
            
            logger.info(f"✅ LLM response: {response_text}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return "I'm sorry, I didn't quite catch that. Could you repeat?"
    
    async def end_call(self, call_id: str):
        """End the call gracefully."""
        logger.info(f"📞 Ending call: {call_id}")
        
        # Say goodbye
        await self.speak(call_id, "Thank you for calling! Goodbye!")
        
        # Stop recording
        self.telnyx.stop_recording(call_id)
        
        # Stop Deepgram streaming
        if self.stt:
            await self.stt.stop_streaming()
        
        # Hang up
        self.telnyx.hangup_call(call_id)
        
        logger.info("✅ Call ended")
    
    async def upload_to_gcp(self, filepath: str) -> str:
        """
        Upload audio file to GCP Storage and return public URL.
        
        This is a placeholder - implement GCP Storage upload.
        
        Args:
            filepath: Local file path
            
        Returns:
            Public URL to audio file
        """
        # TODO: Implement GCP Storage upload
        # from google.cloud import storage
        # client = storage.Client()
        # bucket = client.bucket("your-bucket-name")
        # blob = bucket.blob(f"tts-audio/{os.path.basename(filepath)}")
        # blob.upload_from_filename(filepath)
        # blob.make_public()
        # return blob.public_url
        
        # For now, return placeholder
        return f"https://storage.googleapis.com/your-bucket/{os.path.basename(filepath)}"


# Example usage / testing
if __name__ == "__main__":
    import sys
    
    async def test_agent():
        """Test the AI call agent."""
        
        if len(sys.argv) < 2:
            print("Usage:")
            print("  python ai_call_agent.py call <phone_number>")
            print("  python ai_call_agent.py test")
            sys.exit(1)
        
        command = sys.argv[1]
        
        # Create agent
        agent = AICallAgent(
            agent_name="RecruiterAI Assistant",
            voice="female_professional"
        )
        
        if command == "call" and len(sys.argv) == 3:
            phone = sys.argv[2]
            
            # Start call
            call_id = await agent.start_call(
                phone_number=phone,
                greeting="Hello! This is RecruiterAI. I'm calling to discuss a job opportunity. Do you have a few minutes?"
            )
            
            print(f"✅ Call started: {call_id}")
            print("   Waiting for candidate to answer...")
            
            # In production, Telnyx webhooks would trigger handle_call_answered
            # For testing, simulate it after 5 seconds
            await asyncio.sleep(5)
            await agent.handle_call_answered(call_id)
            
            # Keep call alive for 60 seconds (for testing)
            await asyncio.sleep(60)
            
            # End call
            await agent.end_call(call_id)
        
        elif command == "test":
            # Test text-only conversation (no actual call)
            print("\n🤖 Testing AI Agent (text mode)\n")
            print("Type 'quit' to exit\n")
            
            agent.conversation_history.append({
                "role": "assistant",
                "content": "Hello! I'm RecruiterAI. How can I help you today?"
            })
            
            print(f"Agent: {agent.conversation_history[-1]['content']}\n")
            
            while True:
                user_input = input("You: ")
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nAgent: Goodbye!")
                    break
                
                # Generate response
                response = await agent.generate_response(user_input)
                
                # Add to history
                agent.conversation_history.append({"role": "user", "content": user_input})
                agent.conversation_history.append({"role": "assistant", "content": response})
                
                print(f"\nAgent: {response}\n")
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    
    asyncio.run(test_agent())