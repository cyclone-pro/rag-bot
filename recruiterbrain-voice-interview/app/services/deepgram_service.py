"""
Deepgram streaming speech-to-text service.

Real-time audio transcription using Deepgram Nova-3 model.
Optimized for phone call audio quality.
"""

import os
import logging
import asyncio
from typing import Optional, Callable
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    logger.warning("⚠️  DEEPGRAM_API_KEY not set!")


class DeepgramSTTService:
    """Real-time speech-to-text using Deepgram streaming API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or DEEPGRAM_API_KEY
        
        if not self.api_key:
            raise ValueError("Deepgram API key is required")
        
        # Initialize Deepgram client
        config = DeepgramClientOptions(
            options={"keepalive": "true"}
        )
        self.client = DeepgramClient(self.api_key, config)
        
        # Connection state
        self.connection = None
        self.is_connected = False
        
        logger.info("✅ Deepgram STT service initialized")
    
    async def start_streaming(
        self,
        on_transcript: Callable[[str, bool], None],
        on_error: Optional[Callable[[str], None]] = None
    ):
        """
        Start streaming transcription session.
        
        Args:
            on_transcript: Callback(transcript_text, is_final) called when text is transcribed
            on_error: Optional callback(error_message) for errors
        """
        try:
            # Create live transcription connection
            self.connection = self.client.listen.live.v("1")
            
            # Configure live options for phone calls
            options = LiveOptions(
                model="nova-2",              # Nova-3 is latest but use nova-2 for now
                language="en-US",
                smart_format=True,          # Auto-punctuation
                encoding="mulaw",           # Phone audio format
                sample_rate=8000,           # Phone quality
                channels=1,                 # Mono
                interim_results=True,       # Get partial results
                endpointing=300,            # 300ms silence = end of speech
                utterance_end_ms=1000,      # 1 second silence = utterance end
            )
            
            # Event handlers
            def on_message(self, result, **kwargs):
                """Handle transcription results."""
                sentence = result.channel.alternatives[0].transcript
                
                if len(sentence) == 0:
                    return
                
                is_final = result.is_final
                
                logger.info(f"{'[FINAL]' if is_final else '[INTERIM]'} {sentence}")
                
                # Call user's callback
                if on_transcript:
                    on_transcript(sentence, is_final)
            
            def on_metadata(self, metadata, **kwargs):
                """Handle metadata events."""
                logger.debug(f"Metadata: {metadata}")
            
            def on_error_event(self, error, **kwargs):
                """Handle error events."""
                logger.error(f"Deepgram error: {error}")
                if on_error:
                    on_error(str(error))
            
            def on_close(self, close, **kwargs):
                """Handle connection close."""
                logger.info("Deepgram connection closed")
                self.is_connected = False
            
            # Register event handlers
            self.connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
            self.connection.on(LiveTranscriptionEvents.Error, on_error_event)
            self.connection.on(LiveTranscriptionEvents.Close, on_close)
            
            # Start connection
            if self.connection.start(options) is False:
                raise Exception("Failed to start Deepgram connection")
            
            self.is_connected = True
            logger.info("✅ Deepgram streaming started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Deepgram streaming: {e}")
            raise
    
    async def send_audio(self, audio_chunk: bytes):
        """
        Send audio chunk to Deepgram for transcription.
        
        Args:
            audio_chunk: Raw audio bytes (mulaw, 8kHz)
        """
        if not self.is_connected or not self.connection:
            raise Exception("Not connected to Deepgram")
        
        try:
            self.connection.send(audio_chunk)
        except Exception as e:
            logger.error(f"❌ Failed to send audio: {e}")
            raise
    
    async def stop_streaming(self):
        """Stop streaming session."""
        if self.connection:
            try:
                self.connection.finish()
                logger.info("✅ Deepgram streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping Deepgram: {e}")
            finally:
                self.is_connected = False
                self.connection = None


# Example usage
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_transcription():
        """Test Deepgram with audio file."""
        
        if len(sys.argv) < 2:
            print("Usage: python deepgram_service.py <audio_file.wav>")
            sys.exit(1)
        
        audio_file = sys.argv[1]
        
        # Read audio file
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        
        print(f"📂 Loaded audio file: {len(audio_data)} bytes")
        
        # Callback for transcripts
        transcripts = []
        
        def on_transcript(text, is_final):
            if is_final:
                transcripts.append(text)
                print(f"\n✅ FINAL: {text}\n")
            else:
                print(f"   interim: {text}", end="\r")
        
        # Start streaming
        service = DeepgramSTTService()
        await service.start_streaming(on_transcript=on_transcript)
        
        # Send audio in chunks (simulate real-time)
        chunk_size = 8000  # 1 second of 8kHz audio
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await service.send_audio(chunk)
            await asyncio.sleep(0.1)  # Small delay
        
        # Wait for final transcripts
        await asyncio.sleep(2)
        
        # Stop streaming
        await service.stop_streaming()
        
        print("\n" + "="*60)
        print("FULL TRANSCRIPT:")
        print("="*60)
        print(" ".join(transcripts))
    
    asyncio.run(test_transcription())