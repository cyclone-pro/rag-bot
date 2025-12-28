"""
Google Cloud Text-to-Speech service.

Generates high-quality speech audio from text using WaveNet voices.
Optimized for phone calls (8kHz, mulaw encoding).
"""

import os
import logging
from typing import Optional
from google.cloud import texttospeech
import io
import hashlib

logger = logging.getLogger(__name__)

# Initialize Google TTS client
# Expects GOOGLE_APPLICATION_CREDENTIALS env var pointing to service account JSON
try:
    tts_client = texttospeech.TextToSpeechClient()
    logger.info("✅ Google Cloud TTS client initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Google TTS: {e}")
    tts_client = None


class GoogleTTSService:
    """Service for converting text to speech using Google Cloud TTS."""
    
    def __init__(self):
        self.client = tts_client
        self.cache = {}  # Simple in-memory cache for repeated phrases
    
    def synthesize_speech(
        self,
        text: str,
        voice_name: str = "en-US-Neural2-F",  # Female voice
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        cache_enabled: bool = True
    ) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            voice_name: Google voice name (en-US-Neural2-F = female, en-US-Neural2-J = male)
            speaking_rate: Speed (0.25 to 4.0, default 1.0)
            pitch: Pitch adjustment (-20.0 to 20.0, default 0.0)
            cache_enabled: Use cache for repeated text
            
        Returns:
            Audio bytes in mulaw format (optimized for phone calls)
        """
        if not self.client:
            raise Exception("Google TTS client not initialized. Check GOOGLE_APPLICATION_CREDENTIALS.")
        
        # Check cache
        if cache_enabled:
            cache_key = hashlib.md5(f"{text}_{voice_name}_{speaking_rate}_{pitch}".encode()).hexdigest()
            if cache_key in self.cache:
                logger.info(f"✅ TTS cache hit: {text[:50]}...")
                return self.cache[cache_key]
        
        logger.info(f"🎤 Generating TTS: {text[:100]}...")
        
        try:
            # Set the text input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name
            )
            
            # Select the audio config
            # MULAW format is optimized for telephony (8kHz sample rate)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MULAW,
                sample_rate_hertz=8000,  # Phone quality
                speaking_rate=speaking_rate,
                pitch=pitch
            )
            
            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            audio_bytes = response.audio_content
            
            logger.info(f"✅ TTS generated: {len(audio_bytes)} bytes")
            
            # Cache result
            if cache_enabled:
                self.cache[cache_key] = audio_bytes
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"❌ TTS generation failed: {e}")
            raise Exception(f"Failed to generate speech: {str(e)}")
    
    def save_audio_to_file(self, audio_bytes: bytes, filepath: str) -> str:
        """
        Save audio to file.
        
        Args:
            audio_bytes: Audio data
            filepath: Output path (e.g., "output.wav")
            
        Returns:
            Path to saved file
        """
        with open(filepath, "wb") as out:
            out.write(audio_bytes)
        
        logger.info(f"✅ Audio saved: {filepath}")
        return filepath
    
    def get_available_voices(self, language_code: str = "en-US") -> list:
        """
        Get list of available voices for a language.
        
        Args:
            language_code: Language code (e.g., "en-US")
            
        Returns:
            List of voice names
        """
        if not self.client:
            return []
        
        # Fetch available voices
        response = self.client.list_voices(language_code=language_code)
        
        voices = []
        for voice in response.voices:
            voices.append({
                "name": voice.name,
                "gender": voice.ssml_gender.name,
                "language_codes": voice.language_codes
            })
        
        return voices


# Recommended voices for AI recruiter
RECOMMENDED_VOICES = {
    "female_professional": "en-US-Neural2-F",  # Clear, professional female
    "female_warm": "en-US-Neural2-C",          # Warmer, friendly female
    "male_professional": "en-US-Neural2-J",    # Clear, professional male
    "male_warm": "en-US-Neural2-D",            # Warmer, friendly male
}


# Example usage
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python google_tts_service.py 'Hello, this is a test'")
        print("  python google_tts_service.py list_voices")
        sys.exit(1)
    
    service = GoogleTTSService()
    
    if sys.argv[1] == "list_voices":
        voices = service.get_available_voices()
        print(f"\n✅ Available voices ({len(voices)}):")
        for voice in voices[:10]:  # Show first 10
            print(f"  - {voice['name']} ({voice['gender']})")
    
    else:
        text = " ".join(sys.argv[1:])
        
        # Generate speech
        audio = service.synthesize_speech(
            text=text,
            voice_name=RECOMMENDED_VOICES["female_professional"]
        )
        
        # Save to file
        output_file = "test_output.wav"
        service.save_audio_to_file(audio, output_file)
        
        print(f"\n✅ Audio generated and saved to: {output_file}")
        print(f"   Size: {len(audio)} bytes")
        print(f"   Format: 8kHz mulaw (phone quality)")