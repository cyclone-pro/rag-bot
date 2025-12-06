"""Audio transcription using OpenAI Whisper API."""
import logging
import tempfile
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def transcribe_audio_whisper(audio_bytes: bytes, filename: str = "audio.webm") -> Optional[str]:
    """
    Transcribe audio using OpenAI Whisper API.
    
    Args:
        audio_bytes: Audio file bytes
        filename: Original filename (for format detection)
    
    Returns:
        Transcribed text or None if failed
    """
    from .config import get_openai_client
    
    client = get_openai_client()
    
    if not client:
        logger.error("OpenAI client not available for transcription")
        return None
    
    try:
        # Save audio to temporary file (Whisper API requires file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        logger.info(f"Transcribing audio file: {filename} ({len(audio_bytes)} bytes)")
        
        # Call Whisper API
        with open(tmp_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",  # Can be auto-detected by removing this
                response_format="text"
            )
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        transcribed_text = transcript.strip()
        logger.info(f"✅ Transcription: {transcribed_text}")
        
        return transcribed_text
        
    except Exception as e:
        logger.exception(f"Whisper transcription failed: {e}")
        
        # Clean up temp file on error
        try:
            Path(tmp_path).unlink()
        except:
            pass
        
        return None