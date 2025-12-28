"""
Telnyx integration service for SMS and voice calls.

Handles:
- SMS consent messages
- Outbound call initiation
- Webhook processing
- Call state management
"""

import os
import logging
from typing import Optional, Dict, Any
import telnyx

logger = logging.getLogger(__name__)

# Initialize Telnyx client
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
TELNYX_PHONE_NUMBER = os.getenv("TELNYX_PHONE_NUMBER")  # Your purchased number
TELNYX_CONNECTION_ID = os.getenv("TELNYX_CONNECTION_ID")  # From Telnyx dashboard

if not TELNYX_API_KEY:
    logger.warning("⚠️  TELNYX_API_KEY not set!")
else:
    # For Telnyx v3+, set API key this way
    telnyx.api_key = TELNYX_API_KEY
    logger.info("✅ Telnyx API key configured")


class TelnyxService:
    """Service for Telnyx SMS and voice operations."""
    
    def __init__(self):
        self.api_key = TELNYX_API_KEY
        self.phone_number = TELNYX_PHONE_NUMBER
        self.connection_id = TELNYX_CONNECTION_ID
    
    def send_consent_sms(
        self,
        to_phone: str,
        candidate_name: str,
        job_position: str,
        consent_form_url: str
    ) -> Dict[str, Any]:
        """
        Send SMS consent message to candidate.
        
        Args:
            to_phone: Candidate's phone number (E.164 format: +14155551234)
            candidate_name: Candidate's name
            job_position: Job title
            consent_form_url: URL to consent form
            
        Returns:
            dict with message_id and status
            
        Raises:
            Exception: If SMS fails to send
        """
        # Format message
        message_text = (
            f"Hi {candidate_name}, I am the AI agent from RecruiterAI. "
            f"We are looking for a {job_position} and we thought you were a good fit. "
            f"Would you like to proceed with an AI interview? "
            f"Please review and sign the consent form: {consent_form_url} "
            f"Reply YES or NO."
        )
        
        logger.info(f"📤 Sending consent SMS to {to_phone[:8]}...")
        
        try:
            # Send SMS via Telnyx
            response = telnyx.Message.create(
                from_=self.phone_number,
                to=to_phone,
                text=message_text
            )
            
            logger.info(f"✅ SMS sent successfully: {response.id}")
            
            return {
                "message_id": response.id,
                "status": "sent",
                "to": to_phone,
                "from": self.phone_number,
                "text_length": len(message_text)
            }
            
        except Exception as e:
            logger.error(f"❌ SMS send failed: {e}")
            raise Exception(f"Failed to send SMS: {str(e)}")
    
    def initiate_call(
        self,
        to_phone: str,
        webhook_url: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Initiate outbound call to candidate.
        
        Args:
            to_phone: Candidate's phone number
            webhook_url: Your FastAPI webhook endpoint for call events
            metadata: Additional data to pass (interview_id, candidate_id, etc.)
            
        Returns:
            dict with call_control_id and status
        """
        logger.info(f"📞 Initiating call to {to_phone[:8]}...")
        
        try:
            # Create call
            call = telnyx.Call.create(
                connection_id=self.connection_id,
                to=to_phone,
                from_=self.phone_number,
                webhook_url=webhook_url,
                webhook_url_method="POST",
                # Pass metadata for webhook context
                client_state=str(metadata) if metadata else None
            )
            
            logger.info(f"✅ Call initiated: {call.call_control_id}")
            
            return {
                "call_control_id": call.call_control_id,
                "call_leg_id": call.call_leg_id,
                "call_session_id": call.call_session_id,
                "status": "initiated",
                "to": to_phone
            }
            
        except Exception as e:
            logger.error(f"❌ Call initiation failed: {e}")
            raise Exception(f"Failed to initiate call: {str(e)}")
    
    def answer_call(self, call_control_id: str) -> bool:
        """Answer incoming call."""
        try:
            call = telnyx.Call.retrieve(call_control_id)
            call.answer()
            logger.info(f"✅ Call answered: {call_control_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Answer call failed: {e}")
            return False
    
    def hangup_call(self, call_control_id: str) -> bool:
        """Hangup active call."""
        try:
            call = telnyx.Call.retrieve(call_control_id)
            call.hangup()
            logger.info(f"✅ Call hung up: {call_control_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Hangup failed: {e}")
            return False
    
    def play_audio(
        self,
        call_control_id: str,
        audio_url: str
    ) -> bool:
        """
        Play audio file to caller.
        
        Args:
            call_control_id: Call control ID from Telnyx
            audio_url: Public URL to audio file (must be accessible to Telnyx)
            
        Returns:
            True if playback started successfully
        """
        try:
            call = telnyx.Call.retrieve(call_control_id)
            call.playback_start(audio_url=audio_url)
            logger.info(f"✅ Audio playback started: {call_control_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Audio playback failed: {e}")
            return False
    
    def start_recording(
        self,
        call_control_id: str,
        channels: str = "single"
    ) -> Dict[str, Any]:
        """
        Start recording call.
        
        Args:
            call_control_id: Call control ID
            channels: "single" or "dual" (separate tracks for each party)
            
        Returns:
            dict with recording details
        """
        try:
            call = telnyx.Call.retrieve(call_control_id)
            result = call.record_start(
                channels=channels,
                format="wav"  # WAV format for better Whisper compatibility
            )
            
            logger.info(f"✅ Recording started: {call_control_id}")
            return {
                "status": "recording",
                "channels": channels,
                "format": "wav"
            }
            
        except Exception as e:
            logger.error(f"❌ Start recording failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def stop_recording(self, call_control_id: str) -> Dict[str, Any]:
        """Stop call recording."""
        try:
            call = telnyx.Call.retrieve(call_control_id)
            result = call.record_stop()
            logger.info(f"✅ Recording stopped: {call_control_id}")
            return {"status": "stopped"}
        except Exception as e:
            logger.error(f"❌ Stop recording failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def start_streaming(
        self,
        call_control_id: str,
        stream_url: str
    ) -> bool:
        """
        Start streaming call audio via WebSocket.
        
        Used for real-time transcription.
        
        Args:
            call_control_id: Call control ID
            stream_url: WebSocket URL for audio streaming
            
        Returns:
            True if streaming started
        """
        try:
            call = telnyx.Call.retrieve(call_control_id)
            call.streaming_start(stream_url=stream_url)
            logger.info(f"✅ Audio streaming started: {call_control_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Start streaming failed: {e}")
            return False
    
    def gather_using_speak(
        self,
        call_control_id: str,
        text: str,
        voice: str = "female",
        language: str = "en-US"
    ) -> bool:
        """
        Speak text and gather DTMF input.
        
        Note: Telnyx TTS is basic. For better quality, use Google Cloud TTS
        to generate audio first, then use play_audio().
        
        Args:
            call_control_id: Call control ID
            text: Text to speak
            voice: Voice type
            language: Language code
            
        Returns:
            True if successful
        """
        try:
            call = telnyx.Call.retrieve(call_control_id)
            call.gather_using_speak(
                payload=text,
                voice=voice,
                language=language
            )
            logger.info(f"✅ TTS playback started: {call_control_id}")
            return True
        except Exception as e:
            logger.error(f"❌ TTS playback failed: {e}")
            return False
    
    def get_recording_url(self, recording_id: str) -> Optional[str]:
        """
        Get download URL for call recording.
        
        Args:
            recording_id: Recording ID from Telnyx webhook
            
        Returns:
            Download URL or None
        """
        try:
            # Retrieve recording
            recording = telnyx.Recording.retrieve(recording_id)
            
            # Get download URL
            download_url = recording.download_url
            
            logger.info(f"✅ Recording URL retrieved: {recording_id}")
            return download_url
            
        except Exception as e:
            logger.error(f"❌ Get recording URL failed: {e}")
            return None


# Webhook event types
WEBHOOK_EVENTS = {
    "call.initiated": "Call initiated",
    "call.answered": "Call answered by recipient",
    "call.hangup": "Call ended",
    "call.machine.detection.ended": "Answering machine detection completed",
    "call.recording.saved": "Recording available for download",
    "call.playback.started": "Audio playback started",
    "call.playback.ended": "Audio playback ended",
    "call.gather.ended": "DTMF gather completed",
    "call.speak.started": "TTS started",
    "call.speak.ended": "TTS ended"
}


def process_webhook_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process Telnyx webhook event.
    
    Args:
        payload: Webhook payload from Telnyx
        
    Returns:
        Processed event data
    """
    event_type = payload.get("data", {}).get("event_type")
    event_data = payload.get("data", {}).get("payload", {})
    
    logger.info(f"📨 Webhook event: {event_type}")
    
    return {
        "event_type": event_type,
        "call_control_id": event_data.get("call_control_id"),
        "call_leg_id": event_data.get("call_leg_id"),
        "call_session_id": event_data.get("call_session_id"),
        "client_state": event_data.get("client_state"),
        "from": event_data.get("from"),
        "to": event_data.get("to"),
        "direction": event_data.get("direction"),
        "state": event_data.get("state"),
        "recording_urls": event_data.get("recording_urls"),
        "raw_payload": event_data
    }


# Example usage
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    service = TelnyxService()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python telnyx_service.py test_sms <phone>")
        print("  python telnyx_service.py test_call <phone>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "test_sms" and len(sys.argv) == 3:
        phone = sys.argv[2]
        result = service.send_consent_sms(
            to_phone=phone,
            candidate_name="John Doe",
            job_position="Senior Backend Engineer",
            consent_form_url="https://example.com/consent/123"
        )
        print(f"✅ SMS sent: {result}")
    
    elif command == "test_call" and len(sys.argv) == 3:
        phone = sys.argv[2]
        result = service.initiate_call(
            to_phone=phone,
            webhook_url="https://your-domain.com/webhooks/telnyx/call",
            metadata={"test": True}
        )
        print(f"✅ Call initiated: {result}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)