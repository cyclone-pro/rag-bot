"""
Telephony Service
Handles phone calls via Telnyx and LiveKit SIP integration
"""

import logging
from typing import Optional, Dict, Any
import base64
import telnyx

from app.config.settings import settings


logger = logging.getLogger(__name__)


class TelephonyService:
    """Service for managing phone calls via Telnyx"""
    
    def __init__(self):
        """Initialize Telnyx client"""
        telnyx.api_key = settings.telnyx_api_key
        self.connection_id = settings.telnyx_connection_id
        self.from_number = settings.telnyx_phone_number
    
    async def initiate_call(
        self,
        to_number: str,
        livekit_room: str,
        interview_id: str
    ) -> Optional[str]:
        """
        Initiate a phone call and connect to LiveKit room
        
        Args:
            to_number: Candidate's phone number (E.164 format)
            livekit_room: LiveKit room name
            interview_id: Interview identifier
            
        Returns:
            Call control ID if successful, None otherwise
        """
        
        try:
            logger.info(f"Initiating call to {to_number} for interview {interview_id}")
            
            # Encode the LiveKit room name as base64 for client_state
            client_state_b64 = base64.b64encode(livekit_room.encode()).decode()
            
            # Create call using Telnyx
            call = telnyx.Call.create(
                connection_id=self.connection_id,
                to=to_number,
                from_=self.from_number,
                # Pass the LiveKit room in client_state (must be base64)
                client_state=client_state_b64,
                # Webhook for call status updates
                webhook_url=f"{settings.api_host}/api/v1/webhooks/telnyx",
                webhook_url_method="POST"
            )
            
            call_control_id = call.call_control_id
            
            logger.info(f"Call initiated: {call_control_id}, will bridge to {livekit_room} when answered")
            
            return call_control_id
            
        except telnyx.error.TelnyxError as e:
            logger.error(f"Telnyx error initiating call: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error initiating call: {e}")
            return None
    
    async def bridge_to_livekit(
        self,
        call_control_id: str,
        livekit_room: str
    ) -> bool:
        """
        Bridge Telnyx call to LiveKit SIP room
        
        Args:
            call_control_id: Telnyx call control ID
            livekit_room: LiveKit room name to bridge to
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            # Construct LiveKit SIP URI
            # Format: sip:room_name@your-livekit-sip-domain.livekit.cloud
            livekit_sip_uri = f"sip:{livekit_room}@{settings.livekit_sip_domain}"
            
            logger.info(f"Bridging call {call_control_id} to {livekit_sip_uri}")
            
            # Transfer/bridge the call to LiveKit SIP
            call = telnyx.Call.retrieve(call_control_id)
            call.transfer(to=livekit_sip_uri)
            
            logger.info(f"Call {call_control_id} bridged to LiveKit")
            
            return True
            
        except telnyx.error.TelnyxError as e:
            logger.error(f"Telnyx error bridging call: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error bridging call: {e}")
            return False
    
    async def hangup_call(self, call_control_id: str) -> bool:
        """
        Hang up an active call
        
        Args:
            call_control_id: Telnyx call control ID
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            logger.info(f"Hanging up call {call_control_id}")
            
            call = telnyx.Call.retrieve(call_control_id)
            call.hangup()
            
            logger.info(f"Call {call_control_id} hung up")
            
            return True
            
        except telnyx.error.TelnyxError as e:
            logger.error(f"Telnyx error hanging up call: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error hanging up call: {e}")
            return False
    
    async def get_call_status(self, call_control_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a call
        
        Args:
            call_control_id: Telnyx call control ID
            
        Returns:
            Call status dict if successful, None otherwise
        """
        
        try:
            call = telnyx.Call.retrieve(call_control_id)
            
            return {
                "call_control_id": call.call_control_id,
                "call_leg_id": call.call_leg_id,
                "state": call.state,
                "from": call.from_,
                "to": call.to,
                "direction": call.direction,
                "created_at": call.created_at
            }
            
        except telnyx.error.TelnyxError as e:
            logger.error(f"Telnyx error getting call status: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting call status: {e}")
            return None


# Global telephony service instance
telephony_service = TelephonyService()


# Convenience functions
async def make_interview_call(
    phone_number: str,
    livekit_room: str,
    interview_id: str
) -> Optional[str]:
    """
    Make an interview call
    
    Args:
        phone_number: Candidate's phone number
        livekit_room: LiveKit room name
        interview_id: Interview identifier
        
    Returns:
        Call control ID if successful, None otherwise
    """
    return await telephony_service.initiate_call(
        to_number=phone_number,
        livekit_room=livekit_room,
        interview_id=interview_id
    )


async def end_interview_call(call_control_id: str) -> bool:
    """
    End an interview call
    
    Args:
        call_control_id: Telnyx call control ID
        
    Returns:
        True if successful, False otherwise
    """
    return await telephony_service.hangup_call(call_control_id)