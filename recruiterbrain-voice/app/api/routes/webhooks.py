"""
Telnyx Webhook Handler
Receives call events and bridges to LiveKit
"""

from fastapi import APIRouter, Request
import telnyx
import base64
import logging

from app.config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/webhooks/telnyx")
async def telnyx_webhook(request: Request):
    """
    Handle Telnyx webhook events
    Bridges calls to LiveKit SIP when answered
    """
    
    try:
        payload = await request.json()
        
        event_type = payload.get("data", {}).get("event_type")
        call_payload = payload.get("data", {}).get("payload", {})
        call_control_id = call_payload.get("call_control_id")
        
        logger.info(f"Telnyx webhook received: {event_type}")
        logger.debug(f"Webhook payload: {payload}")
        
        # When candidate answers, bridge to LiveKit
        if event_type == "call.answered":
            logger.info(f"Call answered: {call_control_id}")
            
            # Get the LiveKit room from client_state (base64 encoded)
            client_state_b64 = call_payload.get("client_state", "")
            
            try:
                # Decode base64 client_state
                livekit_room = base64.b64decode(client_state_b64).decode() if client_state_b64 else "default-room"
            except Exception as e:
                logger.error(f"Failed to decode client_state: {e}")
                livekit_room = "default-room"
            
            # Construct LiveKit SIP URI (WITHOUT @ sign - Telnyx adds it)
            # Format: sip:room_name@domain becomes just the room_name part
            livekit_sip_uri = f"sip:{livekit_room}@{settings.livekit_sip_domain}"
            
            logger.info(f"Bridging call {call_control_id} to {livekit_sip_uri}")
            
            try:
                # Retrieve the call
                telnyx.api_key = settings.telnyx_api_key
                call = telnyx.Call.retrieve(call_control_id)
                
                # Transfer with SIP authentication credentials
                # These credentials tell Telnyx HOW to authenticate with LiveKit
                call.transfer(
                    to=livekit_sip_uri,
                    # CRITICAL: Add SIP authentication for LiveKit
                    # These should match your Telnyx FQDN connection credentials
                    #sip_auth_username="Eiteisinc",  # Your FQDN connection username
                    #sip_auth_password="LiveKit2025!Secure"  # Your FQDN connection password
                )
                
                logger.info(f"✅ Call {call_control_id} successfully bridged to LiveKit room: {livekit_room}")
                
            except telnyx.error.TelnyxError as e:
                logger.error(f"❌ Telnyx error bridging call: {e}")
                return {"status": "error", "message": str(e)}
        
        elif event_type == "call.hangup":
            hangup_cause = call_payload.get("hangup_cause", "unknown")
            logger.info(f"Call {call_control_id} hung up: {hangup_cause}")
        
        elif event_type == "call.initiated":
            logger.info(f"Call {call_control_id} initiated")
        
        elif event_type == "call.bridged":
            logger.info(f"✅ Call {call_control_id} bridged successfully")
        
        else:
            logger.info(f"Unhandled event type: {event_type}")
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.get("/webhooks/telnyx/health")
async def webhook_health():
    """Health check endpoint for webhooks"""
    return {
        "status": "ok",
        "service": "telnyx-webhook",
        "livekit_sip_domain": settings.livekit_sip_domain
    }