"""
Telnyx Webhook Handler
Receives call events and bridges to LiveKit
"""

from fastapi import APIRouter, Request
import telnyx
import base64
import logging
import json
import time
from typing import Any, Dict, Optional

from app.config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def _log(level: str, message: str, **fields: Any) -> None:
    payload = {"message": message, **fields}
    record = json.dumps(payload, ensure_ascii=True)
    if level == "warning":
        logger.warning(record)
    elif level == "error":
        logger.error(record)
    else:
        logger.info(record)

def _extract_telnyx_meta(payload: Dict[str, Any]) -> Dict[str, Optional[str]]:
    event_type = payload.get("data", {}).get("event_type")
    call_payload = payload.get("data", {}).get("payload", {})
    call_control_id = call_payload.get("call_control_id")
    return {
        "event_type": event_type,
        "call_control_id": call_control_id,
    }


@router.post("/webhooks/telnyx")
async def telnyx_webhook(request: Request):
    """
    Handle Telnyx webhook events
    Bridges calls to LiveKit SIP when answered
    """
    
    try:
        started_at = time.time()
        payload = await request.json()
        meta = _extract_telnyx_meta(payload)
        
        event_type = payload.get("data", {}).get("event_type")
        call_payload = payload.get("data", {}).get("payload", {})
        call_control_id = call_payload.get("call_control_id")
        
        _log("info", "telnyx_webhook_received", **meta)
        logger.info(f"Telnyx webhook received: {event_type}")
        logger.debug(f"Webhook payload: {payload}")
        
        # When candidate answers, bridge to LiveKit
        if event_type == "call.answered":
            _log("info", "telnyx_call_answered", **meta)
            logger.info(f"Call answered: {call_control_id}")
            
            # Get the LiveKit room from client_state (base64 encoded)
            client_state_b64 = call_payload.get("client_state", "")
            
            try:
                # Decode base64 client_state
                livekit_room = base64.b64decode(client_state_b64).decode() if client_state_b64 else "default-room"
                _log("info", "telnyx_client_state_decoded", **meta, livekit_room=livekit_room)
            except Exception as e:
                _log("error", "telnyx_client_state_decode_failed", **meta, error=str(e))
                logger.error(f"Failed to decode client_state: {e}")
                livekit_room = "default-room"
            
            # Construct LiveKit SIP URI (WITHOUT @ sign - Telnyx adds it)
            # Format: sip:room_name@domain becomes just the room_name part
            livekit_sip_uri = f"sip:{livekit_room}@{settings.livekit_sip_domain}"
            
            _log("info", "telnyx_bridge_started", **meta, livekit_room=livekit_room, sip_uri=livekit_sip_uri)
            logger.info(f"Bridging call {call_control_id} to {livekit_sip_uri}")
            
            try:
                # Retrieve the call
                telnyx.api_key = settings.telnyx_api_key
                bridge_started = time.time()
                _log("info", "telnyx_call_retrieve_started", **meta)
                call = telnyx.Call.retrieve(call_control_id)
                _log("info", "telnyx_call_retrieve_complete", **meta)
                
                # Transfer with SIP authentication credentials
                # These credentials tell Telnyx HOW to authenticate with LiveKit
                call.transfer(
                    to=livekit_sip_uri,
                    # CRITICAL: Add SIP authentication for LiveKit
                    # These should match your Telnyx FQDN connection credentials
                    #sip_auth_username="Eiteisinc",  # Your FQDN connection username
                    #sip_auth_password="LiveKit2025!Secure"  # Your FQDN connection password
                )
                
                _log(
                    "info",
                    "telnyx_bridge_complete",
                    **meta,
                    livekit_room=livekit_room,
                    duration_ms=int((time.time() - bridge_started) * 1000),
                )
                logger.info(f"✅ Call {call_control_id} successfully bridged to LiveKit room: {livekit_room}")
                
            except telnyx.error.TelnyxError as e:
                _log("error", "telnyx_bridge_failed", **meta, error=str(e))
                logger.error(f"❌ Telnyx error bridging call: {e}")
                return {"status": "error", "message": str(e)}
        
        elif event_type == "call.hangup":
            hangup_cause = call_payload.get("hangup_cause", "unknown")
            _log("info", "telnyx_call_hangup", **meta, hangup_cause=hangup_cause)
            logger.info(f"Call {call_control_id} hung up: {hangup_cause}")
        
        elif event_type == "call.initiated":
            _log("info", "telnyx_call_initiated", **meta)
            logger.info(f"Call {call_control_id} initiated")
        
        elif event_type == "call.bridged":
            _log("info", "telnyx_call_bridged", **meta)
            logger.info(f"✅ Call {call_control_id} bridged successfully")
        
        else:
            _log("info", "telnyx_event_unhandled", **meta)
            logger.info(f"Unhandled event type: {event_type}")
        
        _log(
            "info",
            "telnyx_webhook_complete",
            **meta,
            duration_ms=int((time.time() - started_at) * 1000),
        )
        return {"status": "ok"}
        
    except Exception as e:
        _log("error", "telnyx_webhook_failed", error=str(e))
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
