#!/usr/bin/env python3
"""
Update LiveKit Outbound Trunk to use Token Authentication
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Telnyx settings with Token authentication
TELNYX_SIP_ADDRESS = "sip.telnyx.com"
TELNYX_TOKEN = "LiveKitIy4tlewq2qxwz"  # From your Telnyx connection
YOUR_PHONE_NUMBER = "+17792571297"
EXISTING_TRUNK_ID = "ST_gHh3tposVCmZ"  # The trunk we created earlier


async def update_outbound_trunk():
    """Update the existing outbound trunk with Token auth"""
    
    print("=" * 70)
    print("Updating LiveKit Outbound Trunk")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # First, delete the old trunk
        print(f"Deleting old trunk: {EXISTING_TRUNK_ID}")
        await livekit_api.sip.delete_sip_trunk(
            api.DeleteSIPTrunkRequest(sip_trunk_id=EXISTING_TRUNK_ID)
        )
        print("✅ Old trunk deleted")
        print()
        
        # Create new trunk with Token authentication
        print("Creating new trunk with Token authentication...")
        
        trunk_info = api.SIPOutboundTrunkInfo(
            name="Telnyx-Outbound-Token",
            address=TELNYX_SIP_ADDRESS,
            numbers=[YOUR_PHONE_NUMBER],
            auth_username="",  # Not used with Token auth
            auth_password=TELNYX_TOKEN,  # Use token as password
        )
        
        request = api.CreateSIPOutboundTrunkRequest(
            trunk=trunk_info
        )
        
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        print("✅ New trunk created successfully!")
        print()
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Name: {trunk.name}")
        print(f"Address: {trunk.address}")
        print(f"Numbers: {trunk.numbers}")
        print(f"Auth: Token-based")
        print()
        print("=" * 70)
        print("Update your .env file:")
        print("=" * 70)
        print()
        print(f"LIVEKIT_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")
        print()
        
        return trunk
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    trunk = asyncio.run(update_outbound_trunk())
    
    if trunk:
        print()
        print("✅ Trunk updated! Test your interview now.")
        print()
        print("Don't forget to update .env with the new trunk ID!")