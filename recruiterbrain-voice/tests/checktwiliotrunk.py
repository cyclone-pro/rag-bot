#!/usr/bin/env python3
"""
Update LiveKit Trunk with Twilio SIP Domain
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# NEW Twilio SIP Domain configuration
TWILIO_SIP_DOMAIN = "livekitcalls.sip.twilio.com"
TWILIO_USERNAME = "livekit123"  # The username you created in Twilio credentials
TWILIO_PASSWORD = "LiveKit2025!Pass"  # The password you created
TWILIO_PHONE_NUMBER = "+18445529044"


async def update_trunk():
    """Update trunk with Twilio SIP domain"""
    
    print("=" * 70)
    print("Updating LiveKit Trunk for Twilio SIP Domain")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Delete old trunk
        print("Step 1: Removing old trunk...")
        trunks = await livekit_api.sip.list_sip_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
        )
        
        for trunk in trunks.items:
            print(f"  Deleting: {trunk.name}")
            await livekit_api.sip.delete_sip_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=trunk.sip_trunk_id)
            )
        
        print("✅ Cleanup done")
        print()
        
        # Create new trunk with SIP domain
        print("Step 2: Creating trunk with Twilio SIP Domain...")
        print(f"  Domain: {TWILIO_SIP_DOMAIN}")
        print(f"  Username: {TWILIO_USERNAME}")
        print(f"  Number: {TWILIO_PHONE_NUMBER}")
        print()
        
        trunk_info = api.SIPOutboundTrunkInfo(
            name="Twilio-SIP-Domain",
            address=TWILIO_SIP_DOMAIN,
            numbers=[TWILIO_PHONE_NUMBER],
            auth_username=TWILIO_USERNAME,
            auth_password=TWILIO_PASSWORD,
        )
        
        request = api.CreateSIPOutboundTrunkRequest(trunk=trunk_info)
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print()
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Name: {trunk.name}")
        print(f"Address: {trunk.address}")
        print()
        print("=" * 70)
        print("UPDATE YOUR .ENV")
        print("=" * 70)
        print()
        print(f"LIVEKIT_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")
        print()
        print("Now the flow will be:")
        print("  LiveKit → livekitcalls.sip.twilio.com → Twilio → Your phone")
        print()
        
        return trunk
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    print()
    print("IMPORTANT: Update the script with YOUR credentials!")
    print("Edit the TWILIO_USERNAME and TWILIO_PASSWORD values above")
    print()
    input("Press Enter when ready to continue...")
    print()
    
    trunk = asyncio.run(update_trunk())
    
    if trunk:
        print("=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print()
        print("1. Update .env with trunk ID above")
        print("2. Restart API")
        print("3. Test - your phone WILL ring!")
        print()