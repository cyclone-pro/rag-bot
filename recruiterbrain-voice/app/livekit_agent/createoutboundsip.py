#!/usr/bin/env python3
"""
Create LiveKit Outbound Trunk for Telnyx
This allows LiveKit to make calls via Telnyx SIP
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Telnyx SIP credentials from your IP connection
TELNYX_USERNAME = "9kv2m3jg"  # From your screenshot
TELNYX_PASSWORD = "9dgkgvjp69"  # From your screenshot
TELNYX_SIP_ADDRESS = "sip.telnyx.com"  # Telnyx SIP endpoint
YOUR_PHONE_NUMBER = "+17792571297"  # Your Telnyx number


async def create_outbound_trunk():
    """Create LiveKit outbound SIP trunk"""
    
    print("=" * 70)
    print("Creating LiveKit Outbound Trunk for Telnyx")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Create outbound trunk
        trunk_info = api.SIPOutboundTrunkInfo(
            name="Telnyx-Outbound",
            address=TELNYX_SIP_ADDRESS,
            numbers=[YOUR_PHONE_NUMBER],
            auth_username=TELNYX_USERNAME,
            auth_password=TELNYX_PASSWORD,
        )
        
        request = api.CreateSIPOutboundTrunkRequest(
            trunk=trunk_info
        )
        
        print("Creating outbound trunk...")
        print(f"  Name: Telnyx-Outbound")
        print(f"  Address: {TELNYX_SIP_ADDRESS}")
        print(f"  Number: {YOUR_PHONE_NUMBER}")
        print(f"  Username: {TELNYX_USERNAME}")
        print()
        
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        print("✅ Outbound trunk created successfully!")
        print()
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Name: {trunk.name}")
        print(f"Address: {trunk.address}")
        print(f"Numbers: {trunk.numbers}")
        print()
        print("=" * 70)
        print("Next Steps:")
        print("=" * 70)
        print()
        print("1. Save this trunk ID for your code")
        print(f"   LIVEKIT_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")
        print()
        print("2. Update your .env file with this trunk ID")
        print()
        print("3. Modify your interview code to use LiveKit SIP participant")
        print()
        
        return trunk
        
    except Exception as e:
        print(f"❌ Error creating trunk: {e}")
        print()
        print("If trunk already exists, list existing trunks:")
        
        # List existing outbound trunks
        trunks = await livekit_api.sip.list_sip_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
        )
        
        print()
        print("Existing outbound trunks:")
        for t in trunks.items:
            print(f"  - {t.name} (ID: {t.sip_trunk_id})")
        
        return None
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    trunk = asyncio.run(create_outbound_trunk())
    
    if trunk:
        print()
        print("✅ Setup complete! Trunk is ready to use.")
        print()
        print("Add this to your .env file:")
        print(f"LIVEKIT_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")