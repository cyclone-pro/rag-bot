#!/usr/bin/env python3
"""
Create LiveKit Outbound Trunk - PBX Style
Following Telnyx's PBX documentation (like PBXes, FreePBX, etc.)
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Telnyx SIP Configuration (PBX-style)
TELNYX_SIP_SERVER = "sip.telnyx.com"
TELNYX_TOKEN = "LiveKitIy4tlewq2qxwz"  # Your token from IP connection
YOUR_PHONE_NUMBER = "+17792571297"


async def create_pbx_style_trunk():
    """Create trunk following Telnyx PBX documentation"""
    
    print("=" * 70)
    print("Creating LiveKit → Telnyx SIP Trunk (PBX Style)")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Delete existing trunk if any
        print("Cleaning up old trunks...")
        trunks = await livekit_api.sip.list_sip_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
        )
        
        for trunk in trunks.items:
            print(f"  Deleting: {trunk.name} ({trunk.sip_trunk_id})")
            await livekit_api.sip.delete_sip_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=trunk.sip_trunk_id)
            )
        
        print("✅ Cleanup complete")
        print()
        
        # Create trunk with PBX-style config
        # Based on PBXes example: username = token, password = empty
        print("Creating PBX-style trunk...")
        print(f"  Server: {TELNYX_SIP_SERVER}")
        print(f"  Auth: Token-based")
        print(f"  Number: {YOUR_PHONE_NUMBER}")
        print()
        
        trunk_info = api.SIPOutboundTrunkInfo(
            name="Telnyx-PBX-Trunk",
            address=TELNYX_SIP_SERVER,
            numbers=[YOUR_PHONE_NUMBER],
            # PBX-style: token as username, empty password
            auth_username=TELNYX_TOKEN,
            auth_password="",
        )
        
        request = api.CreateSIPOutboundTrunkRequest(trunk=trunk_info)
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        print("✅ Trunk created successfully!")
        print()
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Name: {trunk.name}")
        print(f"Address: {trunk.address}")
        print(f"Numbers: {trunk.numbers}")
        print()
        print("=" * 70)
        print("Configuration")
        print("=" * 70)
        print()
        print("Update your .env:")
        print(f"LIVEKIT_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")
        print()
        print("This trunk is configured like a PBX system:")
        print("- LiveKit acts as the PBX")
        print("- Telnyx acts as the SIP carrier")
        print("- Just like FreePBX, Asterisk, etc.")
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
    trunk = asyncio.run(create_pbx_style_trunk())
    
    if trunk:
        print("=" * 70)
        print("Next Steps")
        print("=" * 70)
        print()
        print("1. Update .env with the trunk ID above")
        print("2. Restart your API")
        print("3. Make sure your interview code uses LiveKit SIP participant")
        print("4. Test!")
        print()
        print("The call flow will be:")
        print("  API → LiveKit Room → LiveKit SIP → Telnyx → Candidate")
        print()