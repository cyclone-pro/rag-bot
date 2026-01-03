#!/usr/bin/env python3
"""
Create LiveKit Outbound Trunk for Telnyx FQDN Connection
Using credentials: Eiteisinc / LiveKit2025!Secure
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Telnyx FQDN Configuration
TELNYX_SIP_ADDRESS = "sip.telnyx.com"
TELNYX_USERNAME = "Eiteisinc"
TELNYX_PASSWORD = "LiveKit2025!Secure"
TELNYX_PHONE_NUMBER = "+17792571297"


async def create_telnyx_fqdn_trunk():
    """Create outbound trunk for Telnyx FQDN connection"""
    
    print("=" * 70)
    print("Creating LiveKit Outbound Trunk for Telnyx FQDN")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Delete ALL existing outbound trunks
        print("Step 1: Cleaning up ALL existing outbound trunks...")
        outbound_trunks = await livekit_api.sip.list_sip_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
        )
        
        for trunk in outbound_trunks.items:
            print(f"  Deleting: {trunk.name} ({trunk.sip_trunk_id})")
            await livekit_api.sip.delete_sip_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=trunk.sip_trunk_id)
            )
        
        print("✅ Cleanup complete")
        print()
        
        # Create Telnyx FQDN trunk
        print("Step 2: Creating Telnyx FQDN outbound trunk...")
        print(f"  Address: {TELNYX_SIP_ADDRESS}")
        print(f"  Username: {TELNYX_USERNAME}")
        print(f"  Number: {TELNYX_PHONE_NUMBER}")
        print()
        
        # Per Telnyx official guide, include X-Telnyx-Username header
        trunk_info = api.SIPOutboundTrunkInfo(
            name="Telnyx-FQDN-Outbound",
            address=TELNYX_SIP_ADDRESS,
            numbers=[TELNYX_PHONE_NUMBER],
            auth_username=TELNYX_USERNAME,
            auth_password=TELNYX_PASSWORD,
            # Critical: X-Telnyx-Username header for proper routing
            headers={
                "X-Telnyx-Username": TELNYX_USERNAME
            }
        )
        
        request = api.CreateSIPOutboundTrunkRequest(trunk=trunk_info)
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        print("=" * 70)
        print("✅ SUCCESS! Trunk Created")
        print("=" * 70)
        print()
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Name: {trunk.name}")
        print(f"Address: {trunk.address}")
        print(f"Username: {TELNYX_USERNAME}")
        print(f"Number: {TELNYX_PHONE_NUMBER}")
        print()
        print("=" * 70)
        print("UPDATE YOUR .ENV FILE")
        print("=" * 70)
        print()
        print(f"LIVEKIT_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")
        print()
        print("IMPORTANT: This trunk uses Telnyx FQDN credentials")
        print("and includes X-Telnyx-Username header for proper authentication")
        print("as specified in the official Telnyx + LiveKit integration guide.")
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
    trunk = asyncio.run(create_telnyx_fqdn_trunk())
    
    if trunk:
        print("=" * 70)
        print("CONFIGURATION COMPLETE!")
        print("=" * 70)
        print()
        print("Now you have:")
        print("  ✅ LiveKit INBOUND trunk (receives calls FROM Telnyx)")
        print("  ✅ LiveKit OUTBOUND trunk (makes calls TO Telnyx)")
        print()
        print("Next steps:")
        print("  1. Update .env with trunk ID above")
        print("  2. Make sure interview.py uses Telnyx approach")
        print("  3. Restart API")
        print("  4. TEST!")
        print()