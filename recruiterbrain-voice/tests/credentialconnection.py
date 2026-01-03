#!/usr/bin/env python3
"""
Recreate LiveKit Trunk with Default Credential Connection
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Telnyx Default Credential Connection
TELNYX_SIP_ADDRESS = "sip.telnyx.com"
TELNYX_USERNAME = "9kv2m3jg"
TELNYX_PASSWORD = "9dgkgvjp69"
YOUR_PHONE_NUMBER = "+17792571297"
EXISTING_TRUNK_ID = "ST_5oNnUMJ2KuYR"


async def recreate_trunk():
    """Recreate trunk with Credential Connection"""
    
    print("=" * 70)
    print("Creating LiveKit Trunk with Credential Connection")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Delete old trunk
        print(f"Deleting old trunk: {EXISTING_TRUNK_ID}")
        try:
            await livekit_api.sip.delete_sip_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=EXISTING_TRUNK_ID)
            )
            print("✅ Old trunk deleted")
        except:
            print("⚠️  Old trunk already deleted or doesn't exist")
        print()
        
        # Create new trunk with Credential Connection
        print("Creating trunk with Credential authentication...")
        
        trunk_info = api.SIPOutboundTrunkInfo(
            name="Telnyx-Credential",
            address=TELNYX_SIP_ADDRESS,
            numbers=[YOUR_PHONE_NUMBER],
            auth_username=TELNYX_USERNAME,
            auth_password=TELNYX_PASSWORD,
        )
        
        request = api.CreateSIPOutboundTrunkRequest(trunk=trunk_info)
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        print("✅ Trunk created successfully!")
        print()
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Name: {trunk.name}")
        print(f"Address: {trunk.address}")
        print(f"Username: {TELNYX_USERNAME}")
        print(f"Numbers: {trunk.numbers}")
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
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    trunk = asyncio.run(recreate_trunk())
    
    if trunk:
        print()
        print("✅ Trunk created! Now:")
        print("1. Update .env with the new trunk ID")
        print("2. Restart your API")
        print("3. Test again!")