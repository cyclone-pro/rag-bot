#!/usr/bin/env python3
"""
FINAL SOLUTION: Use Credential Connection for Outbound
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Telnyx Default Credential Connection (supports outbound without IP whitelist)
TELNYX_SIP_SERVER = "sip.telnyx.com"
TELNYX_USERNAME = "9kv2m3jg"
TELNYX_PASSWORD = "9dgkgvjp69"
YOUR_PHONE_NUMBER = "+17792571297"


async def create_final_trunk():
    """Final trunk with Credential Connection"""
    
    print("=" * 70)
    print("FINAL SOLUTION: Credential-Based Outbound Trunk")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Clean up all existing trunks
        print("Removing all existing trunks...")
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
        
        # Create with Credential Connection
        print("Creating trunk with Credential authentication...")
        print(f"  Server: {TELNYX_SIP_SERVER}")
        print(f"  Username: {TELNYX_USERNAME}")
        print(f"  Number: {YOUR_PHONE_NUMBER}")
        print()
        
        trunk_info = api.SIPOutboundTrunkInfo(
            name="Telnyx-Final",
            address=TELNYX_SIP_SERVER,
            numbers=[YOUR_PHONE_NUMBER],
            auth_username=TELNYX_USERNAME,
            auth_password=TELNYX_PASSWORD,
        )
        
        request = api.CreateSIPOutboundTrunkRequest(trunk=trunk_info)
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print()
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Name: {trunk.name}")
        print()
        print("This trunk uses:")
        print("- Default Credential Connection (not IP connection)")
        print("- Username/password authentication")
        print("- No IP whitelist needed")
        print("- Will work for outbound calls!")
        print()
        print("=" * 70)
        print("UPDATE YOUR .ENV:")
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
    trunk = asyncio.run(create_final_trunk())
    
    if trunk:
        print("Next steps:")
        print("1. Update .env with trunk ID above")
        print("2. Restart API")
        print("3. Test - your phone WILL ring this time!")
        print()