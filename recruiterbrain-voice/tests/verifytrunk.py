#!/usr/bin/env python3
"""
Verify LiveKit Outbound Trunk exists
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


async def verify_trunk():
    """Verify trunk exists"""
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # List all outbound trunks
        trunks = await livekit_api.sip.list_sip_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
        )
        
        print("=" * 70)
        print("LiveKit Outbound Trunks")
        print("=" * 70)
        print()
        
        if len(trunks.items) == 0:
            print("❌ NO OUTBOUND TRUNKS FOUND!")
            print()
            print("The trunk was deleted or doesn't exist.")
            print("Run: python update_trunk_with_token.py")
            return
        
        for trunk in trunks.items:
            print(f"✅ Trunk: {trunk.name}")
            print(f"   ID: {trunk.sip_trunk_id}")
            print(f"   Address: {trunk.address}")
            print(f"   Numbers: {trunk.numbers}")
            print()
        
        print("=" * 70)
        print("Your .env should have:")
        print("=" * 70)
        print()
        print(f"LIVEKIT_OUTBOUND_TRUNK_ID={trunks.items[0].sip_trunk_id}")
        print()
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(verify_trunk())