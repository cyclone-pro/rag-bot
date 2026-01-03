#!/usr/bin/env python3
"""
Check LiveKit Inbound Trunk Configuration
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os
import json

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


async def check_inbound_trunk():
    """Check inbound trunk config"""
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        trunks = await livekit_api.sip.list_sip_inbound_trunk(
            api.ListSIPInboundTrunkRequest()
        )
        
        print("=" * 70)
        print("LiveKit Inbound Trunk Configuration")
        print("=" * 70)
        print()
        
        if len(trunks.items) == 0:
            print("❌ No inbound trunks!")
            return
        
        trunk = trunks.items[0]
        
        print(f"Name: {trunk.name}")
        print(f"ID: {trunk.sip_trunk_id}")
        print(f"Numbers: {trunk.numbers}")
        print(f"Allowed Addresses: {trunk.allowed_addresses}")
        print(f"Allowed Numbers: {trunk.allowed_numbers if hasattr(trunk, 'allowed_numbers') else []}")
        print()
        
        print("Full trunk object:")
        print(trunk)
        print()
        
        # Check configuration
        print("=" * 70)
        print("Configuration Check")
        print("=" * 70)
        print()
        
        if trunk.allowed_addresses and "0.0.0.0/0" in trunk.allowed_addresses:
            print("✅ Allows calls from any IP")
        else:
            print("⚠️  Limited IP addresses")
        
        if trunk.numbers and "+17792571297" in trunk.numbers:
            print("✅ Your number is configured")
        else:
            print("❌ Your number not in trunk!")
        
        print()
        
        # Check dispatch rules
        rules = await livekit_api.sip.list_sip_dispatch_rule(
            api.ListSIPDispatchRuleRequest()
        )
        
        print("Dispatch Rules:")
        for rule in rules.items:
            if trunk.sip_trunk_id in rule.trunk_ids:
                print(f"✅ Rule '{rule.name}' uses this trunk")
            else:
                print(f"⚠️  Rule '{rule.name}' doesn't use this trunk")
        
        print()
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(check_inbound_trunk())