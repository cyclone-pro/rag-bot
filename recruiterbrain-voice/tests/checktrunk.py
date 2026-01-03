#!/usr/bin/env python3
"""
Check LiveKit Trunk Configuration in Detail
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


async def check_trunk_config():
    """Check LiveKit trunk configuration in detail"""
    
    print("=" * 70)
    print("LiveKit SIP Trunk Detailed Configuration")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Get inbound trunks
        trunks = await livekit_api.sip.list_sip_inbound_trunk(
            api.ListSIPInboundTrunkRequest()
        )
        
        print(f"Found {len(trunks.items)} inbound trunk(s)")
        print()
        
        for trunk in trunks.items:
            print(f"📋 Trunk: {trunk.name}")
            print(f"   ID: {trunk.sip_trunk_id}")
            print(f"   Numbers: {trunk.numbers}")
            print(f"   Allowed Addresses: {trunk.allowed_addresses}")
            print(f"   Allowed Numbers: {trunk.allowed_numbers if hasattr(trunk, 'allowed_numbers') else 'N/A'}")
            
            # Check if authentication is configured
            has_auth = False
            if hasattr(trunk, 'inbound_username') and trunk.inbound_username:
                print(f"   ⚠️  Inbound Username: {trunk.inbound_username}")
                has_auth = True
            if hasattr(trunk, 'inbound_password') and trunk.inbound_password:
                print(f"   ⚠️  Inbound Password: [SET]")
                has_auth = True
            
            if not has_auth:
                print(f"   ✅ No authentication required")
            
            print()
            
            # Print full trunk object for debugging
            print("   Full Configuration:")
            print(f"   {trunk}")
            print()
        
        # Get dispatch rules
        rules = await livekit_api.sip.list_sip_dispatch_rule(
            api.ListSIPDispatchRuleRequest()
        )
        
        print("=" * 70)
        print("Dispatch Rules")
        print("=" * 70)
        print()
        
        for rule in rules.items:
            print(f"📋 Rule: {rule.name}")
            print(f"   ID: {rule.sip_dispatch_rule_id}")
            print(f"   Trunk IDs: {rule.trunk_ids}")
            print(f"   Hide Phone Number: {rule.hide_phone_number}")
            print()
        
        print("=" * 70)
        print("Diagnosis")
        print("=" * 70)
        print()
        
        if len(trunks.items) == 0:
            print("❌ NO INBOUND TRUNKS FOUND!")
            print("   You need to create an inbound trunk in LiveKit.")
            return
        
        trunk = trunks.items[0]
        
        # Check allowed addresses
        if not trunk.allowed_addresses or len(trunk.allowed_addresses) == 0:
            print("❌ No allowed addresses configured!")
            print("   Add 'allowedAddresses: [\"0.0.0.0/0\"]' to your trunk.")
        elif "0.0.0.0/0" in trunk.allowed_addresses:
            print("✅ Trunk allows calls from any IP (0.0.0.0/0)")
        else:
            print(f"⚠️  Trunk only allows calls from: {trunk.allowed_addresses}")
            print("   This might block Telnyx servers.")
        
        print()
        
        # Check dispatch rules
        if len(rules.items) == 0:
            print("❌ NO DISPATCH RULES FOUND!")
            print("   You need a dispatch rule to route SIP calls to rooms.")
        else:
            print(f"✅ Found {len(rules.items)} dispatch rule(s)")
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(check_trunk_config())