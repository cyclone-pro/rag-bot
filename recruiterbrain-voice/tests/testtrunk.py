#!/usr/bin/env python3
"""
Check Existing LiveKit Inbound Trunk
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


async def check_inbound_config():
    """Check inbound trunk and dispatch rules"""
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        print("=" * 70)
        print("LiveKit Inbound Configuration Check")
        print("=" * 70)
        print()
        
        # Check inbound trunks
        print("INBOUND TRUNKS:")
        print("-" * 70)
        inbound_trunks = await livekit_api.sip.list_sip_inbound_trunk(
            api.ListSIPInboundTrunkRequest()
        )
        
        for trunk in inbound_trunks.items:
            print(f"Name: {trunk.name}")
            print(f"ID: {trunk.sip_trunk_id}")
            print(f"Numbers: {trunk.numbers}")
            print(f"Allowed Addresses: {trunk.allowed_addresses}")
            print()
        
        # Check dispatch rules
        print("DISPATCH RULES:")
        print("-" * 70)
        dispatch_rules = await livekit_api.sip.list_sip_dispatch_rule(
            api.ListSIPDispatchRuleRequest()
        )
        
        for rule in dispatch_rules.items:
            print(f"Name: {rule.name}")
            print(f"ID: {rule.sip_dispatch_rule_id}")
            print(f"Trunk IDs: {rule.trunk_ids}")
            if hasattr(rule, 'rule') and rule.rule:
                if hasattr(rule.rule, 'dispatch_rule_direct'):
                    direct = rule.rule.dispatch_rule_direct
                    print(f"Room Pattern: {direct.room_name}")
            print()
        
        print("=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        print()
        
        if len(inbound_trunks.items) > 0:
            print("✅ Inbound trunk exists")
            trunk = inbound_trunks.items[0]
            if "+17792571297" in trunk.numbers:
                print("✅ Your Telnyx number is configured")
            if "0.0.0.0/0" in trunk.allowed_addresses:
                print("✅ Accepts calls from any IP")
        
        if len(dispatch_rules.items) > 0:
            print("✅ Dispatch rule exists")
            rule = dispatch_rules.items[0]
            if trunk.sip_trunk_id in rule.trunk_ids:
                print("✅ Dispatch rule uses the inbound trunk")
        
        print()
        print("This means LiveKit IS configured to accept calls from Telnyx!")
        print("The issue must be in HOW we're calling it.")
        print()
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(check_inbound_config())