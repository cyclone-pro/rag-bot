#!/usr/bin/env python3
"""
Create LiveKit INBOUND Trunk for Telnyx
This allows Telnyx to call INTO LiveKit
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Telnyx number
TELNYX_PHONE_NUMBER = "+17792571297"


async def create_inbound_trunk_and_dispatch():
    """Create inbound trunk and dispatch rule"""
    
    print("=" * 70)
    print("Creating LiveKit INBOUND Trunk for Telnyx")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Step 1: Create Inbound Trunk
        print("Step 1: Creating inbound trunk...")
        print(f"  Number: {TELNYX_PHONE_NUMBER}")
        print()
        
        inbound_trunk_info = api.SIPInboundTrunkInfo(
            name="Telnyx-Inbound",
            numbers=[TELNYX_PHONE_NUMBER],
            allowed_addresses=["0.0.0.0/0"],  # Allow from any IP
            allowed_numbers=[],
        )
        
        inbound_request = api.CreateSIPInboundTrunkRequest(trunk=inbound_trunk_info)
        inbound_trunk = await livekit_api.sip.create_sip_inbound_trunk(inbound_request)
        
        print(f"✅ Inbound trunk created: {inbound_trunk.sip_trunk_id}")
        print()
        
        # Step 2: Create Dispatch Rule
        print("Step 2: Creating dispatch rule...")
        print("  This routes incoming calls to rooms based on SIP URI")
        print()
        
        dispatch_rule_info = api.SIPDispatchRuleInfo(
            name="Interview-Dispatch",
            trunk_ids=[inbound_trunk.sip_trunk_id],
            rule=api.SIPDispatchRule(
                dispatch_rule_direct=api.SIPDispatchRuleDirect(
                    room_name="*",  # Wildcard - use room name from SIP URI
                    pin=""
                )
            )
        )
        
        dispatch_request = api.CreateSIPDispatchRuleRequest(rule=dispatch_rule_info)
        dispatch_rule = await livekit_api.sip.create_sip_dispatch_rule(dispatch_request)
        
        print(f"✅ Dispatch rule created: {dispatch_rule.sip_dispatch_rule_id}")
        print()
        
        print("=" * 70)
        print("SUCCESS! Configuration Complete")
        print("=" * 70)
        print()
        print("Inbound Configuration:")
        print(f"  Trunk ID: {inbound_trunk.sip_trunk_id}")
        print(f"  Number: {TELNYX_PHONE_NUMBER}")
        print(f"  Dispatch Rule: {dispatch_rule.sip_dispatch_rule_id}")
        print()
        print("Now Telnyx can call INTO LiveKit!")
        print("When you transfer a call to sip:room_name@54jym8vfe6a.sip.livekit.cloud")
        print("LiveKit will accept it and route to that room.")
        print()
        
        return inbound_trunk, dispatch_rule
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(create_inbound_trunk_and_dispatch())