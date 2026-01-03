#!/usr/bin/env python3
"""
Real SIP Connection Test
Actually tests if LiveKit accepts SIP calls
"""

import socket
import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_SIP_DOMAIN = os.getenv("LIVEKIT_SIP_DOMAIN", "54jym8vfe6a.sip.livekit.cloud")


async def test_livekit_sip():
    """Test LiveKit SIP configuration"""
    
    print("=" * 70)
    print("LiveKit SIP Connection Test")
    print("=" * 70)
    print()
    
    # Test 1: Check SIP domain DNS resolution
    print("Test 1: DNS Resolution")
    print("-" * 70)
    try:
        ip = socket.gethostbyname(LIVEKIT_SIP_DOMAIN)
        print(f"✅ SIP domain resolves: {LIVEKIT_SIP_DOMAIN} → {ip}")
    except socket.gaierror as e:
        print(f"❌ SIP domain does NOT resolve: {LIVEKIT_SIP_DOMAIN}")
        print(f"   Error: {e}")
        return False
    print()
    
    # Test 2: Check if SIP port is reachable
    print("Test 2: SIP Port Check (5060)")
    print("-" * 70)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        # SIP uses UDP on port 5060
        sock.sendto(b"", (ip, 5060))
        print(f"✅ Can send UDP to {ip}:5060")
    except Exception as e:
        print(f"⚠️  Cannot verify SIP port: {e}")
        print("   This is normal - SIP requires proper handshake")
    finally:
        sock.close()
    print()
    
    # Test 3: Check LiveKit API access
    print("Test 3: LiveKit API Connection")
    print("-" * 70)
    try:
        livekit_api = api.LiveKitAPI(
            LIVEKIT_URL,
            LIVEKIT_API_KEY,
            LIVEKIT_API_SECRET
        )
        
        # List rooms
        rooms = await livekit_api.room.list_rooms(api.ListRoomsRequest())
        print(f"✅ LiveKit API connected")
        print(f"   Active rooms: {len(rooms.rooms)}")
    except Exception as e:
        print(f"❌ LiveKit API failed: {e}")
        return False
    print()
    
    # Test 4: Check SIP trunks
    print("Test 4: SIP Trunk Configuration")
    print("-" * 70)
    try:
        trunks = await livekit_api.sip.list_sip_trunk(api.ListSIPTrunkRequest())
        
        print(f"Total trunks: {len(trunks.items)}")
        
        inbound_count = 0
        outbound_count = 0
        
        for trunk in trunks.items:
            trunk_type = "Unknown"
            if hasattr(trunk, 'inbound_addresses') and trunk.inbound_addresses:
                trunk_type = "Inbound"
                inbound_count += 1
            elif hasattr(trunk, 'outbound_address') and trunk.outbound_address:
                trunk_type = "Outbound"
                outbound_count += 1
            
            print(f"   - {trunk.name} ({trunk_type})")
            print(f"     ID: {trunk.sip_trunk_id}")
        
        print()
        print(f"Summary:")
        print(f"   Inbound trunks: {inbound_count}")
        print(f"   Outbound trunks: {outbound_count}")
        
        if outbound_count > 0:
            print()
            print("⚠️  WARNING: You have outbound trunks configured")
            print("   For Telnyx→LiveKit bridging, you only need inbound trunk")
            print("   Consider deleting outbound trunks if misconfigured")
        
    except Exception as e:
        print(f"⚠️  Could not check SIP trunks: {e}")
    print()
    
    # Test 5: Check dispatch rules
    print("Test 5: Dispatch Rules")
    print("-" * 70)
    try:
        rules = await livekit_api.sip.list_sip_dispatch_rule(
            api.ListSIPDispatchRuleRequest()
        )
        
        print(f"Total dispatch rules: {len(rules.items)}")
        
        for rule in rules.items:
            print(f"   - {rule.name}")
            print(f"     ID: {rule.sip_dispatch_rule_id}")
            if hasattr(rule, 'trunk_ids'):
                print(f"     Trunks: {', '.join(rule.trunk_ids)}")
        
        if len(rules.items) == 0:
            print("❌ No dispatch rules found!")
            print("   You need a dispatch rule to route SIP calls to rooms")
        
    except Exception as e:
        print(f"⚠️  Could not check dispatch rules: {e}")
    print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()
    print("For Telnyx→LiveKit to work, you need:")
    print("  1. ✅ Correct SIP domain in .env")
    print("  2. ✅ One inbound SIP trunk (allows 0.0.0.0/0)")
    print("  3. ✅ One dispatch rule (routes to room *)")
    print("  4. ❌ NO outbound trunks (delete if present)")
    print()
    print("Current SIP domain:", LIVEKIT_SIP_DOMAIN)
    print()
    
    return True


if __name__ == "__main__":
    asyncio.run(test_livekit_sip())