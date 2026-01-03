#!/usr/bin/env python3
"""
Setup LiveKit SIP Trunks and Dispatch Rules for Telnyx Integration
This script creates the necessary inbound trunk and dispatch rule for RecruiterBrain voice interviews
"""

import os
import requests
import json
from livekit import api

# Load environment variables
LIVEKIT_API_KEY="APIuNWfZKF6yX7R"
LIVEKIT_API_SECRET="ZuNnTIHiUU1mZM2F15NiP1GwAPY7SLWVtKZ7K96eYzL"
LIVEKIT_URL = "https://beyond-wysdsmxq.livekit.cloud"
TELNYX_PHONE_NUMBER = os.getenv("TELNYX_PHONE_NUMBER")  # Your Telnyx number in E.164 format

def create_inbound_trunk():
    """Create LiveKit Inbound Trunk to accept calls from Telnyx"""
    
    # Initialize LiveKit API
    livekit_api = api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET
    )
    
    # Create inbound trunk configuration
    # This accepts calls from ANY phone number (empty numbers array)
    # You can restrict to specific numbers by adding them to the array
    trunk_config = {
        "trunk": {
            "name": "Telnyx Inbound - RecruiterBrain",
            "numbers": [],  # Empty = accept from any number
            "allowed_addresses": [],  # Empty = accept from any IP (Telnyx uses dynamic IPs)
            "allowed_numbers": [],  # Empty = accept to any destination number
            "metadata": json.dumps({
                "provider": "telnyx",
                "purpose": "voice_interviews"
            })
        }
    }
    
    print("📞 Creating LiveKit Inbound Trunk...")
    print(f"Configuration: {json.dumps(trunk_config, indent=2)}")
    
    try:
        # Make API call to create inbound trunk
        endpoint = f"{LIVEKIT_URL}/twirp/livekit.SIP/CreateSIPInboundTrunk"
        
        # Create access token
        token = livekit_api.access_token.with_grants(
            api.VideoGrants(sip_admin=True)
        ).to_jwt()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            endpoint,
            headers=headers,
            json=trunk_config
        )
        
        response.raise_for_status()
        trunk_info = response.json()
        
        print(f"✅ Inbound Trunk Created Successfully!")
        print(f"Trunk ID: {trunk_info.get('trunk_id', 'N/A')}")
        print(f"Trunk Info: {json.dumps(trunk_info, indent=2)}")
        
        return trunk_info
        
    except Exception as e:
        print(f"❌ Error creating inbound trunk: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        raise


def create_dispatch_rule(trunk_id):
    """Create Dispatch Rule to route incoming calls to interview rooms"""
    
    livekit_api = api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET
    )
    
    # Create dispatch rule that routes calls to rooms based on the SIP request
    # The room name will be extracted from the SIP Request-URI
    dispatch_config = {
        "rule": {
            "dispatchRuleDirect": {
                "roomName": "",  # Empty means use the room from SIP Request-URI
                "pin": ""  # No PIN required
            }
        },
        "trunk_ids": [trunk_id],
        "name": "Interview Room Dispatch",
        "hide_phone_number": False,  # Show caller's phone number in room
        "metadata": json.dumps({
            "app": "recruiterbrain",
            "version": "1.0"
        })
    }
    
    print("\n📋 Creating Dispatch Rule...")
    print(f"Configuration: {json.dumps(dispatch_config, indent=2)}")
    
    try:
        endpoint = f"{LIVEKIT_URL}/twirp/livekit.SIP/CreateSIPDispatchRule"
        
        token = livekit_api.access_token.with_grants(
            api.VideoGrants(sip_admin=True)
        ).to_jwt()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            endpoint,
            headers=headers,
            json=dispatch_config
        )
        
        response.raise_for_status()
        rule_info = response.json()
        
        print(f"✅ Dispatch Rule Created Successfully!")
        print(f"Rule ID: {rule_info.get('sip_dispatch_rule_id', 'N/A')}")
        print(f"Rule Info: {json.dumps(rule_info, indent=2)}")
        
        return rule_info
        
    except Exception as e:
        print(f"❌ Error creating dispatch rule: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        raise


def list_existing_trunks():
    """List all existing SIP trunks"""
    
    livekit_api = api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET
    )
    
    print("\n📋 Listing Existing Trunks and Dispatch Rules...\n")
    
    try:
        # List inbound trunks
        endpoint = f"{LIVEKIT_URL}/twirp/livekit.SIP/ListSIPInboundTrunk"
        
        token = livekit_api.access_token.with_grants(
            api.VideoGrants(sip_admin=True)
        ).to_jwt()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(endpoint, headers=headers, json={})
        response.raise_for_status()
        trunks = response.json()
        
        print("Inbound Trunks:")
        print(json.dumps(trunks, indent=2))
        
        # List dispatch rules
        endpoint = f"{LIVEKIT_URL}/twirp/livekit.SIP/ListSIPDispatchRule"
        response = requests.post(endpoint, headers=headers, json={})
        response.raise_for_status()
        rules = response.json()
        
        print("\nDispatch Rules:")
        print(json.dumps(rules, indent=2))
        
    except Exception as e:
        print(f"❌ Error listing trunks: {e}")


def main():
    print("=" * 60)
    print("LiveKit SIP Trunk Setup for RecruiterBrain Voice Interviews")
    print("=" * 60)
    
    # Validate environment variables
    if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        print("❌ Missing required environment variables:")
        print("   - LIVEKIT_API_KEY")
        print("   - LIVEKIT_API_SECRET")
        return
    
    print(f"\n🔧 LiveKit URL: {LIVEKIT_URL}")
    print(f"🔑 API Key: {LIVEKIT_API_KEY[:10]}...")
    
    # Check if trunks already exist
    print("\n" + "=" * 60)
    list_existing_trunks()
    print("=" * 60)
    
    proceed = input("\n⚠️  Do you want to create new trunk and dispatch rule? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Exiting without changes.")
        return
    
    # Create inbound trunk
    trunk_info = create_inbound_trunk()
    trunk_id = trunk_info.get('trunk_id')
    
    if not trunk_id:
        print("❌ Failed to get trunk ID. Cannot create dispatch rule.")
        return
    
    # Create dispatch rule
    create_dispatch_rule(trunk_id)
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\n📝 Next Steps:")
    print("1. Verify the trunk and dispatch rule were created")
    print("2. Make a test call to your Telnyx number")
    print("3. Check LiveKit Cloud dashboard for incoming SIP participant")
    print("4. Monitor your agent logs for connection")
    print("\n💡 The dispatch rule will automatically route calls to rooms")
    print("   based on the SIP Request-URI (e.g., interview_abc123)")


if __name__ == "__main__":
    main()