#!/usr/bin/env python3
"""
Complete Twilio + LiveKit SIP Integration
This will configure everything needed for voice interviews via Twilio
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os

load_dotenv()

# LiveKit credentials
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Twilio SIP configuration
# For Twilio, you connect to your specific subdomain
TWILIO_SIP_DOMAIN = "sip.twilio.com"


async def setup_twilio_livekit():
    """
    Complete setup for Twilio + LiveKit integration
    """
    
    print("=" * 70)
    print("Twilio + LiveKit SIP Integration Setup")
    print("=" * 70)
    print()
    print("This will configure:")
    print("1. LiveKit outbound trunk for Twilio")
    print("2. Authentication using Twilio credentials")
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    try:
        # Clean up existing trunks
        print("Step 1: Cleaning up old trunks...")
        trunks = await livekit_api.sip.list_sip_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
        )
        
        for trunk in trunks.items:
            print(f"  Deleting: {trunk.name}")
            await livekit_api.sip.delete_sip_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=trunk.sip_trunk_id)
            )
        
        print("✅ Cleanup complete")
        print()
        
        # Create Twilio trunk
        print("Step 2: Creating Twilio SIP trunk...")
        print(f"  Domain: {TWILIO_SIP_DOMAIN}")
        print(f"  Phone: {TWILIO_PHONE_NUMBER}")
        print(f"  Auth: Account SID + Auth Token")
        print()
        
        trunk_info = api.SIPOutboundTrunkInfo(
            name="Twilio-Trunk",
            address=TWILIO_SIP_DOMAIN,
            numbers=[TWILIO_PHONE_NUMBER],
            # Twilio uses Account SID as username, Auth Token as password
            auth_username=TWILIO_ACCOUNT_SID,
            auth_password=TWILIO_AUTH_TOKEN,
        )
        
        request = api.CreateSIPOutboundTrunkRequest(trunk=trunk_info)
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        print("=" * 70)
        print("✅ SUCCESS! Twilio Trunk Created")
        print("=" * 70)
        print()
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Name: {trunk.name}")
        print(f"Address: {trunk.address}")
        print(f"Number: {trunk.numbers}")
        print()
        print("=" * 70)
        print("Configuration Summary")
        print("=" * 70)
        print()
        print("Twilio will:")
        print("  1. Receive outbound call request from LiveKit")
        print("  2. Authenticate using Account SID + Auth Token")
        print("  3. Make the call from your Twilio number")
        print("  4. Connect the candidate to LiveKit room")
        print()
        print("=" * 70)
        print("UPDATE YOUR .ENV FILE")
        print("=" * 70)
        print()
        print(f"LIVEKIT_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")
        print(f"TWILIO_ACCOUNT_SID={TWILIO_ACCOUNT_SID}")
        print(f"TWILIO_AUTH_TOKEN={TWILIO_AUTH_TOKEN}")
        print(f"TWILIO_PHONE_NUMBER={TWILIO_PHONE_NUMBER}")
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
    trunk = asyncio.run(setup_twilio_livekit())
    
    if trunk:
        print("=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print()
        print("1. Add the trunk ID to your .env file (see above)")
        print("2. Restart your API server")
        print("3. Test the interview - YOUR PHONE WILL RING!")
        print()
        print("Twilio trial accounts work great for testing!")
        print("You'll hear a message about it being a trial, then the call connects.")
        print()