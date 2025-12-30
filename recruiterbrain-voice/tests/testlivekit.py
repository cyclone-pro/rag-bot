import requests
import time
import jwt
import json
from dotenv import load_dotenv
import os
# Load environment variables first
load_dotenv()
# Your LiveKit credentials
LIVEKIT_URL = "https://beyond-wysdsmxq.livekit.cloud"
LIVEKIT_API_KEY = "APImy7HkF4sm8eE"
LIVEKIT_API_SECRET = "f9BewpySqeWCgI60PefldOgB8UuzG4ZMFHKgfeG4oq5F"

def create_access_token(api_key, api_secret):
    """Create a JWT access token for LiveKit API"""
    now = int(time.time())
    exp = now + 3600  # Token expires in 1 hour
    
    payload = {
        "iss": api_key,
        "sub": api_key,
        "iat": now,
        "exp": exp,
        "nbf": now,
        "video": {
            "room": "*",
            "roomAdmin": True,
            "canPublish": True,
            "canSubscribe": True
        }
    }
    
    token = jwt.encode(payload, api_secret, algorithm="HS256")
    return token

def test_livekit_credentials():
    """Test LiveKit API credentials"""
    print("🔑 Testing LiveKit API Credentials...")
    print(f"URL: {LIVEKIT_URL}")
    print(f"API Key: {LIVEKIT_API_KEY[:10]}...")
    print(f"API Secret: {LIVEKIT_API_SECRET[:10]}...")
    print()
    
    try:
        # Create access token
        token = create_access_token(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        print("✅ JWT token created successfully")
        print()
        
        # Test API - List rooms
        print("📡 Testing API connection - Listing rooms...")
        url = f"{LIVEKIT_URL}/twirp/livekit.RoomService/ListRooms"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json={})
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        print()
        
        if response.status_code == 200:
            print("✅ API credentials are VALID!")
            data = response.json()
            rooms = data.get("rooms", [])
            print(f"Found {len(rooms)} rooms")
            return True
        else:
            print("❌ API credentials are INVALID or API error")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing credentials: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Make sure PyJWT is installed
    try:
        import jwt
    except ImportError:
        print("Installing PyJWT...")
        import subprocess
        subprocess.check_call(["pip", "install", "PyJWT"])
        import jwt
    
    test_livekit_credentials()