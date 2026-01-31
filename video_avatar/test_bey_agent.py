#!/usr/bin/env python3
"""Test script to debug Bey API agent creation issues."""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BEY_API_URL = os.getenv("BEY_API_URL", "https://api.bey.dev/v1")
BEY_API_KEY = os.getenv("BEY_API_KEY")

# Avatar IDs
AVATARS = {
    "scott": "b63ba4e6-d346-45d0-ad28-5ddffaac0bd0_v2",
    "sam": "1c7a7291-ee28-4800-8f34-acfbfc2d07c0",
    "zara": "694c83e2-8895-4a98-bd16-56332ca3f449",
}

def test_agent_creation(avatar_key: str = "zara"):
    """Test creating an agent with minimal prompt."""
    
    if not BEY_API_KEY:
        print("❌ ERROR: BEY_API_KEY not set in environment")
        return
    
    print(f"🔑 Using API Key: {BEY_API_KEY[:8]}...{BEY_API_KEY[-4:]}")
    print(f"🌐 API URL: {BEY_API_URL}")
    print(f"🎭 Avatar: {avatar_key}")
    print()
    
    avatar_id = AVATARS.get(avatar_key.lower())
    if not avatar_id:
        print(f"❌ Unknown avatar: {avatar_key}")
        print(f"   Available: {list(AVATARS.keys())}")
        return
    
    print(f"📍 Avatar ID: {avatar_id}")
    print()
    
    # Minimal test prompt
    test_prompt = """You are a friendly AI assistant for testing.
    
    Keep responses brief and conversational.
    """
    
    test_greeting = "Hello! I'm a test agent. How can I help you?"
    
    payload = {
        "name": f"Test Agent - {avatar_key}",
        "system_prompt": test_prompt,
        "greeting": test_greeting,
        "avatar_id": avatar_id,
    }
    
    print(f"📝 Prompt length: {len(test_prompt)} chars")
    print(f"👋 Greeting length: {len(test_greeting)} chars")
    print()
    print("🚀 Creating agent...")
    print()
    
    try:
        response = requests.post(
            f"{BEY_API_URL}/agent",
            headers={
                "x-api-key": BEY_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Response Headers:")
        for k, v in response.headers.items():
            if k.lower() in ('content-type', 'x-request-id', 'x-ratelimit-remaining'):
                print(f"   {k}: {v}")
        print()
        
        if response.status_code in (200, 201):
            data = response.json()
            agent_id = data.get("id")
            print(f"✅ SUCCESS! Agent created")
            print(f"   Agent ID: {agent_id}")
            print()
            
            # Try to delete the test agent
            print("🗑️  Cleaning up test agent...")
            delete_resp = requests.delete(
                f"{BEY_API_URL}/agent/{agent_id}",
                headers={"x-api-key": BEY_API_KEY},
                timeout=30,
            )
            if delete_resp.status_code in (200, 204, 404):
                print("   ✅ Deleted")
            else:
                print(f"   ⚠️  Delete status: {delete_resp.status_code}")
            
            return True
        else:
            print(f"❌ FAILED!")
            print(f"   Response: {response.text[:500]}")
            
            # Common error analysis
            if response.status_code == 401:
                print()
                print("   💡 Hint: Check if BEY_API_KEY is valid")
            elif response.status_code == 400:
                print()
                print("   💡 Hint: Check avatar_id or prompt format")
            elif response.status_code == 422:
                print()
                print("   💡 Hint: Validation error - check payload format")
            elif response.status_code == 429:
                print()
                print("   💡 Hint: Rate limited - wait and try again")
            
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Request took too long")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_avatars():
    """Test all avatar IDs."""
    print("=" * 50)
    print("Testing all avatars...")
    print("=" * 50)
    print()
    
    results = {}
    for avatar_key in AVATARS:
        print(f"--- Testing {avatar_key} ---")
        results[avatar_key] = test_agent_creation(avatar_key)
        print()
    
    print("=" * 50)
    print("Results:")
    for avatar_key, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {avatar_key}")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            test_all_avatars()
        else:
            test_agent_creation(sys.argv[1])
    else:
        # Default: test zara (the one that's failing)
        test_agent_creation("zara")