#!/usr/bin/env python3
"""Test to capture FULL Bey API error for 422."""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BEY_API_URL = os.getenv("BEY_API_URL", "https://api.bey.dev/v1")
BEY_API_KEY = os.getenv("BEY_API_KEY")

def test_with_full_error_capture():
    """Send the full prompt and capture complete error response."""
    
    if not BEY_API_KEY:
        print("❌ ERROR: BEY_API_KEY not set")
        return
    
    # Import local agent_prompt
    try:
        from agent_prompt import build_agent_config, get_avatar_config
    except ImportError:
        print("❌ ERROR: agent_prompt.py not found in current directory")
        return
    
    # Build the config exactly as webhook does
    avatar_config = get_avatar_config("zara")
    config = build_agent_config(
        call_history=[],
        username="Wahed",
        agent_name=avatar_config["name"],
        avatar_id=avatar_config["id"],
    )
    
    payload = {
        "name": config["name"],
        "system_prompt": config["system_prompt"],
        "greeting": config["greeting"],
        "avatar_id": config["avatar_id"],
    }
    
    print("=" * 60)
    print("PAYLOAD ANALYSIS")
    print("=" * 60)
    print(f"Name: {payload['name']}")
    print(f"Avatar ID: {payload['avatar_id']}")
    print(f"System prompt length: {len(payload['system_prompt'])} chars")
    print(f"Greeting length: {len(payload['greeting'])} chars")
    print()
    
    # Check for problematic content
    import re
    prompt = payload['system_prompt']
    
    # Check for null bytes
    if '\x00' in prompt:
        print("⚠️  WARNING: Null bytes found in prompt!")
    
    # Check for non-ASCII
    non_ascii = set(re.findall(r'[^\x00-\x7F]', prompt))
    if non_ascii:
        print(f"⚠️  Non-ASCII characters: {non_ascii}")
    else:
        print("✅ No non-ASCII characters")
    
    # Check for very long lines
    lines = prompt.split('\n')
    long_lines = [(i, len(l)) for i, l in enumerate(lines) if len(l) > 500]
    if long_lines:
        print(f"⚠️  Long lines (>500 chars): {long_lines}")
    
    # Check for unbalanced braces (might confuse JSON)
    open_braces = prompt.count('{')
    close_braces = prompt.count('}')
    if open_braces != close_braces:
        print(f"⚠️  Unbalanced braces: {{ = {open_braces}, }} = {close_braces}")
    
    print()
    print("=" * 60)
    print("SENDING TO BEY API...")
    print("=" * 60)
    
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
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers:")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")
        print()
        print("Response Body:")
        print("-" * 40)
        
        try:
            # Try to parse as JSON
            resp_json = response.json()
            print(json.dumps(resp_json, indent=2))
        except:
            # Raw text
            print(response.text)
        
        print("-" * 40)
        
        if response.status_code in (200, 201):
            agent_id = response.json().get("id")
            print(f"\n✅ SUCCESS! Agent ID: {agent_id}")
            
            # Cleanup
            print("Cleaning up...")
            requests.delete(
                f"{BEY_API_URL}/agent/{agent_id}",
                headers={"x-api-key": BEY_API_KEY},
            )
            print("Done!")
        else:
            print(f"\n❌ FAILED with {response.status_code}")
            
            # Suggestions based on error
            if response.status_code == 422:
                print("\n💡 422 Unprocessable Entity means validation failed.")
                print("   Possible causes:")
                print("   - Prompt too long (Bey might have a hard limit)")
                print("   - Invalid characters in prompt")
                print("   - Missing required fields")
                print("   - Invalid avatar_id")
                
                # Try with shorter prompt
                print("\n🔄 Trying with shorter prompt...")
                test_short_prompt(payload["avatar_id"])
                
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()


def test_short_prompt(avatar_id: str):
    """Test with minimal prompt to isolate the issue."""
    
    payload = {
        "name": "Test Agent",
        "system_prompt": "You are a helpful assistant. Keep responses brief.",
        "greeting": "Hello! How can I help?",
        "avatar_id": avatar_id,
    }
    
    print(f"   Prompt length: {len(payload['system_prompt'])} chars")
    
    response = requests.post(
        f"{BEY_API_URL}/agent",
        headers={
            "x-api-key": BEY_API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    
    if response.status_code in (200, 201):
        agent_id = response.json().get("id")
        print(f"   ✅ Short prompt works! Agent ID: {agent_id}")
        requests.delete(f"{BEY_API_URL}/agent/{agent_id}", headers={"x-api-key": BEY_API_KEY})
        
        # Now test incrementally
        print("\n🔍 Finding the breaking point...")
        test_incremental_length(avatar_id)
    else:
        print(f"   ❌ Short prompt also failed: {response.status_code}")
        print(f"   Response: {response.text}")


def test_incremental_length(avatar_id: str):
    """Binary search to find max allowed prompt length."""
    
    base_prompt = "You are a helpful AI assistant. "
    
    # Test different lengths
    lengths = [1000, 2000, 4000, 6000, 8000, 9000, 9500, 9800, 10000]
    
    for target_len in lengths:
        # Build prompt of target length
        prompt = base_prompt
        while len(prompt) < target_len:
            prompt += "Keep responses helpful and concise. "
        prompt = prompt[:target_len]
        
        payload = {
            "name": "Length Test Agent",
            "system_prompt": prompt,
            "greeting": "Hi!",
            "avatar_id": avatar_id,
        }
        
        response = requests.post(
            f"{BEY_API_URL}/agent",
            headers={
                "x-api-key": BEY_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        
        status = "✅" if response.status_code in (200, 201) else "❌"
        print(f"   {status} {target_len} chars: {response.status_code}")
        
        if response.status_code in (200, 201):
            agent_id = response.json().get("id")
            requests.delete(f"{BEY_API_URL}/agent/{agent_id}", headers={"x-api-key": BEY_API_KEY})
        
        if response.status_code not in (200, 201):
            print(f"\n   ⚠️  Max working length is between {lengths[lengths.index(target_len)-1]} and {target_len}")
            break


if __name__ == "__main__":
    test_with_full_error_capture()