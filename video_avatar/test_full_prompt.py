#!/usr/bin/env python3
"""Test script to debug Bey API with FULL production prompt."""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BEY_API_URL = os.getenv("BEY_API_URL", "https://api.bey.dev/v1")
BEY_API_KEY = os.getenv("BEY_API_KEY")

# Import the actual agent_prompt module
try:
    from agent_prompt import build_agent_config, get_avatar_config
    HAS_AGENT_PROMPT = True
except ImportError:
    HAS_AGENT_PROMPT = False
    print("⚠️  agent_prompt.py not found - using inline prompt")


def test_full_prompt(avatar_key: str = "zara", username: str = "Amit"):
    """Test creating an agent with the FULL production prompt."""
    
    if not BEY_API_KEY:
        print("❌ ERROR: BEY_API_KEY not set in environment")
        return False
    
    print(f"🔑 Using API Key: {BEY_API_KEY[:8]}...{BEY_API_KEY[-4:]}")
    print(f"🌐 API URL: {BEY_API_URL}")
    print(f"🎭 Avatar: {avatar_key}")
    print(f"👤 Username: {username}")
    print()
    
    if HAS_AGENT_PROMPT:
        print("📦 Using agent_prompt.py module")
        avatar_config = get_avatar_config(avatar_key)
        agent_config = build_agent_config(
            call_history=[],  # Empty history
            username=username,
            agent_name=avatar_config["name"],
            avatar_id=avatar_config["id"],
        )
        
        payload = {
            "name": agent_config["name"],
            "system_prompt": agent_config["system_prompt"],
            "greeting": agent_config["greeting"],
            "avatar_id": agent_config["avatar_id"],
        }
    else:
        # Fallback minimal prompt if module not available
        AVATARS = {
            "scott": "b63ba4e6-d346-45d0-ad28-5ddffaac0bd0_v2",
            "sam": "1c7a7291-ee28-4800-8f34-acfbfc2d07c0",
            "zara": "694c83e2-8895-4a98-bd16-56332ca3f449",
        }
        avatar_id = AVATARS.get(avatar_key.lower(), AVATARS["scott"])
        
        # Simulated full prompt (copy the key parts)
        test_prompt = """You are Zara, a senior AI recruitment specialist powered by RCRUTR AI from Elite Solutions.

WHO YOU ARE (VERY IMPORTANT - NEVER CONTRADICT THIS):
You ARE the AI recruiter. You personally DO the sourcing, screening, and candidate work. When the user asks "Can you get me candidates?" — the answer is ALWAYS "Absolutely! I'll find the right candidates for you."

Your capabilities:
- I source candidates from multiple job boards, databases, and networks
- I screen and shortlist candidates based on your requirements
- I conduct voice and chat interviews with candidates
- I score and evaluate candidates with explainable insights
- I schedule interviews automatically
- I maintain compliance and audit trails

DUAL MODE OPERATION:

MODE 1 - PRODUCT QUESTIONS (If they ask about you, RCRUTR, or capabilities):
Answer product questions naturally.

MODE 2 - JOB INTAKE (If they want to submit job requirements):

REQUIRED INFO (MUST CAPTURE FOR EVERY ROLE):
1. Job title
2. Job type (Contract, Contract-to-hire, Full-time, etc.)
3. Pay rate OR salary range
4. Work model (remote, onsite, hybrid, flexible)
5. Work authorization requirements
6. Must-have skills (at least 3–5 REAL technical skills)
7. Company/Client name (ALWAYS ASK)

FULL-TIME SPECIFIC QUESTIONS (MUST ASK FOR FTE ROLES):
- Benefits, equity, bonus structure

SKILL VALIDATION:
- Validate skills are REAL and SPECIFIC
- If gibberish, ask to repeat
- Push for 3-5 minimum skills

Keep responses concise and natural.
"""
        
        payload = {
            "name": f"Zara - RCRUTR AI",
            "system_prompt": test_prompt,
            "greeting": f"Hey {username}! I'm Zara, your AI recruitment specialist. I source, screen, and interview candidates. What role are you looking to fill today?",
            "avatar_id": avatar_id,
        }
    
    print(f"📝 Prompt length: {len(payload['system_prompt'])} chars")
    print(f"👋 Greeting length: {len(payload['greeting'])} chars")
    print(f"📍 Avatar ID: {payload['avatar_id']}")
    print(f"🏷️  Agent name: {payload['name']}")
    print()
    
    # Check limits
    if len(payload['system_prompt']) > 10000:
        print(f"❌ PROMPT TOO LONG! {len(payload['system_prompt'])} > 10000")
        print("   This will fail on Bey API")
        return False
    else:
        print(f"✅ Prompt within limit ({len(payload['system_prompt'])}/10000)")
    
    print()
    print("🚀 Creating agent with FULL prompt...")
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
        
        if response.status_code in (200, 201):
            data = response.json()
            agent_id = data.get("id")
            print(f"✅ SUCCESS! Agent created with FULL prompt")
            print(f"   Agent ID: {agent_id}")
            print()
            
            # Cleanup
            print("🗑️  Cleaning up test agent...")
            delete_resp = requests.delete(
                f"{BEY_API_URL}/agent/{agent_id}",
                headers={"x-api-key": BEY_API_KEY},
                timeout=30,
            )
            if delete_resp.status_code in (200, 204, 404):
                print("   ✅ Deleted")
            
            print()
            print("=" * 50)
            print("✅ FULL PROMPT WORKS!")
            print("   The issue is NOT the prompt length or format.")
            print("   The issue is likely in Cloud Run deployment:")
            print("   - Check if BEY_API_KEY secret is set correctly")
            print("   - Check Cloud Run logs for actual error")
            print("=" * 50)
            return True
        else:
            print(f"❌ FAILED!")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    avatar = sys.argv[1] if len(sys.argv) > 1 else "zara"
    username = sys.argv[2] if len(sys.argv) > 2 else "Amit"
    
    test_full_prompt(avatar, username)