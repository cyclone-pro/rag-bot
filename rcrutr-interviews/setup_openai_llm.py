#!/usr/bin/env python3
"""
Register OpenAI API as an external LLM with Beyond Presence.

This allows our agents to use GPT-4o-mini instead of Bey's default LLM.

Usage:
  python setup_openai_llm.py

This creates an LLM API configuration and saves the ID to .env
"""

import os
import requests
from dotenv import load_dotenv, set_key

load_dotenv()

BEY_API_URL = "https://api.bey.dev/v1"
BEY_API_KEY = os.getenv("BEY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def create_openai_llm_config():
    """Register OpenAI API with Beyond Presence."""
    
    print("=" * 60)
    print("REGISTERING OPENAI API WITH BEYOND PRESENCE")
    print("=" * 60)
    
    if not BEY_API_KEY:
        print("❌ BEY_API_KEY not set in .env")
        return None
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set in .env")
        return None
    
    print(f"\n1. Creating LLM API configuration...")
    print(f"   API URL: https://api.openai.com/v1")
    
    response = requests.post(
        f"{BEY_API_URL}/external-api",
        headers={"x-api-key": BEY_API_KEY},
        json={
            "name": "OpenAI GPT-4o-mini for RCRUTR",
            "type": "openai_compatible_llm",
            "url": "https://api.openai.com/v1",
            "api_key": OPENAI_API_KEY,
        },
    )
    
    if response.status_code == 201:
        data = response.json()
        llm_api_id = data["id"]
        llm_api_name = data["name"]
        
        print(f"   ✅ Created successfully!")
        print(f"   Name: {llm_api_name}")
        print(f"   ID: {llm_api_id}")
        
        # Save to .env
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        set_key(env_path, "BEY_LLM_API_ID", llm_api_id)
        print(f"\n2. Saved BEY_LLM_API_ID to .env")
        
        return llm_api_id
    else:
        print(f"   ❌ Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None


def list_llm_configs():
    """List existing LLM API configurations."""
    
    print("\nExisting LLM API configurations:")
    
    response = requests.get(
        f"{BEY_API_URL}/external-api",
        headers={"x-api-key": BEY_API_KEY},
    )
    
    if response.status_code == 200:
        configs = response.json()
        if isinstance(configs, dict):
            configs = configs.get("data") or configs.get("items") or []
        if not configs:
            print("   No configurations found")
        for config in configs:
            if not isinstance(config, dict):
                continue
            name = config.get("name", "unknown")
            config_id = config.get("id", "unknown")
            print(f"   - {name} (ID: {config_id})")
        return configs
    else:
        print(f"   Failed to list: {response.status_code}")
        return []


def main():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       SETUP OPENAI LLM FOR BEYOND PRESENCE                 ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Check if already configured
    existing_id = os.getenv("BEY_LLM_API_ID")
    if existing_id:
        print(f"Found existing BEY_LLM_API_ID: {existing_id}")
        print("Verifying it still exists...")
        
        response = requests.get(
            f"{BEY_API_URL}/external-api/{existing_id}",
            headers={"x-api-key": BEY_API_KEY},
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Valid! Name: {data['name']}")
            return existing_id
        else:
            print("⚠️ Configuration not found, creating new one...")
    
    # List existing
    list_llm_configs()
    
    # Create new
    llm_api_id = create_openai_llm_config()
    
    if llm_api_id:
        print("\n" + "=" * 60)
        print("✅ SETUP COMPLETE!")
        print("=" * 60)
        print(f"""
Your agents will now use GPT-4o-mini for conversations.

LLM API ID: {llm_api_id}

This has been saved to your .env file as BEY_LLM_API_ID.
The bey_client.py will automatically use this when creating agents.
""")
    
    return llm_api_id


if __name__ == "__main__":
    main()
