# test_milvus.py
from pymilvus import MilvusClient, connections
import os
from dotenv import load_dotenv
from pathlib import Path

# Load your V2 .env
BASE_DIR = Path(__file__).resolve().parent / "recruiterbrainv2"
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path, override=True)

uri = os.getenv("MILVUS_URI")
token = os.getenv("MILVUS_TOKEN", "")

print("="*60)
print("Testing Milvus Connection")
print("="*60)
print(f"URI: {uri}")
print(f"Token: {'Yes' if token else 'No'}")
print()

# Try connection
try:
    print(f"Attempting connection to: {uri}")
    
    if token:
        # Zilliz Cloud
        client = MilvusClient(uri=uri, token=token, secure=True)
    else:
        # Local/Self-hosted
        client = MilvusClient(uri=uri)
    
    print("✅ Connected!")
    
    # List collections
    collections = client.list_collections()
    print(f"Collections: {collections}")
    
except Exception as e:
    print(f"❌ Connection failed!")
    print(f"Error: {e}")
    print()
    print("Troubleshooting:")
    print("1. Is Milvus running?")
    print("2. Is the IP/port correct?")
    print("3. Can you ping the server?")
    print(f"4. Try: telnet {uri.split('://')[1].split(':')[0]} 19530")