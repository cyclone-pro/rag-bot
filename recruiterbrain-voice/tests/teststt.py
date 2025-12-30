"""
Test Deepgram STT
"""
from livekit.plugins import deepgram
import os

# Load API key from environment
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

if not deepgram_api_key:
    print("❌ DEEPGRAM_API_KEY not set!")
    exit(1)

print("Testing Deepgram STT initialization...")

try:
    stt = deepgram.STT(
        model="nova-2",
        language="en-US"
    )
    print("✅ Deepgram STT initialized successfully!")
    print(f"   Model: nova-2")
    print(f"   Language: en-US")
except Exception as e:
    print(f"❌ Deepgram STT failed: {e}")
    import traceback
    traceback.print_exc()