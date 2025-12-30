"""
Test OpenAI TTS
"""
from livekit.plugins import openai
import os

# Load API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ OPENAI_API_KEY not set!")
    exit(1)

print("Testing OpenAI TTS initialization...")

try:
    tts = openai.TTS(voice="alloy")
    print("✅ OpenAI TTS initialized successfully!")
    print(f"   Voice: alloy")
except Exception as e:
    print(f"❌ OpenAI TTS failed: {e}")
    import traceback
    traceback.print_exc()