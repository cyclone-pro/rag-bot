"""
Test OpenAI LLM
"""
from livekit.plugins import openai
import os

# Load API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ OPENAI_API_KEY not set!")
    exit(1)

print("Testing OpenAI LLM initialization...")

try:
    llm = openai.LLM()
    print("✅ OpenAI LLM initialized successfully!")
    print(f"   Using default model")
except Exception as e:
    print(f"❌ OpenAI LLM failed: {e}")
    import traceback
    traceback.print_exc()