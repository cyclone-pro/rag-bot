from shared_config import get_openai_client

client = get_openai_client()
print(bool(client))