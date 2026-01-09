from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))

trunk = client.trunking.v1.trunks('TK18e412a60b4f8202c70588743d181b3c').fetch()
print(trunk.transfer_mode)  # Should show codec info