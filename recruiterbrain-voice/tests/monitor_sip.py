#!/usr/bin/env python3
"""
Monitor LiveKit SIP Calls in Real-Time
This will show if LiveKit is receiving and processing SIP calls
"""

import asyncio
from livekit import api
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


async def monitor_sip_calls():
    """Monitor SIP calls in real-time"""
    
    print("=" * 70)
    print("LiveKit SIP Call Monitor")
    print("=" * 70)
    print()
    print("Monitoring LiveKit for SIP call activity...")
    print("Make a test call now and watch for activity here.")
    print()
    print("Press Ctrl+C to stop monitoring")
    print("=" * 70)
    print()
    
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    last_rooms = set()
    last_participants = {}
    
    try:
        while True:
            # Check rooms
            rooms_response = await livekit_api.room.list_rooms(
                api.ListRoomsRequest()
            )
            
            current_rooms = {room.name for room in rooms_response.rooms}
            
            # New rooms
            new_rooms = current_rooms - last_rooms
            if new_rooms:
                for room_name in new_rooms:
                    room = next(r for r in rooms_response.rooms if r.name == room_name)
                    print(f"🆕 NEW ROOM: {room_name}")
                    print(f"   SID: {room.sid}")
                    print(f"   Participants: {room.num_participants}")
                    print()
            
            # Check participants in each room
            for room in rooms_response.rooms:
                if room.name.startswith("interview_"):
                    participants_response = await livekit_api.room.list_participants(
                        api.ListParticipantsRequest(room=room.name)
                    )
                    
                    current_participants = {
                        p.identity for p in participants_response.participants
                    }
                    
                    prev_participants = last_participants.get(room.name, set())
                    
                    # New participants
                    new_participants = current_participants - prev_participants
                    if new_participants:
                        for identity in new_participants:
                            participant = next(
                                p for p in participants_response.participants 
                                if p.identity == identity
                            )
                            
                            is_sip = any(
                                track.source == api.TrackSource.MICROPHONE
                                for track in participant.tracks
                            )
                            
                            participant_type = "🔊 SIP" if "sip" in identity.lower() else "🤖 Agent"
                            
                            print(f"{participant_type} JOINED: {room.name}")
                            print(f"   Identity: {identity}")
                            print(f"   Name: {participant.name}")
                            print(f"   State: {participant.state}")
                            print(f"   Tracks: {len(participant.tracks)}")
                            
                            # Check for SIP-specific attributes
                            if participant.attributes:
                                print(f"   Attributes:")
                                for key, value in participant.attributes.items():
                                    print(f"      {key}: {value}")
                            
                            print()
                    
                    last_participants[room.name] = current_participants
            
            last_rooms = current_rooms
            
            # Poll every 0.5 seconds
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Monitoring stopped")
        print("=" * 70)
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(monitor_sip_calls())