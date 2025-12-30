#!/bin/bash
cd ~/Desktop/stuff/rag3/recruiterbrain-voice
source ~/Desktop/stuff/rag3/rag3/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 app/livekit_agent/worker.py dev
