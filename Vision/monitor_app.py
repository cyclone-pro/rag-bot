import asyncio
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Interview Monitor")
app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

flags_log: List[Dict[str, Any]] = []


class FlagEvent(BaseModel):
    candidate_id: str
    event_type: str
    timestamp: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


suspicious_keywords = ["hey siri", "ok google", "alexa", "chatgpt", "repeat"]

YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
MAX_ALLOWED_PERSONS = int(os.getenv("MAX_ALLOWED_PERSONS", "1"))
SUSPICIOUS_OBJECTS: Dict[str, str] = {
    "cell phone": "MOBILE_DEVICE",
    "book": "STUDY_MATERIAL",
    "laptop": "EXTERNAL_COMPUTER",
    "tv": "SCREEN",
}


@lru_cache(maxsize=1)
def load_yolo_model() -> YOLO:
    logging.info("Loading YOLO model: %s", YOLO_MODEL_NAME)
    return YOLO(YOLO_MODEL_NAME)


def record_flag(
    *,
    candidate_id: str,
    event_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp_ms: Optional[float] = None,
) -> Dict[str, Any]:
    timestamp = (
        datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        if timestamp_ms is not None
        else datetime.now(tz=timezone.utc)
    )
    entry = {
        "candidate_id": candidate_id,
        "event": event_type,
        "timestamp": timestamp.isoformat(),
        "metadata": metadata or {},
    }
    flags_log.append(entry)
    logging.info("[FLAG] %s", entry)
    return entry


def prepare_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Unable to decode image data") from exc
    return np.array(image)


def analyze_frame(image_bytes: bytes, candidate_id: str) -> Dict[str, Any]:
    model = load_yolo_model()
    frame = prepare_image(image_bytes)
    results = model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)

    if not results:
        return {"detections": [], "new_flags": []}

    detections: List[Dict[str, Any]] = []
    new_flags: List[Dict[str, Any]] = []
    result = results[0]
    boxes = result.boxes

    if boxes is None or boxes.xyxy.shape[0] == 0:
        return {"detections": detections, "new_flags": new_flags}

    names = model.model.names  # type: ignore[attr-defined]
    person_count = 0

    for xyxy, cls_id, conf in zip(
        boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
    ):
        class_id = int(cls_id)
        class_name = names.get(class_id, str(class_id))
        detection = {
            "class": class_name,
            "confidence": float(conf),
            "box": {
                "xmin": float(xyxy[0]),
                "ymin": float(xyxy[1]),
                "xmax": float(xyxy[2]),
                "ymax": float(xyxy[3]),
            },
        }
        detections.append(detection)

        if class_name == "person":
            person_count += 1

        if class_name in SUSPICIOUS_OBJECTS:
            new_flags.append(
                record_flag(
                    candidate_id=candidate_id,
                    event_type=f"SUSPICIOUS_OBJECT_{SUSPICIOUS_OBJECTS[class_name]}",
                    metadata={**detection, "reason": SUSPICIOUS_OBJECTS[class_name]},
                )
            )

    if person_count > MAX_ALLOWED_PERSONS:
        new_flags.append(
            record_flag(
                candidate_id=candidate_id,
                event_type="MULTIPLE_PEOPLE_DETECTED",
                metadata={"person_count": person_count},
            )
        )

    return {"detections": detections, "new_flags": new_flags}


@app.post("/api/flag-event")
async def flag_event(event: FlagEvent):
    entry = record_flag(
        candidate_id=event.candidate_id,
        event_type=event.event_type,
        metadata=event.metadata,
        timestamp_ms=event.timestamp,
    )
    return {"status": "flagged", "flag": entry}


@app.get("/api/log")
async def get_log():
    return JSONResponse(content=flags_log)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interview Monitor</title>
    </head>
    <body>
        <h1>Interview Live Monitoring</h1>
        <p>Upload frames via /api/upload-frame to trigger YOLO analysis.</p>
    </body>
    </html>
    """


def scan_transcript(transcript: str, candidate_id: str) -> List[Dict[str, Any]]:
    new_flags: List[Dict[str, Any]] = []
    lowered = transcript.lower()
    for keyword in suspicious_keywords:
        if keyword in lowered:
            new_flags.append(
                record_flag(
                    candidate_id=candidate_id,
                    event_type="KEYWORD_DETECTED",
                    metadata={"keyword": keyword},
                )
            )
    return new_flags


@app.post("/api/upload-transcript")
async def upload_transcript(
    candidate_id: str = Form(...), transcript_file: UploadFile = File(...)
):
    contents = await transcript_file.read()
    text = contents.decode("utf-8")
    new_flags = scan_transcript(text, candidate_id)
    return {"status": "processed", "candidate_id": candidate_id, "flags": new_flags}


@app.post("/api/evaluate-response-time")
async def evaluate_response_time(
    candidate_id: str = Form(...),
    timestamp_question: float = Form(...),
    timestamp_answer: float = Form(...),
):
    response_time = timestamp_answer - timestamp_question
    flag = None
    if response_time < 10:
        flag = record_flag(
            candidate_id=candidate_id,
            event_type="SUSPICIOUS_FAST_REPLY",
            metadata={"response_time": response_time},
        )
    elif response_time > 300:
        flag = record_flag(
            candidate_id=candidate_id,
            event_type="SUSPICIOUS_DELAYED_REPLY",
            metadata={"response_time": response_time},
        )
    return {"status": "flagged" if flag else "ok", "response_time": response_time}


@app.post("/api/upload-frame")
async def upload_frame(
    candidate_id: str = Form(...),
    frame: UploadFile = File(...),
):
    if frame.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    image_bytes = await frame.read()
    try:
        result = await asyncio.to_thread(analyze_frame, image_bytes, candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "processed", **result}
