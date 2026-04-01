"""FastAPI backend for AI Security System - Cloud & WebRTC Compatible."""

import io
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from alert_system import AlertSystem
from utils import FaceEncoder, FaceDetector, create_known_faces_directory
from config import UNKNOWN_PERSON_COLOR, KNOWN_PERSON_COLOR, DISPLAY_CONFIDENCE

app = FastAPI(title="AI Security System API")

# CORS Setup taake Streamlit aur API aapas mein baat kar sakein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Components Initialize
face_encoder = FaceEncoder()
face_detector = FaceDetector()
alert_system = AlertSystem()

# Shared processing state (Sirf stats aur alerts ke liye)
state_lock = threading.Lock() 
state = {
    "stats": {
        "frames_processed": 0,
        "total_detections": 0,
        "unknown_detections": 0,
        "known_persons": 0,
    },
    "current_alert": None,
    "last_alert_time": None
}

@app.on_event("startup")
def startup_event():
    """Startup par database load karna"""
    create_known_faces_directory()
    success = face_encoder.load_known_faces()
    if success:
        with state_lock:
            state["stats"]["known_persons"] = len(set(face_encoder.known_face_names))
    print("Backend Ready: Face Database Loaded.")

@app.get("/")
def root():
    return {"message": "AI Security API is running. Camera is handled by WebRTC Frontend."}

@app.get("/stats")
def stats():
    """Thread-safe retrieval of system statistics."""
    with state_lock:
        return JSONResponse(state["stats"])
        
@app.get("/alerts")
def alerts():
    """Alert summary return karta hai"""
    return JSONResponse(alert_system.get_log_summary())

@app.get("/known")
def known_people():
    """Known faces ki list"""
    return JSONResponse({"known_people": list(set(face_encoder.known_face_names))})

@app.post("/reload_faces")
def reload_faces():
    """Nayi pictures load karne ke liye"""
    success = face_encoder.load_known_faces()
    if success:
        with state_lock:
            state["stats"]["known_persons"] = len(set(face_encoder.known_face_names))
        return {"status": "success", "message": f"Loaded {state['stats']['known_persons']} faces."}
    return {"status": "error", "message": "No faces found."}

# Note: _camera_worker aur MJPEG generator ko hata diya gaya hai 
# kyunke cloud deployment par camera WebRTC (Frontend) sambhalega.