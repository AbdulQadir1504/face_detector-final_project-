import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import requests
import numpy as np
import os  # ADD THIS LINE - For environment variables

# CHANGE THIS LINE - Use environment variable or disable API features
# API_URL = "http://localhost:8000"  # DELETE/COMMENT THIS LINE
API_URL = os.environ.get("API_URL", "")  # ADD THIS LINE - Empty means no backend

st.set_page_config(page_title="AI Security Dashboard", layout="wide")

st.title("🛡️ AI Security System - Live WebRTC Dashboard")

# --- FACE RECOGNITION CALLBACK - COMPLETELY REPLACE THIS FUNCTION ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # STEP 5 CODE - Local face detection (no backend needed)
    # Load face detection model locally
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add text showing number of faces
    cv2.putText(img, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed (WebRTC)")
    webrtc_streamer(
        key="security-camera",
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 System Control & Stats")
    
    # STEP 4 CODE - Modified Sync Button with better error handling
    # Sync Button
    if st.button("🔄 Sync Known Faces"):
        if API_URL:  # Only try if backend URL exists
            try:
                res = requests.post(f"{API_URL}/reload_faces", timeout=5)
                if res.status_code == 200:
                    st.success("Database Updated!")
                else:
                    st.error("Backend Error")
            except requests.exceptions.ConnectionError:  # More specific error
                st.warning("⚠️ Backend API not available. Using local face detection only.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.info("ℹ️ Backend API not configured. Using local face detection.")

    # Stats Display - STEP 4 CODE with better error handling
    st.markdown("---")
    stats_placeholder = st.empty()
    
    if API_URL:  # Only try if backend URL exists
        try:
            r = requests.get(f"{API_URL}/stats", timeout=1)
            if r.status_code == 200:
                stats_placeholder.json(r.json())
            else:
                stats_placeholder.info("Waiting for statistics...")
        except requests.exceptions.ConnectionError:
            stats_placeholder.info("📹 Local face detection active (no backend connection)")
        except:
            stats_placeholder.info("Using local face detection mode")
    else:
        stats_placeholder.success("✅ Local face detection is active! Faces will be detected in real-time.")

st.sidebar.markdown("### Deployment Info")
st.sidebar.info("🎥 Click 'Start' to begin face detection. Green boxes will appear around detected faces.")
