"""Streamlit UI for AI Security System using FastAPI backend."""

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Security Dashboard", layout="wide")

st.title("AI Security System - FastAPI + Streamlit Frontend")

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### Controls")
    load_frame = st.button("Refresh Frame")
    
    # Naya Button for Data Update
    if st.button("🔄 Sync/Update Known Faces"):
        try:
            r = requests.post(f"{API_URL}/reload_faces")
            if r.status_code == 200:
                st.sidebar.success("Database Updated Successfully!")
            else:
                st.sidebar.error("Failed to update faces.")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    interval = st.slider("Refresh interval (seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    show_stream = st.checkbox("Live stream (MJPEG)")

with col1:
    frame_placeholder = st.empty()

status = st.empty()


def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_alert_summary():
    try:
        r = requests.get(f"{API_URL}/alerts", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_frame_bytes():
    try:
        r = requests.get(f"{API_URL}/camera/frame", timeout=2)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None

 def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Yahan hum detection logic ko call kar sakte hain
    # Note: Cloud par 'api_server' ki bajaye direct 'face_detector' call karna behtar hai
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI MEIN REPLACEMENT ---
st.title("Live AI Security System (WebRTC)")

webrtc_streamer(
    key="security-camera",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)    
    while True:
        display_snapshot()
        stats_data = get_stats()
        alert_data = get_alert_summary()

        # 2. Use .container() to overwrite the same spot in the sidebar
        with stats_placeholder.container():
            st.markdown("### System Stats")
            st.write(stats_data)

        with log_placeholder.container():
            st.markdown("### Alert Log Summary")
            st.write(alert_data)

        time.sleep(interval)
if show_stream:
    st.success("Live stream enabled. Make sure FastAPI server is running on port 8000.")
    run_live_loop()
else:
    if load_frame:
        display_snapshot()
        status.info("Snapshot loaded")
    else:
        st.info("Toggle 'Live stream' or click 'Refresh Frame' to start.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Backend endpoints")
st.sidebar.markdown(f"- Camera frame: {API_URL}/camera/frame")
st.sidebar.markdown(f"- MJPEG stream: {API_URL}/camera/stream")
st.sidebar.markdown(f"- Stats: {API_URL}/stats")
st.sidebar.markdown(f"- Alert summary: {API_URL}/alerts")
