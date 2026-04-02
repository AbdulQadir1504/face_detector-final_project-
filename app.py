import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json

# Import your project modules
from utils import FaceDetector
from alert_system import AlertSystem
from config import Config

st.set_page_config(
    page_title="Face Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Security System - Face Detection Dashboard")

# Initialize your components
@st.cache_resource
def init_face_detector():
    return FaceDetector()

@st.cache_resource
def init_alert_system():
    return AlertSystem()

try:
    face_detector = init_face_detector()
    alert_system = init_alert_system()
    st.success("✅ System initialized successfully")
except Exception as e:
    st.error(f"⚠️ Initialization error: {str(e)}")
    st.info("Please ensure known_faces directory has images")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📸 Image Upload", "📹 Live Camera", "📊 Alerts Log", "⚙️ Settings"])

# TAB 1: Image Upload
with tab1:
    st.subheader("Upload Image for Face Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Process image for face detection
        try:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = face_detector.detect_faces(img_bgr)
            
            # Draw rectangles on detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_bgr, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert back to RGB for display
            result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(result_img, caption=f"Detected {len(faces)} Faces", use_column_width=True)
            
            # Log the detection
            if len(faces) > 0:
                alert_system.log_alert(f"Detected {len(faces)} face(s) in uploaded image", "INFO")
                st.success(f"✅ {len(faces)} face(s) detected successfully!")
            else:
                st.warning("⚠️ No faces detected in the image")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# TAB 2: Live Camera (Using OpenCV)
with tab2:
    st.subheader("Live Camera Feed")
    st.warning("Note: For live camera, ensure your browser has camera permissions")
    
    # Simple webcam capture
    run = st.checkbox('Start Camera')
    frame_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        st.info("Camera is running. Click 'Start Camera' checkbox again to stop.")
        
        while run:
            ret, frame = cap.read()
            if ret:
                # Detect faces in real-time
                faces = face_detector.detect_faces(frame)
                
                # Draw rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convert for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            else:
                st.error("Could not access camera")
                break
        cap.release()

# TAB 3: Alert Logs
with tab3:
    st.subheader("Security Alerts Log")
    
    # Display alerts from log file
    if os.path.exists("security_alerts.log"):
        with open("security_alerts.log", "r") as f:
            logs = f.readlines()
        
        if logs:
            # Show last 50 alerts
            for log in logs[-50:]:
                st.text(log.strip())
        else:
            st.info("No alerts logged yet")
    else:
        st.info("Alert log file not found. Alerts will be created when faces are detected.")
    
    if st.button("Clear Logs"):
        if os.path.exists("security_alerts.log"):
            open("security_alerts.log", "w").close()
            st.success("Logs cleared!")
            st.rerun()

# TAB 4: Settings & Info
with tab4:
    st.subheader("System Configuration")
    
    st.write("### Known Faces Directory")
    if os.path.exists("known_faces"):
        known_people = [d for d in os.listdir("known_faces") 
                       if os.path.isdir(os.path.join("known_faces", d))]
        st.write(f"Found {len(known_people)} registered people:")
        for person in known_people:
            st.write(f"- {person}")
    else:
        st.warning("No known_faces directory found")
    
    st.write("### System Status")
    st.json({
        "Status": "Active",
        "Face Detector": "Loaded",
        "Alert System": "Ready",
        "Last Check": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# Sidebar
st.sidebar.markdown("### 📊 Statistics")
st.sidebar.markdown("---")

# Count detections from logs
if os.path.exists("security_alerts.log"):
    with open("security_alerts.log", "r") as f:
        log_lines = f.readlines()
        total_detections = len([l for l in log_lines if "face" in l.lower()])
        st.sidebar.metric("Total Detections", total_detections)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Instructions:**\n"
    "1. Go to 'Image Upload' tab to upload photos\n"
    "2. Use 'Live Camera' for real-time detection\n"
    "3. Check 'Alerts Log' for detection history\n"
    "4. Add known faces to 'known_faces' directory"
)
