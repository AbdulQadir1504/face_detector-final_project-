class Config:
    # Face Detection Settings
    FACE_DETECTION_SCALE_FACTOR = 1.1
    FACE_DETECTION_MIN_NEIGHBORS = 5
    FACE_DETECTION_MIN_SIZE = (30, 30)
    
    # Paths
    KNOWN_FACES_PATH = "known_faces"
    ALERT_LOG_PATH = "security_alerts.log"
    
    # Alert Settings
    ALERT_COOLDOWN_SECONDS = 5
    
    # Display Settings
    DISPLAY_FACE_RECTANGLE = True
    DISPLAY_FACE_NAME = True
    DISPLAY_CONFIDENCE = True
