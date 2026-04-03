"""
Alert system for handling security events
Optimized for Streamlit Cloud deployment
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from config import ALERT_LOG_FILE, ALERT_TRIGGER_COOLDOWN, MAX_ALERT_HISTORY

class AlertSystem:
    """Handles security alerts and logging"""
    
    def __init__(self):
        self.alert_log_file = ALERT_LOG_FILE
        self.alert_history = []
        self.last_alert_time = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(self.alert_log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def can_trigger_alert(self, person_name, cooldown=None):
        """
        Check if an alert can be triggered for a person
        
        Args:
            person_name: Name of the person (or "Unknown")
            cooldown: Cooldown period in seconds
            
        Returns:
            Boolean indicating if alert can be triggered
        """
        if cooldown is None:
            cooldown = ALERT_TRIGGER_COOLDOWN
        
        current_time = time.time()
        last_alert = self.last_alert_time.get(person_name, 0)
        
        return (current_time - last_alert) >= cooldown
    
    def trigger_alert(self, person_name, confidence, location="Entrance"):
        """
        Trigger a security alert
        
        Args:
            person_name: Name of the person detected
            confidence: Recognition confidence
            location: Location of detection
            
        Returns:
            Alert dictionary
        """
        # Check cooldown
        if not self.can_trigger_alert(person_name):
            return None
        
        # Create alert record
        alert = {
            'timestamp': datetime.now(),
            'person': person_name,
            'confidence': confidence,
            'location': location,
            'status': 'INTRUSION_ALERT' if person_name == "Unknown" else 'KNOWN_PERSON'
        }
        
        # Log alert
        if person_name == "Unknown":
            self.logger.warning(
                f"🚨 INTRUSION ALERT | Unknown Person | "
                f"Confidence: {confidence:.2f} | Location: {location}"
            )
        else:
            self.logger.info(
                f"✓ Known Person | {person_name} | "
                f"Confidence: {confidence:.2f} | Location: {location}"
            )
        
        # Store in history
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > MAX_ALERT_HISTORY:
            self.alert_history = self.alert_history[-MAX_ALERT_HISTORY:]
        
        # Update last alert time
        self.last_alert_time[person_name] = time.time()
        
        return alert
    
    def log_known_person(self, person_name, confidence):
        """
        Log a known person detection (without triggering full alert)
        
        Args:
            person_name: Name of known person
            confidence: Recognition confidence
        """
        self.logger.debug(
            f"Known person detected: {person_name} (confidence: {confidence:.2f})"
        )
    
    def get_alert_summary(self):
        """
        Get summary of alerts
        
        Returns:
            Dictionary with alert statistics
        """
        total_alerts = len([a for a in self.alert_history if a['status'] == 'INTRUSION_ALERT'])
        known_detections = len([a for a in self.alert_history if a['status'] == 'KNOWN_PERSON'])
        
        return {
            'total_alerts': total_alerts,
            'known_detections': known_detections,
            'total_events': len(self.alert_history),
            'last_alert': self.alert_history[-1] if self.alert_history else None,
            'log_file': str(self.alert_log_file)
        }
    
    def get_recent_alerts(self, count=10):
        """
        Get recent alerts
        
        Args:
            count: Number of recent alerts to retrieve
            
        Returns:
            List of recent alerts
        """
        return self.alert_history[-count:][::-1]
    
    def clear_history(self):
        """Clear alert history"""
        self.alert_history = []
        self.last_alert_time = {}
        self.logger.info("Alert history cleared")


def display_alert_on_frame(frame, alert):
    """
    Display alert overlay on frame
    
    Args:
        frame: Image frame
        alert: Alert dictionary
        
    Returns:
        Frame with alert overlay
    """
    import cv2
    
    if not alert:
        return frame
    
    height, width = frame.shape[:2]
    
    # Create overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 255), -1)
    
    # Blend overlay
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Add alert text
    alert_text = f"🚨 INTRUSION ALERT! Unknown Person Detected"
    cv2.putText(
        frame,
        alert_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    # Add confidence text
    confidence_text = f"Confidence: {alert.get('confidence', 0):.2f}"
    cv2.putText(
        frame,
        confidence_text,
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    return frame
