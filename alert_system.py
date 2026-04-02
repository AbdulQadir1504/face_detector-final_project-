import logging
from datetime import datetime
import os

class AlertSystem:
    def __init__(self, log_file="security_alerts.log"):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_alert(self, message, level="INFO"):
        """Log an alert message"""
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        
        # Also print for Streamlit
        print(f"[{level}] {message}")
    
    def get_recent_alerts(self, count=10):
        """Get recent alerts from log file"""
        if not os.path.exists(self.log_file):
            return []
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            return lines[-count:]
