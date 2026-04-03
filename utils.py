"""
Face detection and encoding utilities using DeepFace
Optimized for Streamlit Cloud deployment with TensorFlow backend
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
import time
from deepface import DeepFace
from config import (
    KNOWN_FACES_DIR,
    UNKNOWN_FACE_THRESHOLD,
    MAX_FACES_PER_FRAME,
    ENCODING_CACHE_DURATION
)

class FaceDetector:
    """Handles face detection in images using OpenCV"""
    
    def __init__(self):
        # Use OpenCV's built-in face detector (lighter than DeepFace for detection)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ Face detector initialized (OpenCV Haar Cascade)")
    
    def detect_faces(self, image):
        """
        Detect face locations in image
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            List of face locations as (top, right, bottom, left)
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert OpenCV format (x, y, w, h) to face_recognition format (top, right, bottom, left)
        face_locations = []
        for (x, y, w, h) in faces:
            top = y
            right = x + w
            bottom = y + h
            left = x
            face_locations.append((top, right, bottom, left))
        
        # Limit number of faces
        if len(face_locations) > MAX_FACES_PER_FRAME:
            face_locations = face_locations[:MAX_FACES_PER_FRAME]
        
        return face_locations
    
    def get_face_encodings(self, image, face_locations):
        """
        Get face encodings for detected faces using DeepFace
        
        Args:
            image: BGR image from OpenCV
            face_locations: List of face locations
            
        Returns:
            List of face encodings (embeddings)
        """
        if not face_locations:
            return []
        
        encodings = []
        
        for (top, right, bottom, left) in face_locations:
            try:
                # Extract face ROI
                face_roi = image[top:bottom, left:right]
                
                if face_roi.size == 0:
                    encodings.append(None)
                    continue
                
                # Convert BGR to RGB
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                
                # Get embedding using DeepFace
                # Using VGG-Face model (good balance of accuracy and speed)
                embedding = DeepFace.represent(
                    img_path=face_roi_rgb,
                    model_name='VGG-Face',
                    enforce_detection=False,
                    detector_backend='skip'  # Skip detection since we already have face location
                )
                
                if embedding and len(embedding) > 0:
                    encodings.append(np.array(embedding[0]['embedding']))
                else:
                    encodings.append(None)
                    
            except Exception as e:
                print(f"⚠ Error getting face encoding: {str(e)}")
                encodings.append(None)
        
        return encodings
    
    def draw_box_and_label(self, image, face_location, name, distance, is_known, color, show_distance=True):
        """
        Draw bounding box and label on image
        """
        top, right, bottom, left = face_location
        
        # Draw bounding box
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        
        # Prepare label
        if show_distance:
            confidence = 1 - min(distance, 1.0)  # Convert distance to confidence
            label = f"{name} ({confidence:.0%})" if is_known else f"UNKNOWN ({confidence:.0%})"
        else:
            label = name if is_known else "UNKNOWN"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            image,
            (left, top - label_size[1] - 10),
            (left + label_size[0], top),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (left, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return image


class FaceEncoder:
    """Handles face encoding and recognition using DeepFace"""
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.cache_file = KNOWN_FACES_DIR / "face_encodings_cache.pkl"
        self.last_load_time = 0
        
    def load_known_faces(self, force_reload=False):
        """
        Load known faces from directory and generate encodings
        """
        # Check cache first
        if not force_reload and self.cache_file.exists():
            cache_age = time.time() - self.cache_file.stat().st_mtime
            if cache_age < ENCODING_CACHE_DURATION:
                return self._load_from_cache()
        
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Check if directory exists
        if not KNOWN_FACES_DIR.exists():
            print(f"⚠ Directory not found: {KNOWN_FACES_DIR}")
            return False
        
        # Iterate through person folders
        persons_found = 0
        images_processed = 0
        
        for person_dir in KNOWN_FACES_DIR.iterdir():
            if not person_dir.is_dir() or person_dir.name == "README.txt":
                continue
            
            person_name = person_dir.name
            person_encodings = []
            
            # Process images for this person
            for image_path in person_dir.glob("*"):
                if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                try:
                    # Get embedding using DeepFace
                    embedding = DeepFace.represent(
                        img_path=str(image_path),
                        model_name='VGG-Face',
                        enforce_detection=False
                    )
                    
                    if embedding and len(embedding) > 0:
                        person_encodings.append(np.array(embedding[0]['embedding']))
                        images_processed += 1
                        print(f"✓ Processed {image_path.name} for {person_name}")
                    
                except Exception as e:
                    print(f"⚠ Error processing {image_path.name}: {str(e)}")
            
            # Average encodings for this person
            if person_encodings:
                avg_encoding = np.mean(person_encodings, axis=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(person_name)
                persons_found += 1
                print(f"✓ Loaded {person_name} with {len(person_encodings)} images")
        
        # Save to cache
        if self.known_face_encodings:
            self._save_to_cache()
        
        print(f"\n📊 Summary: Loaded {persons_found} persons from {images_processed} images")
        return len(self.known_face_encodings) > 0
    
    def _save_to_cache(self):
        """Save encodings to cache file"""
        try:
            cache_data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'timestamp': time.time()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Saved {len(self.known_face_names)} faces to cache")
        except Exception as e:
            print(f"⚠ Could not save cache: {str(e)}")
    
    def _load_from_cache(self):
        """Load encodings from cache file"""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.known_face_encodings = cache_data['encodings']
            self.known_face_names = cache_data['names']
            
            print(f"✓ Loaded {len(self.known_face_names)} faces from cache")
            return True
        except Exception as e:
            print(f"⚠ Cache load failed: {str(e)}")
            return False
    
    def recognize_face(self, face_encoding, threshold=None):
        """
        Recognize a face by comparing with known encodings using cosine similarity
        """
        if threshold is None:
            threshold = UNKNOWN_FACE_THRESHOLD
        
        if not self.known_face_encodings or face_encoding is None:
            return "Unknown", 1.0, False
        
        # Calculate cosine similarity (higher = more similar)
        similarities = []
        for known_encoding in self.known_face_encodings:
            # Cosine similarity
            similarity = np.dot(face_encoding, known_encoding) / (
                np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding)
            )
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        best_match_index = np.argmax(similarities)
        best_similarity = similarities[best_match_index]
        
        # Convert similarity to distance (1 - similarity)
        distance = 1 - best_similarity
        
        # Check if match is above threshold (using similarity directly)
        if best_similarity >= threshold:
            return self.known_face_names[best_match_index], distance, True
        else:
            return "Unknown", distance, False
    
    def get_statistics(self):
        """Get statistics about known faces"""
        return {
            'known_persons': len(self.known_face_names),
            'known_encodings': len(self.known_face_encodings),
            'persons_list': self.known_face_names
        }


def create_known_faces_directory():
    """Create the known_faces directory if it doesn't exist"""
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    
    # Create a README file with instructions
    readme_file = KNOWN_FACES_DIR / "README.txt"
    if not readme_file.exists():
        with open(readme_file, 'w') as f:
            f.write("""Known Faces Directory Structure
================================

To add known faces for recognition:

1. Create a folder for each person:
   known_faces/Person_Name/

2. Add 2-5 clear face photos of that person:
   known_faces/Person_Name/photo1.jpg
   known_faces/Person_Name/photo2.jpg

3. Supported formats: JPG, JPEG, PNG, BMP

Tips for better recognition:
- Use clear, front-facing photos
- Ensure good lighting
- Include different angles and expressions
- Minimum 2 photos per person recommended

Example:
known_faces/
├── John_Doe/
│   ├── john_1.jpg
│   └── john_2.jpg
└── Jane_Smith/
    ├── jane_1.jpg
    └── jane_2.jpg
""")
    
    return KNOWN_FACES_DIR


def get_system_stats(face_encoder):
    """Get system statistics"""
    stats = face_encoder.get_statistics()
    stats['known_faces_dir'] = str(KNOWN_FACES_DIR)
    return stats
