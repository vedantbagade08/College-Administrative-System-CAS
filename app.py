# Python 3.10 + TensorFlow 2.15.0 + OpenCV + face_recognition & DeepFace Setup  
import os
import sys
import warnings
import logging
# 1. IMMEDIATE WARNING & LOG SUPPRESSION (before other imports)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_MIN_INFO_LOG_LEVEL'] = '3'

# GPU CONFIGURATION (RTX 4050 Optimization)
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"GPU Acceleration Enabled: {gpus}")
    else:
        logging.warning("No GPU found. Running on CPU.")
except Exception as e:
    logging.warning(f"GPU Setup Error: {e}")

# 2. CONFIGURE LOGGING
# Set basic level to INFO to see "Info:" logs as requested
logging.info("Initializing CAS System...") # Test log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# Silence noisy library loggers specifically
for noisy_logger in ['werkzeug', 'absl', 'tensorflow', 'mediapipe', 'google', 'matplotlib', 'urllib3']:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)
logging.captureWarnings(True) # Redirect warnings to logging system

from flask import Flask, request, jsonify, send_from_directory, Response, render_template, redirect, url_for
import pymysql
import json
from datetime import datetime
import hashlib
import logging
from werkzeug.utils import secure_filename
import base64
import time
import re
import shutil

# ML model imports
DEEPFACE_AVAILABLE = False # Default to False to prevent auto-download

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("face_recognition library loaded successfully")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("face_recognition not available")
    
    # Check for DeepFace weights BEFORE importing/using to prevent auto-download
    # DeepFace stores weights in ~/.deepface/weights/
    home_dir = os.path.expanduser("~")
    weights_path = os.path.join(home_dir, ".deepface", "weights", "facenet512_weights.h5")
    
    if os.path.exists(weights_path):
        try:
            from deepface import DeepFace
            DEEPFACE_AVAILABLE = True
            logger_temp.info("DeepFace weights found. DeepFace enabled as fallback.")
        except Exception as e:
            logger_temp.warning(f"Failed to import DeepFace: {e}")
            DEEPFACE_AVAILABLE = False
    else:
        DEEPFACE_AVAILABLE = False
        logger_temp.warning(f"DeepFace weights NOT found at {weights_path}")
        logger_temp.warning("DeepFace fallback DISABLED to prevent auto-download.")
        logger_temp.warning("To enable: Manually download facenet512_weights.h5 to ~/.deepface/weights/")

YOLO_AVAILABLE = False

import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from io import BytesIO
import subprocess
import sys
import threading
from slm_agent import slm_agent
try:
    import train_model
except ImportError:
    train_model = None

# Logger is already configured at the top

# Try to import OpenCV for face/body detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available. Face/body detection will be disabled.")

import mediapipe as mp
MP_AVAILABLE = True

# Try to import flask-cors, fallback to manual CORS if not available
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    logger.warning("flask-cors not installed. Using manual CORS headers.")

# Flask app configuration - Get absolute path to ensure it works from any directory
# Flask app configuration - Get absolute path to ensure it works from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
# Note: static_folder is BASE_DIR because app.py is in 'static' folder in this user setup
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='/static', template_folder=os.path.join(BASE_DIR, 'templates'))

# Set secret key for session management
import secrets
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
logger.info(f"Flask secret key configured")

# Enable CORS for all routes to allow camera frame uploads
if CORS_AVAILABLE:
    CORS(app, resources={r"/api/*": {"origins": "*"}})
else:
    # Manual CORS headers as fallback
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

# MySQL Configuration
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'cas_db',
    'charset': 'utf8mb4',
    'autocommit': True
}

# Data directory for exports and frames
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'exports'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'frames'), exist_ok=True)

# Helper to check active academic session
def get_current_session():
    """
    Check if current time falls within a scheduled session.
    Returns: {'subject': '...', 'room': '...'} or None
    """
    try:
        schedule_path = os.path.join(DATA_DIR, 'schedule.json')
        if not os.path.exists(schedule_path):
            return None
            
        with open(schedule_path, 'r') as f:
            schedule = json.load(f)
            
        now = datetime.now()
        current_time = now.time()
        
        # Simple string parsing for "08:00 AM - 09:00 AM"
        for slot, details in schedule.items():
            time_range = details.get('time', '')
            if '-' in time_range:
                start_str, end_str = time_range.split('-')
                try:
                    start_time = datetime.strptime(start_str.strip(), "%I:%M %p").time()
                    end_time = datetime.strptime(end_str.strip(), "%I:%M %p").time()
                    
                    if start_time <= current_time <= end_time:
                         # Filter out "Free" slots or "Lunch"
                         if "Free" in details['subject'] or "Break" in details['subject']:
                             return None
                         return details
                except ValueError:
                    continue
        return None
    except Exception as e:
        logger.error(f"Session check error: {e}")
        return None

# Load models at startup - placed here to ensure BASE_DIR and logger are available

# Removed obsolete MobileNetV2 loader (using DeepFace now)

def load_cnn_model():
    try:
        path = os.path.join(MODEL_DIR, 'cnn_classifier.h5')
        if os.path.exists(path):
            # Use compile=False to avoid loading optimizer weights that might cause issues
            # Keras 3.x sometimes fails on 'batch_shape' in older H5 files
            try:
                model = tf.keras.models.load_model(path, compile=False)
            except TypeError as te:
                if 'batch_shape' in str(te):
                    logger.warning("Keras version mismatch (batch_shape). Attempting legacy loading...")
                    # Fallback or just ignore if model is not critical
                    return None
                raise te
            
            logger.info("CNN Classifier loaded")
            return model
        return None
    except Exception as e:
        logger.warning(f"Failed to load CNN model: {e}")
        return None

# Global Status Tracking
last_detection_time = 0
faces_detected_count = 0



def load_svm_classifier():
    try:
        path = os.path.join(MODEL_DIR, 'svm_classifier.pkl')
        if os.path.exists(path):
            clf = joblib.load(path)
            logger.info('SVM classifier loaded')
            return clf
        return None
    except Exception as e:
        logger.error(f'Failed to load SVM classifier: {e}')
        return None

def load_label_encoder():
    try:
        path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
        if os.path.exists(path):
            le = joblib.load(path)
            # Ensure classes are loaded
            if hasattr(le, 'classes_'):
                logger.info(f"Label Encoder loaded with classes: {le.classes_}")
            return le
        return None
    except Exception as e:
        logger.error(f"Failed to load Label Encoder: {e}")
        return None

cnn_model = load_cnn_model()
svm_classifier = load_svm_classifier()
label_encoder = load_label_encoder()
# facenet_interpreter = load_facenet_model() # Removed (Using DeepFace)

def recognize_face_from_crop(face_crop):
    """
    Given a BGR face crop (numpy array), extract embedding using DeepFace
    and classify using SVM classifier.
    Returns: (name, confidence) or ('Unknown', 0.0)
    """
    global svm_classifier, label_encoder
    
    if svm_classifier is None or label_encoder is None:
        return ('Unknown', 0.0)
    
    try:
        # Convert BGR to RGB for DeepFace
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Get embedding using VGG-Face model (4096 features - matches SVM training)
        embedding_result = DeepFace.represent(
            face_rgb, 
            model_name='VGG-Face',  # Changed from Facenet512 to match SVM
            enforce_detection=False,  # Already cropped
            detector_backend='skip'   # Skip detection since we have a crop
        )
        
        if not embedding_result:
            return ('Unknown', 0.0)
        
        embedding = np.array(embedding_result[0]['embedding']).reshape(1, -1)
        
        # Classify using SVM
        prediction = svm_classifier.predict(embedding)
        
        # Get confidence (probability)
        if hasattr(svm_classifier, 'predict_proba'):
            proba = svm_classifier.predict_proba(embedding)
            confidence = float(np.max(proba))
        else:
            # Use decision function as fallback
            confidence = 0.8  # Default confidence
        
        # Decode label
        name = label_encoder.inverse_transform(prediction)[0]
        
        # Threshold - if confidence is low, return Unknown
        if confidence < 0.5:
            return ('Unknown', confidence)
        
        return (str(name), confidence)
        
    except Exception as e:
        logger.warning(f"Face recognition error: {e}")
        return ('Unknown', 0.0)

# Cache for recognition results (to reduce lag)
recognition_cache = {}
recognition_cache_timeout = 2.0  # seconds

# Initialize SLM in background with error handling
def init_slm():
    try:
        slm_agent.load_model()
    except Exception as e:
        logger.warning(f"SLM initialization error (will use templates): {e}")

# threading.Thread(target=init_slm, daemon=True).start()
logger.info("SLM disabled to save memory for Face Recognition")

os.makedirs(os.path.join(DATA_DIR, 'exports'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'frames'), exist_ok=True)

# Initialize OpenCV cascade classifiers for detection
face_cascade = None
body_cascade = None

def init_detection_models():
    """Initialize OpenCV Haar cascade classifiers for face and body detection"""
    global face_cascade, body_cascade
    if not CV2_AVAILABLE:
        logger.warning("OpenCV not available - detection models not initialized")
        return False
    
    try:
        # Load face cascade
        face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(face_path):
            face_cascade = cv2.CascadeClassifier(face_path)
            logger.info("Face detection model loaded successfully")
        else:
            logger.warning(f"Face cascade not found at {face_path}")
        
        # Load body cascade (try fullbody first, then upperbody)
        body_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        if os.path.exists(body_path):
            body_cascade = cv2.CascadeClassifier(body_path)
            logger.info("Body detection model (fullbody) loaded successfully")
        else:
            alt_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            if os.path.exists(alt_path):
                body_cascade = cv2.CascadeClassifier(alt_path)
                logger.info("Body detection model (upperbody) loaded successfully")
            else:
                logger.warning(f"Body cascade not found at {body_path} or {alt_path}")
        
        return face_cascade is not None or body_cascade is not None
    except Exception as e:
        logger.error(f"Error initializing detection models: {e}")
        return False

# MediaPipe Initialization
mp_face_detection = None
face_detector = None

def init_mediapipe():
    global mp_face_detection, face_detector
    if MP_AVAILABLE:
        try:
            mp_face_detection = mp.solutions.face_detection
            face_detector = mp_face_detection.FaceDetection(
                model_selection=1, 
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe Face Detection initialized successfully")
            return True
        except Exception as e:
            logger.error(f"MediaPipe init failed: {e}")
            return False
    return False





# YOLO Initialization Removed
yolo_model = None
def init_yolo():
    return False

# 1. Email to Name Mapping (Login)
# Password for all students: student123
# Password for Faculty: faculty123
# Password for Admin: admin123

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode('utf-8')).hexdigest()

# Default Hardcoded Users (Admin & Faculty)
EMAIL_TO_USER = {
    # Faculty
    "DemoFacultyZealPoly@gmail.com": {"name": "Demo Faculty", "id": "FAC001", "dept": "Computer Science", "role": "faculty", "password_hash": hash_password("faculty123")},
    # Admin
    "Admin123@gmail.com": {"name": "System Admin", "id": "ADM001", "dept": "Administration", "role": "admin", "password_hash": hash_password("admin123")}
}

def load_student_emails():
    """Load student emails from datasets/Email ID's.txt"""
    try:
        file_path = os.path.join(BASE_DIR, 'datasets', "Email ID's.txt")
        if not os.path.exists(file_path):
            logger.warning(f"Student email file not found: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Expected format: "1. Email - Profile name : Name"
                # Robust parsing strategy
                try:
                    # Remove numbering "1. "
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        content = parts[1].strip()
                    else:
                        content = line
                    
                    # Split by " - " to get Email
                    email_parts = content.split(' - ')
                    if len(email_parts) < 2: continue
                    
                    email = email_parts[0].strip()
                    rest = email_parts[1]
                    
                    # Split by ":" to get Name
                    if ':' in rest:
                        name = rest.split(':', 1)[1].strip()
                    else:
                        name = "Unknown Student"
                    
                    # Add to directory with default student props
                    EMAIL_TO_USER[email] = {
                        "name": name,
                        "id": f"STU{len(EMAIL_TO_USER)+1:03d}", # Generate ID
                        "dept": "Artificial Intelligence and Machine Learning",
                        "role": "student",
                        "password_hash": hash_password("student123")
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse line '{line}': {e}")
                    
        logger.info(f"Loaded {len(EMAIL_TO_USER)} users (including Admin/Faculty)")
        
    except Exception as e:
        logger.error(f"Error loading student emails: {e}")

# Load students at startup
load_student_emails()

# 2. Face ID to Name Mapping (Recognition)
# Assuming dataset folders 01-10 map somewhat to the order or specific IDs.
# Based on datasets:
# 2. Face ID to Name Mapping (Recognition)
# Updated to include ALL students 01-11
FACE_ID_MAP = {
    "01": "Shreyas Gaikar",
    "02": "Vaibhav Vatane",
    "03": "Atharav Vahutre",
    "04": "Dipak Bandgar",
    "05": "Shivam Banngar",
    "06": "Siddhi Khatal",
    "07": "Ojas Marathe",
    "08": "Vedant Bagade",
    "09": "Varun Chavan",
    "10": "Tanmay Temghare",
    "11": "Harshal Ghadage",
    "Prof_Susmita": "Prof. Susmita",
    "HoD-Prof_vijay_moohite": "HoD Prof. Vijay Mohite"
}

# Load Student Embeddings from datasets/Student
STUDENT_EMBEDDINGS = {} # {id: [emb1, emb2]}

def load_student_embeddings():
    """
    Load images from datasets/Student/{id} and datasets/faculty and generate reference encodings.
    Uses face_recognition library for faster processing.
    """
    global STUDENT_EMBEDDINGS
    
    if not FACE_RECOGNITION_AVAILABLE:
        if DEEPFACE_AVAILABLE:
            logger.warning("face_recognition not available, using DeepFace fallback")
            load_student_embeddings_deepface()
        else:
            logger.error("face_recognition NOT available and DeepFace weights missing. Face recognition DISABLED.")
        return
    
    # Load Students
    student_base_dir = os.path.join(BASE_DIR, 'datasets', 'Student')
    
    if os.path.exists(student_base_dir):
        logger.info("Loading student reference images with face_recognition...")
        
        # Iterate over ALL student folders (01-11)
        for student_id in os.listdir(student_base_dir):
            # Process all student IDs 01-11
            if student_id not in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]: 
                continue 
            
            student_dir = os.path.join(student_base_dir, student_id)
            if not os.path.isdir(student_dir):
                continue
                
            encodings = []
            # Support common image formats
            for img_file in os.listdir(student_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(student_dir, img_file)
                    try:
                        # Load image and get face encodings
                        img = face_recognition.load_image_file(img_path)
                        face_encodings = face_recognition.face_encodings(img)
                        
                        if face_encodings:
                            encodings.append(face_encodings[0])
                            logger.info(f"Loaded reference: {img_file} for {student_id}")
                        else:
                            logger.warning(f"No face detected in {img_file}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
            
            if encodings:
                STUDENT_EMBEDDINGS[student_id] = encodings
                logger.info(f"Loaded {len(encodings)} references for ID {student_id}")
    
    # Load Faculty
    faculty_dir = os.path.join(BASE_DIR, 'datasets', 'faculty')
    
    if os.path.exists(faculty_dir):
        logger.info("Loading faculty reference images...")
        
        for img_file in os.listdir(faculty_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(faculty_dir, img_file)
                # Extract ID from filename (e.g., "Prof_Susmita.jpg" -> "Prof_Susmita")
                faculty_id = os.path.splitext(img_file)[0]
                
                try:
                    img = face_recognition.load_image_file(img_path)
                    face_encodings = face_recognition.face_encodings(img)
                    
                    if face_encodings:
                        STUDENT_EMBEDDINGS[faculty_id] = [face_encodings[0]]
                        logger.info(f"Loaded faculty reference: {img_file}")
                    else:
                        logger.warning(f"No face detected in {img_file}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load faculty {img_path}: {e}")

    logger.info(f"Total references loaded: {len(STUDENT_EMBEDDINGS)}")

def load_student_embeddings_deepface():
    """
    Fallback: Load with DeepFace if face_recognition is not available
    """
    global STUDENT_EMBEDDINGS
    
    if not DEEPFACE_AVAILABLE:
        logger.error("Attempted to use DeepFace but DEEPFACE_AVAILABLE is False. Aborting.")
        return

    from deepface import DeepFace
    
    student_base_dir = os.path.join(BASE_DIR, 'datasets', 'Student')
    
    if os.path.exists(student_base_dir):
        logger.info("Loading student reference images with DeepFace (fallback)...")
        for student_id in os.listdir(student_base_dir):
            if student_id not in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]: 
                continue 
            student_dir = os.path.join(student_base_dir, student_id)
            if not os.path.isdir(student_dir):
                continue
            embeddings = []
            for img_file in os.listdir(student_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(student_dir, img_file)
                    try:
                        embedding_objs = DeepFace.represent(
                            img_path=img_path, 
                            model_name="Facenet512", 
                            enforce_detection=False,
                            detector_backend='opencv',
                            align=True
                        )
                        if embedding_objs:
                            emb = embedding_objs[0]["embedding"]
                            embeddings.append(emb)
                            logger.info(f"Loaded reference: {img_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
            if embeddings:
                STUDENT_EMBEDDINGS[student_id] = embeddings
    
    # Load Faculty
    faculty_dir = os.path.join(BASE_DIR, 'datasets', 'faculty')
    if os.path.exists(faculty_dir):
        for img_file in os.listdir(faculty_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(faculty_dir, img_file)
                faculty_id = os.path.splitext(img_file)[0]
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=img_path, 
                        model_name="Facenet512", 
                        enforce_detection=False,
                        detector_backend='opencv',
                        align=True
                    )
                    if embedding_objs:
                        emb = embedding_objs[0]["embedding"]
                        STUDENT_EMBEDDINGS[faculty_id] = [emb]
                        logger.info(f"Loaded faculty reference: {img_file}")
                except Exception as e:
                    logger.warning(f"Failed to load faculty {img_path}: {e}")

    logger.info(f"Total references loaded (DeepFace): {len(STUDENT_EMBEDDINGS)}")

# Load embeddings once so recognition works from your dataset (e.g. when run via gunicorn/flask run)
_embeddings_loaded = False
def _ensure_embeddings_loaded():
    global _embeddings_loaded
    if not _embeddings_loaded:
        load_student_embeddings()
        _embeddings_loaded = True

# Dynamically load classes from training (Optional fallback)
try:
    classes_path = os.path.join(MODEL_DIR, 'classes.npy')
    if os.path.exists(classes_path):
        trained_classes = np.load(classes_path, allow_pickle=True)
        # ... logic if needed
except Exception as e:
    pass

def get_face_identity(face_img):
    """
    Recognize face using hybrid approach (face_recognition + DeepFace).
    Returns: (id, name, confidence)
    """
    _ensure_embeddings_loaded()  # Ensure dataset loaded (e.g. when run via gunicorn)
    sid, name, conf = "Unknown", "Unknown", 0
    
    # 1. Try Dlib (face_recognition) - Fast & Accurate for known poses
    if FACE_RECOGNITION_AVAILABLE:
        try:
            # Convert BGR to RGB (OpenCV to face_recognition)
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_img)
            
            if face_encodings:
                target_encoding = face_encodings[0]
                best_match_id = "Unknown"
                min_distance = 1.0
                threshold = 0.45  # Reset to 0.45 
                
                for student_id, ref_encodings in STUDENT_EMBEDDINGS.items():
                    for ref_encoding in ref_encodings:
                        # Calculate face distance
                        distance = face_recognition.face_distance([ref_encoding], target_encoding)[0]
                        if distance < min_distance:
                            min_distance = distance
                            best_match_id = student_id
                
                if min_distance < threshold:
                    sid = best_match_id
                    name = FACE_ID_MAP.get(sid, f"Student {sid}")
                    conf = max(0, (1 - min_distance) * 100)
                    logger.info(f"Dlib Match: {name} ({conf:.1f}%)")
                    return sid, name, conf
                else:
                    logger.info(f"Dlib Near Miss: {best_match_id if best_match_id != 'Unknown' else 'None'} dist {min_distance:.3f}")
        except Exception as e:
            logger.error(f"Dlib identity error: {e}")

    # 2. Try DeepFace (Fallback/Mixed) - Better for difficult angles/lighting
    if sid == "Unknown" and DEEPFACE_AVAILABLE:
        try:
            logger.info("Dlib failed/no match. Trying DeepFace (Facenet512) fallback...")
            sid, name, conf = get_face_identity_deepface(face_img)
            if sid != "Unknown":
                logger.info(f"DeepFace Match: {name} ({conf:.1f}%)")
        except Exception as e:
            logger.error(f"DeepFace fallback error: {e}")

    return sid, name, conf

def get_face_identity_light(face_img):
    """
    Light face recognition: dlib (face_recognition) only, no DeepFace.
    Use in camera stream for low lag. Returns (id, name, confidence).
    """
    _ensure_embeddings_loaded()
    if not FACE_RECOGNITION_AVAILABLE or not STUDENT_EMBEDDINGS:
        return "Unknown", "Unknown", 0
    try:
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_img)
        if not face_encodings:
            return "Unknown", "Unknown", 0
        target_encoding = face_encodings[0]
        best_match_id = "Unknown"
        min_distance = 1.0
        threshold = 0.45
        for student_id, ref_encodings in STUDENT_EMBEDDINGS.items():
            for ref_encoding in ref_encodings:
                distance = face_recognition.face_distance([ref_encoding], target_encoding)[0]
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = student_id
        if min_distance < threshold:
            name = FACE_ID_MAP.get(best_match_id, f"Student {best_match_id}")
            conf = max(0, (1 - min_distance) * 100)
            return best_match_id, name, conf
    except Exception:
        pass
    return "Unknown", "Unknown", 0

def get_face_identity_deepface(face_img):
    """
    Fallback recognition using DeepFace
    Returns: (id, name, confidence)
    """
    if not DEEPFACE_AVAILABLE:
        return "Unknown", "Unknown", 0
        
    try:
        from deepface import DeepFace
        from scipy.spatial.distance import cosine
        
        # Performance: Pass numpy array directly instead of writing/reading file
        embedding_objs = DeepFace.represent(
            img_path=face_img, # DeepFace supports numpy arrays
            model_name="Facenet512", 
            enforce_detection=False,
            align=False
        )
        
        if not embedding_objs:
            return "Unknown", "Unknown", 0
        
        target_embedding = embedding_objs[0]["embedding"]
        best_match_id = "Unknown"
        min_distance = 1.0
        threshold = 0.45 # Reset to 0.45
        
        for student_id, ref_embeddings in STUDENT_EMBEDDINGS.items():
            for ref_emb in ref_embeddings:
                dist = cosine(target_embedding, ref_emb)
                if dist < min_distance:
                    min_distance = dist
                    best_match_id = student_id
        
        if min_distance < threshold:
            name = FACE_ID_MAP.get(best_match_id, f"Student {best_match_id}")
            confidence = max(0, (1 - min_distance) * 100)
            return best_match_id, name, confidence
        else:
            logger.info(f"DeepFace Near Miss: {best_match_id} dist {min_distance:.3f}")
        
        return "Unknown", "Unknown", 0
    
    except Exception as e:
        logger.error(f"Error in DeepFace fallback: {e}")
        return "Unknown", "Unknown", 0

# Global Camera Instance
video_camera = None



def detect_faces_and_bodies(image_data):
    """
    Detect faces and bodies in an image using OpenCV Haar cascades.
    Returns: {'faces': [...], 'bodies': [...]} where each item is {'x', 'y', 'w', 'h'}
    """
    if not CV2_AVAILABLE or (face_cascade is None and body_cascade is None):
        return {'faces': [], 'bodies': []}
    
    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("Failed to decode image for detection")
            return {'faces': [], 'bodies': []}
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detections = {'faces': [], 'bodies': []}
        
        # Detect faces (Priority: MediaPipe -> Haar)
        faces_found = False
        
        # 1. Try MediaPipe first
        if MP_AVAILABLE and face_detector:
            try:
                results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.detections:
                    h_img, w_img, _ = img.shape
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        detections['faces'].append({
                            'x': int(bboxC.xmin * w_img),
                            'y': int(bboxC.ymin * h_img),
                            'w': int(bboxC.width * w_img),
                            'h': int(bboxC.height * h_img)
                        })
                    faces_found = True
            except Exception as e:
                logger.warning(f"MediaPipe detection failed in helper: {e}")

        # 2. Fallback to Haar (More Sensitive)
        if not faces_found and face_cascade is not None:
            # Balanced settings: minNeighbors 4
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            for (x, y, w, h) in faces:
                detections['faces'].append({
                    'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)
                })
        
        # Detect bodies removed
        
        return detections
    except Exception as e:
        logger.error(f"Error during face/body detection: {e}")
        return {'faces': [], 'bodies': []}

# Legacy students JSON
STUDENTS_JSON_PATH = os.path.join(BASE_DIR, 'data', 'students.json')

def load_students():
    if not os.path.exists(STUDENTS_JSON_PATH):
        logger.error(f"Students file missing at {STUDENTS_JSON_PATH}")
        return {}
        
    try:
        with open(STUDENTS_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading students.json: {e}")
        return {}

def get_db_connection(use_database=True, max_retries=3):
    """
    Get a database connection with retry logic.
    :param use_database: If True, connects to the 'cas_db' database. If False, connects only to the server.
    :param max_retries: Number of retry attempts for transient errors.
    """
    config = MYSQL_CONFIG.copy()
    if not use_database:
        config.pop('database', None)
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return pymysql.connect(**config)
        except pymysql.Error as e:
            last_error = e
            logger.warning(f"DB Connection attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(1) # Wait a bit before retrying
            
    logger.error(f"Failed to connect to DB after {max_retries} attempts. Last error: {last_error}")
    raise last_error

def init_db():
    """Initialize database and tables if they don't exist."""
    # 1. Connect WITHOUT selecting a database to ensure it exists
    conn = None
    try:
        conn = get_db_connection(use_database=False)
        with conn.cursor() as cur:
            cur.execute("CREATE DATABASE IF NOT EXISTS cas_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            logger.info("Database 'cas_db' ensured.")
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return # Cannot proceed if DB creation fails
    finally:
        if conn: conn.close()

    # 2. Connect WITH the database to create tables
    try:
        conn = get_db_connection(use_database=True)
        with conn.cursor() as cur:
            # Users table (Modern, used by Admin Portal and New Registration)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255),
                    password VARCHAR(255),
                    role VARCHAR(50) DEFAULT 'pending',
                    face_encoding_status TINYINT(1) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Accounts table (Legacy compatibility)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    email VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255),
                    password_hash VARCHAR(255),
                    role VARCHAR(50) DEFAULT 'student'
                )
            """)
            
            # Events table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE,
                    title VARCHAR(255),
                    notes TEXT,
                    owner VARCHAR(255)
                )
            """)
            
            # Attendance table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    course VARCHAR(50),
                    date DATE,
                    actor VARCHAR(255),
                    rows_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Audit table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    actor VARCHAR(255),
                    action VARCHAR(100),
                    details JSON
                )
            """)
            
            logger.info("Tables initialized successfully.")
    except pymysql.Error as e:
        logger.error(f"DB Table Init error: {e}")
    finally:
        if conn: conn.close()



def save_attendance_csv(course, date_str, attendance_rows, out_path=None):
    if out_path is None:
        filename = f"attendance_{course}_{date_str}.csv".replace(' ', '_')
        out_path = os.path.join(DATA_DIR, 'exports', filename)
    
    fieldnames = ['timestamp', 'student_id', 'name', 'course', 'status', 'confidence', 'notes']
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in attendance_rows:
            writer.writerow({
                'timestamp': datetime.utcnow().isoformat(),
                'student_id': r.get('id'),
                'name': r.get('name'),
                'course': course,
                'status': r.get('status'),
                'confidence': r.get('confidence', ''),
                'notes': r.get('notes', '')
            })
    return out_path

def append_audit(actor, action, details):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (actor, action, details) VALUES (%s, %s, %s)",
                (actor, action, json.dumps(details))
            )
    finally:
        conn.close()

@app.route('/api/audit/logs', methods=['GET'])
def get_audit_logs():
    """Fetch recent auto-attendance logs"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT timestamp, actor, details FROM audit_log WHERE action="auto_attendance" ORDER BY timestamp DESC LIMIT 20')
            rows = cur.fetchall()
            logs = []
            for r in rows:
                try:
                    details = json.loads(r[2]) if isinstance(r[2], str) else r[2]
                except:
                    details = {}
                
                # Extract actual student/person identifier
                # If actor is 'System', look in details for 'student_id' or 'name' or from message
                person = r[1]
                if person == 'System' or person is None:
                    person = details.get('student_id') or details.get('name') or 'System'
                
                logs.append({
                    'timestamp': r[0].isoformat() if r[0] else None,
                    'student': person, 
                    'message': details.get('slm_log', 'Attendance marked'),
                    'details': details # Pass full details just in case
                })
    finally:
        conn.close()
    return jsonify({'logs': logs})

# Routes for serving HTML pages
@app.route('/')
def index():
    try:
        # User requested this be the main page
        file_path = os.path.join(BASE_DIR, 'Home', 'CASWare-Home.html')
        if not os.path.exists(file_path):
            logger.error(f"Main landing page not found at: {file_path}")
            return "File not found: CASWare-Home.html", 404
        return send_from_directory(os.path.join(BASE_DIR, 'Home'), 'CASWare-Home.html')
    except Exception as e:
        logger.error(f"Error serving main landing page: {e}")
        return f"Error: {str(e)}", 500

@app.route('/old-index')
def old_index():
    try:
        return send_from_directory(BASE_DIR, 'index.html')
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/account')
def account():
    try:
        return send_from_directory(BASE_DIR, 'account.html')
    except Exception as e:
        logger.error(f"Error serving account.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/account.html')
def account_html():
    return account()

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    
    if not email:
        return "Email required", 400

    # 1. Check MySQL Database first (Persistent Users)
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Note: password_hash is sha256 of the password
            cur.execute("SELECT name, email, role, password_hash FROM accounts WHERE email = %s", (email,))
            user_record = cur.fetchone()
            
            if user_record:
                db_name, db_email, db_role, db_pw_hash = user_record
                if db_pw_hash == hash_password(password):
                    from flask import session
                    user = {
                        'name': db_name,
                        'email': db_email,
                        'role': db_role,
                        'dept': 'General' # Default
                    }
                    session['user'] = user
                    
                    if db_role == "admin":
                        return redirect(url_for('administrator'))
                    elif db_role == "faculty":
                        return redirect(url_for('faculty_dashboard'))
                    else:
                        return redirect(url_for('student'))
                else:
                    return f"Invalid Password for {email}", 401
    except Exception as e:
        logger.error(f"Login DB error: {e}")
    finally:
        if conn: conn.close()

    # 2. Check Pre-defined / Hardcoded Users (Legacy/Demo fallback)
    # Case-insensitive lookup
    email_key = next((k for k in EMAIL_TO_USER if k.lower() == email.lower()), None)
    
    if email_key:
        user = EMAIL_TO_USER[email_key]
        # Verify Password
        if user.get("password_hash") == hash_password(password):
            from flask import session
            session['user'] = user
            
            # Role-Based Redirect
            role = user.get("role", "student")
            if role == "admin":
                return redirect(url_for('administrator'))
            elif role == "faculty":
                return redirect(url_for('faculty_dashboard'))
            else:
                return redirect(url_for('student'))
        else:
             return f"Invalid Password for {email}", 401
    
    # 2. Smart / Dynamic Login (Demo Mode) assuming password is provided
    # Allow ANY password for these dynamic demo accounts to simplify testing
    if password: 
        import re
        from flask import session
        
        # A) Faculty Detection
        if "faculty" in email.lower():
            user = {
                'name': 'Faculty Member',
                'email': email,
                'role': 'faculty',
                'dept': 'General'
            }
            session['user'] = user
            return redirect(url_for('faculty_dashboard'))
            
        # B) Student Pattern: Name + ID (e.g., Ojas42Zeal@gmail.com)
        # Regex: Starts with letters, then digits, then anything, @gmail.com
        match = re.match(r"^([A-Za-z]+)(\d+)(.*)@gmail\.com$", email, re.IGNORECASE)
        if match:
            name_part = match.group(1)
            id_part = match.group(2)
            
            # Capitalize name
            formatted_name = name_part.capitalize()
            
            user = {
                'name': formatted_name,
                'id': id_part,
                'email': email,
                'role': 'student',
                'dept': 'Computer Science' 
            }
            session['user'] = user
            return redirect(url_for('student'))
            
        # C) ZRP / Standard Roll Number
        if email.upper().startswith("ZRP"):
            user = {
                'name': 'Student User',
                'id': email.split('@')[0].upper(), # Use part before @ as ID
                'email': email,
                'role': 'student',
                'dept': 'Unknown'
            }
            session['user'] = user
            return redirect(url_for('student'))

    return f"Invalid Email or User not found: {email}. Try 'Ojas42Zeal@gmail.com' or see documentation.", 401

@app.route('/admin')
def admin_dashboard():
    from flask import session, abort
    user = session.get('user')
    # Basic role check
    if not user or user.get('role') != 'admin':
        return "Access Denied: Admin privileges required.", 403
        
    try:
        return send_from_directory(BASE_DIR, 'admin.html')
    except Exception as e:
        return f"Error loading admin dashboard: {e}", 500


# --- Faculty API Endpoints (JSON Storage) ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def load_json(filename, default=[]):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path): return default
    try:
        with open(path, 'r') as f: return json.load(f)
    except: return default

def save_json(filename, data):
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/api/quick-links', methods=['GET', 'POST'])
def api_quick_links():
    if request.method == 'GET':
        return jsonify(load_json('quick_links.json', []))
    
    data = request.json
    links = load_json('quick_links.json', [])
    links.append(data)
    save_json('quick_links.json', links)
    return jsonify({'ok': True})

@app.route('/api/quick-links/<int:index>', methods=['DELETE'])
def api_delete_quick_link(index):
    links = load_json('quick_links.json', [])
    if 0 <= index < len(links):
        links.pop(index)
        save_json('quick_links.json', links)
    return jsonify({'ok': True})

@app.route('/api/feedback/forms', methods=['GET'])
def api_feedback_forms():
    return jsonify(load_json('feedback_forms.json', []))

@app.route('/api/feedback/submit', methods=['POST'])
def api_feedback_submit():
    data = request.json
    responses = load_json('feedback_responses.json', [])
    responses.append({
        'form_id': data.get('form_id'),
        'user': data.get('user_email'),
        'responses': data.get('response'),
        'timestamp': datetime.now().isoformat()
    })
    save_json('feedback_responses.json', responses)
    return jsonify({'ok': True})

@app.route('/api/faculty/courses', methods=['GET'])
def api_faculty_courses():
    courses = load_json('courses.json', [])
    # In real app, filter by session user. For demo, return all.
    return jsonify({'courses': courses})

@app.route('/api/faculty/courses/manage', methods=['POST'])
def api_manage_courses():
    action = request.json.get('action') # 'add', 'delete'
    courses = load_json('courses.json', [])
    
    if action == 'add':
        courses.append(request.json.get('course'))
    elif action == 'delete':
        code = request.json.get('code')
        courses = [c for c in courses if c.get('code') != code]
    
    save_json('courses.json', courses)
    return jsonify({'ok': True})

@app.route('/api/faculty/courses/syllabus', methods=['POST'])
def api_syllabus_update():
    code = request.json.get('code')
    progress = request.json.get('progress')
    courses = load_json('courses.json', [])
    
    for c in courses:
        if c.get('code') == code:
            c['completion'] = int(progress)
            break
            
    save_json('courses.json', courses)
    return jsonify({'ok': True})

# Division (class) to course filter: TY-A -> TY-AN, FY-A -> FY, etc.
DIVISION_TO_COURSE = {
    'TY-A': 'TY-AN', 'FY-A': 'FY', 'FY-B': 'FY', 'FY-C': 'FY', 'FY-D': 'FY', 'FY-E': 'FY', 'FY-F': 'FY',
    'SY-A': 'SY', 'SY-B': 'SY', 'SY-C': 'SY',
}

@app.route('/api/grades/student/<string:student_id>', methods=['GET'])
def api_student_grades(student_id):
    all_grades = load_json('grades.json', {})
    return jsonify(all_grades.get(student_id, []))

@app.route('/api/grades/student/<string:student_id>/summary', methods=['GET'])
def api_student_grades_summary(student_id):
    """Return semester summary (CGPA, percentage) for student, set by faculty."""
    summaries = load_json('student_academic_summary.json', {})
    return jsonify(summaries.get(student_id, {}))

@app.route('/api/grades/student/summary', methods=['POST'])
def api_student_grades_summary_save():
    """Save semester summary (semester label, CGPA, percentage) for a student."""
    data = request.get_json()
    student_id = data.get('student_id')
    semester_label = (data.get('semester_label') or '').strip() or 'Semester'
    try:
        cgpa = float(data.get('cgpa', 0))
        percentage = float(data.get('percentage', 0))
    except (TypeError, ValueError):
        cgpa = 0.0
        percentage = 0.0
    if not student_id:
        return jsonify({'ok': False, 'error': 'student_id required'}), 400
    summaries = load_json('student_academic_summary.json', {})
    summaries[student_id] = {
        'semester_label': semester_label,
        'cgpa': round(cgpa, 2),
        'percentage': round(percentage, 2),
        'updated_at': datetime.now().isoformat()
    }
    save_json('student_academic_summary.json', summaries)
    return jsonify({'ok': True})

@app.route('/api/grades/update', methods=['POST'])
def api_update_grades():
    data = request.json
    student_id = data.get('student_id')
    subject_id = data.get('subject_id')
    
    all_grades = load_json('grades.json', {})
    if student_id not in all_grades:
        all_grades[student_id] = []
        
    student_record = all_grades[student_id]
    
    # Check if subject exists for student, else create
    target = next((x for x in student_record if x.get('subject_id') == subject_id), None)
    if not target:
        target = {
            'subject_id': subject_id, 
            'subject_name': data.get('subject_name'),
            'ut1': 0, 'ut2': 0, 'sem': 0
        }
        student_record.append(target)
        
    target['ut1'] = int(data.get('ut1', 0))
    target['ut2'] = int(data.get('ut2', 0))
    target['sem'] = int(data.get('sem', 0))
    
    save_json('grades.json', all_grades)
    return jsonify({'ok': True})

@app.route('/api/faculty/payroll', methods=['GET'])
def api_faculty_payroll():
    # Helper to get payroll for currently logged in faculty
    from flask import session
    user = session.get('user')
    email = user.get('email') if user else "DemoFacultyZealPoly@gmail.com"
    
    payroll = load_json('payroll.json', {})
    return jsonify(payroll.get(email, {}))

# --- Faculty Payroll Documents (upload / list / view / delete) ---
PAYROLL_DOCS_DIR = os.path.join(DATA_DIR, 'payroll_docs')
os.makedirs(PAYROLL_DOCS_DIR, exist_ok=True)
ALLOWED_PAYROLL_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

def get_faculty_payroll_dir():
    from flask import session
    user = session.get('user')
    raw = (user or {}).get('email')
    email = (str(raw).strip() if raw else "") or "DemoFacultyZealPoly@gmail.com"
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', email)
    path = os.path.join(PAYROLL_DOCS_DIR, safe_id)
    os.makedirs(path, exist_ok=True)
    return path

def payroll_manifest_path():
    return os.path.join(get_faculty_payroll_dir(), 'manifest.json')

def load_payroll_manifest():
    path = payroll_manifest_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_payroll_manifest(entries):
    with open(payroll_manifest_path(), 'w') as f:
        json.dump(entries, f, indent=2)

@app.route('/api/faculty/payroll/documents', methods=['GET'])
def api_faculty_payroll_documents_list():
    entries = load_payroll_manifest()
    base_url = url_for('api_faculty_payroll_document_file', filename='')
    for e in entries:
        e['url'] = base_url + e.get('filename', '')
    return jsonify({'ok': True, 'documents': entries})

@app.route('/api/faculty/payroll/documents', methods=['POST'])
def api_faculty_payroll_upload():
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'ok': False, 'error': 'No file selected'}), 400
    ext = (file.filename.rsplit('.', 1)[-1] or '').lower()
    if ext not in ALLOWED_PAYROLL_EXTENSIONS:
        return jsonify({'ok': False, 'error': f'Allowed: {", ".join(ALLOWED_PAYROLL_EXTENSIONS)}'}), 400
    title = (request.form.get('title') or '').strip() or file.filename
    stored_name = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
    if not stored_name.lower().endswith(f'.{ext}'):
        stored_name += f'.{ext}'
    dirpath = get_faculty_payroll_dir()
    filepath = os.path.join(dirpath, stored_name)
    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
    entries = load_payroll_manifest()
    entries.append({
        'id': stored_name,
        'filename': stored_name,
        'title': title[:200],
        'uploaded_at': datetime.now().isoformat()
    })
    save_payroll_manifest(entries)
    return jsonify({'ok': True, 'filename': stored_name})

@app.route('/api/faculty/payroll/documents/file/<path:filename>', methods=['GET'])
def api_faculty_payroll_document_file(filename):
    dirpath = get_faculty_payroll_dir()
    path = os.path.join(dirpath, filename)
    if not os.path.isfile(path) or not os.path.realpath(path).startswith(os.path.realpath(dirpath)):
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory(dirpath, filename, as_attachment=False)

@app.route('/api/faculty/payroll/documents/<path:doc_id>', methods=['DELETE'])
def api_faculty_payroll_document_delete(doc_id):
    dirpath = get_faculty_payroll_dir()
    entries = load_payroll_manifest()
    entry = next((e for e in entries if e.get('filename') == doc_id or e.get('id') == doc_id), None)
    if not entry:
        return jsonify({'ok': False, 'error': 'Not found'}), 404
    path = os.path.join(dirpath, entry.get('filename', ''))
    if os.path.isfile(path):
        try:
            os.remove(path)
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    entries = [e for e in entries if e.get('filename') != doc_id and e.get('id') != doc_id]
    save_payroll_manifest(entries)
    return jsonify({'ok': True})

@app.route('/faculty')
def faculty_dashboard():
    from flask import session, render_template
    user = session.get('user')
    
    # Allow access if role is faculty OR admin (for testing)
    if not user or (user.get('role') != 'faculty' and user.get('role') != 'admin'):
         # Fallback for demo purposes if no user in session
         # return "Access Denied: Faculty privileges required.", 403
         pass

    try:
        # Mock user if missing (for direct access testing without login)
        if not user:
            user = {
                'name': 'Demo Faculty', 
                'email': 'demo@faculty.com', 
                'role': 'faculty',
                'id': 'FAC-DEMO'
            }
        
        return render_template('faculty.html', user=user)
    except Exception as e:
        logger.error(f"Error serving faculty dashboard: {e}")
        return f"Error loading faculty dashboard: {e}", 500

@app.route('/student')
def student():
    try:
        # Get User from Session or Default
        from flask import session
        user = session.get('user', {
            'name': 'Ojas Marathe',
            'id': 'AN20210001',
            'dept': 'Computer Science'
        })

        # Get Month for calendar (default current)
        month_str = request.args.get('month')
        today = datetime.now()
        
        if month_str:
            try:
                year, month = map(int, month_str.split('-'))
                curr_date = datetime(year, month, 1)
            except:
                curr_date = today
        else:
            curr_date = today

        # Calendar Logic
        import calendar
        cal = calendar.monthcalendar(curr_date.year, curr_date.month)
        calendar_days = []
        for week in cal:
            for day in week:
                if day == 0:
                    calendar_days.append({'day': '', 'empty': True})
                else:
                    is_today = (day == today.day) and (curr_date.month == today.month) and (curr_date.year == today.year)
                    calendar_days.append({'day': day, 'empty': False, 'is_today': is_today})

        # Next/Prev Links (Standard Lib Safe)
        if curr_date.month == 12:
            next_m = datetime(curr_date.year + 1, 1, 1)
        else:
            next_m = datetime(curr_date.year, curr_date.month + 1, 1)
            
        if curr_date.month == 1:
            prev_m = datetime(curr_date.year - 1, 12, 1)
        else:
            prev_m = datetime(curr_date.year, curr_date.month - 1, 1)
            
        next_str = next_m.strftime("%Y-%m")
        prev_str = prev_m.strftime("%Y-%m")
        
        cal_title = curr_date.strftime("%B %Y")
        
        # Get Data
        grades_resp = get_grades().json['grades']
         
        # Events
        conn = get_db_connection()
        events = []
        try:
             with conn.cursor() as cur:
                 query_date = curr_date.strftime("%Y-%m")
                 cur.execute('SELECT id, date, title, notes, owner FROM events WHERE date LIKE %s', (f"{query_date}%",))
                 rows = cur.fetchall()
                 events = [{'id': r[0], 'date': str(r[1]), 'title': r[2], 'notes': r[3], 'owner': r[4]} for r in rows]
        finally:
             conn.close()

        # Logs
        logs_resp = get_audit_logs().json['logs']
        
        # User session handling
        if not user or 'name' not in user:
             # Fallback if no session
             user = {'name': 'Ojas Marathe', 'email': 'ojas36Zeal@gmail.com', 'dept': 'AIML', 'id': 'AN2024001'}

        return render_template('student.html', 
                             user=user,
                             grades=grades_resp,
                             events=events,
                             logs=logs_resp,
                             calendar_days=calendar_days,
                             calendar_title=cal_title,
                             next_month=next_str,
                             prev_month=prev_str)
                             
    except Exception as e:
        logger.error(f"Error serving student.html: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route('/student/event/add', methods=['POST'])
def student_add_event():
    title = request.form.get('title')
    date = request.form.get('date')
    if title and date:
        conn = get_db_connection()
        try:
             with conn.cursor() as cur:
                 cur.execute('INSERT INTO events (date, title, notes, owner) VALUES (%s, %s, %s, %s)',
                            (date, title, 'Personal Event', 'Ojas'))
        finally:
             conn.close()
    return redirect(url_for('student'))

@app.route('/student/event/delete', methods=['POST'])
def student_delete_event():
    eid = request.form.get('id')
    if eid:
        conn = get_db_connection()
        try:
             with conn.cursor() as cur:
                 cur.execute('DELETE FROM events WHERE id=%s', (eid,))
        finally:
             conn.close()
    return redirect(url_for('student'))

@app.route('/student-demo')
def student_demo():
    try:
        return send_from_directory(os.path.join(BASE_DIR, 'Home'), 'student-demo.html')
    except Exception as e:
        logger.error(f"Error serving student.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/faculty')
def faculty():
    try:
        return send_from_directory(BASE_DIR, 'faculty.html')
    except Exception as e:
        logger.error(f"Error serving faculty.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/faculty-demo')
def faculty_demo():
    try:
        return send_from_directory(os.path.join(BASE_DIR, 'Home'), 'faculty-demo.html')
    except Exception as e:
        logger.error(f"Error serving faculty.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/app')
def main_app():
    try:
        return send_from_directory(os.path.join(BASE_DIR, 'Home'), 'CASWare-Home.html')
    except Exception as e:
        logger.error(f"Error serving main app: {e}")
        return f"Error: {str(e)}", 500

@app.route('/administrator')
def administrator():
    try:
        file_path = os.path.join(BASE_DIR, 'Administrator.html')
        if not os.path.exists(file_path):
             return "Administrator.html not found", 404
        return send_from_directory(BASE_DIR, 'Administrator.html')
    except Exception as e:
        logger.error(f"Error serving Administrator.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/admin')
def admin():
    try:
        file_path = os.path.join(BASE_DIR, 'admin.html')
        if not os.path.exists(file_path):
            logger.error(f"admin.html not found at: {file_path}")
            return f"File not found: admin.html<br>Base directory: {BASE_DIR}<br>Files: {', '.join(os.listdir(BASE_DIR)[:10])}", 404
        return send_from_directory(BASE_DIR, 'admin.html')
    except Exception as e:
        logger.error(f"Error serving admin.html: {e}")
        import traceback
        return f"Error: {str(e)}<br><pre>{traceback.format_exc()}</pre>", 500

@app.route('/admin-demo')
def admin_demo():
    try:
        return send_from_directory(os.path.join(BASE_DIR, 'Home'), 'admin_demo.html')
    except Exception as e:
        logger.error(f"Error serving admin_demo.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/documentation')
def documentation():
    return send_from_directory(BASE_DIR, 'documentation.html')

@app.route('/help')
def help_center():
    return send_from_directory(os.path.join(BASE_DIR, 'Home'), 'help_center.html')

@app.route('/privacy')
def privacy_policy():
    return send_from_directory(os.path.join(BASE_DIR, 'Home'), 'privacy_policy.html')

@app.route('/terms')
def terms_of_service():
    return send_from_directory(os.path.join(BASE_DIR, 'Home'), 'terms_of_service.html')

# Serve images from Home/img directory
@app.route('/img/<path:filename>')
def serve_images(filename):
    """Serve images from the Home/img directory"""
    try:
        img_dir = os.path.join(BASE_DIR, 'Home', 'img')
        return send_from_directory(img_dir, filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return f"Image not found: {filename}", 404

# Camera frame endpoint - INTEGRATED VERSION
# Server-Side Camera Routes

# Shared Registry for Recognition Throttling
GLOBAL_LAST_LOG_TIME = {}

def auto_log_presence(sid, name, conf, actor="System"):
    """Unify auto-attendance logic for all camera sources (Local, Browser, CCTV)"""
    if not sid or sid == "Unknown" or name == "Detecting...":
        return False

    now = time.time()
    if now - GLOBAL_LAST_LOG_TIME.get(sid, 0) < 60:
        return False # Cooldown

    is_student = False
    is_faculty = False
    
    if "Prof" in sid or "FAC" in sid.upper() or "Faculty" in name:
        is_faculty = True
    elif sid.isdigit() or sid.upper().startswith(('STU', 'AN')):
        is_student = True
    
    should_log = False
    log_details = {}
    subject = "General"
    
    if is_faculty:
        should_log = True
        subject = "Faculty Presence"
        log_details = {
            'slm_log': f"Faculty {name} detected via {actor}",
            'student_id': sid,
            'confidence': conf,
            'course': "N/A",
            'status': 'Active'
        }
    elif is_student:
        active_session = get_current_session()
        # Handle both dict-returning and string-returning get_current_session variants
        if isinstance(active_session, dict):
            should_log = True
            subject = active_session.get('subject', 'Unknown Class')
            room = active_session.get('room', '')
            log_details = {
                'slm_log': f"Attended {subject} (via {actor})",
                'student_id': sid,
                'confidence': conf,
                'course': subject,
                'room': room,
                'status': 'Present'
            }
        elif isinstance(active_session, str):
             # For the S1-S6 variant, we assume it's a valid session match
             should_log = True
             subject = f"Session {active_session}"
             log_details = {
                'slm_log': f"Attended {active_session} (via {actor})",
                'student_id': sid,
                'confidence': conf,
                'course': subject,
                'status': 'Present'
            }
             
    if should_log:
        GLOBAL_LAST_LOG_TIME[sid] = now
        append_audit(actor, 'auto_attendance', log_details)
        logger.info(f"Auto-marked ({actor}): {name} ({sid}) - {subject}")
        return True
    return False

# VideoCamera Class for Server-Side Streaming
# VideoCamera Class with Multi-Threading (Smooth UI + Async AI)
import threading
import queue

class VideoCamera:
    def __init__(self, camera_index=None, url=None):
        self.camera_index = -1
        self.cap = None
        self.is_running = False
        
        # Threading primitives
        self.lock = threading.Lock()
        self.current_frame = None
        self.processed_frame = None # Frame with overlays for display
        self.last_ai_result = [] # [(x,y,w,h, label, color), ...]
        
        # Connection params
        self.p_url = url
        self.p_index = camera_index
        
        # Ensure dataset embeddings loaded so recognition works from your dataset
        _ensure_embeddings_loaded()
        # Initialize connection (blocking attempt first to fail fast if needed)
        self._connect_camera()
        
        if self.cap:
             self.is_running = True
             self._get_frame_count = 0  # For if-else: run light recognition every Nth frame
             # Only capture thread; recognition runs inline in get_frame() to avoid multithreading issues
             self.thread_capture = threading.Thread(target=self._capture_loop, daemon=True)
             self.thread_capture.start()
             logger.info("VideoCamera started (capture thread + inline light face recognition)")
        else:
             logger.error("Failed to initialize camera")

    def _connect_camera(self):
        """Internal helper to connect/reconnect to camera"""
        if self.p_url:
            try:
                self.cap = cv2.VideoCapture(self.p_url)
                if self.cap.isOpened():
                    self.camera_index = -99
                    logger.info(f"Connected to external camera: {self.p_url}")
                    return True
            except: pass
        
        # Local Camera Search
        indices = [self.p_index] if self.p_index is not None else [0, 1, 2]
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for idx in indices:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        # Patient Connection: Wait for hardware warmup
                        # Give it up to 5 attempts over ~2 seconds
                        connected = False
                        for attempt in range(5):
                            ret, _ = cap.read()
                            if ret:
                                connected = True
                                break
                            time.sleep(0.4)
                        
                        if connected:
                            self.cap = cap
                            self.camera_index = idx
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Smooth 15 FPS if driver supports it
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Low buffer for responsive frames
                            logger.info(f"Connected to local camera {idx} after {attempt+1} attempts")
                            return True
                        cap.release()
                except: pass
        return False

    def __del__(self):
        self.stop()

    def stop(self):
        """Stops the camera and AI threads, ensuring hardware is released first"""
        self.is_running = False
        
        # Release hardware immediately to prevent thread hang on Windows
        if self.cap:
             try:
                 if self.cap.isOpened():
                    self.cap.release()
                 logger.info("Camera hardware released")
             except Exception as e:
                 logger.error(f"Error releasing camera: {e}")
             self.cap = None

        if hasattr(self, 'thread_capture'):
            self.thread_capture.join(timeout=1.0)
        logger.info("VideoCamera stopped")

    def _capture_loop(self):
        """Capture frames using read(); mirror and update shared frame."""
        black_frame_count = 0
        while self.is_running:
            if self.cap and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        center = frame[h//3:2*h//3, w//3:2*w//3]
                        if center.mean() < 0.4:
                            black_frame_count += 1
                            if black_frame_count > 450:
                                logger.warning("Persistent black frame; attempting hardware reset...")
                                self.cap.release()
                                time.sleep(1.0)
                                self._connect_camera()
                                black_frame_count = 0
                        else:
                            black_frame_count = 0
                        processed_frame = cv2.flip(frame, 1)
                        with self.lock:
                            self.current_frame = processed_frame
                    else:
                        time.sleep(0.03)
                except Exception as e:
                    logger.error(f"Capture loop error: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.5)

    def _no_signal_bytes(self):
        """Return JPEG bytes for NO SIGNAL placeholder so stream never breaks."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(image, "Waiting for camera...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes() if ret else b''

    def get_frame(self):
        """Returns the latest frame with overlays. Light face detection + recognition inline (if-else every 6th frame)."""
        global last_detection_time, faces_detected_count
        frame_copy = None
        with self.lock:
            if self.current_frame is not None:
                frame_copy = self.current_frame.copy()
        if frame_copy is None:
            for _ in range(6):
                time.sleep(0.05)
                with self.lock:
                    if self.current_frame is not None:
                        frame_copy = self.current_frame.copy()
                        break
            if frame_copy is None:
                return self._no_signal_bytes()

        # Detection every 6th frame (fast); recognition every 30th frame only (slow - avoid blocking stream)
        self._get_frame_count = getattr(self, '_get_frame_count', 0) + 1
        run_detection = (self._get_frame_count % 6 == 0) and frame_copy.mean() >= 0.4
        run_recognition = (self._get_frame_count % 30 == 0)
        if run_detection:
            raw_faces = []
            scale_factor = 0.25
            small_img = cv2.resize(frame_copy, (0, 0), fx=scale_factor, fy=scale_factor)
            if MP_AVAILABLE and face_detector:
                try:
                    rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                    results = face_detector.process(rgb)
                    if results.detections:
                        h, w, _ = frame_copy.shape
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            raw_faces.append((int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)))
                except Exception:
                    pass
            if not raw_faces and face_cascade:
                gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
                rects = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                for (x, y, w, h) in rects:
                    raw_faces.append((int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)))
            faces_found = []
            if raw_faces:
                last_detection_time = time.time()
                faces_detected_count = len(raw_faces)
                for (x, y, w, h) in raw_faces:
                    label = "..."
                    color = (0, 165, 255)
                    if run_recognition:
                        y1, y2 = max(0, y), min(frame_copy.shape[0], y + h)
                        x1, x2 = max(0, x), min(frame_copy.shape[1], x + w)
                        face_roi = frame_copy[y1:y2, x1:x2]
                        if face_roi.size > 0:
                            sid, name, conf = get_face_identity_light(face_roi)
                            if name != "Unknown":
                                label = name
                                color = (0, 255, 0)
                                auto_log_presence(sid, name, conf, actor="Server-Cam")
                            else:
                                label = "Unknown"
                    faces_found.append((x, y, w, h, label, color))
            self.last_ai_result = faces_found

        ai_data = self.last_ai_result
        for (x, y, w, h, label, color) in ai_data:
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, 2)
            if label:
                 cv2.putText(frame_copy, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Frame Stats
        cv2.putText(frame_copy, "LIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ret or jpeg is None:
            return self._no_signal_bytes()
        return jpeg.tobytes()

def gen(camera):
    """MJPEG stream at 12 FPS; get_frame() always returns bytes (frame or placeholder)."""
    TARGET_FPS = 12
    frame_interval = 1.0 / TARGET_FPS
    while True:
        start_time = time.time()
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if not camera.is_running:
            break
        elapsed = time.time() - start_time
        sleep_time = max(0.005, frame_interval - elapsed)
        time.sleep(sleep_time)

@app.route('/api/camera/control', methods=['POST'])
def camera_control():
    global video_camera
    data = request.json or {}
    action = data.get('action')
    
    if action == 'stop':
        if video_camera:
            video_camera.stop()
            video_camera = None # Force re-init next time
        return jsonify({'status': 'stopped'})
    
    elif action == 'start':
        if video_camera is None:
            video_camera = VideoCamera()
        return jsonify({'status': 'started'})
        
    return jsonify({'error': 'Invalid action'}), 400

@app.route('/video_feed')
def video_feed():
    """Serve MJPEG stream. Create camera if None so stream works after Start (handles multi-worker)."""
    global video_camera
    if video_camera is None:
        try:
            video_camera = VideoCamera()
        except Exception as e:
            logger.error(f"video_feed: failed to create camera: {e}")
            return "Camera not available.", 503
    if not video_camera.is_running:
        return "Camera not running.", 503
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/reset', methods=['POST'])
def reset_camera_feed():
    """Forces the camera to release and re-initialize."""
    global video_camera
    try:
        if video_camera:
            del video_camera # Trigger __del__ to release cap
            video_camera = None
        logger.info("Camera reset requested manually.")
        return jsonify({'status': 'success', 'message': 'Camera reset. Refresh page.'})
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detection_status')
def get_detection_status():
    """Returns whether faces are currently being detected"""
    global last_detection_time, faces_detected_count, video_camera
    # Active if detection happened in last 2 seconds
    is_active = (time.time() - last_detection_time) < 2.0
    
    # Check if camera is running
    c_count = 0
    if video_camera and video_camera.is_running:
        c_count = 1
        
    return jsonify({
        'active': is_active,
        'count': faces_detected_count if is_active else 0,
        'camera_active': c_count > 0,
        'camera_count': c_count
    })

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global video_camera
    data = request.get_json() or {}
    url = data.get('url')
    
    if video_camera is None:
        try:
            video_camera = VideoCamera(url=url)
            return jsonify({'status': 'started', 'camera_index': video_camera.camera_index})
        except RuntimeError as e:
            logger.error(f"Failed to start camera: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 503
    return jsonify({'status': 'already_running', 'camera_index': video_camera.camera_index})

@app.route('/api/camera/list', methods=['GET'])
def list_cameras():
    """List available cameras for debugging"""
    available = []
    for idx in range(5):  # Check indices 0-4
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        available.append({'index': idx, 'backend': str(backend), 'readable': True})
                        break
            except Exception as e:
                pass
    return jsonify({'cameras': available})

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users for the drag-and-drop management UI."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT email, name, role, face_encoding_status FROM users')
            rows = cur.fetchall()
            users = []
            for r in rows:
                users.append({
                    'email': r[0],
                    'name': r[1],
                    'role': r[2],
                    'has_face': bool(r[3])
                })
            return jsonify({'users': users})
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/users/update_role', methods=['POST'])
def update_user_role():
    """Update a user's role."""
    data = request.json
    email = data.get('email')
    new_role = data.get('role')
    
    if not email or not new_role:
        return jsonify({'error': 'Email and role required'}), 400
        
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Update modern table
            cur.execute('UPDATE users SET role = %s WHERE email = %s', (new_role, email))
            
            # Sync to legacy table if exists
            try:
                cur.execute('UPDATE accounts SET role = %s WHERE email = %s', (new_role, email))
            except:
                pass
                
            conn.commit()
            
            # Update In-Memory Cache
            global EMAIL_TO_USER
            # Find the user key (case-insensitive search)
            user_key = next((k for k in EMAIL_TO_USER if k.lower() == email.lower()), None)
            
            if user_key:
                EMAIL_TO_USER[user_key]['role'] = new_role
                logger.info(f"Updated role for {email} to {new_role} (Cache updated)")
            else:
                # If not in cache, try to fetch full user info to add them
                cur.execute('SELECT name, role FROM users WHERE email = %s', (email,))
                user_data = cur.fetchone()
                if user_data:
                    EMAIL_TO_USER[email] = {
                        'name': user_data[0],
                        'role': user_data[1],
                        'password_hash': 'DB_MANAGED' # Placeholder
                    }
                logger.warning(f"Updated role for {email} to {new_role} but user not in RAM cache - Syncing now")
                
            return jsonify({'status': 'success', 'ok': True, 'message': f'Role updated to {new_role}'})
    except Exception as e:
        logger.error(f"Error updating role: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# === Faculty Courses Management ===
@app.route('/api/faculty/courses', methods=['GET'])
def get_faculty_courses():
    """Get courses assigned to faculty member."""
    email = request.args.get('email')
    if not email:
        return jsonify({'error': 'Email parameter required'}), 400
    
    conn = get_db_connection()
    try:
        # Ensure table exists
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS faculty_courses (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    faculty_email VARCHAR(255),
                    course_code VARCHAR(50),
                    course_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_assignment (faculty_email, course_code)
                )
            ''')
            conn.commit()
            
            # Fetch courses for this faculty
            cur.execute(
                'SELECT course_code, course_name FROM faculty_courses WHERE faculty_email = %s',
                (email,)
            )
            rows = cur.fetchall()
            courses = [{'code': r[0], 'name': r[1]} for r in rows]
            
            return jsonify({'courses': courses})
    except Exception as e:
        logger.error(f"Error fetching faculty courses: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/faculty/courses/assign', methods=['POST'])
def assign_faculty_course():
    """Assign a course to faculty."""
    data = request.json
    email = data.get('email')
    course_code = data.get('code')
    course_name = data.get('name')
    
    if not email or not course_code or not course_name:
        return jsonify({'error': 'Email, code, and name required'}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'INSERT INTO faculty_courses (faculty_email, course_code, course_name) VALUES (%s, %s, %s)',
                (email, course_code, course_name)
            )
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Course assigned'})
    except Exception as e:
        logger.error(f"Error assigning course: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/faculty/courses/remove', methods=['DELETE'])
def remove_faculty_course():
    """Remove course assignment from faculty."""
    email = request.args.get('email')
    course_code = request.args.get('code')
    
    if not email or not course_code:
        return jsonify({'error': 'Email and code required'}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'DELETE FROM faculty_courses WHERE faculty_email = %s AND course_code = %s',
                (email, course_code)
            )
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Course removed'})
    except Exception as e:
        logger.error(f"Error removing course: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/courses/available', methods=['GET'])
def get_available_courses():
    """Get list of all available courses."""
    # Predefined course list - you can make this dynamic from DB later
    courses = [
        {'code': 'CS101', 'name': 'Introduction to Computer Science'},
        {'code': 'CS102', 'name': 'Data Structures'},
        {'code': 'CS201', 'name': 'Database Management Systems'},
        {'code': 'CS202', 'name': 'Operating Systems'},
        {'code': 'EE101', 'name': 'Basic Electrical Engineering'},
        {'code': 'ME101', 'name': 'Engineering Mechanics'},
        {'code': 'MATH101', 'name': 'Engineering Mathematics I'},
        {'code': 'MATH102', 'name': 'Engineering Mathematics II'}
    ]
    return jsonify({'courses': courses})

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Returns system statistics for the dashboard."""
    try:
        # 1. Total Students & Faculty
        total_students = len([u for u in EMAIL_TO_USER.values() if u.get('role') == 'student'])
        total_faculty = len([u for u in EMAIL_TO_USER.values() if u.get('role') == 'faculty'])
        
        # 2. Present Today (Unique students in attendance logs today)
        conn = get_db_connection()
        present_count = 0
        try:
            with conn.cursor() as cur:
                today_str = datetime.now().strftime('%Y-%m-%d')
                # Count unique students from attendance table for today
                cur.execute("SELECT COUNT(DISTINCT actor) FROM attendance WHERE date = %s", (today_str,))
                result = cur.fetchone()
                if result:
                    present_count = result[0]
                
                # Also check audit logs for auto-attendance if not in main table yet
                # (This is an approximation, ideally we join or union)
        except Exception as e:
            logger.error(f"Error fetching attendance stats: {e}")
        finally:
            conn.close()

        return jsonify({
            'total_students': total_students,
            'total_faculty': total_faculty,
            'present_today': present_count,
            'system_uptime': '99.9%' # Static for now
        })
    except Exception as e:
        logger.error(f"Error serving dashboard stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/health', methods=['GET'])
def get_system_health():
    """Return dynamic disk usage and system health."""
    try:
        total, used, free = shutil.disk_usage("/")
        # Convert to GB and percentages
        disk_total = round(total / (2**30), 1)
        disk_used = round(used / (2**30), 1)
        disk_percent = round((used / total) * 100, 1)
        
        return jsonify({
            'status': 'success',
            'disk': {
                'total': f"{disk_total} GB",
                'used': f"{disk_used} GB",
                'percent': disk_percent
            },
            'cpu': 'Healthy', # Static fallback for CPU without psutil
            'time': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error serving system health: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/accounts/create', methods=['POST'])
def create_account():
    """Create a new account with dynamic role assignment based on strict email patterns."""
    try:
        data = request.json
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        name = data.get('name', '').strip()

        if not email or not password or not name:
            return jsonify({'status': 'error', 'ok': False, 'message': 'All fields are required', 'error': 'fields_required'}), 400

        # Strict Email Pattern: NameRollNumberCollegeName@gmail.com
        # Example: Ojas36Zeal@gmail.com
        student_pattern = r'^[A-Za-z]+\d+[A-Za-z]+@gmail\.com$'
        
        email_lower = email.lower()
        role = 'pending' # Default to pending for manual approval

        # Role Assignment Logic
        if 'admin' in email_lower:
            role = 'admin'
        elif 'faculty' in email_lower or 'prof' in email_lower:
            role = 'faculty'
        elif re.match(student_pattern, email, re.IGNORECASE):
            role = 'pending' # Students MUST be approved
        else:
            return jsonify({'status': 'error', 'ok': False, 'message': 'Invalid Email Format. Use NameRollNumberCollegeName@gmail.com', 'error': 'invalid_format'}), 400

        # Hash password using app's standard sha256
        hashed_pw = hash_password(password)

        # Connect to database
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Check if user already exists in 'users' table
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    return jsonify({'status': 'error', 'ok': False, 'message': 'User already exists', 'error': 'user_exists'}), 400

                # Insert user into 'users' table (modern table used by Admin Portal)
                cursor.execute(
                    "INSERT INTO users (email, name, password, role) VALUES (%s, %s, %s, %s)",
                    (email, name, hashed_pw, role)
                )
                
                # Also insert into legacy 'accounts' table for compatibility if needed
                # (Assuming 'accounts' has similar schema but use password_hash)
                # This is risky if tables are out of sync, but better than breakages.
                try:
                    cursor.execute(
                        "INSERT INTO accounts (email, name, password_hash, role) VALUES (%s, %s, %s, %s)",
                        (email, name, hashed_pw, role)
                    )
                except:
                    pass # Ignore if table doesn't exist or fails
                    
                conn.commit()
        finally:
            conn.close()

        # Update In-Memory Cache
        global EMAIL_TO_USER
        EMAIL_TO_USER[email] = {
            'email': email,
            'name': name,
            'password': hashed_pw,
            'role': role
        }

        logger.info(f"Created account for {email} with role {role} (Strict Pattern)")

        # Auto-login: Store in session
        session['user'] = {'email': email, 'name': name, 'role': role}

        # Return redirect based on role
        redirect_url = '/'
        if role == 'admin':
            redirect_url = '/administrator'
        elif role == 'faculty':
            redirect_url = '/faculty'
        elif role == 'pending':
            redirect_url = '/account?pending=true'
        else:
            redirect_url = '/student'

        return jsonify({'status': 'success', 'ok': True, 'redirect': redirect_url, 'redirect_url': redirect_url})

    except Exception as e:
        logger.error(f"Error creating account: {e}")
        return jsonify({'status': 'error', 'ok': False, 'message': str(e), 'error': 'server_error'}), 500

    except Exception as e:
        logger.error(f"Error creating account: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global video_camera
    # Check if 'video_camera' is in globals() to avoid NameError
    if 'video_camera' in globals() and video_camera:
        del video_camera
        video_camera = None
    return jsonify({'status': 'stopped'})

# Legacy/Unused Frame Endpoint (Kept to prevent 404s if JS still calls it initially)
@app.route('/api/camera/frame', methods=['POST', 'OPTIONS'])
def receive_camera_frame():
    return jsonify({'status': 'ignored_server_side_streaming_active'})

@app.route('/api/recognize', methods=['POST'])
def handle_recognition():
    """Handle one-shot recognition request from Admin Dashboard"""
    data = request.get_json()
    image_b64 = data.get('image')
    
    if not image_b64:
        return jsonify({'status': 'error', 'error': 'No image data'}), 400
        
    try:
        # Decode base64
        if 'base64,' in image_b64:
            image_b64 = image_b64.split('base64,')[1]
            
        import base64
        image_bytes = base64.b64decode(image_b64)
        
        # Convert bytes to numpy array for cropping
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'status': 'error', 'error': 'Invalid image data'}), 400

        # Use existing detection logic
        detections = detect_faces_and_bodies(image_bytes)
        
        # Add identity info
        for face in detections.get('faces', []):
            try:
                # Need to crop ROI for identification
                y1, y2 = max(0, face['y']), min(image.shape[0], face['y']+face['h'])
                x1, x2 = max(0, face['x']), min(image.shape[1], face['x']+face['w'])
                face_roi = image[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    sid, name, conf = get_face_identity(face_roi)
                    face['identity'] = name
                    face['sid'] = sid
                    face['confidence'] = conf
                    
                    # Auto-log if matched (Cloud/Browser Mode)
                    auto_log_presence(sid, name, conf, actor="Browser-Camera")
                else:
                    face['identity'] = 'Unknown'
            except Exception as e:
                logger.warning(f"Feature identification error: {e}")
                face['identity'] = 'Unknown'
            
        return jsonify({'status': 'success', 'detections': detections})
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# --- ATTENDANCE FEATURE ENDPOINTS (Hybrid & Stats) ---

def get_current_session():
    """Map current time to session ID (S1-S6) based on college schedule."""
    now = datetime.now().time()
    
    # helper to check if time is between two strings
    def is_between(start_str, end_str):
        start = datetime.strptime(start_str, "%H:%M").time()
        end = datetime.strptime(end_str, "%H:%M").time()
        return start <= now <= end

    if is_between("08:00", "09:00"): return "S1"
    if is_between("09:00", "10:00"): return "S2"
    if is_between("10:30", "11:30"): return "S3"
    if is_between("11:30", "12:30"): return "S4"
    if is_between("13:45", "14:45"): return "S5"
    if is_between("14:45", "15:45"): return "S6"
    return None

@app.route('/api/attendance/live', methods=['GET'])
def get_live_attendance():
    """
    Get list of students detected *today* by the camera.
    Sources: Audit logs with action='auto_attendance' and today's date.
    """
    try:
        updated_students = []
        today_str = datetime.now().strftime('%Y-%m-%d')
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Fetch distinct student IDs detected today
                # JSON_UNQUOTE ensures we get clean strings
                cur.execute("""
                    SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(details, '$.student_id')) as sid 
                    FROM audit_log 
                    WHERE action = 'auto_attendance' 
                    AND DATE(timestamp) = %s
                """, (today_str,))
                rows = cur.fetchall()
                detected_ids = [r[0] for r in rows if r[0]]
                
                return jsonify({
                    'status': 'success', 
                    'detected_ids': detected_ids,
                    'current_session': get_current_session(),
                    'server_time': datetime.now().strftime('%H:%M:%S')
                })
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching live attendance: {e}")
        return jsonify({'status': 'error', 'detected_ids': []})

@app.route('/api/attendance/logs', methods=['GET'])
def get_student_logs():
    """
    Get detailed logs of student detections for the current day.
    """
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, JSON_UNQUOTE(JSON_EXTRACT(details, '$.student_id')), JSON_UNQUOTE(JSON_EXTRACT(details, '$.slm_log'))
                    FROM audit_log 
                    WHERE action = 'auto_attendance' 
                    AND DATE(timestamp) = %s
                    # AND details->>'$.student_id' IS NOT NULL # Allow all IDs (Faculty + Student)
                    ORDER BY timestamp DESC
                    LIMIT 50
                """, (today_str,))
                rows = cur.fetchall()
                logs = [{'time': r[0].strftime('%H:%M:%S'), 'id': r[1], 'msg': r[2]} for r in rows]
                return jsonify({'status': 'success', 'logs': logs})
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({'status': 'error', 'logs': []})

@app.route('/api/attendance/finalize', methods=['POST'])
def finalize_attendance():
    """
    Save the final attendance sheet (Hybrid: Camera + Manual).
    Expects: { 'date': 'YYYY-MM-DD', 'records': [{ 'student_id': '...', 'status': 1/0 }] }
    """
    data = request.get_json()
    date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    records = data.get('records', [])
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # First, clear existing records for this subject/date to allow updates
            # Assuming 'CS-301' for demo. In real app, subject would be passed.
            cur.execute("DELETE FROM attendance WHERE date = %s AND course = 'CS-301'", (date_str,))
            
            # Insert new records
            # Since our simple table stores 'rows_json', we'll just store the whole blob 
            # OR inserts individual rows if we changed schema.
            # Current Schema: id, course, date, actor, rows_json
            
            # For this demo, let's store the bulk record
            cur.execute("""
                INSERT INTO attendance (course, date, actor, rows_json)
                VALUES (%s, %s, %s, %s)
            """, ('CS-301', date_str, 'Faculty-Hybrid', json.dumps(records)))
            
        conn.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Finalize attendance error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/attendance/stats/<student_id>', methods=['GET'])
def get_student_stats(student_id):
    """
    Calculate reverse attendance stats.
    Logic: Start at 100%. Deduct for absences.
    Rules: Sunday is Holiday (no penalty). Saturday is neutral (optional).
    """
    try:
        # Real-time calculation from DB
        conn = get_db_connection()
        total_sessions = 0
        present_days = 0
        absent_days = 0
        
        try:
            with conn.cursor() as cur:
                # 1. Get Auto-Attendance Logs (Audit Table)
                # Count unique automated check-ins for this student
                cur.execute("""
                    SELECT DISTINCT DATE(timestamp), JSON_UNQUOTE(JSON_EXTRACT(details, '$.course'))
                    FROM audit_log 
                    WHERE action = 'auto_attendance' 
                    AND JSON_UNQUOTE(JSON_EXTRACT(details, '$.student_id')) = %s
                """, (student_id,))
                
                auto_rows = cur.fetchall()
                # auto_sessions = set((date, course))
                auto_sessions = {(r[0], r[1]) for r in auto_rows}
                
                # 2. Get Finalized Manual/Hybrid Records (Attendance Table)
                cur.execute("SELECT rows_json, date, course FROM attendance")
                manual_rows = cur.fetchall()
                
                manual_present_sessions = set()
                manual_absent_sessions = set()
                all_manual_sessions = set()
                
                for r in manual_rows:
                    try:
                        sess_date = r[1]
                        sess_course = r[2]
                        all_manual_sessions.add((sess_date, sess_course))
                        
                        record_data = json.loads(r[0]) if isinstance(r[0], str) else r[0]
                        student_record = next((item for item in record_data if str(item.get('student_id')) == str(student_id)), None)
                        
                        if student_record:
                            status = str(student_record.get('status', '')).lower()
                            if status in ['1', 'true', 'present', 'p']:
                                manual_present_sessions.add((sess_date, sess_course))
                            else:
                                manual_absent_sessions.add((sess_date, sess_course))
                    except: 
                        continue

                # 3. Merge Logic (Union of Automated + Manual)
                # If manual record exists, it takes precedence (e.g. override).
                # If no manual record, but auto-log exists -> Present.
                
                # Total known sessions = All Manual Sessions UNION All Auto Sessions (System-wide? No, let's assume auto-log implies a session happened)
                
                final_present = manual_present_sessions.union(auto_sessions)
                
                # Absences are tricky without a master schedule history.
                # Use explicit manual absences + (Implicit absences? No, let's stick to what we know)
                # For "Dynamic Demopage", showing 100% until proven absent is safer, 
                # OR we calculate "Total Sessions" as count of present + count of explicit absent.
                
                total_sessions_count = len(final_present) + len(manual_absent_sessions)
                present_days = len(final_present)
                absent_days = len(manual_absent_sessions)
                
                # Correction: If a session is in BOTH present and absent (conflict), prioritize manual.
                # (handled by union logic implicitly if we trust manual_absent)
                # If I was auto-present but manual-absent -> manual wins.
                conflict_sessions = final_present.intersection(manual_absent_sessions)
                for s in conflict_sessions:
                    # Remove from present, keep in absent
                    present_days -= 1
                    # Total stays same
                    
        finally:
            conn.close()
            
        # Default to 100% if no sessions yet
        if total_sessions == 0:
            percentage = 100.0
        else:
            percentage = (present_days / total_sessions) * 100
        
        return jsonify({
            'status': 'success',
            'student_id': student_id,
            'stats': {
                'total_working_days': total_sessions,
                'present': present_days,
                'absent': absent_days,
                'percentage': round(percentage, 1),
                'holidays': 0 # Dynamic calculation complex, leaving 0 for now
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# --- EXISTING API ENDPOINTS ---

@app.route('/api/camera/fallback', methods=['POST'])
def launch_fallback_camera():
    """Launch Python Tkinter camera app with live logs"""
    try:
        camera_script = os.path.join(BASE_DIR, 'camera_fallback.py')
        if not os.path.exists(camera_script):
            return jsonify({'ok': False, 'error': f'camera_fallback.py not found at {camera_script}'}), 404
        
        # Launch in separate process so it doesn't block
        subprocess.Popen(['python', camera_script], shell=False, cwd=BASE_DIR)
        logger.info(f"Launched camera_fallback.py from {BASE_DIR}")
        return jsonify({'ok': True, 'message': 'Camera fallback application launched'})
    except Exception as e:
        logger.error(f"Failed to launch camera app: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


# API Endpoints
@app.route('/api/students', methods=['GET'])
def handle_students():
    students = load_students()
    division = request.args.get('division', '').strip()
    if division and division in DIVISION_TO_COURSE:
        course_filter = (DIVISION_TO_COURSE[division] or '').strip().upper()
        def matches(course_val):
            c = (course_val or '').strip().upper().replace('-', '')
            f = (course_filter or '').replace('-', '')
            return c == f or c.startswith(f) or f.startswith(c)
        students = {sid: info for sid, info in students.items() if matches(info.get('course'))}
    return jsonify({'students': students})

@app.route('/api/students/update', methods=['POST'])
def handle_update_student():
    data = request.get_json()
    student_id = data.get('id')
    name = data.get('name')
    course = data.get('course')
    
    if not student_id:
        return jsonify({'ok': False, 'error': 'id_required'}), 400
        
    students = load_students()
    
    if student_id in students:
        students[student_id]['name'] = name
        students[student_id]['course'] = course
        
        # Save back to file
        try:
            with open(os.path.join(DATA_DIR, 'students.json'), 'w') as f:
                json.dump(students, f, indent=2)
            return jsonify({'ok': True})
        except Exception as e:
            logger.error(f"Failed to save students: {e}")
            return jsonify({'ok': False, 'error': 'save_failed'}), 500
    else:
        return jsonify({'ok': False, 'error': 'not_found'}), 404

# Legacy creation route removed - redirected to unified /api/accounts/create

@app.route('/api/accounts/login', methods=['POST'])
def handle_login_account():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'ok': False, 'error': 'email and password required'}), 400
    
    pw_hash = hash_password(password)
    
    # HARDCODED DEMO USERS (Nikita as Faculty for User Management drag-to-faculty + login)
    demo_users = {
        'ojas36Zeal@gmail.com': {'name': 'Ojas Marathe', 'role': 'student', 'pass': '1234'},
        'DemoFacultyZealPoly@gmail.com': {'name': 'Demo Faculty', 'role': 'faculty', 'pass': '1234'},
        'Admin123@gmail.com': {'name': 'System Admin', 'role': 'admin', 'pass': '1234'},
        'Nikita21Zeal@gmail.com': {'name': 'Nikita', 'role': 'faculty', 'pass': '1234'}
    }
    
    # Case-insensitive demo lookup
    demo_key = next((k for k in demo_users if k.lower() == email.strip().lower()), None)
    
    if demo_key:
        user = demo_users[demo_key]
        if password == user['pass']:
            return jsonify({'ok': True, 'email': email, 'name': user['name'], 'role': user['role'], 'dept': 'AIML'})
        else:
            return jsonify({'ok': False, 'error': 'invalid_credentials'}), 401

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check modern 'users' table first
            cur.execute('SELECT name, role, password FROM users WHERE email = %s', (email,))
            row = cur.fetchone()
            
            if row:
                name, role, stored_hash = row
                if stored_hash == pw_hash:
                    return jsonify({'ok': True, 'email': email, 'name': name, 'role': role, 'dept': 'AIML'})
                else:
                    return jsonify({'ok': False, 'error': 'invalid_credentials'}), 401
            
            # Fallback to legacy 'accounts' table
            cur.execute('SELECT name, role, password_hash FROM accounts WHERE email = %s', (email,))
            row = cur.fetchone()
            if not row:
                return jsonify({'ok': False, 'error': 'not_found'}), 404
            name, role, stored_hash = row
            if stored_hash != pw_hash:
                return jsonify({'ok': False, 'error': 'invalid_credentials'}), 401
            row = cur.fetchone()
            if not row:
                return jsonify({'ok': False, 'error': 'not_found'}), 404
            name, role, stored_hash = row
            if stored_hash != pw_hash:
                return jsonify({'ok': False, 'error': 'invalid_credentials'}), 401
    except Exception as e:
        logger.warning(f"DB Login failed, but might be expected if no DB: {e}")
        # Fallback for ANY login if DB fails (for demo purposes)
        if password == "1234": 
             return jsonify({'ok': True, 'email': email, 'name': 'Demo User', 'role': 'student', 'dept': 'AIML'})
        return jsonify({'ok': False, 'error': 'db_error'}), 500
    finally:
        if conn: conn.close()
    
    append_audit(email, 'login', {'role': role})
    dept = 'Administration' if role == 'admin' else 'Unknown'
    return jsonify({'ok': True, 'email': email, 'name': name, 'role': role, 'dept': dept})

@app.route('/api/accounts', methods=['GET'])
def handle_get_accounts():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT email, name, role FROM accounts')
            rows = cur.fetchall()
            accounts = [{'email': r[0], 'name': r[1], 'role': r[2]} for r in rows]
    finally:
        conn.close()
    return jsonify({'accounts': accounts})

@app.route('/api/events', methods=['GET'])
def handle_get_events():
    month = request.args.get('month')
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if month:
                cur.execute('SELECT id, date, title, notes, owner FROM events WHERE date LIKE %s', (f"{month}%",))
            else:
                cur.execute('SELECT id, date, title, notes, owner FROM events')
            rows = cur.fetchall()
            events = [{'id': r[0], 'date': str(r[1]), 'title': r[2], 'notes': r[3], 'owner': r[4]} for r in rows]
    finally:
        conn.close()
    return jsonify({'events': events})

@app.route('/api/events', methods=['POST'])
def handle_post_event():
    data = request.get_json()
    action = data.get('action', 'create')
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if action == 'create':
                cur.execute(
                    'INSERT INTO events (date, title, notes, owner) VALUES (%s, %s, %s, %s)',
                    (data.get('date'), data.get('title'), data.get('notes'), data.get('owner'))
                )
                eid = cur.lastrowid
                conn.close()
                append_audit(data.get('owner', 'unknown'), 'create_event', {'id': eid, 'date': data.get('date')})
                return jsonify({'ok': True, 'id': eid})
            elif action == 'update':
                cur.execute(
                    'UPDATE events SET date=%s, title=%s, notes=%s WHERE id=%s',
                    (data.get('date'), data.get('title'), data.get('notes'), data.get('id'))
                )
                return jsonify({'ok': True})
            elif action == 'delete':
                cur.execute('DELETE FROM events WHERE id=%s', (data.get('id'),))
                return jsonify({'ok': True})
            else:
                return jsonify({'ok': False, 'error': 'unknown_action'}), 400
    finally:
        conn.close()

# Default Schedule Data matching Faculty Slots and Student Subjects
DEFAULT_SCHEDULE = {
    'S1': {'time': '08:00 AM - 09:00 AM', 'subject': 'Management (MAN)', 'room': '204'},
    'S2': {'time': '09:00 AM - 10:00 AM', 'subject': 'Big Data Analytics (BDA)', 'room': 'Lab 1'},
    'S3': {'time': '10:30 AM - 11:30 AM', 'subject': 'Image Processing (PIP)', 'room': '205'},
    'S4': {'time': '11:30 AM - 12:30 PM', 'subject': 'Mobile Dev (MAD)', 'room': 'Lab 2'},
    'S5': {'time': '01:45 PM - 02:45 PM', 'subject': 'Capstone Project (CPE)', 'room': 'Project Lab'},
    'S6': {'time': '02:45 PM - 03:45 PM', 'subject': 'Free', 'room': '-'}
}

@app.route('/api/schedule', methods=['GET', 'POST'])
def handle_schedule():
    schedule_file = os.path.join(DATA_DIR, 'schedule.json')
    
    if request.method == 'GET':
        if os.path.exists(schedule_file):
            with open(schedule_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify(DEFAULT_SCHEDULE)
        
    elif request.method == 'POST':
        data = request.get_json()
        try:
            with open(schedule_file, 'w') as f:
                json.dump(data, f, indent=2)
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/quick-links', methods=['GET', 'POST'])
def handle_quick_links():
    links_file = os.path.join(DATA_DIR, 'quick_links.json')
    
    if request.method == 'GET':
        if os.path.exists(links_file):
            with open(links_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify([])
        
    elif request.method == 'POST':
        new_link = request.get_json()
        if not new_link.get('id'):
            new_link['id'] = str(int(time.time() * 1000))
            
        links = []
        if os.path.exists(links_file):
            with open(links_file, 'r') as f:
                links = json.load(f)
        
        links.append(new_link)
        
        try:
            with open(links_file, 'w') as f:
                json.dump(links, f, indent=2)
            return jsonify({'ok': True, 'link': new_link})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/feedback/forms', methods=['GET', 'POST'])
def handle_feedback_forms():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if request.method == 'GET':
                # Faculty only see published forms, Admin sees all
                role = request.args.get('role', 'faculty')
                if role == 'admin':
                    cur.execute("SELECT id, title, structure_json, status, created_at FROM feedback_forms ORDER BY created_at DESC")
                else:
                    cur.execute("SELECT id, title, structure_json, status, created_at FROM feedback_forms WHERE status = 'published' ORDER BY created_at DESC")
                
                rows = cur.fetchall()
                forms = []
                for r in rows:
                    forms.append({
                        'id': r[0],
                        'title': r[1],
                        'structure': json.loads(r[2]),
                        'status': r[3],
                        'created_at': r[4].isoformat() if r[4] else None
                    })
                return jsonify(forms)
                
            elif request.method == 'POST':
                data = request.get_json()
                form_id = data.get('id')
                title = data.get('title')
                structure = data.get('structure')
                status = data.get('status', 'draft')
                
                if form_id: # Update
                    cur.execute("""
                        UPDATE feedback_forms 
                        SET title = %s, structure_json = %s, status = %s 
                        WHERE id = %s
                    """, (title, json.dumps(structure), status, form_id))
                    return jsonify({'ok': True, 'message': 'Form updated'})
                else: # Create
                    cur.execute("""
                        INSERT INTO feedback_forms (title, structure_json, status)
                        VALUES (%s, %s, %s)
                    """, (title, json.dumps(structure), status))
                    return jsonify({'ok': True, 'id': cur.lastrowid})
    except Exception as e:
        logger.error(f"Feedback form error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/feedback/forms/<int:form_id>', methods=['DELETE'])
def delete_feedback_form(form_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Permanent delete as requested
            cur.execute("DELETE FROM feedback_forms WHERE id = %s", (form_id,))
            return jsonify({'ok': True, 'message': 'Form deleted permanently'})
    except Exception as e:
        logger.error(f"Delete form error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/feedback/submit', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    form_id = data.get('form_id')
    user_email = data.get('user_email')
    response = data.get('response')
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback_responses (form_id, user_email, response_json)
                VALUES (%s, %s, %s)
            """, (form_id, user_email, json.dumps(response)))
            return jsonify({'ok': True, 'message': 'Response submitted successfully'})
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/feedback/responses/<int:form_id>', methods=['GET'])
def get_feedback_responses(form_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, user_email, response_json, submitted_at 
                FROM feedback_responses 
                WHERE form_id = %s 
                ORDER BY submitted_at DESC
            """, (form_id,))
            rows = cur.fetchall()
            responses = []
            for r in rows:
                responses.append({
                    'id': r[0],
                    'user_email': r[1],
                    'response': json.loads(r[2]),
                    'submitted_at': r[3].isoformat() if r[3] else None
                })
            return jsonify(responses)
    except Exception as e:
        logger.error(f"Get responses error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/quick-links/<link_id>', methods=['DELETE'])
def delete_quick_link(link_id):
    links_file = os.path.join(DATA_DIR, 'quick_links.json')
    
    if not os.path.exists(links_file):
        return jsonify({'ok': False, 'error': 'not_found'}), 404
        
    try:
        with open(links_file, 'r') as f:
            links = json.load(f)
        
        new_links = [l for l in links if l.get('id') != link_id]
        
        if len(new_links) == len(links):
            return jsonify({'ok': False, 'error': 'not_found'}), 404
            
        with open(links_file, 'w') as f:
            json.dump(new_links, f, indent=2)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/attendance', methods=['GET'])
def handle_attendance_proposal():
    course = request.args.get('course', 'CS-301')
    date = request.args.get('date', datetime.now().date().isoformat())
    
    students = load_students()
    proposals = []
    for sid, info in students.items():
        proposals.append({
            'id': sid,
            'name': info.get('name'),
            'proposed_status': 'present' if hash(sid + date) % 5 != 0 else 'absent',
            'confidence': float((hash(sid) % 100) / 100.0),
            'explanation': 'Temporal presence & face match (simulated)',
            'camera': 1
        })
    
    return jsonify({'course': course, 'date': date, 'proposals': proposals})

@app.route('/api/attendance/logs', methods=['GET'])
def get_attendance_logs():
    """Fetch recent attendance logs from database"""
    camera_filter = request.args.get('camera')
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT rows_json, timestamp FROM attendance ORDER BY timestamp DESC LIMIT 100')
            rows = cur.fetchall()
            logs = []
            for r in rows:
                try:
                    row_data = json.loads(r[0]) if isinstance(r[0], str) else r[0]
                    if isinstance(row_data, list):
                        for item in row_data:
                            item['timestamp'] = r[1].isoformat() if r[1] else datetime.now().isoformat()
                            logs.append(item)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse log row: {e}")
                    continue
    finally:
        conn.close()
    return jsonify({'logs': logs})

@app.route('/api/attendance/save', methods=['POST'])
def handle_save_attendance():
    """Save attendance records to database"""
    data = request.get_json()
    course = data.get('course')
    date = data.get('date')
    actor = data.get('actor', 'unknown')
    rows = data.get('rows', [])
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'INSERT INTO attendance (course, date, actor, rows_json) VALUES (%s, %s, %s, %s)',
                (course, date, actor, json.dumps(rows))
            )
    finally:
        conn.close()
    
    out_path = save_attendance_csv(course or 'unknown', date or datetime.now().date().isoformat(), rows)
    
    append_audit(actor, 'save_attendance', {
        'course': course, 
        'date': date, 
        'rows_saved': len(rows), 
        'out_path': out_path
    })
    
    logger.info(f"Saved attendance: {len(rows)} records for {course} on {date}")
    
    return jsonify({'ok': True, 'out_path': out_path})

# Camera fallback endpoint - Launch Tkinter application
@app.route('/api/camera/fallback', methods=['POST'])
def launch_camera_fallback():
    """
    Launches the Tkinter camera fallback application
    This is used when browser camera is not available
    """
    try:
        # Get the path to the camera_fallback.py file
        fallback_script = os.path.join(os.path.dirname(__file__), 'camera_fallback.py')
        
        if not os.path.exists(fallback_script):
            logger.error(f"Camera fallback script not found at: {fallback_script}")
            return jsonify({'ok': False, 'error': 'Fallback script not found'}), 404
        
        # Launch the Tkinter app in a separate process
        def launch_app():
            try:
                if sys.platform == 'win32':
                    # Windows: use pythonw for GUI apps or python
                    subprocess.Popen([sys.executable, fallback_script], 
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    # Linux/Mac: use python
                    subprocess.Popen([sys.executable, fallback_script])
                logger.info("Camera fallback application launched successfully")
            except Exception as e:
                logger.error(f"Failed to launch camera fallback: {e}")
        
        # Launch in a separate thread to avoid blocking
        thread = threading.Thread(target=launch_app, daemon=True)
        thread.start()
        
        return jsonify({
            'ok': True,
            'message': 'Camera fallback application is launching...',
            'note': 'A new window should open with the camera feed'
        }), 200
        
    except Exception as e:
        logger.error(f"Error launching camera fallback: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/grades', methods=['GET'])
def get_grades():
    """Return mock grades for student dashboard"""
    # Mock data - in real app would query DB based on student ID
    grades = {
        'MAN': {'code': '315301', 'title': 'Management', 'mid': 24, 'end': 60, 'total': 84},
        'BDA': {'code': '316318', 'title': 'Big Data Analytics', 'mid': 26, 'end': 62, 'total': 88},
        'PIP': {'code': '316319', 'title': 'Principles of Image Processing', 'mid': 25, 'end': 65, 'total': 90},
        'MAD': {'code': '316006', 'title': 'Mobile Application Development', 'mid': 28, 'end': 68, 'total': 96},
        'NMA': {'code': '316007', 'title': 'Network Management', 'mid': 22, 'end': 58, 'total': 80},
        'CPE': {'code': '316004', 'title': 'Capstone Project', 'mid': 29, 'end': 69, 'total': 98},
        'AAM': {'code': '316320', 'title': 'Advanced Algorithm in AI & ML', 'mid': 27, 'end': 63, 'total': 90}
    }
    return jsonify({'ok': True, 'grades': grades})

# Serve static files (JS, CSS, images, etc.)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

# Also serve JS files directly from root for easier access
@app.route('/<filename>.js')
def serve_js(filename):
    """Serve JavaScript files directly from root path"""
    js_path = os.path.join(BASE_DIR, f'{filename}.js')
    if os.path.exists(js_path):
        return send_from_directory(BASE_DIR, f'{filename}.js')
    return f"File not found: {filename}.js", 404


@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Receive an image, detect faces, and recognize using loaded models.
    Returns JSON with detections and optional identity predictions.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    # Decode base64 image
    image_base64 = data['image']
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    try:
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Image decode error in /api/recognize: {e}")
        return jsonify({'error': 'Invalid image data'}), 400
    # Detect faces using existing cascade
    # Detect faces using MediaPipe (Primary) or Haar (Fallback)
    detections = {'faces': []}
    faces_list = []
    
    if MP_AVAILABLE and face_detector:
        try:
             results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
             if results.detections:
                 h_img, w_img, _ = img.shape
                 for detection in results.detections:
                     bboxC = detection.location_data.relative_bounding_box
                     faces_list.append((
                         int(bboxC.xmin * w_img),
                         int(bboxC.ymin * h_img),
                         int(bboxC.width * w_img),
                         int(bboxC.height * h_img)
                     ))
        except Exception as e:
            logger.warning(f"MediaPipe error in /api/recognize: {e}")

    if not faces_list and CV2_AVAILABLE and face_cascade is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Less sensitive params
        faces_haar = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        for (x, y, w, h) in faces_haar:
            faces_list.append((x, y, w, h))
            
    for (x, y, w, h) in faces_list:
            face_roi = img[y:y+h, x:x+w]
            # Preprocess for facenet (resize to 112x112, normalize)
            try:
                face_resized = cv2.resize(face_roi, (112, 112))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                input_data = np.expand_dims(face_rgb.astype(np.float32) / 255.0, axis=0)
                # Run through facenet interpreter if available
                identity = None
                if facenet_interpreter is not None:
                    if isinstance(facenet_interpreter, tf.lite.Interpreter):
                        input_index = facenet_interpreter.get_input_details()[0]['index']
                        output_index = facenet_interpreter.get_output_details()[0]['index']
                        facenet_interpreter.set_tensor(input_index, input_data)
                        facenet_interpreter.invoke()
                        embedding = facenet_interpreter.get_tensor(output_index)
                    else:
                        # Fallback for Keras model (MobileNetV2)
                        # MobileNetV2 expects specific preprocess, but we already have [0,1] floating point
                        # Let's adjust input if needed. MobileNetV2 preprocess usually expects [-1, 1]
                        # but if we initialized with weights='imagenet', we should follow its specific requirements.
                        # However, for simplicity and consistency with our training script's fallback:
                        # We will re-normalize to what tf.keras.applications.mobilenet_v2.preprocess_input expects
                        # which takes [0, 255] inputs. 
                        # Since we already divided by 255.0, let's revert or just feed raw 0-1 if compatible.
                        # Safest is to use the exact same preprocessing as Training script:
                        # train_model.py fallback uses: tf.keras.applications.mobilenet_v2.preprocess_input(face_rgb as float)
                        
                        # Re-create correct input for Keras MobileNetV2
                        input_data_keras = np.expand_dims(face_rgb.astype(np.float32), axis=0) # [0-255] range
                        input_data_keras = tf.keras.applications.mobilenet_v2.preprocess_input(input_data_keras)
                        embedding = facenet_interpreter.predict(input_data_keras, verbose=0)
                    
                    # Flatten embedding
                    if len(embedding.shape) > 1:
                        embedding = embedding.flatten()
                        
                    # Reshape for sklearn (1, -1)
                    embedding = embedding.reshape(1, -1)
                    # Classify with SVM if available
                    if svm_classifier is not None:
                        pred_prob = svm_classifier.predict_proba(embedding)
                        max_prob = np.max(pred_prob)
                        pred = svm_classifier.predict(embedding)
                        ident = str(pred[0])
                        
                        # Use SLM to generate a log if confidence is high
                        if max_prob > 0.6:
                            log_msg = slm_agent.generate_log(ident, int(max_prob*100))
                            identity = f"{ident} ({int(max_prob*100)}%) - {log_msg}"
                            # Also append to audit log automatically
                            append_audit(ident, 'auto_attendance', {'confidence': float(max_prob), 'slm_log': log_msg})
                        else:
                            identity = f"Unknown ({int(max_prob*100)}%)"
                    # Fallback to CNN if SVM not loaded
                    elif cnn_model is not None:
                        pred = cnn_model.predict(embedding)
                        identity = str(np.argmax(pred, axis=1)[0])
                detections['faces'].append({
                    'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'identity': identity
                })
            except Exception as e:
                logger.warning(f"Recognition error for a face: {e}")
                detections['faces'].append({
                    'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'identity': None
                })
    else:
        logger.warning("OpenCV not available or face cascade missing for recognition.")
    return jsonify({'status': 'success', 'detections': detections})

@app.route('/api/train', methods=['POST'])
def train_svm_model():
    """Trigger the supervised training pipeline"""
    def run_training():
        result = train_model.train()
        if result:
            # Reload the model in the main process
            global svm_classifier
            svm_classifier = load_svm_classifier()
            logger.info("SVM model reloaded after training")
            
    # Run in background to avoid blocking
    threading.Thread(target=run_training).start()
    return jsonify({'status': 'ok', 'message': 'Training started in background'})


    
def print_startup_banner():
    print("\n" + "="*60)
    print("    COLLEGE ADMINISTRATIVE SYSTEM (CAS) - ACTIVE")
    print("="*60)
    print(f"   Root:     {os.path.basename(BASE_DIR)}")
    print("   Status:   Video Processing & AI Ready")
    print("-" * 60)
    print("   MAIN APP:   http://localhost:5000/")
    print("   LOGIN:      http://localhost:5000/account")
    print("   ADMIN:      http://localhost:5000/admin")
    print("   STUDENT:    http://localhost:5000/student")
    print("="*60 + "\n")

if __name__ == '__main__':
    # Initialize DB with Graceful Failure
    try:
        init_db()
    except Exception as e:
        logger.error(f"DATABASE ERROR: {e}")
        logger.warning("App starting WITHOUT database connection. Login/Attendance saving will fail.")

    # Initialize models and data ONLY once
    _ensure_embeddings_loaded()
    init_mediapipe()
    init_detection_models()
    init_yolo() 
    
    # Pre-warm DeepFace to avoid lag on first request
    if DEEPFACE_AVAILABLE:
        try:
             import numpy as np
             empty_img = np.zeros((112, 112, 3), dtype=np.uint8)
             DeepFace.represent(empty_img, model_name='VGG-Face', enforce_detection=False, detector_backend='skip')
             logger.info("DeepFace AI Engine pre-warmed")
        except: pass
    print_startup_banner()
    
    # Auto-launch browser (Localhost only)
    # Check WERKZEUG_RUN_MAIN to avoid opening twice during reload
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        import webbrowser
        from threading import Timer
        def open_browser():
            if not os.environ.get("WERKZEUG_RUN_MAIN"):
                webbrowser.open_new('http://localhost:5000/')
        Timer(1.5, open_browser).start()

    # Run server - Bind to 0.0.0.0 for Cloud/Network access
    # use_reloader=False is CRITICAL to prevent launching two copies of the AI engine (TensorFlow/Mediapipe)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)