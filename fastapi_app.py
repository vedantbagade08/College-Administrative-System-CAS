import os
import cv2
import numpy as np
import base64
import json
import time
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymysql
from datetime import datetime

# ML specific imports
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI-AI")

app = FastAPI(title="CAS AI Service")

# CORS Configuration for Domain/Cloud Hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration (Mirrored from app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'cas_db',
    'charset': 'utf8mb4',
    'autocommit': True
}

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

STUDENT_EMBEDDINGS = {}

# MediaPipe Initialization
face_detector = None
if MP_AVAILABLE:
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Haar Cascade Fallback
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_student_embeddings():
    global STUDENT_EMBEDDINGS
    student_base_dir = os.path.join(BASE_DIR, 'datasets', 'Student')
    if not os.path.exists(student_base_dir):
        logger.warning("Student datasets not found")
        return

    logger.info("Loading student reference images...")
    for student_id in os.listdir(student_base_dir):
        if student_id not in FACE_ID_MAP: continue
        student_dir = os.path.join(student_base_dir, student_id)
        if not os.path.isdir(student_dir): continue
        
        encodings = []
        for img_file in os.listdir(student_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = face_recognition.load_image_file(os.path.join(student_dir, img_file))
                    face_encodings = face_recognition.face_encodings(img)
                    if face_encodings:
                        encodings.append(face_encodings[0])
                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")
        if encodings:
            STUDENT_EMBEDDINGS[student_id] = encodings

@app.on_event("startup")
async def startup_event():
    load_student_embeddings()

class ImageRequest(BaseModel):
    image: str  # Base64 string
    camera_id: Optional[str] = "0"
    source: Optional[str] = "browser" # browser, cctv, app

@app.get("/")
async def root():
    return {"status": "AI Service Running", "type": "FastAPI", "capabilities": ["MP_Detection", "Face_Recognition"]}

@app.post("/api/recognize")
async def recognize(request: ImageRequest):
    try:
        # 1. Decode Image
        image_base64 = request.image
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 2. Detect Faces
        detections = []
        faces_rects = []
        
        if MP_AVAILABLE and face_detector:
            results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.detections:
                h_img, w_img, _ = img.shape
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    faces_rects.append((
                        int(bboxC.xmin * w_img),
                        int(bboxC.ymin * h_img),
                        int(bboxC.width * w_img),
                        int(bboxC.height * h_img)
                    ))
        
        if not faces_rects: # Fallback to Haar
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_rects = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))

        # 3. Recognize Each Face
        for (x, y, w, h) in faces_rects:
            face_img = img[y:y+h, x:x+w]
            identity, name, confidence = "Unknown", "Unknown", 0
            
            if FACE_RECOGNITION_AVAILABLE:
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_img)
                if face_encodings:
                    target_encoding = face_encodings[0]
                    min_dist = 1.0
                    for sid, ref_encs in STUDENT_EMBEDDINGS.items():
                        dists = face_recognition.face_distance(ref_encs, target_encoding)
                        dist = min(dists)
                        if dist < min_dist:
                            min_dist = dist
                            identity = sid
                    
                    if min_dist < 0.45: # Threshold
                        name = FACE_ID_MAP.get(identity, f"Student {identity}")
                        confidence = (1 - min_dist) * 100
                        # Log to DB
                        log_detection_to_db(identity, request.camera_id, request.source)
                    else:
                        identity = "Unknown"

            detections.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "identity": identity,
                "name": name,
                "confidence": round(confidence, 2)
            })

        return {
            "ok": True,
            "detections": detections,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return {"ok": False, "error": str(e)}

def log_detection_to_db(student_id, camera_id, source):
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (actor, action, details) VALUES (%s, %s, %s)",
                ("FastAPI-AI", "auto_attendance", json.dumps({
                    "student_id": student_id, 
                    "camera": camera_id, 
                    "source": source,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }))
            )
        conn.close()
    except Exception as e:
        logger.error(f"DB Log error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
