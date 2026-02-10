import os
import cv2
import numpy as np
import joblib
from deepface import DeepFace
import tensorflow as tf # Keep for now if needed, but DeepFace handles backend
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import logging
import mediapipe as mp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FACENET_MODEL_PATH = os.path.join(MODEL_DIR, 'mobilefacenet.tflite')

def init_detection():
    global mp_face_detection, face_detection, face_cascade
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        logger.info("MediaPipe loaded")
    except Exception as e:
        logger.warning(f"MediaPipe not available ({e}), falling back to Haar Cascade")
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
    return True

def load_facenet():
    if os.path.exists(FACENET_MODEL_PATH):
        try:
            interpreter = tf.lite.Interpreter(model_path=FACENET_MODEL_PATH)
            interpreter.allocate_tensors()
            logger.info("Loaded MobileFaceNet (TFLite)")
            return interpreter
        except Exception as e:
            logger.error(f"Error loading FaceNet TFLite: {e}")
    
    # Fallback to MobileNetV2
    logger.info("MobileFaceNet not found. Loading MobileNetV2 as fallback feature extractor...")
    try:
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(112, 112, 3))
        logger.info("Loaded MobileNetV2 (Keras)")
        return base_model
    except Exception as e:
        logger.error(f"Error loading MobileNetV2: {e}")
        return None

def get_embedding(model, face_img):
    try:
        # Preprocess: resize to 112x112
        face_resized = cv2.resize(face_img, (112, 112))
        
        if isinstance(model, tf.lite.Interpreter):
            # TFLite path
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(face_rgb.astype(np.float32) / 255.0, axis=0)
            
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            model.set_tensor(input_details[0]['index'], input_data)
            model.invoke()
            embedding = model.get_tensor(output_details[0]['index'])
            return embedding.flatten()
        else:
            # Keras Model path (MobileNetV2)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(face_rgb.astype(np.float32), axis=0)
            input_data = tf.keras.applications.mobilenet_v2.preprocess_input(input_data)
            
            embedding = model.predict(input_data, verbose=0)
            return embedding.flatten()
            
    except Exception as e:
        logger.warning(f"Failed to get embedding: {e}")
        return None

def detect_face_robust(image):
    """Detect single face using MediaPipe or Haar Cascade"""
    h, w, c = image.shape
    
    # 1. MediaPipe
    if face_detection:
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)
            
            if results.detections:
                best_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                bboxC = best_detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                dw = int(bboxC.width * w)
                dh = int(bboxC.height * h)
                
                # Padding
                pad_x = int(dw * 0.1)
                pad_y = int(dh * 0.1)
                
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                dw = min(w - x, dw + 2*pad_x)
                dh = min(h - y, dh + 2*pad_y)
                
                return image[y:y+dh, x:x+dw]
        except Exception as e:
            logger.warning(f"MP Detection error: {e}")

    # 2. Haar Fallback
    if face_cascade:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            (x,y,dw,dh) = faces[0] # Take first
            return image[y:y+dh, x:x+dw]

    # 3. Fallback: Center crop? Or None.
    # If no face detected, returning None skips this bad image
    return None

def build_cnn_classifier(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    logger.info("Starting training pipeline...")
    
    # Lazy Init of detection models only when training starts
    init_detection()
    
    if not os.path.exists(DATASET_DIR):
        logger.error(f"Dataset directory not found: {DATASET_DIR}")
        return False
        
    interpreter = load_facenet()
    if not interpreter:
        logger.error("Cannot train without FaceNet model")
        return False
        
    X = []
    labels = []
    
    # Walk through dataset - Robust logic for mixed structures
    # Structure 1: Student/01/img.jpg -> Label: 01
    # Structure 2: Faculty/Prof_Name.jpg -> Label: Prof_Name
    
    for category in os.listdir(DATASET_DIR): # e.g. Student, faculty
        category_dir = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(category_dir):
            continue
            
        logger.info(f"Scanning category: {category}")
        
        # Walk potentially nested directories
        for root, dirs, files in os.walk(category_dir):
            for img_name in files:
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(root, img_name)
                
                # Determine Label
                rel_path = os.path.relpath(root, category_dir)
                if rel_path == '.':
                    # Flat file in category (e.g. Faculty/Prof.jpg) -> Label = Filename
                    label = os.path.splitext(img_name)[0]
                else:
                    # Nested folder (e.g. Student/01) -> Label = Folder Name
                    # Use the first part of the relative path as ID
                    label = rel_path.split(os.sep)[0]
                
                try:
                    # Detect face using robust method (DeepFace handles this, or uses MediaPipe if backend specified)
                    # We can pass img_path directly to DeepFace
                    
                    logger.info(f"Processing {img_name} with DeepFace...")
                    embedding_objs = DeepFace.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=False)
                    
                    if embedding_objs and len(embedding_objs) > 0:
                        emb = embedding_objs[0]["embedding"]
                        X.append(emb)
                        labels.append(label)
                    else:
                         logger.warning(f"No face detected in {img_name}")
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_name}: {e}")
                    pass # Skip bad files
                    
    logger.info(f"Total extracted faces: {len(X)}")

    if len(X) < 2:
        logger.error("Insufficient training data (need at least 2 samples).")
        return False
        
    X = np.array(X)
    
    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    
    # Check class counts for stratification safety
    unique, counts = np.unique(y_encoded, return_counts=True)
    min_count = np.min(counts)
    
    logger.info(f"Class distribution: {dict(zip(le.classes_, counts))}")

    # Train/Test Split logic
    if min_count < 2:
        logger.warning("Some classes have only 1 sample. Disabling Stratified Split and Cross-Validation.")
        X_train, y_train = X, y_encoded
        X_test, y_test = X, y_encoded # Test on train (overfitting warning, but necessary for 1-shot)
        cv_folds = 2
    else:
        # Standard robust split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
        cv_folds = min(3, min_count)

    # 1. Train SVM with Optimization (GridSearch)
    logger.info("Tuning SVM hyperparameters...")
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'C': [0.1, 1, 10, 100], 
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    # Adjust CV based on data size
    if len(X_train) < 5 or min_count < 2:
        # Too small for CV -> just fit directly
        logger.info("Dataset too small for GridSearch. Training default SVM.")
        best_svm = SVC(kernel='linear', C=1.0, probability=True)
        best_svm.fit(X_train, y_train)
    else:
        grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=1, cv=cv_folds)
        grid.fit(X_train, y_train)
        best_svm = grid.best_estimator_
        logger.info(f"Best SVM Parameters: {grid.best_params_}")
    
    # Accuracy Report
    train_acc = best_svm.score(X_train, y_train)
    test_acc = best_svm.score(X_test, y_test)
    logger.info(f"Training Accuracy: {train_acc*100:.2f}%")
    logger.info(f"Test Accuracy:     {test_acc*100:.2f}%")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(best_svm, os.path.join(MODEL_DIR, 'svm_classifier.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    logger.info("Optimized SVM model saved.")
    
    # Save classes list
    np.save(os.path.join(MODEL_DIR, 'classes.npy'), le.classes_)
    
    return True

if __name__ == "__main__":
    train()
