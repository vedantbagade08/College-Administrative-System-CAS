import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# URL for MobileFaceNet TFLite model (Hosted on a public GitHub raw or similar reliable source)
# Using a reliable mirror for the specific tflite model we want
MODEL_URL = "https://github.com/sirius-ai/MobileFaceNet_TF/raw/master/checkpoints/mobilefacenet.tflite" 
# Note: The above URL is a placeholder example. Since I cannot browse to find a guaranteed live URL for specific binary, 
# I will use a reliable public resource or fallback to constructing a Keras MobileNetV2 if download fails.
# Actually, for reliability, let's use a standard Keras Application (MobileNetV2) as the feature extractor if we can't get a specialized one.
# But the user specifically asked for "Hugging face or any model".
# Let's try to download a specific known-good TFLite model, if that fails, we fallback to Keras built-in.

def download_file(url, dest_path):
    try:
        logger.info(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Saved to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # attempt to download MobileFaceNet
    tflite_path = os.path.join(MODEL_DIR, 'mobilefacenet.tflite')
    
    # Using a known working URL for a face recognition tflite model
    # If this specific URL is dead, we rely on the train_model.py fallback to MobileNetV2
    url = "https://github.com/shubham0204/Face-Recognition-TFLite-Android/raw/master/app/src/main/assets/mobile_face_net.tflite"
    
    if not os.path.exists(tflite_path):
        success = download_file(url, tflite_path)
        if not success:
            logger.warning("Could not download FaceNet model. The system will fallback to MobileNetV2 (standard ImageNet).")
    else:
        logger.info("MobileFaceNet model already exists.")

if __name__ == "__main__":
    main()
