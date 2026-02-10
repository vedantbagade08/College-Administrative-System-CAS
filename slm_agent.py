import logging
import random
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Fallback templates if HF model fails or is too slow
TEMPLATES = [
    "Marked {name} present at {time}. Confidence: {conf}%",
    "Student {name} detected in class at {time}.",
    "Attendance recorded for {name}. (Match: {conf}%)",
    "Verified identity: {name} at {time}."
]

class SLMAgent:
    def __init__(self):
        self.model_name = "distilgpt2" # Lightweight model
        self.generator = None
        self.initialized = False

    def load_model(self):
        """Load the HF model. Call this in a background thread to avoid blocking."""
        try:
            # Set environment variable to use tf-keras (backwards compatible)
            os.environ['TF_KERAS'] = '1'
            
            from transformers import pipeline
            logger.info(f"Loading SLM model: {self.model_name}...")
            # Use a text-generation pipeline. 
            # Note: This requires 'transformers' and 'torch' or 'tensorflow' installed.
            self.generator = pipeline('text-generation', model=self.model_name, device=-1)
            self.initialized = True
            logger.info("SLM model loaded successfully.")
        except ImportError as e:
            if 'tf-keras' in str(e).lower() or 'keras' in str(e).lower():
                logger.warning(f"Keras compatibility issue detected. Using fallback templates. Error: {e}")
            else:
                logger.warning(f"Failed to load SLM model: {e}. Using fallback templates.")
            self.initialized = False
        except Exception as e:
            logger.warning(f"Failed to load SLM model: {e}. Using fallback templates.")
            self.initialized = False

    def generate_log(self, name, confidence):
        """Generate a log entry for the student."""
        time_str = datetime.now().strftime("%I:%M %p")
        
        if self.initialized and self.generator:
            try:
                # Prompt engineering for the model
                prompt = f"Log entry: Student {name} arrived at {time_str} with {confidence}% match. Action:"
                
                # Generate text
                result = self.generator(prompt, max_length=50, num_return_sequences=1)
                generated_text = result[0]['generated_text']
                
                # Clean up the output (basic post-processing)
                # We want just the relevant part if the model rambles
                summary = generated_text.replace(prompt, "").strip()
                if not summary:
                    summary = f"Confirmed presence."
                
                return f"{prompt} {summary}"
            except Exception as e:
                logger.error(f"Error generating SLM log: {e}")
        
        # Fallback
        template = random.choice(TEMPLATES)
        return template.format(name=name, time=time_str, conf=int(confidence*100))

# Singleton instance
slm_agent = SLMAgent()
