import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLASS_NAMES = ['angry', 'fear', 'happy', 'neutral', 'sad']  # Update based on your use case

class MLModel:
    def __init__(self):
        super().__init__()
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        logging.info('Model Load successfuly!')
        return self.model
    
    def predictions(self, image_array, model):
        predictions = model.predict(image_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])
        confidence = round(confidence, 2) * 100
        return [predicted_class, confidence]
    