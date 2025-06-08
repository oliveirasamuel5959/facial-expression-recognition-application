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
        '''
        Load keras model weights in .h5 format
        '''
        self.model = tf.keras.models.load_model(path)
        logging.info('Model Load successfuly!')
        
        return self.model
    
    def make_predictions(self, image_array, model):
        '''
        model load and stored in model variable
        image must be in the format: (1, 224, 224, 3)
        class names must be in the same order that was trained
        
        return class name prediction and accuracy
        '''
        predictions = model.predict(image_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])
        confidence = round(confidence, 2) * 100
        
        return [predicted_class, confidence]
    