import tensorflow as tf


class MLModel:
    def __init__(self):
        super().__init__()
    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        return self.model