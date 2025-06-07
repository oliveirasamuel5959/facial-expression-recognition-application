import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def crop_face(frame, pos, dim):
    '''
    pos = [x, y]
    dim = [w, h]
    '''
    faces = frame[pos[1]:pos[1] + dim[1], pos[0]: pos[0] + dim[0]]
    return faces

def image_preprocessing(face_image):
    try:
        image = cv2.resize(face_image, (224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize
        return image
    except Exception as e:
        raise ValueError("Invalid image data") from e
    
    
def save_image(image):
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    if resized.dtype != 'uint8':
        resized = (resized * 255).astype('uint8')  # Se for float, normaliza
    cv2.imwrite("resized_frame.png", resized)
