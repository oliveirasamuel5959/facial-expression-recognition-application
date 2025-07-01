import cv2
import requests
import json

import base64
from io import BytesIO

URL_POST = 'https://ai.emovio.com.br/api/v1/predictions'
URL_GET = 'https://ai.emovio.com.br/api/v1/predictions/Samuel'

# URL_POST_LOCAL = 'http://192.168.0.16/v1/predictions'
# URL_GET_LOCAL = 'http://192.168.0.16/v1/predictions/Samuel'

headers = {'content-type': 'application/json'}

import numpy as np
import tensorflow as tf
import logging


logging.basicConfig(filename='face.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
def save_image_crop(image_array):
    logging.info(f"Start face detection. Image shape is {image_array.shape}.")
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype('uint8')
    detect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    faces = detect.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    print("faces:", faces)
    i = 0
    if len(faces) == 0:
        logging.info(f"face(s) not found for image {gray_image.shape}.")
    else:
        logging.info(f"this is face return variable {faces}.")
        logging.info(f"this is face return variable shape {faces.shape}.")
        num_faces = faces.shape[0]
        for (x, y, w, h) in faces:
            pos = (int(x), int(y))
            dim = (int(w), int(h))
            
            cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 3)
            face_image = crop_face(image_array, pos=pos, dim=dim)
            cv2.imwrite(f"images/result/face_image_crop_{i}.png", face_image)
            i += 1

        cv2.imwrite("images/result/face_image.png", image_array)
        
        logging.info(f"Completed face detection. Face shape {faces} and found {num_faces} face(s).")    
        return face_image, num_faces, faces
    
    return 0, 0, 0

def image64_encode(base_image, name):
    try:
        buffered = BytesIO()
        base_image.save(buffered, format='JPEG')
        
        image_bytes = buffered.getvalue()
        base64_bytes = base64.b64encode(image_bytes)
        base64_encoded = base64_bytes.decode()
        
        data = {
            'image':
                {
                    'name': str(name),
                    'timestamp': 1215456,
                    'content': base64_encoded
                },
            'collection_name': 'Image base64 for data analysis'
        }

        print(data['image']['content'][0:20])
        return data
        
    except Exception as e:
        print("Error: ", e)

def send_image_api(data_json):
    response = requests.post(url=URL_POST, data=json.dumps(data_json), headers=headers)
    return response    
    
def get_predictions():
    response = requests.get(url=URL_GET, headers=headers)
    return response