import cv2
import base64
import requests
import json
import asyncio

import base64
from PIL import Image
from io import BytesIO

import time
from .utils import send_image_api
from .utils import get_predictions
from .utils import image64_encode

font = cv2.FONT_HERSHEY_SIMPLEX

def remote_conn(frame):
    pil_image = Image.fromarray(frame)
        
    data = image64_encode(pil_image, "Samuel")
    print(data['image']['content'][0:20])
    
    start_post = time.time()
    post_response = send_image_api(data)
    end_time_post = time.time()

    if post_response.status_code == 201:
        print("Successfully created data")
    else:
        print(f"Error: {post_response.status_code} - {post_response.reason}")

    start_get = time.time()
    get_response = get_predictions()
    get_data = get_response.json()
    end_time_get = time.time()

    print(get_data)

    print(f"Elapsed Time post and data: {end_time_post - start_post}s")
    print(f"Elapsed Time request data: {end_time_get - start_get}s")
    
    if not get_data:
        cv2.putText(frame, 'No face(s) found', (50, 50), fontFace=font, fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    else:
        # return rectangle from face
        ret = cv2.rectangle(
            frame, 
            (get_data["image-props"]["faces_positions"][0][0], get_data["image-props"]["faces_positions"][0][1]), 
            (get_data["image-props"]["faces_positions"][0][0] + get_data["image-props"]["faces_positions"][0][2], get_data["image-props"]["faces_positions"][0][0] + get_data["image-props"]["faces_positions"][0][3]), 
            color=(0, 255, 0), 
            thickness=2
        )
        
        cv2.putText(frame, f'{get_data["prediction"][0]}', (get_data["image-props"]["faces_positions"][0][0], get_data["image-props"]["faces_positions"][0][1]), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f'{get_data["accuracy"][0]}', (get_data["image-props"]["faces_positions"][0][0], get_data["image-props"]["faces_positions"][0][1] + 30), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
    return frame
