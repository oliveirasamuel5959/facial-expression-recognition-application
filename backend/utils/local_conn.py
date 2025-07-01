import cv2
import time
import requests

from utils.ml_classifier import EmotionDetection
from utils.utils import crop_face
from utils.utils import image_preprocessing

# Detect face object haarcascade
detect_face = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')

# Text formatting
font = cv2.FONT_HERSHEY_SIMPLEX

# model path
model_path = '../model/model-26-0.7175.h5'

# Machine Learning Model class
CLASS_NAMES = ['angry', 'fear', 'happy', 'neutral', 'sad'] 

emd = EmotionDetection(class_names=CLASS_NAMES)
model = emd.load(model_path)

# Check if Camera was found
def local_conn(frame, cam):
    # Start the time counter
    prev_frame_time = time.time() 
    new_frame_time = 0
    
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Put the FPS text on the top-right corner of the frame
    print(f"frame shape: {frame.shape[:2]}      {frame_width}, {frame_height}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('media/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    
    while True:
        # detect face from frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face from frame
        face = detect_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))
        
        # Calculate the FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Convert FPS to an integer for display
        fps = int(fps)
        fps_text = f'FPS: {fps}'
        
        # iterate over position and dimensions of the rectangle 
        # from cascade classifier
        for (x, y, w, h) in face:
            # get frame position and dimensions
            pos = [x, y]
            dim = [w, h]
            
            # define text position coordinates
            text_pos_x = x
            text_pos_y = y + h + 20
            
            face_image_crop = crop_face(frame=frame, pos=pos, dim=dim)
            image_array = image_preprocessing(face_image_crop)

            start_time = time.time()
            class_name, confidence = emd.make_predictions(image_array, model)
            end_time = time.time()
            print(f"Time taken for prediction: {round(end_time - start_time, 2)}s")
            
            # return rectangle from face
            ret = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'{class_name}', (text_pos_x, text_pos_y), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f'{confidence}%', (text_pos_x, text_pos_y + 30), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)            

        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Write the frame to the output file
        out.write(frame)
        
        return frame