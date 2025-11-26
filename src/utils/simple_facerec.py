import cv2
import os
import logging
# import face_recognition
from ultralytics import YOLO

# from src.core.logging import setup_logging
# from utils import crop_face

# setup_logging()

yolo_model = YOLO("model/yolo/yolov8n-face.pt")

# Detect face object haarcascade
detect_face = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

# Open the default camera
cap = cv2.VideoCapture(0)

class SimpleFaceRec:
    def __init__(self):
        pass

    def face_from_facerec(self, frame):
        # face_locations = face_recognition.face_locations(frame)
        # return face_locations
        pass

    def face_from_yolo(self, frame):
        results = yolo_model.predict(source=frame)
        boxes = results[0].boxes
        return boxes

    def face_from_haarcascade(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_face.detectMultiScale(gray, 1.3, 3)
        if len(faces) == 0:
            logging.info("No faces detected in this frame.")
            return ()
        else:
            return faces

# sfr = SimpleFaceRec()

# while True:
#     ret, frame = cap.read()

#     face_locations = sfr.face_from_haarcascade(frame)

#     for face_loc in face_locations:
#         x, y, w, h = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2.imshow("Frame", frame)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()