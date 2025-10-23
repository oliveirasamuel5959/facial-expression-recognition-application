import logging
import os
import time

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

# from ai.keras.ml_classifier import EmotionDetection
from ai.keras.utils import crop_face, image_preprocessing, save_image
from utils.simple_facerec import SimpleFaceRec
from ai.torch.nn_eval import EmotionDetection
from core.logging import setup_logging
from db.database import EmotionDatabase

setup_logging()
load_dotenv()

db = EmotionDatabase()

print(os.getenv("CAM_IP"))

rtsp = f"rtsp://{os.getenv('CAM_USER')}:{os.getenv('CAM_PASSWORD')}@{os.getenv('CAM_IP')}:{os.getenv('CAM_PORT')}/cam/realmonitor?channel=1&subtype=1"

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("output_3.mp4", fourcc, 20.0, (frame_width, frame_height))

# Define rectangle positions
x0, y0 = 200, 300
x1, y1 = 300, 400

# Text formatting
font = cv2.FONT_HERSHEY_SIMPLEX

# model path
# model_path = "../model/model-26-0.7175.h5"
model_path = "../model"

# Machine Learning Model class
CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad"]

emd = EmotionDetection(class_names=CLASS_NAMES)
model = emd.load(model_path)
sfr = SimpleFaceRec()
yolo_model = YOLO("../model/yolo/yolov8n-face.pt")

# Check if Camera was found
if not cam.isOpened():
    print("Error: Could not open video source.")
    exit()

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    # get camera read boolean
    # and each camera frame
    ret, frame = cam.read()

    new_frame_time = time.time()

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(
        frame,
        f"FPS {fps}",
        (frame_width - 120, 30),
        fontFace=font,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    boxes = sfr.face_from_yolo(frame)
    # boxes = sfr.face_from_haarcascade(frame)

    for box in boxes:

        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bottom_right_x = int(box.xyxy.tolist()[0][2])
        bottom_right_y = int(box.xyxy.tolist()[0][3])

        print("Faces: ", box.xyxy.tolist())

        face_image_crop = crop_face(frame=frame, x=(top_left_x, top_left_y), y=(bottom_right_x, bottom_right_y))
        # image_array = image_preprocessing(face_image_crop)

        start_time = time.time()
        # class_name, confidence = emd.make_predictions(image_array, model=model)
        class_name, confidence = emd.make_predictions(face_image_crop)
        end_time = time.time()

        if confidence > 0.75:
            db.add_emotion(class_name, confidence)

        logging.info(f"Time taken for prediction: {round(end_time - start_time, 2)}s")

        # return rectangle from face
        ret = cv2.rectangle(
            frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color=(0, 255, 0), thickness=2
        )

        cv2.putText(
            frame,
            f"{class_name}",
            (top_left_x, bottom_right_y + 20),
            fontFace=font,
            fontScale=0.65,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"{confidence}",
            (top_left_x, bottom_right_y + 40),
            fontFace=font,
            fontScale=0.65,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        save_image(image=face_image_crop)

        # Write the frame to the output file
        # out.write(frame)

    # Display the captured frame
    cv2.imshow("Camera", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord("q"):
        db.close()
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
