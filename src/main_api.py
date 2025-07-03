import cv2
import time
from PIL import Image

from ml_classifier import EmotionDetection
from utils import image64_encode
from utils import send_image_api
from utils import get_predictions

# Open the default camera

cam = cv2.VideoCapture(0)

# Detect face object haarcascade
detect_face = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Define rectangle positions
x0, y0 = 200, 300
x1, y1 = 300, 400

# Text formatting
font = cv2.FONT_HERSHEY_SIMPLEX

# Check if Camera was found
if not cam.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # get camera read boolean
    # and each camera frame
    ret, frame = cam.read()

    # detect face from frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

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
        
        # Write the frame to the output file
        out.write(frame)
        
        # Display the captured frame
        cv2.imshow('Camera', frame)
    
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        # return rectangle from face
        ret = cv2.rectangle(frame, 
                            (get_data["image-props"]["faces_positions"][0][0], get_data["image-props"]["faces_positions"][0][1]), 
                            (get_data["image-props"]["faces_positions"][0][0] + get_data["image-props"]["faces_positions"][0][2], get_data["image-props"]["faces_positions"][0][0] + get_data["image-props"]["faces_positions"][0][3]), 
                            color=(0, 255, 0), 
                            thickness=2
                            )
        
        cv2.putText(frame, f'{get_data["prediction"][0]}', (get_data["image-props"]["faces_positions"][0][0], get_data["image-props"]["faces_positions"][0][1]), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f'{get_data["accuracy"][0]}', (get_data["image-props"]["faces_positions"][0][0], get_data["image-props"]["faces_positions"][0][1] + 30), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Write the frame to the output file
        out.write(frame)
            
        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()