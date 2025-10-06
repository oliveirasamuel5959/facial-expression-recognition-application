import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from starlette.responses import JSONResponse
from starlette.requests import Request
from pydantic import BaseModel
import os
import cv2
import time
from dotenv import load_dotenv

load_dotenv(override=True)

from utils.remote_conn import remote_conn
from utils.local_conn import local_conn
from utils.ml_classifier import EmotionDetection

from utils.utils import crop_face
from utils.utils import image_preprocessing

font = cv2.FONT_HERSHEY_SIMPLEX

app = FastAPI()

# CORS (Cross-Origin Resource Sharing) middleware configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "https://example.com",
    "https://www.example.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class EmotionResult(BaseModel):
    x: int
    y: int
    w: int
    h: int
    emotion: str
    
remote = False

video_path = 'media/people_street_2.mp4'

print(os.getenv('CAM_IP'))

rtsp = f"rtsp://{os.getenv('CAM_USER')}:{os.getenv('CAM_PASSWORD')}@{os.getenv('CAM_IP')}:{os.getenv('CAM_PORT')}/cam/realmonitor?channel=1&subtype=1"

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

# Start the time counter
prev_frame_time = time.time() 
new_frame_time = 0

vid_fmt = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

def gen_frames():
    cam = cv2.VideoCapture(rtsp)
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    cam.set(cv2.CAP_PROP_FOURCC, vid_fmt)
    
    
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter('media/output_4.mp4', fourcc, 20.0, (frame_width, frame_height))
    
    # Start the time counter
    prev_frame_time = time.time() 
    new_frame_time = 0
    
    while True:
        success, frame = cam.read()
        
        if not success:
            break
        
        try:
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
            
            
            print(f"Frame: {frame.shape[:2]}")
            
            # Write the frame to the output file
            out.write(frame)
            
        except Exception as e:
            print(f"Prediction error: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/users")
async def get_users(req: Request):
    try:
        return JSONResponse({'username': os.getenv('ADMIN_USER')})   
    except Exception as e:
        return JSONResponse({"message": "Failed"})


@app.post("/login")
async def login(req: Request):
    if req.method == 'POST':
        admin_user = os.getenv('ADMIN_USER')
        admin_password = os.getenv('ADMIN_PASSWORD')
        data = await req.json()
        if admin_user == data['name'] and admin_password == data['password']:
            return JSONResponse({'message': 'correct credentials', 'success': True})
        else:
            return JSONResponse({'message': 'login or password incorrect', 'success': False})
    else:
        return JSONResponse({"message": "Failed"})

# @app.get("/")
# async def index():
#     return HTMLResponse("""
#     <html>
#         <head>
#             <title>Emotion Stream</title>
#         </head>
#         <body>
#             <h1>Live Camera Emotion Detection</h1>
#             <img src="/video_feed" width="800" />
#         </body>
#     </html>
#     """)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)