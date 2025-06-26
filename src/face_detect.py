import cv2
import time

from ml_classifier import EmotionDetection
from utils import crop_face
from utils import image_preprocessing
from utils import save_image
from utils import save_image_crop

# Open the default camera
try:
    image_array = cv2.imread('images/sample/me_and_kids.jpeg')    
except FileNotFoundError:
    print("File not found")
    
face, num_faces, pos_faces = save_image_crop(image_array)

print(face)
print(num_faces)
print(pos_faces)