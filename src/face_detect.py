import cv2
import time

from src.ai.ml_classifier import EmotionDetection
from src.ai.utils import crop_face
from src.ai.utils import image_preprocessing
from src.ai.utils import save_image
from src.ai.utils import save_image_crop

# Open the default camera
try:
    image_array = cv2.imread('images/sample/happy.jpg')    
except FileNotFoundError:
    print("File not found")
    
face, num_faces, pos_faces = save_image_crop(image_array)

print(face)
print(num_faces)
print(pos_faces)