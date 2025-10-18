import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch import nn, optim
from PIL import Image

from src.ai.torch.nn_eval import EmotionDetection
from torchvision import transforms

model_path = "../model/"

# Machine Learning Model class
CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad"]
ITERATIONS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emd = EmotionDetection(class_names=CLASS_NAMES)
model = emd.load(model_path)

data_transform = transforms.Compose(
    [
        # It is necessary to resize the image to match the network's input size.
        transforms.Resize(230),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Mean and standard deviation of ImageNet
        # These are required as we will use a model pre-trained on ImageNet.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def model_benchmark():
    pil_image = Image.open("image/test.jpg")
    transformed_image = data_transform(pil_image)
    input_tensor = transformed_image.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)
    for _ in range(ITERATIONS):
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
    result = (start_time - end_time) / ITERATIONS
    print("RESULT: ", result)

model_benchmark()