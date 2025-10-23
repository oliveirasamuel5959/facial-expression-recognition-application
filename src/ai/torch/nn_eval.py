import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch import nn, optim
from PIL import Image

from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets, utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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

class EmotionDetection:

    def __init__(self, class_names):
        self.ARCH_NAME = "resnet18"
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(pretrained=True).to(self.device)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 5).to(self.device)

    def load(self, path):
        """
        Load keras model weights in .h5 format
        """
        self.model.load_state_dict(
            torch.load(
                os.path.join(path, f"fer-classification-{self.ARCH_NAME}-model.pth"),
                map_location="cpu",
            )
        )

        self.model.to(self.device)
        logging.info("Model Load successfuly!")

        return self.model

    def make_predictions(self, image_array):
        """
        model load and stored in model variable
        image must be in the format: (1, 224, 224, 3)
        class names must be in the same order that was trained

        return class name prediction and accuracy
        """
        logging.info("Start Prediction...")
        pil_image = Image.fromarray(image_array)
        transformed_image = data_transform(pil_image)
        input_tensor = transformed_image.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)  # converte logits em probabilidades

            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = probabilities[0, predicted_class].item()
            # predicted_index = torch.argmax(output, dim=1).item()
            # # convert Torch.tensor type to a number Integer
            # confidence = round(output[0][predicted_index].item(), 2)
            logging.info(f"Output tensor: {output} Confidence: {predicted_prob}")
        logging.info("Return Prediction!")
        return [self.class_names[predicted_class], round(predicted_prob, 2)]
