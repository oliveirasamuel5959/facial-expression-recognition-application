# Computer Vision: Facial emotion recognition application

- The application will load the pre-trained model stored in a .h5 file in the project root directory and make predictions on video stream camera using opencv library
- The results will show on the screen and will be one of the 5 classes: (Happy, Sad, Surprise, Angry, Neutral).

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [License](#license)

## Requirements

Before you start, make sure you have the following installed:

- [VS Code](https://code.visualstudio.com/download)
- [Python 3.10](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started)

## Installation

Instruction on how to install and set up the project.

## Project Structure
------------
    ├── src                          <- Main entry of the project
    │   ├── main.py                  <- Main application code
        ├── utils.py                 <- Main application code
        ├── ml_classifier.py         <- Model Class
    ├── model
    │   ├── haarcascade         <- haarcascade xml config file for face detection
    │   ├── model-21-0.7172     <- model weights
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │

--------
