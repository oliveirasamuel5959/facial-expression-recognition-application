# **Computer Vision: Facial emotion recognition application**

## **Introduction**
This AI-powered application is designed to assist healthcare professionals in monitoring and understanding the emotional states of patients who experience communication challenges, such as intellectual disabilities, hearing impairments, or other conditions that hinder verbal expression.

The core of the application is a deep learning model developed using Python with TensorFlow and Keras. The model is trained to classify human facial expressions into five distinct emotional categories:
- Happy
- Sad
- Surprise
- Angry
- Neutral

The application connect to a camera using opencv library to capture live video streams, detects faces in real time, and performs emotion classification on the detected facial regions. This tool aims to support healthcare providers in delivering more empathetic and personalized care by offering additional insights into a patient's emotional well-being.

## **Table of Contents**

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## **Requirements**

Before you start, make sure you have the following installed:

- [VS Code](https://code.visualstudio.com/download)(Or use your favorite code editor)
- [Python 3.10](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started)

## **Installation**

Instruction on how to install and set up the project:

1. Clone the repository: **`git clone`**
2. Navigate to the project directory: **`cd fer_expression`**
3. Create a virtual envirornment: **`Python -m venv .venv`**
4. Activate the virtual envirornment: **`Source .venv/bin/activate`**
5. Install the requirements packages: **`pip install -r requirements.txt`**

## **Usage**

To use Project, follow these steps:

1. Open the project in your favorite code editor.
2. Modify the source code to fit your needs.
3. Run the project: **`Python src/main.py`**
5. Use the project as desired.
--------
