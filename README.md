# Facial emotion recognition application

Video stream facial recognition application using camera and opencv library.
The model was trained using CNN for feature extraction and Deep Learning for classification

## Table of Contents

- [Requirements](#requirements)
- [Project Instruction](#project-instruction)
- [Usage](#usage)
- [License](#license)

## Requirements

Before you start, make sure you have the following installed:

- [VS Code](https://code.visualstudio.com/download)
- [Python 3.10](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started)

## Installation

Instruction on how to install and set up the project.

## Project Instruction

├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interm         <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── guide              <- A set of markdown files with documented best practices, guidelines and rools for collaborative projects
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
└── da-project         <- Source code for use in this project.
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
