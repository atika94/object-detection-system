# Real-Time Vehicle & Traffic Light Detection System

A computer vision system for detecting vehicles and traffic lights in images and real-time webcam streams using a pretrained YOLO object detection model.

The project will initially use a pretrained model for basic object detection. Custom fine-tuning and additional features may be added later.

## Detection Classes

The initial system will detect:

* Car
* Bus
* Truck
* Motorcycle
* Bicycle
* Traffic Light

## Features

### Current Scope

* Image-based object detection
* Real-time webcam detection
* Bounding boxes around detected objects
* Object class names
* Confidence scores

### Planned Enhancements

* Custom model fine-tuning
* Video file detection
* Object tracking
* Performance/FPS optimization
* Streamlit web interface

## Technologies

* Python
* Ultralytics YOLO
* OpenCV
* NumPy
* Matplotlib
* Streamlit
* Git & GitHub

## Project Workflow

```text
Input Image / Webcam
        ↓
Pretrained YOLO Model
        ↓
Object Detection
        ↓
Bounding Boxes
        ↓
Class Names + Confidence Scores
        ↓
Detection Results
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Instructions for image detection, webcam detection, and the Streamlit application will be added as the project is developed.

## Model

The project initially uses a pretrained YOLO model. The model weights will be downloaded automatically when required.

Custom fine-tuning will be considered after the basic detection system is working.


## Future Improvements

* Custom dataset and fine-tuning
* Improved detection accuracy
* Video detection
* Object tracking
* Real-time FPS monitoring
* Streamlit deployment