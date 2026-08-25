# 🔍 Object Detection System

A beginner-friendly object detection project built with **Python** and **Ultralytics YOLOv8**.  
The pipeline covers training on a custom dataset, running inference, evaluating model performance, and serving results through an interactive **Streamlit** web app.

## 📁 Project Structure

object-detection-system/
├── dataset/
│   ├── images/
│   │   ├── train/        # Training images
│   │   ├── val/          # Validation images
│   │   └── test/         # Test images
│   └── labels/
│       ├── train/        # YOLO-format training labels
│       ├── val/          # YOLO-format validation labels
│       └── test/         # YOLO-format test labels
├── models/               # Saved trained model weights
├── train.py              # Fine-tune YOLO on a custom dataset
├── predict.py            # Run inference on image / video / webcam
├── evaluate.py           # Evaluate model on the test set
├── app.py                # Streamlit web app
├── data.yaml             # Dataset configuration for YOLO
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md