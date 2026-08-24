# 🔍 Object Detection System

A beginner-friendly object detection project built with **Python** and **Ultralytics YOLOv8**.  
The pipeline covers training on a custom dataset, running inference, evaluating model performance, and serving results through an interactive **Streamlit** web app.

---
## 📁 Project Structure
```
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
```

##  Setup
### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/object-detection-system.git
cd object-detection-system
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

##  Usage

### Train
```bash
python train.py
```
### Predict (inference)
```bash
python predict.py --source dataset/images/test/sample.jpg
```

### Evaluate
```bash
python evaluate.py
```

### Run the web app
```bash
streamlit run app.py
```
## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| [Ultralytics YOLO](https://docs.ultralytics.com/) | Object detection model |
| [OpenCV](https://opencv.org/) | Image & video processing |
| [NumPy](https://numpy.org/) | Array operations |
| [Pandas](https://pandas.pydata.org/) | Data handling |
| [Matplotlib](https://matplotlib.org/) | Plotting & visualisation |
| [PyYAML](https://pyyaml.org/) | YAML config parsing |
| [Streamlit](https://streamlit.io/) | Interactive web UI |

## 📌 Roadmap

- [x] Project structure initialised
- [ ] Prepare custom dataset
- [ ] Fine-tune YOLOv8 on custom data
- [ ] Evaluate model performance
- [ ] Build full Streamlit web app

## 📄 License

This project is for educational purposes.