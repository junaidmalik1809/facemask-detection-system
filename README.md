# Face Mask Detection System

A real-time computer vision application that detects faces from a live webcam feed and classifies whether a person is wearing a face mask.

The project uses **Python, OpenCV, and TensorFlow/Keras** and is designed as a practical computer vision project with potential applications in **access control, workplace safety, surveillance, and health-compliance monitoring**.

## Demo

The application provides real-time face-mask classification through a webcam feed.

The system detects faces in the video stream and classifies each detected face as:

- **With Mask**
- **Without Mask**

The application was tested locally using both masked and unmasked faces. Personal test images are intentionally excluded from the public repository for privacy reasons.`r`n`r`n---

## Features

* Real-time webcam-based face detection
* Face localization using OpenCV Haar Cascade
* Face-mask classification using a trained TensorFlow/Keras model
* Separate training and testing datasets
* Modular Python source code
* Model training and evaluation workflow
* Real-time computer vision pipeline
* Easily extensible for additional detection classes or deployment environments

---

## How It Works

The system follows a simple computer vision pipeline:

```text
Webcam
   │
   ▼
Capture Video Frame
   │
   ▼
Face Detection
   │
   ▼
Extract Detected Face
   │
   ▼
Preprocess Face Image
   │
   ▼
Mask Classification Model
   │
   ▼
With Mask / Without Mask
   │
   ▼
Display Detection Result
```

### 1. Video Capture

OpenCV captures frames from the computer's webcam.

### 2. Face Detection

The application uses a Haar Cascade classifier to locate faces within each video frame.

### 3. Face Preprocessing

Detected face regions are extracted and prepared in the format expected by the mask-classification model.

### 4. Mask Classification

The processed face image is passed to a TensorFlow/Keras model trained to distinguish between:

* `with_mask`
* `without_mask`

### 5. Real-Time Output

The classification result is displayed on the webcam feed.

---

## Technology Stack

| Technology         | Purpose                                      |
| ------------------ | -------------------------------------------- |
| Python 3.11        | Application and machine-learning development |
| OpenCV             | Computer vision and webcam processing        |
| TensorFlow / Keras | Machine-learning model development           |
| NumPy              | Numerical and image-array processing         |
| scikit-learn       | Machine-learning utilities and evaluation    |
| Matplotlib         | Visualization                                |
| Pillow             | Image processing                             |
| imutils            | OpenCV utility functions                     |
| tqdm               | Progress indicators                          |
| Git / GitHub       | Version control and project hosting          |

---

## Project Structure

```text
facemask-detection-system/
│
├── data/
│   ├── train/
│   │   ├── with_mask/
│   │   └── without_mask/
│   └── test/
│       ├── with_mask/
│       └── without_mask/
│
├── models/
│   └── haarcascade_frontalface_default.xml
│
│
├── src/
│   ├── __init__.py
│   ├── face_detector.py
│   ├── mask_classifier.py
│   ├── pipeline.py
│   ├── train.py
│   └── utils.py
│
├── notebooks/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** The `data/` directory and trained `.keras` model files are intentionally excluded from Git because datasets and trained model artifacts can significantly increase repository size.

---

## Dataset

The project uses a two-class image dataset organized into training and testing directories:

```text
data/
├── train/
│   ├── with_mask/
│   └── without_mask/
│
└── test/
    ├── with_mask/
    └── without_mask/
```

Each class contains images representing the corresponding category.

The dataset is kept locally and is excluded from GitHub through `.gitignore`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/junaidmalik1809/facemask-detection-system.git
cd facemask-detection-system
```

### 2. Create a Python virtual environment

Python 3.11 is recommended for this project.

On Windows:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(venv)
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Dataset Setup

Place the training and testing images into the following structure:

```text
data/
├── train/
│   ├── with_mask/
│   └── without_mask/
│
└── test/
    ├── with_mask/
    └── without_mask/
```

The dataset is intentionally not included in this repository.

---

## Model Training

The project contains a dedicated training script:

```text
src/train.py
```

Run the training process from the project root:

```powershell
python src/train.py
```

The training workflow prepares the images, trains the classification model, evaluates its performance, and saves the resulting model locally.

Because trained model files can be large, they are excluded from GitHub using:

```gitignore
*.keras
```

---

## Running the Application

After the required dependencies, dataset, and trained model have been configured, start the real-time application with:

```powershell
python app.py
```

The application accesses the computer's webcam and processes the incoming video frames.

To stop the webcam application, use the key specified by the application, such as:

```text
q
```

---

## Example Output

The system can classify detected faces into:

```text
With Mask
```

or:

```text
Without Mask
```

The application was tested locally using both masked and unmasked faces. Personal test images are intentionally excluded from the public repository for privacy reasons.

---

## Security and Access-Control Relevance

Although this project is primarily a computer vision application, the underlying concepts are relevant to cybersecurity and physical security systems.

Potential applications include:

* Access-control compliance monitoring
* Workplace safety monitoring
* Restricted-area monitoring
* Automated visual compliance checks
* Security-camera analytics
* Computer-vision-based monitoring systems

For example, a security system could use a similar architecture to detect whether an individual satisfies a predefined visual requirement before allowing access to a controlled environment.

This project therefore provides practical experience with:

* Computer vision
* Machine learning
* Image preprocessing
* Real-time video processing
* Python application development
* Modular software architecture
* Security-oriented automation

---

## Limitations

This project is intended as an educational and portfolio project.

Real-world deployment would require additional considerations, including:

* Different lighting conditions
* Camera quality and positioning
* Face angles and partial occlusion
* False positives and false negatives
* Model performance across different environments
* Privacy and data-protection requirements
* Secure handling of captured images and video
* Performance optimization for production systems

The model's predictions should therefore not be treated as a perfect or standalone security decision.

---

## Future Improvements

Possible future improvements include:

* Improve model accuracy and generalization
* Add confidence scores to predictions
* Add logging of detection events
* Add configurable detection thresholds
* Improve performance for low-end hardware
* Add GPU acceleration where available
* Replace Haar Cascade with a modern face detector
* Add automated testing
* Add Docker support
* Build a web-based dashboard
* Store detection events in a database
* Add role-based access to monitoring features
* Deploy the system as a production-ready service

---

## Learning Objectives

This project was developed to gain practical experience with:

1. Python programming
2. OpenCV
3. Computer vision pipelines
4. Image preprocessing
5. TensorFlow/Keras
6. Machine-learning model training
7. Model evaluation
8. Real-time webcam applications
9. Git and GitHub
10. Security-oriented application development

---

## Author

**Muhammad Junaid**

Cybersecurity & IT Support

GitHub: [@junaidmalik1809](https://github.com/junaidmalik1809)

---

## License

This project is intended for educational and portfolio purposes.



