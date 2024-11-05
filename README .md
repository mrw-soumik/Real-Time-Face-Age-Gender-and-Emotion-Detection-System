
# Real-Time Face, Age, Gender, and Emotion Detection System

This project is a real-time detection system that identifies faces, predicts gender, estimates age, and analyzes emotions in video feeds. It utilizes OpenCV for computer vision, pre-trained models for age and gender prediction, and a deep learning model for emotion detection.

## Project Overview

The project provides valuable insights into demographics and emotional responses, which could be useful in fields like retail, market analysis, and user experience research. It uses:
- **OpenCV's DNN module** for face detection.
- **Pre-trained Caffe models** for age and gender classification.
- **Keras** to load a pre-trained emotion detection model.

### Main Components

- **`age_gender_detection.py`**: Loads pre-trained models for age and gender classification, processes the input face, and predicts age and gender.
- **`detect_faces.py`**: Uses a Haar cascade to detect faces in real-time from a video feed.
- **`emotion_detection.py`**: Loads a deep learning model for emotion detection and predicts emotions for detected faces.
- **`real_time_detector.py`**: Integrates face, age, gender, and emotion detection to process each frame of the video feed and display results in real-time.
- **`utils.py`**: Contains utility functions for loading models and drawing labels on the video frames.

## Setup and Installation

Follow these steps to set up the project and run it on your machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/face_gender_age_emotion_detection.git
cd face_gender_age_emotion_detection
```

### Step 2: Set Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies
Install the necessary Python libraries from `requirements.txt`.

```bash
pip install -r requirements.txt
```

> **Note:** Ensure that you have OpenCV with the `opencv-contrib-python` package to support DNN modules and Caffe model loading.

### Step 4: Download Large Model Files
Due to GitHub’s file size limits, the following large model files are not included in the repository. Please download them from the links below and place them in the `data/` directory:

- `age_net.caffemodel` - Age detection model ([Download here](https://drive.google.com/drive/u/0/folders/1CibG5LWevqiM0oQ0pe2sznvyqjDUj8Da))
- `gender_net.caffemodel` - Gender detection model ([Download here](https://drive.google.com/drive/u/0/folders/1CibG5LWevqiM0oQ0pe2sznvyqjDUj8Da))
- `emotion_model.hdf5` - Emotion detection model

### Step 5: Run the Project
Use the following command to start the real-time face, age, gender, and emotion detection:

```bash
python main.py
```

This will open a video feed from your webcam, detect faces, and display the predicted gender, age range, and emotion for each detected face. Press `q` to exit the application.

## How It Works

- **Face Detection**: The program captures each frame from the video feed and uses a Haar cascade to detect faces.
- **Gender and Age Prediction**: For each detected face, the program uses a pre-trained Caffe model to classify gender and estimate age.
- **Emotion Detection**: The program uses a Keras model to classify the emotion of the detected face.
- **Display**: The results are displayed in real-time on the video feed, with bounding boxes around faces and labels for gender, age, and emotion.

## Contributing

If you'd like to contribute to this project, please fork the repository and make a pull request.

---

Enjoy building with this real-time detection system!
