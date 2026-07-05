# Real-Time Face, Age, Gender, and Emotion Detection System

A Python application that uses a webcam feed to detect faces in real time and estimate each person's age range, gender, and facial emotion, overlaying the results directly on the video.

## Overview

The system combines three separate computer-vision models in a single OpenCV video loop:

- **Face detection** with an OpenCV Haar cascade classifier.
- **Age and gender classification** with pre-trained Caffe models run through OpenCV's DNN module.
- **Emotion classification** with a Keras/TensorFlow model.

For every frame captured from the webcam, each detected face is cropped, passed through the age/gender and emotion models, and annotated with a bounding box and a text label showing the predicted gender, age bracket, and emotion.

## Project Structure

```
main.py                                 # Entry point
requirements.txt                        # Python dependencies
data/
  haarcascade_frontalface_default.xml   # Face detection cascade (placeholder in this repo, see note below)
  age_deploy.prototxt                   # Age model architecture (included)
  gender_deploy.prototxt                # Gender model architecture (included)
  emotion_model.hdf5                    # Emotion classification model (included)
  age_net.caffemodel                    # Age model weights (NOT included, must be downloaded)
  gender_net.caffemodel                 # Gender model weights (NOT included, must be downloaded)
src/
  detect_faces.py                       # FaceDetector: Haar cascade face detection
  age_gender_detection.py               # AgeGenderDetector: Caffe DNN inference for age/gender
  emotion_detection.py                  # EmotionDetector: Keras model inference for emotion
  real_time_detector.py                 # RealTimeDetector: main webcam capture/inference loop
  utils.py                              # Helpers for loading the cascade and drawing labels
```

## How It Works

1. `main.py` defines the paths to the model files and constructs a `RealTimeDetector`, then calls `run()`.
2. `RealTimeDetector.run()` (`src/real_time_detector.py`) opens the default webcam with `cv2.VideoCapture(0)` and reads frames in a loop.
3. `FaceDetector` (`src/detect_faces.py`) converts each frame to grayscale and runs `cv2.CascadeClassifier.detectMultiScale` using the Haar cascade to find face bounding boxes.
4. For each detected face region:
   - `AgeGenderDetector` (`src/age_gender_detection.py`) builds a 227x227 blob from the cropped face and runs it through the Caffe `gender_net` and `age_net` models via `cv2.dnn.readNetFromCaffe`. Gender is predicted as `Male`/`Female`, and age is predicted as one of eight fixed brackets: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`.
   - `EmotionDetector` (`src/emotion_detection.py`) resizes the face to 64x64, converts it to grayscale, normalizes pixel values, and runs it through a Keras model (`tensorflow.keras.models.load_model`) loaded from `emotion_model.hdf5`. The output is one of seven emotions: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.
5. `utils.draw_label` (`src/utils.py`) draws a bounding box and a text label (`"<Gender>, Age: <range>, Emotion: <emotion>"`) on the frame for each detected face.
6. The annotated frame is shown in an OpenCV window. The loop exits and releases the webcam when the `q` key is pressed.

All processing runs locally in the video loop; the code does not write frames to disk or send data over the network.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/mrw-soumik/Real-Time-Face-Age-Gender-and-Emotion-Detection-System.git
cd Real-Time-Face-Age-Gender-and-Emotion-Detection-System
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

A Python 3.8+ environment compatible with the TensorFlow version you install is recommended. The repository does not pin exact package versions in `requirements.txt`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `opencv-python`, `opencv-contrib-python` (required for the DNN module used to load the Caffe models), `tensorflow`, and `numpy`.

### 4. Obtain the required model files

The `data/` directory contains the network architecture files (`age_deploy.prototxt`, `gender_deploy.prototxt`) and the emotion model (`emotion_model.hdf5`), but the following files must be added manually because of GitHub file size limits or because the tracked copy is not a usable model file:

| File | Status in this repo | How to get it |
|---|---|---|
| `data/age_net.caffemodel` | Not included | [Download](https://drive.google.com/file/d/15VCN5Tng4MTzoYBeI17_BQwscFUXbnVB/view?usp=drive_link) and place in `data/` |
| `data/gender_net.caffemodel` | Not included | [Download](https://drive.google.com/file/d/18VtvvfSpILO6IlhqagMdSKepoizYCXTZ/view?usp=drive_link) and place in `data/` |
| `data/haarcascade_frontalface_default.xml` | Present but empty (placeholder, not a valid cascade file) | Replace with the official file from [opencv/opencv](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml), or copy it from your local OpenCV install (`cv2.data.haarcascades` path) |

The `data/age_deploy.prototxt`, `data/gender_deploy.prototxt`, and `data/emotion_model.hdf5` files are already tracked in the repository and require no additional download.

### 5. Run the application

```bash
python main.py
```

This opens your default webcam, detects faces in the live feed, and overlays the predicted gender, age range, and emotion for each face. Press `q` in the video window to exit.

## Privacy and Ethics

This application performs real-time facial analysis on live webcam video. Anyone building on or deploying it should consider:

- **Consent**: Obtain informed consent from anyone whose face will be captured and analyzed, especially outside a private, personal-use context.
- **Local processing**: Based on the code, frames are processed in memory and displayed in an OpenCV window; the application does not write images to disk or transmit them over a network. Anyone modifying the code to add storage, logging, or network transmission of faces should evaluate the applicable privacy obligations before doing so.
- **Binary gender output**: The gender model exposes only two labels, `Male` and `Female`, which does not represent the full range of gender identities. Predictions should be treated as a rough classification from a general-purpose model, not an authoritative label.
- **Demographic accuracy**: No accuracy metrics or evaluation results are included in this repository. Face, age, gender, and emotion classifiers are known to vary in accuracy across skin tone, lighting, age group, and camera quality, so outputs should not be relied on for decisions that materially affect individuals (e.g., hiring, law enforcement, access control, or credit).
- **Appropriate use**: This project is intended for learning and experimentation with computer vision pipelines, not as a production identification, surveillance, or decision-making system.

## Limitations

- Face detection uses a Haar cascade classifier, which is less robust than modern deep-learning face detectors, particularly for non-frontal poses, occlusion, or difficult lighting.
- Age is predicted as one of eight coarse, fixed ranges rather than an exact value, and the underlying Caffe models are not evaluated or benchmarked in this repository.
- Emotion classification is limited to seven basic categories and may not capture subtle, mixed, or culturally variable expressions.
- The two Caffe `.caffemodel` weight files are external downloads and are not verified or hosted by this repository beyond the linked source; users should verify file integrity before use.
- No automated tests, benchmarks, or sample output images are currently included in this repository.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Fork the repository, make your changes, and open a pull request.
