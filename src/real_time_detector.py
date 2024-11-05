import cv2
from src.detect_faces import FaceDetector
from src.age_gender_detection import AgeGenderDetector
from src.emotion_detection import EmotionDetector
from src.utils import draw_label

class RealTimeDetector:
    def __init__(self, face_model_path, age_proto_path, age_model_path, gender_proto_path, gender_model_path, emotion_model_path):
        self.face_detector = FaceDetector(face_model_path)
        self.age_gender_detector = AgeGenderDetector(age_proto_path, age_model_path, gender_proto_path, gender_model_path)
        self.emotion_detector = EmotionDetector(emotion_model_path)

    def run(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = self.face_detector.detect_faces(frame)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                # Predict age, gender, and emotion
                gender, age = self.age_gender_detector.predict_age_gender(face)
                emotion = self.emotion_detector.predict_emotion(face)

                # Draw bounding box and labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                draw_label(frame, f"{gender}, Age: {age}, Emotion: {emotion}", x, y - 10)

            # Display the resulting frame
            cv2.imshow('Face, Age, Gender, and Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
