import cv2


class FaceDetector:
    def __init__(self, model_path: str):
        self.face_cascade = cv2.CascadeClassifier(model_path)

        if self.face_cascade.empty():
            raise ValueError(
                f"Could not load face detection model: {model_path}"
            )

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        return faces


if __name__ == "__main__":
    detector = FaceDetector(
        "models/haarcascade_frontalface_default.xml"
    )

    print("Face detector loaded successfully.")