import cv2
from pathlib import Path
from src.face_detector import FaceDetector
from src.mask_classifier import MaskClassifier

def main():
    # Paths
    face_model_path = Path("models") / "haarcascade_frontalface_default.xml"
    mask_model_path = Path("models") / "mask_detector.keras"

    # Load models
    print("[INFO] Loading face detector...")
    face_detector = FaceDetector(str(face_model_path))

    print("[INFO] Loading mask classifier...")
    mask_classifier = MaskClassifier(str(mask_model_path))

    # Open camera (1 = external, 0 = built-in)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[INFO] External camera not found. Trying built-in camera...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open any webcam.")
        return

    print("[INFO] Webcam opened successfully.")
    print("[INFO] Real-time Face + Mask Detection is running...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect faces
        faces = face_detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            label, confidence = mask_classifier.predict(face_roi)

            # Color: Green for Mask, Red for No Mask
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Text with confidence
            text = f"{label}: {confidence*100:.1f}%"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - 25), (x + text_width, y), color, -1)
            cv2.putText(frame, text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show result
        cv2.imshow("Face Mask Detection - Cybersecurity Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program ended.")

if __name__ == "__main__":
    main()