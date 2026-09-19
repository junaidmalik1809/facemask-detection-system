import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class MaskClassifier:
    def __init__(self, model_path: str):
        """
        Load the trained mask detection model.
        """
        self.model = load_model(model_path)
        # Important: order must match training class_indices
        # {'with_mask': 0, 'without_mask': 1}
        self.labels = ["Mask", "No Mask"]

    def predict(self, face_image):
        """
        Predict Mask or No Mask from a cropped face image.
        
        Returns:
            label (str), confidence (float)
        """
        # Convert BGR (OpenCV) → RGB
        face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Resize to the size used during training (128x128)
        face = cv2.resize(face, (128, 128))
        
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0   # Normalize

        # Prediction
        pred = self.model.predict(face, verbose=0)[0]
        
        label_index = np.argmax(pred)
        confidence = float(pred[label_index])
        label = self.labels[label_index]

        return label, confidence