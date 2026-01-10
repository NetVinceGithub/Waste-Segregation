import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "garbage_model.h5"
IMAGE_SIZE = 224
CLASS_NAMES = ["BIO", "NON-BIO", "RECYCLABLE"]

# Load model
model = load_model(MODEL_PATH)

# Capture one frame from camera
cap = cv2.VideoCapture(1)
ret, frame = cap.read()
cap.release()

if ret:
    # Preprocess
    img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img)[0]
    
    print("\n=== MODEL PREDICTION ===")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {predictions[i]*100:.2f}%")
    
    print(f"\nPredicted: {CLASS_NAMES[np.argmax(predictions)]}")
    print("="*30)