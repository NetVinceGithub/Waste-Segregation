import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# SETTINGS
# =========================
MODEL_PATH = "garbage_model.h5"
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
CLASS_NAMES = ["BIO", "NON-BIO", "RECYCLABLE"]

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("✅ Camera started. Press 'Q' to quit.")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:  # Fixed indentation
        break
    
    # --- Preprocess frame ---
    img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    # --- Predict ---
    predictions = model.predict(img, verbose=0)[0]
    confidence = np.max(predictions)
    class_id = np.argmax(predictions)
    label = CLASS_NAMES[class_id]
    
    # --- Display ---
    color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Garbage Classification (Vision Only)", frame)
    
    # Quit - Fixed indentation
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()