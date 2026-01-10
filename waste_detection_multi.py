import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# =========================
# SETTINGS
# =========================
MODEL_PATH = "garbage_model.h5"
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.30  # Lowered for weak model
CLASS_NAMES = ["BIO", "NON-BIO"]
CLASS_COLORS = {
    "BIO": (0, 255, 0),          # Green
    "NON-BIO": (0, 0, 255),      # Red
}

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# =========================
# BACKGROUND SUBTRACTION SETUP
# =========================
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500, 
    varThreshold=50, 
    detectShadows=True
)

# =========================
# CAMERA SETUP - IP WEBCAM
# =========================
# IMPORTANT: Update this IP address to match your IP Webcam app
# PHONE_CAMERA_URL = "http://10.52.190.176:8080/video"
PHONE_CAMERA_URL = 0
print(f"Connecting to IP Webcam: {PHONE_CAMERA_URL}")
cap = cv2.VideoCapture(PHONE_CAMERA_URL)

# Set buffer size to 1 to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Wait a bit for connection
time.sleep(2)

if not cap.isOpened():
    print("❌ Camera not found")
    print("\nTroubleshooting:")
    print("1. Make sure IP Webcam app is running on your phone")
    print("2. Check if phone and PC are on same WiFi")
    print("3. Update the IP address in line 38 to match the IP shown in your app")
    print("4. Test the URL in your browser first")
    exit()

# Test if we can read frames
ret, test_frame = cap.read()
if not ret or test_frame is None:
    print("❌ Cannot read frames from camera")
    print("Try these alternatives:")
    print("1. http://YOUR_IP:8080/video")
    print("2. http://YOUR_IP:8080/videofeed")
    print("\nMake sure to replace YOUR_IP with the IP shown in IP Webcam app")
    cap.release()
    exit()

print("✅ Camera connected successfully!")
print("Press 'Q' to quit.")
print("📦 Detecting multiple objects with bounding boxes...")
print("\n⚠️ Note: Model has low accuracy - you may need to retrain it")

# =========================
# DETECTION SETTINGS
# =========================
MIN_AREA = 5000  # Minimum object size
MAX_AREA = 100000  # Maximum object size

# =========================
# HELPER FUNCTIONS
# =========================
def detect_objects(frame, fg_mask):
    """Detect objects using contour detection"""
    # Find contours
    contours, _ = cv2.findContours(
        fg_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter small contours (noise)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (avoid very thin objects)
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 5:
                objects.append((x, y, w, h))
    
    return objects

def classify_object(frame, bbox, model):
    """Classify a single object from bounding box"""
    x, y, w, h = bbox
    
    # Add padding to bbox
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    
    # Crop object
    obj_img = frame[y1:y2, x1:x2]
    
    if obj_img.size == 0:
        return None, 0.0, None
    
    # Preprocess
    img = cv2.resize(obj_img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img, verbose=0)[0]
    confidence = np.max(predictions)
    class_id = np.argmax(predictions)
    label = CLASS_NAMES[class_id]
    
    # Return all predictions for debugging
    return label, confidence, predictions

def draw_detection(frame, bbox, label, confidence, predictions=None):
    """Draw bounding box and label"""
    x, y, w, h = bbox
    
    # Get color based on classification
    color = CLASS_COLORS.get(label, (255, 255, 255))
    
    # Draw bounding box
    thickness = 3 if confidence > CONFIDENCE_THRESHOLD else 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Prepare label text
    text = f"{label} {confidence*100:.0f}%"
    
    # Draw label background
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        frame, 
        (x, y - text_h - 10), 
        (x + text_w + 10, y), 
        color, 
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame, 
        text, 
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255), 
        2
    )
    
    # Show all predictions if available
    if predictions is not None:
        y_offset = y + h + 20
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
            pred_text = f"{name}: {prob*100:.0f}%"
            cv2.putText(
                frame,
                pred_text,
                (x, y_offset + i*15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

# =========================
# MAIN LOOP
# =========================
frame_count = 0
skip_frames = 3  # Process every 3rd frame for better performance
consecutive_failures = 0
max_failures = 30

print("Starting detection loop...")
print("Wait 10-15 seconds for background calibration...")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            consecutive_failures += 1
            print(f"⚠️ Failed to read frame ({consecutive_failures}/{max_failures})")
            
            if consecutive_failures >= max_failures:
                print("❌ Too many frame read failures. Connection lost.")
                break
            
            time.sleep(0.1)
            continue
        
        # Reset failure counter on successful read
        consecutive_failures = 0
        frame_count += 1
        
        # Print status every 90 frames
        if frame_count % 90 == 0:
            print(f"Processing frame {frame_count}...")
        
        display_frame = frame.copy()
        
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)
        
        # Denoise
        fg_mask = cv2.morphologyEx(
            fg_mask, 
            cv2.MORPH_OPEN, 
            np.ones((5, 5), np.uint8)
        )
        fg_mask = cv2.morphologyEx(
            fg_mask, 
            cv2.MORPH_CLOSE, 
            np.ones((5, 5), np.uint8)
        )
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Detect objects
        objects = detect_objects(frame, fg_mask)
        
        # Classify and draw each object
        if frame_count % skip_frames == 0:  # Skip frames for performance
            for bbox in objects:
                label, confidence, predictions = classify_object(frame, bbox, model)
                if label:
                    draw_detection(display_frame, bbox, label, confidence, predictions)
        
        # Draw object count
        cv2.putText(
            display_frame,
            f"Objects: {len(objects)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw connection status
        cv2.putText(
            display_frame,
            "IP Webcam Connected",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        # Show calibration message
        if frame_count < 300:  # First 10 seconds at 30fps
            cv2.putText(
                display_frame,
                "Calibrating... Keep still!",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
        
        # Show frames
        try:
            cv2.imshow("Waste Detection - Multi Object", display_frame)
            cv2.imshow("Foreground Mask", fg_mask)
        except Exception as e:
            print(f"Display error: {e}")
            break
        
        # Quit on 'Q' key
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q')]:
            print("Quit key pressed")
            break

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped")
print("\n💡 Tip: Your model needs retraining for better accuracy!")