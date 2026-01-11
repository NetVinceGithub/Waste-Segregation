import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time


MODEL_PATH = "garbage_model.h5"
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.30  # Lowered for weak model
CLASS_NAMES = ["BIO", "NON-BIO"]
CLASS_COLORS = {
    "BIO": (0, 255, 0),          # Green
    "NON-BIO": (0, 0, 255),      # Red
}


print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded")


back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500, 
    varThreshold=50, 
    detectShadows=True
)


# Don't forget to update this IP address to match your IP Webcam app
# PHONE_CAMERA_URL = "http://10.52.190.176:8080/video", kapag yung mismong camera ng laptop is 0
PHONE_CAMERA_URL = 0
print(f"Connecting to IP Webcam: {PHONE_CAMERA_URL}")
cap = cv2.VideoCapture(PHONE_CAMERA_URL)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

time.sleep(2)

if not cap.isOpened():
    print("Camera not found")
    exit()

ret, test_frame = cap.read()
if not ret or test_frame is None:
    print("Cannot read frames from camera")

    cap.release()
    exit()

print("Camera connected successfully!")
print("Press 'Q' to quit.")
print("Detecting multiple objects with bounding boxes...")
print("\n Note: Model has low accuracy - you may need to retrain it")


MIN_AREA = 5000  
MAX_AREA = 100000  


def detect_objects(frame, fg_mask):
    """Detect objects using contour detection"""
    contours, _ = cv2.findContours(
        fg_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 5:
                objects.append((x, y, w, h))
    
    return objects

def classify_object(frame, bbox, model):
    """Classify a single object from bounding box"""
    x, y, w, h = bbox
    
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    
    obj_img = frame[y1:y2, x1:x2]
    
    if obj_img.size == 0:
        return None, 0.0, None
    
    
    img = cv2.resize(obj_img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img, verbose=0)[0]
    confidence = np.max(predictions)
    class_id = np.argmax(predictions)
    label = CLASS_NAMES[class_id]
    
    return label, confidence, predictions

def draw_detection(frame, bbox, label, confidence, predictions=None):
    """Draw bounding box and label"""
    x, y, w, h = bbox
    
    color = CLASS_COLORS.get(label, (255, 255, 255))
    
    thickness = 3 if confidence > CONFIDENCE_THRESHOLD else 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    text = f"{label} {confidence*100:.0f}%"
    
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
    
    cv2.putText(
        frame, 
        text, 
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255), 
        2
    )
    
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

frame_count = 0
skip_frames = 3 
consecutive_failures = 0
max_failures = 30

print("Starting detection loop...")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            consecutive_failures += 1
            print(f"Failed to read frame ({consecutive_failures}/{max_failures})")
            
            if consecutive_failures >= max_failures:
                print("Too many frame read failures. Connection lost.")
                break
            
            time.sleep(0.1)
            continue
        
        consecutive_failures = 0
        frame_count += 1

        if frame_count % 90 == 0:
            print(f"Processing frame {frame_count}...")
        
        display_frame = frame.copy()
        
        fg_mask = back_sub.apply(frame)
        
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
        
        fg_mask[fg_mask == 127] = 0
        
        objects = detect_objects(frame, fg_mask)
        
        if frame_count % skip_frames == 0:  
            for bbox in objects:
                label, confidence, predictions = classify_object(frame, bbox, model)
                if label:
                    draw_detection(display_frame, bbox, label, confidence, predictions)
        
        cv2.putText(
            display_frame,
            f"Objects: {len(objects)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            display_frame,
            "IP Webcam Connected",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        if frame_count < 300:  
            cv2.putText(
                display_frame,
                "Calibrating... Keep still!",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
        
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
    print("\n Interrupted ")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

cap.release()
cv2.destroyAllWindows()
print("Detection stopped")
