import cv2

print("Searching for cameras...")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ Camera {i} works - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"⚠️ Camera {i} opens but can't read frames")
        cap.release()
    else:
        print(f"❌ Camera {i} not found")