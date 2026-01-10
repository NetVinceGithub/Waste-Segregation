import cv2

cap = cv2.VideoCapture(1)
if cap.isOpened():
    print("Camera 1 opened successfully")
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera 1 Test", frame)
            print(f"Frame {i+1} received - Shape: {frame.shape}")
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            print(f"Failed to read frame {i+1}")
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Failed to open camera 1")