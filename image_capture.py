import cv2
import os

label = "nonbio"  # change to recyclable / nonbio
save_dir = f"dataset/{label}"
os.makedirs(save_dir, exist_ok=True)

# cap = cv2.VideoCapture("http://10.181.225.204:8080/video") ito ay kapag naka hotspot
# cap = cv2.VideoCapture("http://192.168.2.165:8080/video")
cap = cv2.VideoCapture("http://192.168.2.165:8080/video")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture - Press SPACE", frame)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
        print(f"Saved {count}")
        count += 1

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
