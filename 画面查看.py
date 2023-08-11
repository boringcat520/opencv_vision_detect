import cv2
import time

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("img", cv2.WINDOW_NORMAL)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    cv2.imshow("img", frame)
    
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    if elapsed_time > 1:  # Calculate frame rate every second
        frame_rate = frame_count / elapsed_time
        print(f"Frame Rate: {frame_rate:.2f} fps")
        frame_count = 0
        start_time = time.time()

    if cv2.waitKey(1) == 27:
        break

    if cv2.waitKey(1) == ord("s"):
        cv2.imwrite("test.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()
