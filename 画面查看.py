import cv2

cap=cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("img", cv2.WINDOW_NORMAL)  # 创建可调节大小的窗口

while True:
    ret,frame=cap.read()
    if not ret:
        print("Can't receive frame")
        break
    cv2.imshow("img",frame)
    if cv2.waitKey(1)==27:
        break

    if cv2.waitKey(1)==ord("s"):
        cv2.imwrite("test.jpg",frame)
        break

cap.release()
cv2.destroyAllWindows()