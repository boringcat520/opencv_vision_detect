import cv2
import numpy as np

def classify_color(hsv_color):
    h, s, v = hsv_color

    if 0 <= h < 30:
        return "red"
    elif 30 <= h < 90:
        return "green"
    elif 90 <= h < 150:
        return "blue"
    else:
        return "unknown"

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2
        roi = frame[center_y - 50:center_y + 50, center_x - 50:center_x + 50]

        # 转换为HSV颜色空间
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 计算颜色的平均值
        avg_color = np.mean(hsv_roi, axis=(0, 1)).astype(int)
        h, s, v = avg_color

        classified_color = classify_color(avg_color)

        cv2.rectangle(frame, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (0, 255, 0), 2)
        cv2.putText(frame, classified_color, (center_x - 40, center_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
