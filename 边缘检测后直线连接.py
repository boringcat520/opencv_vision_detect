import cv2
import numpy as np

def update_thresholds(value):
    global threshold1, threshold2, hough_threshold
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Settings')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Settings')
    hough_threshold = cv2.getTrackbarPos('Hough Threshold', 'Settings')
    detect_lines()

def detect_lines():
    global threshold1, threshold2, hough_threshold
    edges = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)
    line_img = np.copy(image)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=hough_threshold)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.namedWindow('Detected Lines', cv2.WINDOW_FREERATIO)
    cv2.imshow('Detected Lines', line_img)

# 读取图像
image_path = r"C:\Users\admin\Downloads\1.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 创建一个窗口
cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Settings', 400, 300)

# 创建滑动条
threshold1 = 50
threshold2 = 150
hough_threshold = 100
cv2.createTrackbar('Threshold1', 'Settings', threshold1, 255, update_thresholds)
cv2.createTrackbar('Threshold2', 'Settings', threshold2, 255, update_thresholds)
cv2.createTrackbar('Hough Threshold', 'Settings', hough_threshold, 500, update_thresholds)

# 初始化
update_thresholds(None)

cv2.waitKey(0)
cv2.destroyAllWindows()
