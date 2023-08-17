import cv2
import numpy as np

# 读取图像
image = cv2.imread(r"C:\Users\admin\Downloads\1.jpg")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 边缘检测
edges = cv2.Canny(gray, 50, 150)

# 轮廓检测
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []  # 存储检测到的直角
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:  # 如果逼近后的多边形有四个顶点，认为是直角
        rectangles.append(approx)

# 绘制直角
for rectangle in rectangles:
    cv2.polylines(image, [rectangle], isClosed=True, color=(0, 255, 0), thickness=2)

# 显示结果
cv2.namedWindow('Rectangles Detected', cv2.WINDOW_FREERATIO)
cv2.imshow('Rectangles Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
