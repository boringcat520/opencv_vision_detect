import cv2
import numpy as np

def update_params(_):
    global maxCorners, qualityLevel, minDistance, lines
    
    maxCorners = cv2.getTrackbarPos('maxCorners', 'Detected Lines')
    qualityLevel = cv2.getTrackbarPos('qualityLevel', 'Detected Lines') / 100.0
    minDistance = cv2.getTrackbarPos('minDistance', 'Detected Lines')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
    corners = np.int0(corners)
    
    lines = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            x1, y1 = corners[i].ravel()
            x2, y2 = corners[j].ravel()
            lines.append((x1, y1, x2, y2))
    
    image_with_lines = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow('Detected Lines', image_with_lines)

# 读取图像
image = cv2.imread(r"C:\Users\admin\Downloads\1.jpg")

# 创建窗口和滑动条
cv2.namedWindow('Detected Lines', cv2.WINDOW_FREERATIO)
cv2.namedWindow('Detected Lines')
cv2.createTrackbar('maxCorners', 'Detected Lines', 50, 500, update_params)
cv2.createTrackbar('qualityLevel', 'Detected Lines', 1, 100, update_params)
cv2.createTrackbar('minDistance', 'Detected Lines', 10, 100, update_params)

maxCorners = 50
qualityLevel = 0.01
minDistance = 10
lines = []

update_params(None)  # 初始更新一次

cv2.waitKey(0)
cv2.destroyAllWindows()
