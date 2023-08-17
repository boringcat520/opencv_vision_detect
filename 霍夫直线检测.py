import cv2
import numpy as np

# 回调函数，用于处理滑动条变化事件
def update_threshold(value):
    _, binary_image = cv2.threshold(image_gray, value, max_value, cv2.THRESH_BINARY)
    
    # 添加腐蚀操作
    kernel = np.ones((6, 6), np.uint8)  # 腐蚀核大小和形状
    eroded_image = cv2.erode(binary_image, kernel, iterations=2)
    
    # 进行霍夫直线检测
    lines = cv2.HoughLinesP(eroded_image, 1, np.pi / 90, threshold=90, minLineLength=100, maxLineGap=20)
    
    # 在彩色图像上绘制检测到的直线
    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 画出红色的直线
    cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image', eroded_image)
    cv2.imshow('Hough Lines', line_image)

# 读取图像
image_path = r"C:\Users\admin\Downloads\1.jpg"
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 创建一个窗口
cv2.namedWindow('Threshold Adjuster', cv2.WINDOW_NORMAL)  # 使用WINDOW_NORMAL以允许调整窗口大小

# 创建一个滑动条
initial_threshold = 134
max_value = 255
cv2.createTrackbar('Threshold', 'Threshold Adjuster', initial_threshold, max_value, update_threshold)

# 初始化二值化图像
_, binary_image = cv2.threshold(image_gray, initial_threshold, max_value, cv2.THRESH_BINARY)

# 获取屏幕尺寸
screen_width, screen_height = 1920, 1080  # 这里以1920x1080为例，你可以根据实际情况修改

# 计算适应屏幕的图像大小
image_scale = min(screen_width / image.shape[1], screen_height / image.shape[0])
window_width = int(image.shape[1] * image_scale)
window_height = int(image.shape[0] * image_scale)

# 显示原始图像和二值化图像
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image)

# 调整窗口大小
cv2.resizeWindow('Original Image', window_width, window_height)
cv2.resizeWindow('Binary Image', window_width, window_height)

cv2.waitKey(0)
cv2.destroyAllWindows()
