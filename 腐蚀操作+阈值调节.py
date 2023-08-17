import cv2
import numpy as np

def update_parameters(x):
    global lab_image, binary, edges

    # 获取颜色范围滑动条的值
    lower_l = cv2.getTrackbarPos('Lower L', 'edges')
    lower_a = cv2.getTrackbarPos('Lower A', 'edges')
    lower_b = cv2.getTrackbarPos('Lower B', 'edges')
    upper_l = cv2.getTrackbarPos('Upper L', 'edges')
    upper_a = cv2.getTrackbarPos('Upper A', 'edges')
    upper_b = cv2.getTrackbarPos('Upper B', 'edges')

    # 获取Canny参数滑动条的值
    canny_lower = cv2.getTrackbarPos('Canny Lower', 'edges')
    canny_upper = cv2.getTrackbarPos('Canny Upper', 'edges')

    # 腐蚀和膨胀操作等预处理步骤
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(resized_image, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # 图像预处理
    blur = cv2.GaussianBlur(dilated, (5, 5), 0)
    lab_image = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)

    # 设置颜色范围，提取所需颜色
    lower_yellow = np.array([lower_l, lower_a, lower_b])
    upper_yellow = np.array([upper_l, upper_a, upper_b])
    mask = cv2.inRange(lab_image, lower_yellow, upper_yellow)

    # 自适应阈值法二值化处理等后续步骤
    binary = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 边缘检测
    edges = cv2.Canny(binary, canny_lower, canny_upper)

    # 显示边缘图
    cv2.imshow('edges', edges)

# 读取图像
image = cv2.imread(r"C:\Users\admin\Downloads\1.jpg")
resized_image = cv2.resize(image, (1080, 720))

# 创建窗口
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)

# 创建颜色范围滑动条
cv2.createTrackbar('Lower L', 'edges', 130, 255, update_parameters)
cv2.createTrackbar('Lower A', 'edges', 20, 255, update_parameters)
cv2.createTrackbar('Lower B', 'edges', 120, 255, update_parameters)
cv2.createTrackbar('Upper L', 'edges', 255, 255, update_parameters)
cv2.createTrackbar('Upper A', 'edges', 126, 255, update_parameters)
cv2.createTrackbar('Upper B', 'edges', 180, 255, update_parameters)

# 创建Canny参数滑动条
cv2.createTrackbar('Canny Lower', 'edges', 60, 255, update_parameters)
cv2.createTrackbar('Canny Upper', 'edges', 150, 255, update_parameters)

# 初始化图像处理
lab_image = np.zeros_like(resized_image)
binary = np.zeros_like(resized_image)
edges = np.zeros_like(resized_image)

# 初始化参数更新
update_parameters(0)

while True:
    # 检测按键事件，按Esc退出循环
    key = cv2.waitKey(1)
    if key == 27:
        break

# 关闭窗口
cv2.destroyAllWindows()
