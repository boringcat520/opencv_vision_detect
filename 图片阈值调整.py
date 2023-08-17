import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

# 读取图像
image = cv2.imread(r"C:\Users\admin\Downloads\1.jpg")
resized_image = cv2.resize(image, (1080, 720))

# 腐蚀和膨胀操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))# (1, 1) 是结构元素的大小，表示一个 1x1 的矩形。这个结构元素用于形态学操作。
eroded = cv2.erode(resized_image, kernel, iterations=2)# iterations 参数表示腐蚀操作的迭代次数
dilated = cv2.dilate(eroded, kernel, iterations=2)#iterations 参数表示膨胀操作的迭代次数

# 图像预处理
blur = cv2.GaussianBlur(dilated, (5, 5), 0)
lab_image = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
'''
blur = cv2.GaussianBlur(dilated, (5, 5), 0): 这行代码使用高斯模糊Gaussian Blur对经过膨胀操作的图像 dilated 进行处理。
高斯模糊是一种常用的图像模糊方法，通过应用高斯核来平滑图像，从而降低图像中的噪声和细节。
参数 (5, 5) 指定了高斯核的大小，表示一个 5x5 的核，这个大小会影响模糊程度。
最后一个参数 0 是指在 x 和 y 方向上的高斯标准差 如果设为0 则函数会自动计算。
'''
# 设置颜色范围，提取所需颜色
lower_yellow = np.array([130, 20, 120])
upper_yellow = np.array([255, 126, 180])
mask = cv2.inRange(lab_image, lower_yellow, upper_yellow)

# 自适应阈值法二值化处理
binary = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 边缘检测
edges = cv2.Canny(binary, 60, 150)
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()