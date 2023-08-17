import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(r"C:\Users\admin\Downloads\1.jpg", cv2.IMREAD_GRAYSCALE)

# 执行极线变换
theta_values = np.linspace(0, 180, max(image.shape), endpoint=False)
radon_transform = cv2.radon(image, angles=theta_values, circle=True)

# 在Radon空间中找到峰值（直线）
peak_angles = np.argmax(radon_transform, axis=0)

# 在原始图像上绘制检测到的直线
for angle in peak_angles:
    angle_radians = np.deg2rad(angle)
    cos_val = np.cos(angle_radians)
    sin_val = np.sin(angle_radians)
    x0 = cos_val * image.shape[1]
    y0 = sin_val * image.shape[0]
    x1 = int(x0 + 1000 * (-sin_val))
    y1 = int(y0 + 1000 * (cos_val))
    x2 = int(x0 - 1000 * (-sin_val))
    y2 = int(y0 - 1000 * (cos_val))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示原始图像和检测到的直线
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(radon_transform, cmap='gray'), plt.title('Radon Transform')
plt.show()
