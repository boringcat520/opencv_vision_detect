import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

# 读取图像
image = cv2.imread(r"C:\Users\admin\Downloads\4.jpg")
resized_image = cv2.resize(image, (1080, 720))
new_img=resized_image.copy()
# 腐蚀和膨胀操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
eroded = cv2.erode(resized_image, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

# 图像预处理
blur = cv2.GaussianBlur(dilated, (5, 5), 0)
lab_image = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)

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
# 霍夫直线变换
lines = cv2.HoughLines(edges, 1, np.pi/360, threshold=130)

# 提取直线的起始点和终点坐标，计算角度，并进行拟合
if lines is not None:
    points = []
    near_lines = []
    far_points = []
    far_lines = []
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        cv2.line(resized_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 使用红色线条绘制所有直线
        cv2.namedWindow('resized_image', cv2.WINDOW_NORMAL)
        cv2.imshow('resized_image', resized_image)
        # 判断与已有直线的角度差异

        angle_deg = np.rad2deg(theta)
        angle_diffs = [np.abs(angle_deg - line[4]) for line in near_lines]
        if len(near_lines) == 0 or min(angle_diffs) < 5:
            points.append([x1, y1])
            points.append([x2, y2])
            near_lines.append((x1, y1, x2, y2, angle_deg))
        else:
            far_points.append([x1, y1])
            far_points.append([x2, y2])
            far_lines.append((x1, y1, x2, y2, angle_deg))
            
    # 对角度相近的直线进行拟合
    if len(near_lines) > 0:
        x_values = np.array([line[0] for line in near_lines] + [line[2] for line in near_lines]).reshape(-1, 1)
        y_values = np.array([line[1] for line in near_lines] + [line[3] for line in near_lines]).reshape(-1, 1)
        regressor = LinearRegression()
        regressor.fit(x_values, y_values)
        x1_fit = 0
        y1_fit = int(regressor.predict([[x1_fit]]))
        x2_fit = resized_image.shape[1] - 1
        y2_fit = int(regressor.predict([[x2_fit]]))
        cv2.line(new_img, (x1_fit, y1_fit), (x2_fit, y2_fit), (0, 255, 0), 2)  # 使用绿色线条绘制参与拟合的直线
        print("起始点坐标:", x1_fit, y1_fit)
        print("终点坐标:", x2_fit, y2_fit)

        angle_rad = np.arctan(regressor.coef_[0][0])
        angle_deg = np.rad2deg(angle_rad)
        print("直线夹角（角度）:", angle_deg)
        # 对未参与拟合的直线进行拟合并输出信息

    if len(near_lines) < len(lines):
        remaining_lines = [line for line in lines[:, 0] if tuple(line) not in near_lines]
        remaining_x_values = []
        remaining_y_values = []
        for line in remaining_lines:
            if len(line) == 4:  # 检查直线长度是否为4
                remaining_x_values.append(line[0])
                remaining_x_values.append(line[2])
                remaining_y_values.append(line[1])
                remaining_y_values.append(line[3])
        remaining_x_values = np.array(remaining_x_values).reshape(-1, 1)
        remaining_y_values = np.array(remaining_y_values).reshape(-1, 1)
    
        if len(remaining_x_values) > 0 and len(remaining_y_values) > 0:  # 检查是否有足够的数据来拟合直线
            remaining_regressor = LinearRegression()
            remaining_regressor.fit(remaining_x_values, remaining_y_values)
            remaining_x1_fit = 0
            remaining_y1_fit = int(remaining_regressor.predict([[remaining_x1_fit]]))
            remaining_x2_fit = resized_image.shape[1] - 1
            remaining_y2_fit = int(remaining_regressor.predict([[remaining_x2_fit]]))
            cv2.line(new_img, (remaining_x1_fit, remaining_y1_fit), (remaining_x2_fit, remaining_y2_fit), (0, 255, 255), 2)  # 使用绿色线条绘制未参与拟合的直线
            print("有垂直的直线:")
            print("起始点坐标:", remaining_x1_fit, remaining_y1_fit)
            print("终点坐标:", remaining_x2_fit, remaining_y2_fit)
            print("斜率:", remaining_regressor.coef_[0][0])
            print("截距:", remaining_regressor.intercept_[0])
        else:
            print("有垂直的直线:")
            print("起始点坐标:", far_points[0][0], far_points[0][1])
            print("终点坐标:", far_points[1][0], far_points[1][1])
            
            if far_points[1][0] != far_points[0][0]:
                slope = (far_points[1][1] - far_points[0][1]) / (far_points[1][0] - far_points[0][0])
                
            else:
                # 分母为零，将直线定义为垂直于x轴的竖直线
                slope_remaining = float('inf')
                intercept_remaining = None
            angle_rad = np.arctan(slope)
            angle_deg = np.rad2deg(angle_rad)
            print("直线夹角（角度）:", angle_deg)
            # 使用黄色线条绘制未参与拟合的直线 
            cv2.line(new_img, (far_points[0][0],far_points[0][1]), (far_points[1][0], far_points[1][1]), (0, 255, 255), 2)  
             # 已参与拟合的直线
            slope_fit = regressor.coef_[0][0]
            intercept_fit = regressor.intercept_[0]

            # 未参与拟合的直线（使用俩点式计算斜率和截距）
            if len(far_lines) > 0:
                x1_remaining, y1_remaining = far_points[0]
                x2_remaining, y2_remaining = far_points[1]
                slope_remaining = slope
                intercept_remaining = far_points[0][1] - slope * far_points[0][0]

# 计算直线交点坐标
            if len(far_lines) > 0:
                x_intersect = (intercept_remaining - intercept_fit) / (slope_fit - slope_remaining)
                y_intersect = slope_fit * x_intersect + intercept_fit
    
                x_intersect = int(x_intersect)
                y_intersect = int(y_intersect)
    
                print("交点坐标:", x_intersect, y_intersect)
        
        
        
            # 显示结果

cv2.namedWindow("Lines", cv2.WINDOW_NORMAL)
cv2.imshow("Lines", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()