import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)  # 打开摄像头
new_width =1280
new_height =960
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)  # 设置摄像头帧宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)  # 设置摄像头帧高度



while True:
    left_center_x_list = []  # 存储x坐标
    right_center_x_list = []  # 存储x坐标

    left_center_y_list = []
    right_center_y_list = []  # 存储y坐标
    ret, frame = cap.read()  # 读取视频帧
    #cv2.imshow("img",frame)
    kernel = np.ones((13, 13), np.uint8)
    img_dilation = cv2.dilate(frame, kernel, iterations=1)  # 膨胀操作
    img_erosion = cv2.erode(frame, kernel, iterations=1)  # 腐蚀操作
    gray = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    #cv2.imshow("gray", gray)
    blur = cv2.GaussianBlur(gray, (17, 17), 0)  # 高斯模糊
    ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)  # 二值化图像
    #cv2.imshow("thresh", thresh)
    edgeImage = cv2.Canny(thresh, 50, 220)  # 边缘检测
    cv2.imshow("canny",edgeImage)

    image_width = edgeImage.shape[1]

    # 将图像分成左右两半
    offset_x = image_width // 2
    left_half = edgeImage[:, :offset_x]
    right_half = edgeImage[:, offset_x:]

    # 显示左右两半图像
    #cv2.imshow("Left Half", left_half)
    #cv2.imshow("Right Half", right_half)
    contours_left, hierarchy = cv2.findContours(image=left_half, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    contours_right, hierarchy = cv2.findContours(image=right_half, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    frame_copy = frame.copy()  # 复制一份原始帧用于绘制结果

    for cnt in contours_right:#右边
        if 6000 < cv2.contourArea(cnt) < 40000:  # 选择面积满足条件的轮廓

            (right_center_x, right_center_y), (a, b), angle = cv2.fitEllipse(cnt)
            #cv2.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆

            ellipse=(right_center_x + offset_x, right_center_y),(a, b), angle
            right_center_x_list.append(right_center_x + offset_x)  # 将中心点的x坐标添加到列表中
            right_center_y_list.append(right_center_y)  # 将中心点的y坐标添加到列表中
            cv2.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆
            #print(f'Circle Center: ({center_x}, {center_y})')

    for cnt in contours_left:#左边
        if 7000 < cv2.contourArea(cnt) < 30000:  # 选择面积满足条件的轮廓
            ellipse = cv2.fitEllipse(cnt)  # 拟合椭圆
            cv2.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆
            (left_center_x, left_center_y), _, _ = ellipse
            left_center_x_list.append(left_center_x)  # 将中心点的x坐标添加到列表中
            left_center_y_list.append(left_center_y)  # 将中心点的y坐标添加到列表中

            #print(f'Circle Center: ({center_x}, {center_y})')

    cv2.waitKey(1)

    cv2.imshow("frame", frame_copy)  # 显示帧



    # 计算直线的左右端点像素位置
    if len(left_center_x_list)>=2:
        '''
        left_point_x = sum(left_center_x_list) / len(left_center_x_list)
        right_point_x = sum(right_center_x_list) / len(right_center_x_list)

        left_point_y = sum(left_center_y_list) / len(left_center_y_list)
        right_point_y = sum(right_center_y_list) / len(right_center_y_list)
        '''
        left_point_x = np.mean(left_center_x_list)
        right_point_x = np.mean(right_center_x_list)

        left_point_y = np.mean(left_center_y_list)
        right_point_y = np.mean(right_center_y_list)
            # 计算直线的角度
        slope = (right_point_y - left_point_y) / (right_point_x - left_point_x)
        angle = np.arctan(slope) * 180 / np.pi

        print(f'Left Point: ({left_point_x}, {left_point_y})')
        print(f'Right Point: ({right_point_x}, {right_point_y})')
        print(f'Angle: {angle}')
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下 'q' 键则退出循环
            break


cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 关闭窗口






