import numpy as np
import cv2
import matplotlib.pyplot as plt
import serial
import time
import re


def send_data(data):  # 串口发送数据，data为字符串
    if data is not None:
        ser = serial.Serial("/dev/ttyAMA0", 115200)  # set up serial
        ser.write(data.encode('utf-8'))  # write data


def receive_data():  # 串口接收数据
    ser = serial.Serial("/dev/ttyAMA0", 115200)  # set up serial

    while True:
        # 获得接收缓冲区字符
        count = ser.inWaiting()  # 获取接收缓冲区字符的字节长度，返回值为整型，可用于判断是否接收到数据
        if count != 0:
            # 读取内容并回显
            recv = ser.read(count)

            return recv
            # 清空接收缓冲区
        ser.flushInput()
        # 必要的软件延时
        time.sleep(0.1)


while True:
    start_time = time.time()
    cap = cv2.VideoCapture(0)  # 打开摄像头
    new_width = 1280
    new_height = 960
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)  # 设置摄像头帧宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)  # 设置摄像头帧高度
    center_x_list = []  # 存储x坐标
    center_y_list = []  # 存储y坐标
    center_x_list1 = []  # 存储x坐标
    center_y_list1 = []  # 存储y坐标

    while True:
        ret, frame = cap.read()  # 读取视频帧
        cv2.waitKey(1)
        kernel = np.ones((13, 13), np.uint8)
        img_dilation = cv2.dilate(frame, kernel, iterations=2)  # 膨胀操作
        img_erosion = cv2.erode(frame, kernel, iterations=2)  # 腐蚀操作
        gray = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
        blur = cv2.GaussianBlur(gray, (17, 17), 0)  # 高斯模糊
        ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow("frame", thresh)
        edgeImage = cv2.Canny(thresh, 50, 220)  # 边缘检测
        contours, hierarchy = cv2.findContours(image=edgeImage, mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
        frame_copy = frame.copy()  # 复制一份原始帧用于绘制结果

        for cnt in contours:
            if 15000 < cv2.contourArea(cnt) < 60000:  # 选择面积满足条件的轮廓
                ellipse = cv2.fitEllipse(cnt)  # 拟合椭圆
                (center_x, center_y), (major_axis, minor_axis), angle = ellipse
                print(center_x, center_y)
                cv2.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆
                center_x_list.append(center_x)
                center_y_list.append(center_y)
            cv2.imshow("frame2", frame_copy)  # 显示帧

        if len(center_x_list) == 30 and len(center_y_list) == 30:  # 如果30次则退出循环
            break

    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口

    center_x_mean = np.mean(center_x_list)  # 计算x坐标的均值
    center_y_mean = np.mean(center_y_list)  # 计算y坐标的均值
    std_x = np.std(center_x_list)
    std_y = np.std(center_y_list)
    print(std_x, std_y)
    print(f'Mean X: {center_x_mean}')  # 输出x坐标的均值
    print(f'Mean Y: {center_y_mean}')  # 输出y坐标的均值
    '''
    plt.scatter(center_x_list, center_y_list)
    plt.title("Scatter plot of coordinates")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    '''
    for i, j in zip(center_x_list, center_y_list):
        distance_x = abs(i - center_x_mean)
        distance_y = abs(j - center_y_mean)
        if distance_x < std_x and distance_y < std_y:
            center_x_list1.append(i)
            center_y_list1.append(j)

    center_x_mean1 = np.mean(center_x_list1)  # 计算x坐标的均值
    center_y_mean1 = np.mean(center_y_list1)  # 计算y坐标的均值
    std_x1 = np.std(center_x_list1)
    std_y1 = np.std(center_y_list1)

    send_data("center_x_mean: ")
    send_data(str(center_x_mean1)+"\n")
    send_data("center_y_mean: ")
    send_data(str(center_y_mean1)+"\n")

    if len(center_x_list1) > 1 and len(center_y_list1) > 1:
        print(f'Mean X1: {center_x_mean1}')  # 输出x坐标的均值
        print(f'Mean Y1: {center_y_mean1}')  # 输出y坐标的均值
        print(f'Std X1: {std_x1}')
        print(f'Std Y1: {std_y1}')

    else:
        print("Insufficient data points.")
    end_time = time.time()
    run_time = end_time - start_time
    print("代码运行时间：", run_time, "秒")

    '''
    plt.scatter(center_x_list1, center_y_list1)
    plt.title("Scatter plot of coordinates1")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    #time.sleep(4)
    #plt.close()
    '''

