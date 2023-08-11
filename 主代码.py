import cv2 as cv
import numpy as np
import serial
import time
import re
import matplotlib.pyplot as plt


def circle_det():  # 仅检测圆环
    cap = cv.VideoCapture(0)  # 打开摄像头
    new_width = 1280
    new_height = 960
    cap.set(cv.CAP_PROP_FRAME_WIDTH, new_width)  # 设置摄像头帧宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, new_height)  # 设置摄像头帧高度
    center_x_list = []  # 存储x坐标
    center_y_list = []  # 存储y坐标

    while True:
        ret, frame = cap.read()  # 读取视频帧

        kernel = np.ones((13, 13), np.uint8)
        img_dilation = cv.dilate(frame, kernel, iterations=2)  # 膨胀操作
        img_erosion = cv.erode(frame, kernel, iterations=2)  # 腐蚀操作
        gray = cv.cvtColor(img_erosion, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图

        blur = cv.GaussianBlur(gray, (17, 17), 0)  # 高斯模糊

        ret, thresh = cv.threshold(blur, 90, 255, cv.THRESH_BINARY)
        # ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY ) # 二值化图像
        cv.imshow("frame", thresh)
        edgeImage = cv.Canny(thresh, 50, 220)  # 边缘检测
        # cv2.imshow("frame1", edgeImage)
        contours, hierarchy = cv.findContours(image=edgeImage, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)  # 寻找轮廓

        frame_copy = frame.copy()  # 复制一份原始帧用于绘制结果

        for cnt in contours:
            if 10000 < cv.contourArea(cnt) < 30000:  # 选择面积满足条件的轮廓
                ellipse = cv.fitEllipse(cnt)  # 拟合椭圆
                cv.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆
                (center_x, center_y), (major_axis, minor_axis), angle = ellipse
                center_x_list.append(center_x)  # 将中心点的x坐标添加到列表中
                center_y_list.append(center_y)  # 将中心点的y坐标添加到列表中
                print(center_x, center_y)
        cv.imshow("frame2", frame_copy)  # 显示帧

        if cv.waitKey(1) & 0xFF == ord('q'):  # 如果按下 'q' 键则退出循环
            break

    cap.release()  # 释放摄像头
    cv.destroyAllWindows()  # 关闭窗口

    center_x_mean = np.mean(center_x_list)  # 计算x坐标的均值
    center_y_mean = np.mean(center_y_list)  # 计算y坐标的均值

    print(f'Mean X: {center_x_mean}')  # 输出x坐标的均值
    print(f'Mean Y: {center_y_mean}')  # 输出y坐标的均值
    diff_x = new_width / 2 - center_x_mean
    diff_y = new_height / 2 - center_y_mean
    print(f'Difference X: {diff_x}')
    print(f'Difference Y: {diff_y}')

    send_data("diff_x:")  # 发送数据
    send_data(str(diff_x))
    send_data("\ndiff_y:")
    send_data(str(diff_y))

    threshold = 0.1

    center_x_filtered = [x for x in center_x_list if
                         abs(x - center_x_mean) < threshold * np.std(center_x_list)]  # 过滤后的x坐标
    center_y_filtered = [y for y in center_y_list if
                         abs(y - center_y_mean) < threshold * np.std(center_y_list)]  # 过滤后的y坐标

    min_len = min(len(center_x_filtered), len(center_y_filtered))
    center_x_filtered = center_x_filtered[:min_len]
    center_y_filtered = center_y_filtered[:min_len]

    if len(center_x_filtered) > 1 and len(center_y_filtered) > 1:
        print(f'Std X: {np.std(center_x_filtered)}')
        print(f'Std Y: {np.std(center_y_filtered)}')
        plt.scatter(center_x_filtered, center_y_filtered)
        plt.title("Scatter plot of coordinates")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    else:
        print("Insufficient data points after filtering.")


def mass_det(src):  # 使用霍夫圆检测物料或者圆环
    if src is None:
        print("load image fail!")
        return
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 9)
    binary = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 2)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(binary, kernel)
    dilation = cv.dilate(erosion, kernel)

    circle = cv.HoughCircles(erosion, cv.HOUGH_GRADIENT, 1.4, minDist=3700, param1=50, param2=50, minRadius=50,
                             maxRadius=300)
    if circle is not None:
        for i in circle[0, :]:
            cv.circle(src, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 255), 20)
            cv.circle(src, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)
            break

    print("core:", int(i[0]), int(i[1]))  # 要把它存入一个数组，然后取平均值
    return (src)  # 返回画好圆的图像

    # return (int(i[0]), int(i[1]))  # 返回圆心


# 处理摄像头画面
def process_video_mass():
    capture = cv.VideoCapture(0)  # 打开摄像头
    capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    capture.set(cv.CAP_PROP_EXPOSURE, -100000)  # 曝光度

    while True:
        ret, frame = capture.read()  # 读取一帧画面
        if not ret:
            break

        # 在每一帧上检测圆环和圆心
        get_center_coordinates(frame)
        img = mass_det(frame)
        cv.namedWindow("img", cv.WINDOW_NORMAL)  # 创建可调节大小的窗口
        cv.resizeWindow("img", 800, 600)  # 调整窗口的初始大小
        cv.imshow("img", img)

        if cv.waitKey(20) == 27:  # 50ms一帧，按下ESC键退出循环
            break

        # cv.imshow("Video", frame)
        # if cv.waitKey(1) == 27:  # 按下ESC键退出循环
        #   break

    capture.release()
    cv.destroyAllWindows()


def get_center_coordinates(src):
    center_x = src.shape[1] // 2
    center_y = src.shape[0] // 2
    print("Center:", center_x, center_y)
    return center_x, center_y


def set_camera_expousre():  # 设置曝光度，但是没有启用
    capture = cv.VideoCapture(0)  # 打开摄像头
    capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    capture.set(cv.CAP_PROP_EXPOSURE, -6)


def QR_code_detect(src):
    qrcode = cv.QRCodeDetector()
    retval, decoded_info, points = qrcode.detectAndDecode(src)
    if retval:
        print(retval)
        return retval


def process_video_QRcode():  # 处理二维码
    capture = cv.VideoCapture(0)  # 打开摄像头
    capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    capture.set(cv.CAP_PROP_EXPOSURE, -100000)  # 曝光度

    while True:
        ret, frame = capture.read()  # 读取一帧画面
        if not ret:
            break

        # 在每一帧上检测二维码

        cv.namedWindow("img", cv.WINDOW_NORMAL)  # 创建可调节大小的窗口
        cv.resizeWindow("img", 800, 600)  # 调整窗口的初始大小
        cv.imshow("img", frame)

        if cv.waitKey(1) == 27:  # 30ms一帧，按下ESC键退出循环
            break
        result = QR_code_detect(frame)
        send_data(numbers_process(result))  # 发送数据
        if result is not None and len(result) > 0:
            break
        # cv.imshow("Video", frame)
        # if cv.waitKey(1) == 27:  # 按下ESC键退出循环
        #   break

    capture.release()
    cv.destroyAllWindows()

def color_det():  # 颜色检测

    blue_lower = np.array([83, 27, 0])
    blue_upper = np.array([111, 255, 255])
    red_lower = np.array([0, 47, 177])
    red_upper = np.array([180, 255, 255])
    green_lower = np.array([54, 47, 35])
    green_upper = np.array([89, 255, 255])

    # 打开摄像头
    cap = cv.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        # 读取摄像头画面
        ret, frame = cap.read()

        # 获取画面中心100*100区域
        height, width, _ = frame.shape
        center_x, center_y = int(width/2), int(height/2)
        roi = frame[center_y-50:center_y+50, center_x-50:center_x+50]

        # 将区域转换为HSV格式
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # 在区域中提取蓝色、红色、绿色区域
        blue_mask = cv.inRange(hsv, blue_lower, blue_upper)
        red_mask = cv.inRange(hsv, red_lower, red_upper)
        green_mask = cv.inRange(hsv, green_lower, green_upper)

        # 计算蓝色、红色、绿色区域的像素数量
        blue_pixels = cv.countNonZero(blue_mask)
        red_pixels = cv.countNonZero(red_mask)
        green_pixels = cv.countNonZero(green_mask)
        color = "unkonw"
        # 判断哪种颜色占据了最多的像素数量
        if blue_pixels > red_pixels and blue_pixels > green_pixels:
            color = "blue"
        elif red_pixels > blue_pixels and red_pixels > green_pixels:
            color = "red"
        elif green_pixels > blue_pixels and green_pixels > red_pixels:
            color = "green"
        else:
            color = "unkonw"
        print(blue_pixels, red_pixels, green_pixels)
        # 在画面中心显示检测结果
        cv.putText(frame, color, (center_x-50, center_y-60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.rectangle(frame, (center_x-50, center_y-50), (center_x+50, center_y+50), (0, 255, 0), 2)
        cv.imshow("frame", frame)

        # 检测键盘输入，按下"q"键或"Esc"键退出程序
        key = cv.waitKey(1)
        if key == ord('q') or key == 27:
            break

        # 释放摄像头资源
        cap.release()

    # 关闭所有OpenCV创建的窗口
    cv.destroyAllWindows()

def line_det():  # 直线检测
    cap = cv.VideoCapture(0)  # 打开摄像头
    new_width =1280
    new_height =960
    cap.set(cv.CAP_PROP_FRAME_WIDTH, new_width)  # 设置摄像头帧宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, new_height)  # 设置摄像头帧高度



    while True:
        left_center_x_list = []  # 存储x坐标
        right_center_x_list = []  # 存储x坐标

        left_center_y_list = []
        right_center_y_list = []  # 存储y坐标
        ret, frame = cap.read()  # 读取视频帧
        #cv2.imshow("img",frame)
        kernel = np.ones((13, 13), np.uint8)
        img_dilation = cv.dilate(frame, kernel, iterations=1)  # 膨胀操作
        img_erosion = cv.erode(frame, kernel, iterations=1)  # 腐蚀操作
        gray = cv.cvtColor(img_erosion, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图
        #cv2.imshow("gray", gray)
        blur = cv.GaussianBlur(gray, (17, 17), 0)  # 高斯模糊
        ret, thresh = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)  # 二值化图像
        #cv2.imshow("thresh", thresh)
        edgeImage = cv.Canny(thresh, 50, 220)  # 边缘检测
        cv.imshow("canny",edgeImage)

        image_width = edgeImage.shape[1]

        # 将图像分成左右两半
        offset_x = image_width // 2
        left_half = edgeImage[:, :offset_x]
        right_half = edgeImage[:, offset_x:]

        # 显示左右两半图像
        #cv2.imshow("Left Half", left_half)
        #cv2.imshow("Right Half", right_half)
        contours_left, hierarchy = cv.findContours(image=left_half, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)  # 寻找轮廓
        contours_right, hierarchy = cv.findContours(image=right_half, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)  # 寻找轮廓
        frame_copy = frame.copy()  # 复制一份原始帧用于绘制结果

        for cnt in contours_right:#右边
            if 6000 < cv.contourArea(cnt) < 40000:  # 选择面积满足条件的轮廓

                (right_center_x, right_center_y), (a, b), angle = cv.fitEllipse(cnt)
                #cv2.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆

                ellipse=(right_center_x + offset_x, right_center_y),(a, b), angle
                right_center_x_list.append(right_center_x + offset_x)  # 将中心点的x坐标添加到列表中
                right_center_y_list.append(right_center_y)  # 将中心点的y坐标添加到列表中
                cv.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆
                #print(f'Circle Center: ({center_x}, {center_y})')

        for cnt in contours_left:#左边
            if 7000 < cv.contourArea(cnt) < 30000:  # 选择面积满足条件的轮廓
                ellipse = cv.fitEllipse(cnt)  # 拟合椭圆
                cv.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆
                (left_center_x, left_center_y), _, _ = ellipse
                left_center_x_list.append(left_center_x)  # 将中心点的x坐标添加到列表中
                left_center_y_list.append(left_center_y)  # 将中心点的y坐标添加到列表中

                #print(f'Circle Center: ({center_x}, {center_y})')

        cv.waitKey(1)

        cv.imshow("frame", frame_copy)  # 显示帧



        # 计算直线的左右端点像素位置
        if len(left_center_x_list)>=2:
          
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
            if cv.waitKey(1) & 0xFF == ord('q'):  # 如果按下 'q' 键则退出循环
                break


    cap.release()  # 释放摄像头
    cv.destroyAllWindows()  # 关闭窗口
    

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


def decode_result(data):  # 除去异常符号 ps：未启用
    if data is not None:
        # print(type(data))
        # print(repr(data))
        new_data = data.replace("＋", "")
        # print(new_data)
        return new_data


def numbers_process(data):  # 处理数据:除去异常符号，提取数字
    if data is not None:
        numbers = re.findall(r'\d+', data)
        result = ''.join(numbers)
        print(result)
        return result


# def data_process():#计算并减小误差

def main():  # 主函数

    while 1:
        request = receive_data()
        print(request)
        if request == b'1':  # 在这里修改命令
            process_video_QRcode()
        if request == b'2':
            circle_det()
        if request == b'3':
            process_video_mass()
        if request == 4:
            print('4')
        if request == 3:
            print('5')


if __name__ == '__main__':
    # process_video_circle()
    # process_video_QRcode()
    main()
    # request=receive_data()
    # print(request)