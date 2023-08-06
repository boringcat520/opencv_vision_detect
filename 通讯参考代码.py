import cv2 as cv
import numpy as np
import serial
import time
import re
#import matplotlib.pyplot as plt
import struct


def circle_det():
    cap = cv.VideoCapture(0)  # 打开摄像头
    new_width = 1280
    new_height = 960
    cap.set(cv.CAP_PROP_FRAME_WIDTH, new_width)  # 设置摄像头帧宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, new_height)  # 设置摄像头帧高度
    center_x_list = []  # 存储x坐标
    center_y_list = []  # 存储y坐标
    #cs_code_shuju()
    #c=40
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    #ser.write(c)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    r=b'2\r\n'
    while r==b'2\r\n':
        ret, frame = cap.read()  # 读取视频帧
        #cv.waitKey(1)
        kernel = np.ones((13, 13), np.uint8)
        img_dilation = cv.dilate(frame, kernel, iterations=2)  # 膨胀操作
        img_erosion = cv.erode(frame, kernel, iterations=2)  # 腐蚀操作
        gray = cv.cvtColor(img_erosion, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图
   
        blur = cv.GaussianBlur(gray, (17, 17), 0)  # 高斯模糊
           
        ret, thresh = cv.threshold(blur, 90, 255, cv.THRESH_BINARY )
        #ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY ) # 二值化图像
        #cv.imshow("frame", thresh)
        edgeImage = cv.Canny(thresh, 50, 220)  # 边缘检测
        #cv.imshow("frame1", edgeImage)
        contours, hierarchy = cv.findContours(image=edgeImage, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)  # 寻找轮廓

        frame_copy = frame.copy()  # 复制一份原始帧用于绘制结果

        for cnt in contours:
            if 10000 < cv.contourArea(cnt) < 60000:  # 选择面积满足条件的轮廓
                ellipse = cv.fitEllipse(cnt)  # 拟合椭圆
                cv.ellipse(frame_copy, ellipse, (0, 0, 255), 2)  # 在复制的帧上绘制椭圆
                (center_x, center_y), (major_axis, minor_axis), angle = ellipse
                center_x_list.append(center_x)  # 将中心点的x坐标添加到列表中
                center_y_list.append(center_y)  # 将中心点的y坐标添加到列表中
                #print(center_x,center_y)
               
                center_x_mean = np.mean(center_x_list)  # 计算x坐标的均值
                center_y_mean = np.mean(center_y_list)  # 计算y坐标的均值
                center_x = width/2
                center_y = height/2
                distance_x=center_x_mean-center_x
                distance_y=center_y_mean-center_y
                # 将距离转换为像素点数目
                pixel_distance_x = round(distance_x,3)
                pixel_distance_y = round(distance_y,3)
                data_packet = struct.pack('<2s6f2s', b'\x0D\x0A',0.0, 0.0, 0.0, pixel_distance_x, pixel_distance_y, 0.0, b'\x0A\x0D')
                #ser = serial.Serial("/dev/ttyAMA0", 115200)#set up serial
                #ser.write(data_packet)
                #time.sleep(1)
                send_data(data_packet)
               
                print(pixel_distance_x,pixel_distance_y)
                count2 = ser.inWaiting()
                if count2 != 0:
                    break
               
        #cv.imshow("frame2", frame_copy)
        #time.sleep(16)# 显示帧
        #if r==b'2\r\n':
         #   r=receive_data()
        count1 = ser.inWaiting()
        if count1 != 0:
            break
        if cv.waitKey(1) & 0xFF == ord('q'): # 如果按下 'q' 键则退出循环
            break

    cap.release()  # 释放摄像头
    cv.destroyAllWindows()  # 关闭窗口



def set_camera_expousre():
    capture = cv.VideoCapture(0)  # 打开摄像头
    capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    capture.set(cv.CAP_PROP_EXPOSURE, -6)
   
def QR_code_detect(src):
    qrcode=cv.QRCodeDetector()
    retval,decoded_info,points=qrcode.detectAndDecode(src)
    if retval:
        print(retval)
        return retval
 
   
   
def process_video_QRcode():#处理二维码
    capture = cv.VideoCapture(0)  # 打开摄像头
    capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    capture.set(cv.CAP_PROP_EXPOSURE, -100000)#曝光度

    while True:
        ret, frame = capture.read()  # 读取一帧画面
        if not ret:
            break

        # 在每一帧上检测二维码
       

        cv.namedWindow("img", cv.WINDOW_NORMAL)  # 创建可调节大小的窗口
        cv.resizeWindow("img", 800, 600)  # 调整窗口的初始大小
        cv.imshow("img", frame)
       
        if cv.waitKey(1)== 27:  # 30ms一帧，按下ESC键退出循环
            break
        result=QR_code_detect(frame)
        send_data(numbers_process(result))#发送数据
        if result is not None and len(result)>0:
            break
        #cv.imshow("Video", frame)
        #if cv.waitKey(1) == 27:  # 按下ESC键退出循环
         #   break

    capture.release()
    cv.destroyAllWindows()


def send_data(data):#串口发送数据
    if data is not None:
       
        ser = serial.Serial("/dev/ttyAMA0", 115200)#set up serial
        #ser.write(data.encode('utf-8'))#write data
        ser.write(data)#write data

def receive_data():#串口接收数据
    ser = serial.Serial("/dev/ttyAMA0", 115200)#set up serial

    while True:
        # 获得接收缓冲区字符
        count = ser.inWaiting()#获取接收缓冲区字符的字节长度，返回值为整型，可用于判断是否接收到数据
        if count != 0:
            # 读取内容并回显
            recv = ser.read(count)
            return recv         
        # 清空接收缓冲区
        ser.flushInput()
        # 必要的软件延时
        time.sleep(0.1)
       
       

def decode_result(data):#除去异常符号 ps：未启用
    if data is not None:
        #print(type(data))
        #print(repr(data))
        new_data=data.replace("＋","")
        #print(new_data)
        return new_data

def numbers_process(data):#处理数据:除去异常符号，提取数字
    if data is not None:
        numbers=re.findall(r'\d+',data)
        result=''.join(numbers)
        print(result)
        return result

#def data_process():#计算并减小误差

def cs_code_shuju():
    a=30
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    ser.write(a)

def main():#主函数
   
    while 1:
        request=receive_data()
        print(request)
        if request==b'1\r\n':#在这里修改命令
            process_video_QRcode()
        if request==b'SMP0':
            circle_det()
            print(request)
        if request==b'3':
            cs_code_shuju()
            print('3')
        if request==4:
            print('4')
        if request==3:
            print('5')



if __name__ == '__main__':
   
    #process_video_circle()
    #process_video_QRcode()
    main()
    #request=receive_data()
    #print(request)
