import cv2
import numpy as np
import serial
import time

def send_data(data):#串口发送数据
    if data is not None:
        #data=numbers_process(data)
        ser = serial.Serial("/dev/ttyAMA0", 115200)#set up serial
        ser.write(data)#write data

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





def detect():
    
    # 定义蓝色、红色、绿色的HSV值范围
    blue_lower = np.array([83, 27, 0])
    blue_upper = np.array([111, 255, 255])
    red_lower = np.array([0, 47, 177])
    red_upper = np.array([180, 255, 255])
    green_lower = np.array([54, 47, 35])
    green_upper = np.array([89, 255, 255])

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    global send
    send=0
    while True:
        # 读取摄像头画面
        ret, frame = cap.read()

        # 获取画面中心100*100区域
        height, width, _ = frame.shape
        center_x, center_y = int(width/2), int(height/2)
        roi = frame[center_y-50:center_y+50, center_x-50:center_x+50]

        # 将区域转换为HSV格式
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 在区域中提取蓝色、红色、绿色区域
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # 计算蓝色、红色、绿色区域的像素数量
        blue_pixels = cv2.countNonZero(blue_mask)
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        color = "unkonw"
        # 判断哪种颜色占据了最多的像素数量
        cv2.imshow("frame", frame)
        if blue_pixels > red_pixels and blue_pixels > green_pixels:
            color = "blue"
            if send==0:
                
                send_data(b'3')
                send=1
                
                
        elif red_pixels > blue_pixels and red_pixels > green_pixels:
            color = "red"
            if send==0:
                
                
                send_data(b'1')
                send=1
                
                
        elif green_pixels > blue_pixels and green_pixels > red_pixels:
            color = "green"
            if send==0:
                
                
                send_data(b'2')
                send=1
            
        else:
            color = "unkonw"

        #print(blue_pixels, red_pixels, green_pixels)
        # 在画面中心显示检测结果
        cv2.putText(frame, color, (center_x-50, center_y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (center_x-50, center_y-50), (center_x+50, center_y+50), (0, 255, 0), 2)
        cv2.imshow("frame", frame)
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))
        # 检测键盘输入，按下"q"键或"Esc"键退出程序
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or elapsed_time>4:

            break
     # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV创建的窗口
    cv2.destroyAllWindows()    
        
   

def main():  # 主函数
    
    while 1:
        detect()
        request = receive_data()
        send=request
        print(request)
        if request == b'0':  # 在这里修改命令
            request=b'1'
            detect()
            


if __name__ == '__main__':
    # process_video_circle()
    # process_video_QRcode()
    #detect()
    main()
    # request=receive_data()
    # print(request)
    #color_det()
