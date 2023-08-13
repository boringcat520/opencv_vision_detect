
import cv2
import numpy as  np
import math
#import serial

# lower_blue = np.array([110, 50, 50])
# upper_blue = np.array([130, 255, 255])
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()#读取图像一帧
    # load the image, convert it to grayscale, blur it slightly,
    # and threshold it
    # frame = cv2.imread('wan0.png')
    #frame= np.array(frame0[::-1])#摄像头安装倒置，沿X轴反转图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#opencv读取的为BGR格式，转化为灰度
    cv2.imshow("arg",gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)#高斯滤波算法
    thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)[1]#二值化
    cv2.imshow("erzhi", thresh)
    # cv2.imshow("二值化",thresh)
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)#找寻边界
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = cnts[1]
    # loop over the contours
    for c in cnts:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):#q 跳出循环
        cv2.imwrite("8080.jpg", frame)
        break
cv2.destroyAllWindows()#关闭所有窗口