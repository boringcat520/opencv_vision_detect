import cv2
import numpy as np

def nothing(x):
    pass

# 创建一个空窗口用于阈值编辑器
cv2.namedWindow('Threshold Editor')

# 创建滑条来调整颜色阈值
cv2.createTrackbar('Hue Lower', 'Threshold Editor', 0, 180, nothing)
cv2.createTrackbar('Hue Upper', 'Threshold Editor', 180, 180, nothing)

cv2.createTrackbar('Saturation Lower', 'Threshold Editor', 0, 255, nothing)
cv2.createTrackbar('Saturation Upper', 'Threshold Editor', 255, 255, nothing)

cv2.createTrackbar('Value Lower', 'Threshold Editor', 0, 255, nothing)
cv2.createTrackbar('Value Upper', 'Threshold Editor', 255, 255, nothing)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头画面
    ret, frame = cap.read()

    # 将画面转换为HSV格式
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 复制画面用于显示
    frame_show = frame.copy()

    # 读取滑条值
    hue_lower = cv2.getTrackbarPos('Hue Lower', 'Threshold Editor')
    hue_upper = cv2.getTrackbarPos('Hue Upper', 'Threshold Editor')

    saturation_lower = cv2.getTrackbarPos('Saturation Lower', 'Threshold Editor')
    saturation_upper = cv2.getTrackbarPos('Saturation Upper', 'Threshold Editor')

    value_lower = cv2.getTrackbarPos('Value Lower', 'Threshold Editor')
    value_upper = cv2.getTrackbarPos('Value Upper', 'Threshold Editor')

    # 设置颜色阈值范围
    lower_bound = np.array([hue_lower, saturation_lower, value_lower])
    upper_bound = np.array([hue_upper, saturation_upper, value_upper])

    # 在HSV画面中提取选定颜色范围
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 将mask应用到原始画面上，保留选定颜色，其他颜色变为黑色
    result = cv2.bitwise_and(frame_show, frame_show, mask=mask)

    # 显示阈值编辑器和画面
    cv2.imshow('Threshold Editor', mask)
    cv2.imshow('Camera View', result)

    # 检测键盘输入，按下"q"键退出程序
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
