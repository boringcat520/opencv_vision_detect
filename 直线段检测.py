import cv2
import numpy as np

def update_params(_):
    global scale, sigma_scale, quant, ang_th, log_eps
    
    scale = cv2.getTrackbarPos('scale', 'Detected Line Segments') / 10.0
    sigma_scale = cv2.getTrackbarPos('sigma_scale', 'Detected Line Segments') / 10.0
    quant = cv2.getTrackbarPos('quant', 'Detected Line Segments') / 10.0
    ang_th = cv2.getTrackbarPos('ang_th', 'Detected Line Segments')
    log_eps = cv2.getTrackbarPos('log_eps', 'Detected Line Segments') / 100.0
    
    lsd = cv2.createLineSegmentDetector(scale=scale, sigma_scale=sigma_scale, quant=quant, ang_th=ang_th, log_eps=log_eps)
    
    lines, width, _, _ = lsd.detect(gray)
    
    image_with_lines = image.copy()
    for line in lines:
        x1, y1, x2, y2 = np.int0(line[0])
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow('Detected Line Segments', image_with_lines)

# 读取图像
image = cv2.imread(r"C:\Users\admin\Downloads\1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 创建窗口和滑动条
cv2.namedWindow('Detected Line Segments', cv2.WINDOW_FREERATIO)
cv2.namedWindow('Detected Line Segments')
cv2.createTrackbar('scale', 'Detected Line Segments', 8, 20, update_params)
cv2.createTrackbar('sigma_scale', 'Detected Line Segments', 6, 20, update_params)
cv2.createTrackbar('quant', 'Detected Line Segments', 10, 20, update_params)
cv2.createTrackbar('ang_th', 'Detected Line Segments', 15, 90, update_params)
cv2.createTrackbar('log_eps', 'Detected Line Segments', 10, 100, update_params)

scale = 0.8
sigma_scale = 0.6
quant = 1.0
ang_th = 15
log_eps = 0.1

update_params(None)  # 初始更新一次

cv2.waitKey(0)
cv2.destroyAllWindows()
