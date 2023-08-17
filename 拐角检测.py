import numpy as np
import cv2 as cv

filename = r"C:\Users\admin\Downloads\1.jpg"
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)
# result用于标记角点，并不重要
dst = cv.dilate(dst, None)
# 最佳值的阈值，它可能因图像而异。
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv.namedWindow('dst', cv.WINDOW_FREERATIO)
cv.imshow('dst', img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
