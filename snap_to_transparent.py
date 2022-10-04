import cv2
import numpy as np

src = cv2.imread("face_test_stationary_centered.png")

tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,15,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(src)
rgba = [b,g,r, alpha]
dst = cv2.merge(rgba,4)

cv2.imwrite("test.png",dst)
cv2.imshow("test",dst)
cv2.waitKey(0)
