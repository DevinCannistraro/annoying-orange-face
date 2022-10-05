import cv2
import numpy as np


def makeTransparent(path,outputPath):
    src = cv2.imread(path)

    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,15,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)

    cv2.imwrite(outputPath,dst)
