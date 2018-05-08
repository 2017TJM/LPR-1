import cv2
import numpy as np
from findRegion import findPlateNumberRegion

def correct_image(img):
    cv2.imshow('card',img)
    cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
    dilation = cv2.dilate(binary, element2, iterations=1)
    dilation2 = cv2.dilate(dilation, element2, iterations=1)
    # cv2.imshow('', dilation2)
    # cv2.waitKey(0)
    # thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow('',thresh)
    # cv2.waitKey(0)

    coords = np.column_stack(np.where(dilation2 > 0))

    angle = cv2.minAreaRect(coords)
    angle = angle[-1]
    if angle < -45:

        angle = -(90 + angle)

    else:

        angle = -angle

    (h, w) = img.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)

    return rotated