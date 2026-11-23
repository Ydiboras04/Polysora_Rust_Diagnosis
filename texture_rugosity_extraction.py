import cv2
import numpy as np

from hsv_extraction import leaf_mask, to_hsv

BORDER_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

def sobel_magnitude(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)

def inner_leaf_mask(img_bgr):
    leaf = leaf_mask(to_hsv(img_bgr))
    return cv2.erode(leaf, BORDER_KERNEL)

def rugosite(img_bgr):
    magnitude = sobel_magnitude(img_bgr)
    inner = inner_leaf_mask(img_bgr)
    values = magnitude[inner > 0]
    if values.size == 0:
        return 0.0
    return float(values.mean())