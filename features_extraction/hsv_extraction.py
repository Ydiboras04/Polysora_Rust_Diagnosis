import cv2
import numpy as np

from load_dataset import iterate_dataset

LEAF_LOWER = np.array([5, 40, 40])
LEAF_UPPER = np.array([90, 255, 255])

RUST_LOWER = np.array([5, 80, 50])
RUST_UPPER = np.array([28, 255, 255])

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def to_hsv(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def leaf_mask(hsv):
    mask = cv2.inRange(hsv, LEAF_LOWER, LEAF_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    return mask

def rust_mask(hsv, leaf):
    rust = cv2.inRange(hsv, RUST_LOWER, RUST_UPPER)
    return cv2.bitwise_and(rust, leaf)

def pct_rouille(img_bgr):
    hsv = to_hsv(img_bgr)
    leaf = leaf_mask(hsv)
    rust = rust_mask(hsv, leaf)

    n_leaf = np.count_nonzero(leaf)
    if n_leaf == 0:
        return 0.0
    return np.count_nonzero(rust) / n_leaf