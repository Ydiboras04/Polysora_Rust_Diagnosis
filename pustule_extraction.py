import cv2
import numpy as np

from hsv_extraction import leaf_mask, rust_mask, to_hsv

# Spots smaller than this (in pixels) are considered noise, not pustules
MIN_PUSTULE_AREA = 3


def find_pustules(rust):
    # 8-connectivity: pixels touching by a side OR a corner are in the same group
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rust, connectivity=8)
    # Label 0 is the background (black pixels): skip it
    areas = stats[1:, cv2.CC_STAT_AREA]
    return areas[areas >= MIN_PUSTULE_AREA]


def densite_pustules(img_bgr):
    hsv = to_hsv(img_bgr)
    leaf = leaf_mask(hsv)
    rust = rust_mask(hsv, leaf)

    n_leaf = np.count_nonzero(leaf)
    if n_leaf == 0:          # no leaf detected: avoid dividing by zero
        return 0.0
    n_pustules = len(find_pustules(rust))
    return n_pustules / n_leaf * 1000
