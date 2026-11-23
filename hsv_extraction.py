import argparse
from pathlib import Path

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

def save_debug(image_id, img_bgr, out_dir):
    hsv = to_hsv(img_bgr)
    leaf = leaf_mask(hsv)
    rust = rust_mask(hsv, leaf)

    leaf_vis = cv2.cvtColor(leaf, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    overlay[rust > 0] = (0, 0, 255)          # rust pixels painted red

    panel = np.hstack([img_bgr, leaf_vis, overlay])
    pct = pct_rouille(img_bgr)
    cv2.putText(panel, f"pct_rouille={pct:.3f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_dir.mkdir(exist_ok=True)
    name = image_id.replace("/", "_") + ".png"
    cv2.imwrite(str(out_dir / name), panel)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--debug", type=int, default=0,
                        help="Save the masks of N images per class in debug_masks/")
    args = parser.parse_args()

    if args.debug:
        out_dir = Path("debug_masks")
        saved = {0: 0, 1: 0}
        for image_id, img, label in iterate_dataset(args.dataset, max_per_class=50):
            if saved[label] < args.debug:
                save_debug(image_id, img, out_dir)
                saved[label] += 1
            if all(v >= args.debug for v in saved.values()):
                break
        print(f"Debug images saved in {out_dir}/")
        return

    values = {0: [], 1: []}
    for image_id, img, label in iterate_dataset(args.dataset, args.max_per_class):
        values[label].append(pct_rouille(img))

    for label, name in [(0, "malades"), (1, "saines")]:
        v = np.array(values[label])
        print(f"{name:8s} (n={len(v)}): pct_rouille mean={v.mean():.3f}  "
              f"median={np.median(v):.3f}  min={v.min():.3f}  max={v.max():.3f}")


if __name__ == "__main__":
    main()