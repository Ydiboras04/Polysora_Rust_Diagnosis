import argparse
from pathlib import Path

import cv2
import numpy as np

from hsv_extraction import leaf_mask, pct_rouille, rust_mask, to_hsv
from load_dataset import iterate_dataset
from pustule_extraction import densite_pustules
from texture_rugosity_extraction import inner_leaf_mask, rugosite, sobel_magnitude

# Features computed for each image
FEATURES = {
    "pct_rouille": pct_rouille,
    "rugosite": rugosite,
    "densite_pustules": densite_pustules,
}


def save_debug(image_id, img_bgr, out_dir):
    """Save [original | leaf mask | rust in red | Sobel gradient] side by
    side to check visually that the features behave as expected."""
    hsv = to_hsv(img_bgr)
    leaf = leaf_mask(hsv)
    rust = rust_mask(hsv, leaf)

    leaf_vis = cv2.cvtColor(leaf, cv2.COLOR_GRAY2BGR)

    overlay = img_bgr.copy()
    overlay[rust > 0] = (0, 0, 255)          # rust pixels painted red

    # Gradient shown only inside the leaf, scaled to 0-255 for display
    mag = sobel_magnitude(img_bgr)
    mag[inner_leaf_mask(img_bgr) == 0] = 0
    mag_vis = cv2.convertScaleAbs(mag, alpha=255.0 / max(mag.max(), 1e-6))
    mag_vis = cv2.cvtColor(mag_vis, cv2.COLOR_GRAY2BGR)

    panel = np.hstack([img_bgr, leaf_vis, overlay, mag_vis])
    text = (f"pct_rouille={pct_rouille(img_bgr):.3f}  "
            f"rugosite={rugosite(img_bgr):.1f}  "
            f"densite={densite_pustules(img_bgr):.2f}")
    cv2.putText(panel, text, (10, 25),
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

    values = {name: {0: [], 1: []} for name in FEATURES}
    for image_id, img, label in iterate_dataset(args.dataset, args.max_per_class):
        for name, func in FEATURES.items():
            values[name][label].append(func(img))

    for name in FEATURES:
        print(f"\n{name}")
        for label, cls in [(0, "healthy"), (1, "diseased")]:
            v = np.array(values[name][label])
            print(f"  {cls:8s} (n={len(v)}): mean={v.mean():.3f}  "
                  f"median={np.median(v):.3f}  min={v.min():.3f}  max={v.max():.3f}")


if __name__ == "__main__":
    main()