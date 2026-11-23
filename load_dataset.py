import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = {
    "saines": 0,
    "malades": 1,
}

EXTENSIONS = {
    ".jpg", 
    ".jpeg",
    ".png"
}

IMG_SIZE = (256, 256)

#only list dataset that will be used (never load image)
def list_images(dataset_dir, max_per_class=None, seed=42):
    dataset_dir = Path(dataset_dir)
    per_class = {}

    for folder, label in CLASSES.items():
        class_dir = dataset_dir / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing Folder: {class_dir}")
        paths = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
        per_class[label] = paths

    if max_per_class is not None:
        n = min(max_per_class, *(len(p) for p in per_class.values()))
        rng = random.Random(seed)
        per_class = {
            label: sorted(rng.sample(paths, n))
            for label, paths in per_class.items()    
        }

    return [
        (path, label)
        for label, paths in per_class.items()    
        for path in paths
    ]

def load_image(path, size=IMG_SIZE):
    img = cv2.imread(str(path))
    if img is None:
        return None

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

#hands back the images from list_images one at a time
def iterate_dataset(dataset_dir, max_per_class=None, seed=42):
    for path, label in list_images(dataset_dir, max_per_class, seed):
        img = load_image(path)
        if img is None:
            print(f"[WARNING] unreadable image skipeed: {path}")
            continue
        yield f"{path.parent.name}/{path.stem}", img, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--max-per-class", type=int, default=None, help="Balance the classes by limiting images per class")
    args = parser.parse_args()

    items = list_images(args.dataset, args.max_per_class)
    labels = np.array([label for _, label in items])
    print(f"Image selected: {len(items)}")
    print(f" saine (0): {np.sum(labels == 0)}")
    print(f" malade (1): {np.sum(labels == 1)}")

    for i, (image_id, img, label) in enumerate (iterate_dataset(args.dataset, args.max_per_class)):
        print(  
            f"{image_id:30s} label={label} shape={img.shape}" 
            f"mean BGR={np.round(img.mean(axis=(0, 1)), 1)}"
        )

        if i == 4:
            break

if __name__ == "__main__":
    main()