import argparse
import time

import pandas as pd

from features import FEATURES
from load_dataset import iterate_dataset, list_images

COLUMNS = ["ID_image", *FEATURES.keys(), "label_malade"]

def build_dataframe(dataset_dir, max_per_class=None):
    total = len(list_images(dataset_dir, max_per_class))
    rows = []
    start = time.time()

    for i, (image_id, img, label) in enumerate(
        iterate_dataset(dataset_dir, max_per_class), start=1
    ):
        row = {"ID_image": image_id}
        for name, func in FEATURES.items():
            row[name] = func(img)
        row["label_malade"] = label
        rows.append(row)

        if i % 50 == 0 or i == total:
            print(f" {i}/{total} images processed ({time.time() - start:.0f}s)")
    return pd.DataFrame(rows, columns=COLUMNS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--out", default="features.csv")
    args = parser.parse_args()

    print("Extracting features...")
    df = build_dataframe(args.dataset, args.max_per_class)

    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}\n")

    print("First rows:")
    print(df.head().to_string(index=False))

    print("\nClass balance:")
    print(df["label_malade"].value_counts().rename({0: "healthy (0)", 1: "diseased (1)"}))

    print("\nMean of each feature by class:")
    print(df.groupby("label_malade")[list(FEATURES)].mean().round(3)) 

if __name__ == "__main__":
    main()