# Polysora Rust Diagnosis

Detects common rust on corn leaves from images using hand-crafted features
(rust colour percentage, texture roughness, pustule density).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The dataset is not stored in the repository. Download it from the
[`Dataset - v1` release](https://github.com/Ydiboras04/Polysora_Rust_Diagnosis/releases/tag/Dataset)
and unzip it at the project root:

```bash
gh release download Dataset
unzip dataset_v1.zip
```

Expected layout:

```
dataset/
├── malades/   # diseased leaves (label 1)
└── saines/    # healthy leaves (label 0)
```

## Usage

```bash
python build_dataframe.py                  # writes features.csv
python build_dataframe.py --max-per-class 50
```
