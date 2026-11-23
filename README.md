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
[`dataset-v1` release](https://github.com/Ydiboras04/Polysora_Rust_Diagnosis/releases/tag/dataset-v1)
and unzip it at the project root:

```bash
gh release download dataset-v1
unzip dataset-v1.zip
```

Expected layout:

```
dataset/
├── Common_Rust/   # diseased leaves (label 1)
└── saines/        # healthy leaves (label 0)
```

## Usage

```bash
python build_dataframe.py                  # writes features.csv
python build_dataframe.py --max-per-class 50
```
