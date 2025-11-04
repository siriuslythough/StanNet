# StanNet: Fully Complex CNNs for Generic Image Classification

_A EE604 Course Project — IIT Kanpur_

> End‑to‑end PyTorch pipeline built around **StanNet**, our custom **fully complex‑valued convolutional architecture** for general image classification. Includes CLI training, one‑shot image validation, and a lightweight Tkinter GUI. Baselines (ResNet/VGG/AlexNet) are provided for comparison.

---

## Table of Contents
- [Features](#features)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Dataset](#dataset)
- [Quickstart](#quickstart)
- [Training Options](#training-options)
- [Validate / Inference](#validate--inference)
- [GUI (Tkinter)](#gui-tkinter)
- [Implementation Notes (Complex CNNs)](#implementation-notes-complex-cnns)
- [Reproducibility](#reproducibility)
- [Results & Tracking](#results--tracking)
- [Troubleshooting](#troubleshooting)
- [License & Usage](#license--usage)
- [Citation](#citation)
- [Authors](#authors)

---

## Features
- **StanNet**: our custom **fully complex‑valued CNN** designed to handle complex color representations and phase information for **any multi‑class image classification** task.
- **Baselines included** for apples‑to‑apples comparisons (ResNet/VGG/AlexNet).
- **Single entry script** for training with clean CLI and sensible defaults.
- **Robust augmentations** and stratified train/val split for fair evaluation.
- **One‑shot validator** for running a saved model on a single image from the CLI.
- **Simple GUI** (Tkinter) for interactive demos.

## Repository Layout
```
.
├── main.py                  # Train models (complex & real-valued baselines)
├── validation.py            # Load a saved model and classify a single image
├── UI_main.py               # Tkinter-based demo UI
├── utils.py                 # HSV → complex image transforms, progress bar, helpers
├── networks.py              # Backbone definitions / wiring
├── complexnn.py             # Complex-valued layers & ops
├── complex_activations.py   # Complex activation functions
├── stanNet.py               # StanNet: our primary custom fully complex CNN architecture
├── requirements.txt         # Exact versions pinned for reproducibility
├── README.md                # This file
└── YogaPoses/               # Dataset root (download separately)
```

> **Tip:** Keep trained weights in `saved_models/` (created automatically by `main.py`).

## Setup
1. **Python**: 3.8+ recommended.
2. (Optional) Create a virtual environment.
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3. **Install deps**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
Bring your own dataset structured as **class‑per‑folder** (ImageFolder‑style):
```
DATASET_ROOT/
  ├── ClassA/
  │    ├── img_0001.jpg
  │    └── ...
  ├── ClassB/
  └── ...
```
Use the path to this folder with `--root`.

> Example source: public YogaPoses datasets work out‑of‑the‑box, but StanNet is not restricted to poses.

## Quickstart
### Train (example)
```bash
python main.py \
  --root DATASET_ROOT \
  --arch stannet \
  --batch_size 16 \
  --epochs 50 \
  --lr 3e-4 \
  --modelName stannet_v1
```
This will create `saved_models/stannet_v1_*.pth` and print training/validation metrics epoch‑wise.

## Training Options
`main.py` accepts the following key arguments (see in‑file help for full list):

| Flag | Type | Default | Notes |
|---|---|---:|---|
| `--root` | str | **required** | Path to dataset root (folder with class subfolders). |
| `--arch` | str | `resnet18` | One of: `stannet`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`, `wide_resnet50_2`, `wide_resnet101_2`, `alexnet`, `vgg11`, `vgg13`, `vgg16`, `vgg19`, `cds_e` |
| `--image_size` | int | `224` | Input image size. |
| `--val_ratio` | float | `0.2` | Validation split ratio (stratified). |
| `--batch_size` | int | `16` | Batch size. |
| `--epochs` | int | `40` | Number of epochs. |
| `--lr` | float | `3e-4` | Learning rate. |
| `--weight_decay` | float | `1e-5` | Weight decay (L2). |
| `--num_workers` | int | `4` | DataLoader workers. |
| `--lamb_phase` | float | `0.05` | Weight for phase loss term (set `0.0` to disable). |
| `--strong_aug` | flag | _off_ | Enable stronger spatial augmentations (flip/rotation). |
| `--seed` | int | `42` | Random seed for split & training. |
| `--out_dir` | str | `saved_models/` | Output directory for checkpoints. |
| `--modelName` | str | `best_model` | Base name for saved checkpoint(s). |
| `--num_params` | set/unset | `not required` | returns the number of parameter of --arch |

> Note: You can also specify CIFAR10 as the value for the --root argument. The script will then automatically download the CIFAR-10 dataset and proceed to train the model on it.

To check the number of parameter of a specific architecture
```bash
python main.py \
  --root DATASET_ROOT \
  --arch stanet \
  --num_params
```
`Note`  --root is required because no. of parameters depends on number of classes in the dataset.

> Device is auto‑detected (CUDA if available then MPS if available otherwise CPU) inside the scripts.

## Validate / Inference
Classify a single image with a saved model:
```bash
python validation.py --model_dir ./saved_models --image path/to/test.jpg
```
Useful options:
```bash
python validation.py --model_dir ./saved_models --image test.jpg --model_name best.pth
python validation.py --model_dir ./saved_models --image test.jpg --device cuda
python validation.py --list-archs   # Show available architectures
```
The script prints Top‑K predictions with probabilities.

## GUI (Tkinter)
Run a simple desktop demo:
```bash
python UI_main.py
```
Select a checkpoint from `saved_models/` and an image file; predictions appear in the UI.

## Implementation Notes (Complex CNNs)
- **StanNet (primary):** Implemented in `stanNet.py`, built from complex layers in `complexnn.py` and activations in `complex_activations.py`.
- **Complex image construction:** RGB → HSV (custom), then stacked into a 3‑channel **complex** tensor using `ToHSV` and `ToComplex` from `utils.py`.
- **Backbones:** Real‑valued baselines (ResNet/VGG/AlexNet) provided alongside `stannet` for ablations.
- **Loss:** Magnitude classification loss + optional **phase consistency** term weighted by `--lamb_phase`.
- **Transforms:** Strong augments gateable via `--strong_aug`.

## Reproducibility
- Use `--seed` for deterministic data splits; PyTorch/CUDA determinism flags are set where sensible.
- All package versions are pinned in `requirements.txt` for reproducible installs.

## Results & Tracking
- Store checkpoints in `saved_models/` (default).
- If you enable experiment tracking (e.g., W&B listed in `requirements.txt`), integrate it inside `main.py` as needed.

## Troubleshooting
- **Out of memory:** Reduce `--batch_size` or `--image_size`.
- **Class imbalance:** Adjust `--val_ratio`, consider weighted loss/sampler.
- **Slow data loading:** Increase `--num_workers` (balance with CPU cores).
- **CUDA not used:** Ensure PyTorch + CUDA build, or run with CPU (auto‑fallback).

## License & Usage
This repository is for **academic use only** as part of IIT Kanpur EE604 coursework. Unauthorised commercial use is prohibited. © 2024 Mohd Amir, Nishant Pandey & Tanmay Siddharth. All rights reserved.

## Citation
> S. Yadav and K. R. Jerripothula, “FCCNs: Fully Complex‑valued Convolutional Networks using Complex‑valued Color Model and Loss Function,” ICCV 2023.

## Authors
- Mohd Amir — mmamir22@iitk.ac.in
- Nishant Pandey — nishantp22@iitk.ac.in
- Tanmay Siddharth — tanmays22@iitk.ac.in

---

