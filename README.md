# AI-Based Smart Campus Safety Detection System — Advanced Iteration

**Course:** BCS407 – Artificial Intelligence  
**Institution:** Canadian University Dubai  
**Project:** Model Improvement & Research (Project 2)  
**Baseline Repository (Project 1):** [safety-detection-yolov8](https://github.com/Hassan-Mujtabata/safety-detection-yolov8)

---

## Overview

This repository contains the training scripts, dataset pipeline, and result visualisation tools for the advanced iteration of an AI-based campus safety monitoring system. The system uses YOLOv8 to detect 15 safety-relevant classes across four categories: helmets, fire alarms, emergency exits, and safety signage.

This project builds directly on Project 1, introducing two structured improvement experiments — dataset expansion with label remapping, and enhanced augmentation with a larger backbone — each evaluated against the Project 1 baseline.

---

## Detected Classes

| # | Class |
|---|-------|
| 0 | Fire alarm |
| 1 | Left Exit |
| 2 | Left/Right Exit |
| 3 | Right Exit |
| 4 | Straight Exit |
| 5 | blocked emergency exit |
| 6 | boxes blocking exit |
| 7 | emergency exit door |
| 8 | exit block violation |
| 9 | exit-sign |
| 10 | head |
| 11 | helmet |
| 12 | objects blocked emergency exit |
| 13 | person |
| 14 | sign |

---

## Repository Structure

```
BCS407-Campus-Safety-Detection-P2/
│
├── Train.py                  # Experiment 1 training script
├── Train_Exp2.py             # Experiment 2 training script
├── remap_labels.py           # Remaps class label IDs to match the unified 15-class schema
│                             # (used when merging the additional dataset into Exp 2)
├── make_grid.py              # Combines result images into a single grid (both experiments)
├── visualize_results.py      # Generates per-class and summary visualisations (both experiments)
├── run_model.py
│
├── train/                    # Experiment 1 dataset
│   ├── data.yaml             # Class definitions and split paths for Exp 1
│   ├── images/               # All raw images
│   ├── labels/               # All YOLO-format label files
│   ├── train/                # Training split
│   ├── valid/                # Validation split
│   └── test/                 # Test split
│
│
└── runs/                     # Training outputs
    ├── final_project_model/  # Experiment 1 model outputs
    └── exp2_expanded_dataset/ # Experiment 2 model outputs
```

---

## Models & Datasets

All model weights and datasets are hosted on Hugging Face.

### Experiment 1

| Resource | Description | Link |
|----------|-------------|------|
| `best.pt` | Best checkpoint from Exp 1 training | [Download](https://huggingface.co/Hassanmujtabat/campus-safety-detection-exp1/blob/main/best.pt) |
| `last.pt` | Final epoch checkpoint from Exp 1 | [Download](https://huggingface.co/Hassanmujtabat/campus-safety-detection-exp1/blob/main/last.pt) |
| Dataset | Original dataset used for Exp 1 (YOLOv8 format) | [Download](https://huggingface.co/Hassanmujtabat/campus-safety-detection-exp1/blob/main/safety%20helmet.yolov8.zip) |

### Experiment 2

| Resource | Description | Link |
|----------|-------------|------|
| `best.pt` | Best checkpoint from Exp 2 training | [Download](https://huggingface.co/Hassanmujtabat/campus-safety-detection-exp2/blob/main/best.pt) |
| `last.pt` | Final epoch checkpoint from Exp 2 | [Download](https://huggingface.co/Hassanmujtabat/campus-safety-detection-exp2/blob/main/last.pt) |
| Additional dataset | Emergency exit images added for Exp 2 | [Download](https://huggingface.co/Hassanmujtabat/campus-safety-detection-exp2/blob/main/Emergency%20Exit%20Signs.yolov8.zip) |
| Full combined dataset | All images and labels merged, labels remapped and corrected | [Download](https://huggingface.co/Hassanmujtabat/campus-safety-detection-exp2/blob/main/train.rar) |

---

## Requirements

- Python 3.12
- PyTorch (CUDA-enabled)
- Ultralytics YOLOv8
- OpenCV, Matplotlib, Pandas, PyYAML

Install dependencies:

```bash
pip install ultralytics opencv-python matplotlib pandas pyyaml
```

---

## How to Run

> All scripts must be run using **Python 3.12**.

### Experiment 1 — Train

```bash
py -3.12 Train.py
```

### Experiment 2 — Train

```bash
py -3.12 Train_Exp2.py
```

### Remap Dataset Labels

Run this if you are merging the additional dataset with the original and need to realign class IDs to the unified 15-class schema:

```bash
py -3.12 remap_labels.py
```

### Generate Result Visualisations (both experiments)

```bash
py -3.12 visualize_results.py
```

### Create Results Grid Image (both experiments)

```bash
py -3.12 make_grid.py
```

---

## Dataset Notes

- **Experiment 1:** Uses the original Project 1 dataset. Train/validation/test split is handled within the `train/` folder as defined in `data.yaml`.
- **Experiment 2:** Expands the dataset with additional emergency exit images. Labels from the added dataset were remapped using `remap_labels.py` to align with the unified 15-class schema. The split for Exp 2 was performed separately from Exp 1.
- All images comply with ethical guidelines: no identifiable faces, no license plates, no personal data.

---

## Reproducibility

- Random seeds are set at the start of each training script.
- Environment: Windows 11, Python 3.12, CUDA-enabled GPU, Ultralytics YOLOv8
- All hyperparameters are documented within the training scripts.

---

## Academic Integrity

All experimental results are from genuine training runs. All submitted work complies with the academic integrity policy of Canadian University Dubai.

---

## License

This project was developed for academic purposes at Canadian University Dubai. Not licensed for commercial use.
