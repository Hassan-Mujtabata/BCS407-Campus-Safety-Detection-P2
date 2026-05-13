# AI-Based Smart Campus Safety Detection System — Advanced Iteration

**Course:** BCS407 – Artificial Intelligence  
**Institution:** Canadian University Dubai  
**Project:** Model Improvement & Research (Project 2)  

---

## Overview

This repository contains the training scripts, dataset pipeline, and result visualisation tools for the advanced iteration of an AI-based campus safety monitoring system. The system uses YOLOv8 to detect 15 safety-relevant classes across four categories: helmets, fire alarms, emergency exits, and safety signage.

This project builds directly on Project 1, introducing two structured improvement experiments — dataset expansion with label remapping, and enhanced augmentation with a larger backbone — evaluated against the Project 1 baseline.

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
├── remap_labels.py           # Remaps class label IDs in the dataset
├── make_grid.py              # Combines result images into a single grid (used for both experiments)
├── visualize_results.py      # Generates per-class and summary visualisations (used for both experiments)
│
├── train/                    # Experiment 1 dataset
│   ├── data.yaml             # Class definitions and absolute split paths (Exp 1)
│   ├── data_split.yaml       # Split configuration used for Exp 1
│   ├── images/               # All raw images
│   ├── labels/               # All YOLO-format label files
│   ├── train/                # Training split
│   ├── valid/                # Validation split
│   └── test/                 # Test split
│
└── runs/                     # Training outputs
    ├── final_project_model/  # Experiment 1 model outputs
    └── exp2_expanded_dataset/ # Experiment 2 model outputs
```

---

## Models & Dataset

Model weights and datasets are hosted externally due to file size.

### Experiment 1

| Resource | Link |
|----------|------|
| Model weights (`best.pt`) | [Download — Hugging Face]([EXP1_MODEL_LINK]) |
| Dataset | [Download — Hugging Face]([EXP1_DATASET_LINK]) |

### Experiment 2

| Resource | Link |
|----------|------|
| Model weights (`best.pt`) | [Download — Hugging Face]([EXP2_MODEL_LINK]) |
| Dataset (expanded) | [Download — Hugging Face]([EXP2_DATASET_LINK]) |

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

- **Experiment 1:** Dataset split (train/valid/test) was performed directly inside the `train/` folder using `data_split.yaml`.
- **Experiment 2:** Dataset was expanded with additional images targeting underperforming classes. The split was performed separately and is available via the Experiment 2 dataset link above.
- All images comply with ethical guidelines: no identifiable faces, no license plates, no personal data.

---

## Reproducibility

- Random seeds are set at the start of each training script.
- Environment: Windows 11, Python 3.12, CUDA 11.x, Ultralytics YOLOv8
- All hyperparameters are documented within the training scripts.

---

## Academic Integrity

AI writing assistance (Claude by Anthropic) was used for report drafting and README generation, as declared in the submitted report. All experimental results are from genuine training runs.

---

## License

This project was developed for academic purposes at Canadian University Dubai. Not licensed for commercial use.
