"""
Emergency Exit Detection — YOLOv8l Experiment 2: Expanded Dataset
=================================================================
Changes from Exp 1:
 - Seed fixed to 42 for full reproducibility
 - Fresh start from yolov8l.pt (no checkpoint resume)
 - Expanded dataset (2,167 images vs original 1,386)
 - Flat dataset auto-split into train/valid/test
 - Project name: exp2_expanded_dataset
Everything else identical to Exp 1 for isolation.
"""

import os
import sys
import shutil
import random
import time
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import yaml

# ─── REPRODUCIBILITY SEED ─────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

# ─── INSTALL DEPS IF MISSING ──────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    os.system(f"{sys.executable} -m pip install ultralytics --quiet")
    from ultralytics import YOLO

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
DATASET_ROOT  = Path(r"C:\Users\sands\Downloads\Artificial Intelligence\final project\train")
DATA_YAML     = Path(r"C:\Users\sands\Downloads\Artificial Intelligence\final project\data.yaml")
SPLIT_RATIOS  = (0.70, 0.20, 0.10)
EPOCHS        = 50
PATIENCE      = 10
MIN_WARN      = 30
PROJECT       = "runs"
NAME          = "exp2_expanded_dataset"
IMGSZ         = 640
# ─────────────────────────────────────────────────────────────────────────────

LOG_CSV        = Path(PROJECT) / NAME / "epoch_log.csv"
EXCLUSION_FILE = Path(PROJECT) / NAME / "excluded_images.txt"

_state = {
    "low_util_streak": 0,
    "csv_file": None,
    "csv_writer": None,
    "epoch_t0": None,
    "train_t0": None,
}


# ════════════════════════════════════════════════════════════════════════════
# 1. GPU HELPERS
# ════════════════════════════════════════════════════════════════════════════

def get_gpu_info():
    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU found — training on CPU (very slow).")
        return "cpu", 0.0
    idx = 0
    name = torch.cuda.get_device_name(idx)
    vram = torch.cuda.get_device_properties(idx).total_memory / 1e9
    print(f"✅ GPU : {name}")
    print(f"   VRAM: {vram:.1f} GB")
    return "0", vram


def gpu_utilization() -> int:
    if NVML_AVAILABLE:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        except Exception:
            pass
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return int(used / total * 100)
    return 0


def vram_usage():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return used, total
    return 0.0, 0.0


# ════════════════════════════════════════════════════════════════════════════
# 2. DATASET HELPERS
# ════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def all_images(folder: Path):
    return [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS]


def has_annotation(img: Path, labels_dir: Path) -> bool:
    lbl = labels_dir / (img.stem + ".txt")
    if not lbl.exists():
        return False
    return bool(lbl.read_text().strip())


def load_exclusions() -> set:
    if EXCLUSION_FILE.exists():
        return set(EXCLUSION_FILE.read_text().splitlines())
    return set()


def save_exclusions(names: set):
    EXCLUSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXCLUSION_FILE.write_text("\n".join(sorted(names)))


def is_flat(root: Path) -> bool:
    return (root / "images").exists() and not (root / "train").exists()


def split_and_organise(root: Path, ratios=(0.70, 0.20, 0.10), seed=42):
    images_dir = root / "images"
    labels_dir = root / "labels"

    imgs      = all_images(images_dir)
    annotated = [i for i in imgs if has_annotation(i, labels_dir)]
    unanno    = [i for i in imgs if not has_annotation(i, labels_dir)]
    excluded  = load_exclusions()

    print(f"\n📂 Dataset scan:")
    print(f"   Total        : {len(imgs)}")
    print(f"   Annotated    : {len(annotated)}")
    print(f"   Unannotated  : {len(unanno)}  → test only")
    print(f"   Excluded     : {len(excluded)} (zero-detection from prior run)")

    annotated = [i for i in annotated if i.name not in excluded]

    random.seed(seed)
    random.shuffle(annotated)
    n       = len(annotated)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    splits = {
        "train": annotated[:n_train],
        "valid": annotated[n_train: n_train + n_val],
        "test" : annotated[n_train + n_val:] + unanno,
    }

    for split_name, split_imgs in splits.items():
        img_out = root / split_name / "images"
        lbl_out = root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in split_imgs:
            dst = img_out / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            lbl_src = labels_dir / (img.stem + ".txt")
            if lbl_src.exists():
                lbl_dst = lbl_out / lbl_src.name
                if not lbl_dst.exists():
                    shutil.copy2(lbl_src, lbl_dst)

    print(f"\n✅ Split: Train={len(splits['train'])}  "
          f"Valid={len(splits['valid'])}  Test={len(splits['test'])}")
    return splits


def make_split_yaml(root: Path, src_yaml: Path) -> Path:
    with open(src_yaml) as f:
        data = yaml.safe_load(f)
    data["train"] = str((root / "train" / "images").resolve())
    data["val"]   = str((root / "valid" / "images").resolve())
    data["test"]  = str((root / "test"  / "images").resolve())
    out = root / "data_split.yaml"
    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUTION REPORT
# ════════════════════════════════════════════════════════════════════════════

def class_distribution(root: Path, class_names: list) -> dict:
    report = {}
    for split in ("train", "valid", "test"):
        ldir = root / split / "labels"
        counts   = Counter()
        n_images = 0
        if ldir.exists():
            files    = list(ldir.glob("*.txt"))
            n_images = len(files)
            for f in files:
                for line in f.read_text().splitlines():
                    parts = line.strip().split()
                    if parts:
                        counts[int(parts[0])] += 1
        report[split] = {"n_images": n_images, "counts": counts}
    return report


def print_report(report: dict, class_names: list):
    W = 65
    print("\n" + "=" * W)
    print("📊  DATASET DISTRIBUTION REPORT")
    print("=" * W)
    print(f"{'Class':<33} {'Train':>7} {'Valid':>7} {'Test':>7} {'Total':>7}")
    print("-" * W)
    for cid, name in enumerate(class_names):
        tr = report["train"]["counts"].get(cid, 0)
        va = report["valid"]["counts"].get(cid, 0)
        te = report["test"]["counts"].get(cid, 0)
        print(f"{name:<33} {tr:>7} {va:>7} {te:>7} {tr+va+te:>7}")
    print("-" * W)
    print(f"{'Images':<33} "
          f"{report['train']['n_images']:>7} "
          f"{report['valid']['n_images']:>7} "
          f"{report['test']['n_images']:>7}")
    print("=" * W)


def warn_minority(report: dict, class_names: list, threshold: int):
    print(f"\n⚠️  Minority class check (threshold={threshold} train samples):")
    ok = True
    for cid, name in enumerate(class_names):
        n = report["train"]["counts"].get(cid, 0)
        if n < threshold:
            print(f"   ⚠️  '{name}': {n} train samples — risk of underfitting")
            ok = False
    if ok:
        print("   ✅ All classes above threshold.")


def class_weights(report: dict, n_classes: int) -> list:
    counts = [max(1, report["train"]["counts"].get(i, 0)) for i in range(n_classes)]
    total  = sum(counts)
    w      = [total / (n_classes * c) for c in counts]
    mx     = max(w)
    return [round(x / mx, 4) for x in w]


def estimate_time(n_train: int, batch: int, epochs: int, vram_gb: float):
    ms_per_img = 3.0 if vram_gb >= 8 else 3.5
    total_s = (n_train * ms_per_img * epochs) / 1000
    lo, hi  = total_s / 60, total_s * 1.4 / 60
    print(f"\n⏱️  Estimated training time: {lo:.0f}–{hi:.0f} min  "
          f"({n_train} imgs × {epochs} epochs, batch {batch})")


# ════════════════════════════════════════════════════════════════════════════
# 4. CALLBACKS
# ════════════════════════════════════════════════════════════════════════════

def cb_train_start(trainer):
    _state["train_t0"] = time.time()
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOG_CSV, "w", newline="")
    w = csv.writer(f)
    w.writerow(["epoch", "box_loss", "cls_loss", "dfl_loss",
                "precision", "recall", "mAP50", "mAP50-95",
                "gpu_util_%", "vram_used_gb", "epoch_s"])
    _state["csv_file"]   = f
    _state["csv_writer"] = w
    print(f"\n📝 Epoch log → {LOG_CSV}")


def cb_epoch_start(trainer):
    _state["epoch_t0"] = time.time()


def cb_epoch_end(trainer):
    epoch   = trainer.epoch + 1
    util    = gpu_utilization()
    vu, vt  = vram_usage()
    elapsed = time.time() - _state["epoch_t0"]

    if util < 50:
        _state["low_util_streak"] += 1
        streak = _state["low_util_streak"]
        print(f"\n⚠️  GPU util {util}% at epoch {epoch} (streak={streak})")
        if streak >= 2:
            torch.cuda.empty_cache()
            os.environ["OMP_NUM_THREADS"] = "4"
            print("   🔧 Auto-adjust: cache cleared, OMP_NUM_THREADS=4")
    else:
        _state["low_util_streak"] = 0

    print(f"   GPU {util}%  VRAM {vu:.1f}/{vt:.1f} GB  [{elapsed:.1f}s]")

    m = trainer.metrics if hasattr(trainer, "metrics") else {}
    if _state["csv_writer"]:
        _state["csv_writer"].writerow([
            epoch,
            round(float(m.get("train/box_loss", 0) or 0), 5),
            round(float(m.get("train/cls_loss", 0) or 0), 5),
            round(float(m.get("train/dfl_loss", 0) or 0), 5),
            round(float(m.get("metrics/precision(B)", 0) or 0), 4),
            round(float(m.get("metrics/recall(B)",    0) or 0), 4),
            round(float(m.get("metrics/mAP50(B)",     0) or 0), 4),
            round(float(m.get("metrics/mAP50-95(B)",  0) or 0), 4),
            util,
            round(vu, 2),
            round(elapsed, 1),
        ])
        _state["csv_file"].flush()


def cb_train_end(trainer):
    if _state["csv_file"]:
        _state["csv_file"].close()
    total = time.time() - _state["train_t0"]
    print(f"\n✅ Training finished in {total/60:.1f} min")


# ════════════════════════════════════════════════════════════════════════════
# 5. POST-TRAINING
# ════════════════════════════════════════════════════════════════════════════

def rename_weights(project: str, name: str) -> Path:
    weights_dir = Path(project) / name / "weights"
    src = weights_dir / "best.pt"
    dst = weights_dir / "exp2_best_model.pt"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"\n✅ Best weights → {dst}")
    else:
        print("⚠️  best.pt not found — check training output.")
        dst = src
    return dst


def evaluate_test_set(model_path: Path, data_yaml: Path,
                      class_names: list, device: str):
    print("\n" + "=" * 65)
    print("🧪  TEST SET EVALUATION")
    print("=" * 65)
    model   = YOLO(str(model_path))
    results = model.val(data=str(data_yaml), split="test",
                        device=device, verbose=False)
    box = results.box
    print(f"\n{'Class':<33} {'Precision':>10} {'Recall':>10} {'mAP50':>8}")
    print("-" * 65)
    ap_idx = getattr(box, "ap_class_index", None)
    if ap_idx is not None and len(ap_idx):
        for i, cid in enumerate(ap_idx):
            name = class_names[cid] if cid < len(class_names) else f"cls{cid}"
            p    = float(box.p[i])    if hasattr(box, "p")    else 0.0
            r    = float(box.r[i])    if hasattr(box, "r")    else 0.0
            ap   = float(box.ap50[i]) if hasattr(box, "ap50") else 0.0
            print(f"{name:<33} {p:>10.3f} {r:>10.3f} {ap:>8.3f}")
    print("-" * 65)
    print(f"  Overall mAP50    : {box.map50:.4f}")
    print(f"  Overall mAP50-95 : {box.map:.4f}")
    print("=" * 65)


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("🚦  EXP 2 — YOLOv8l + EXPANDED DATASET (seed=42)")
    print("=" * 65)

    device, vram_gb = get_gpu_info()
    batch = 8

    # ── Dataset split ─────────────────────────────────────────────────────
    if is_flat(DATASET_ROOT):
        print(f"\n📁 Flat structure detected → splitting {SPLIT_RATIOS}")
        split_and_organise(DATASET_ROOT, SPLIT_RATIOS, seed=SEED)
    else:
        print("\n📁 Pre-split structure detected → using as-is")

    data_yaml_split = make_split_yaml(DATASET_ROOT, DATA_YAML)

    with open(data_yaml_split) as f:
        yaml_data   = yaml.safe_load(f)
    class_names = yaml_data.get("names", [])
    n_classes   = len(class_names)

    report = class_distribution(DATASET_ROOT, class_names)
    print_report(report, class_names)
    warn_minority(report, class_names, MIN_WARN)

    w = class_weights(report, n_classes)
    print(f"\n⚖️  Class weights (minority boost, logged for reference):")
    for name, wi in zip(class_names, w):
        bar = "█" * int(wi * 10)
        print(f"   {name:<33} {wi:.3f}  {bar}")

    n_train = report["train"]["n_images"]
    estimate_time(n_train, batch, EPOCHS, vram_gb)

    # Always fresh — never resume for Exp 2
    print("\n🆕 Starting fresh from yolov8l.pt (pretrained COCO weights)")
    model = YOLO("yolov8l.pt")

    model.add_callback("on_train_start",       cb_train_start)
    model.add_callback("on_train_epoch_start", cb_epoch_start)
    model.add_callback("on_train_epoch_end",   cb_epoch_end)
    model.add_callback("on_train_end",         cb_train_end)

    model.train(
        data          = str(data_yaml_split),
        epochs        = EPOCHS,
        imgsz         = IMGSZ,
        batch         = batch,
        device        = device,
        workers       = 7,
        project       = PROJECT,
        name          = NAME,
        exist_ok      = True,
        cache         = True,
        amp           = True,
        patience      = PATIENCE,
        resume        = False,
        plots         = True,
        save          = True,
        val           = True,
        verbose       = True,
        seed          = SEED,
        optimizer     = "AdamW",
        lr0           = 0.001,
        cos_lr        = True,
        warmup_epochs = 3,
        augment       = True,
        degrees       = 10.0,
        fliplr        = 0.5,
        flipud        = 0.1,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        mosaic        = 1.0,
        mixup         = 0.1,
        copy_paste    = 0.1,
    )

    final_weights = rename_weights(PROJECT, NAME)
    evaluate_test_set(final_weights, data_yaml_split, class_names, device)

    print(f"\n🏁 Done.")
    print(f"   Model : {final_weights}")
    print(f"   Logs  : {LOG_CSV}")
    print(f"   Plots : {Path(PROJECT) / NAME}")