"""
Emergency Exit Detection — Inference Script
"""

import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    os.system(f"{sys.executable} -m pip install ultralytics --quiet")
    from ultralytics import YOLO

import torch

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = Path(r"C:\Users\sands\Downloads\Artificial Intelligence\final project\runs\detect\runs\exp2_expanded_dataset\weights\best.pt")
SOURCE     = r"C:\Users\sands\Downloads\Artificial Intelligence\final project\New folder"
SAVE_DIR   = Path(r"C:\Users\sands\Downloads\Artificial Intelligence\final project\New folder\results")

CONF       = 0.25
IOU        = 0.45
IMGSZ      = 640
# ─────────────────────────────────────────────────────────────────────────────


def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"✅ GPU : {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU — running on CPU")

    if not MODEL_PATH.exists():
        print(f"\n❌ Model not found at:\n   {MODEL_PATH}")
        print("\n   Run this in terminal to find your model:")
        print(r'   dir /s /b "C:\Users\sands\Downloads\Artificial Intelligence\final project\runs\*.pt"')
        return

    print(f"\n📦 Loading model...")
    model = YOLO(str(MODEL_PATH))
    print(f"   Classes: {list(model.names.values())}")

    print(f"\n🔍 Running on all images in:\n   {SOURCE}")

    results = model.predict(
        source   = SOURCE,
        conf     = CONF,
        iou      = IOU,
        imgsz    = IMGSZ,
        device   = device,
        save     = True,
        show     = False,
        project  = str(SAVE_DIR.parent),
        name     = SAVE_DIR.name,
        exist_ok = True,
        line_width = 2,
        verbose  = True,
    )

    print("\n" + "=" * 55)
    print("📊  DETECTION SUMMARY")
    print("=" * 55)

    class_counts = {}
    total = 0

    for r in results:
        for cls_id in r.boxes.cls.tolist():
            name = model.names[int(cls_id)]
            class_counts[name] = class_counts.get(name, 0) + 1
            total += 1

    if class_counts:
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"   {cls_name:<35} {count:>5} detection(s)")
    else:
        print("   No detections found. Try lowering CONF to 0.15.")

    print("-" * 55)
    print(f"   Total detections : {total}")
    print(f"   Images processed : {len(results)}")
    print("=" * 55)
    print(f"\n✅ Results saved to:\n   {SAVE_DIR}")


if __name__ == "__main__":
    main()