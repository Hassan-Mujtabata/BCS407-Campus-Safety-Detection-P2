from PIL import Image, ImageDraw, ImageFont
import os
import glob

# ── AUTO-DETECT: finds the latest runs/detect/train* folder next to this script
BASE       = os.path.dirname(os.path.abspath(__file__))
train_dirs = sorted(glob.glob(os.path.join(BASE, "runs", "detect", "train*")))

if not train_dirs:
    raise FileNotFoundError(
        f"No train* folder found under {os.path.join(BASE, 'runs', 'detect')}\n"
        "Make sure this script sits in your project root (same level as the 'runs' folder)."
    )

OUTPUT_DIR = train_dirs[-1]   # picks the highest-numbered / most-recent train run
SAVE_PATH  = os.path.join(BASE, "all_results.jpg")

print(f"  Using run folder : {OUTPUT_DIR}")
print(f"  Saving grid to   : {SAVE_PATH}\n")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Images to include (filename → label)
IMAGES = [
    ("results.png",                    "results"),
    ("confusion_matrix.png",           "confusion_matrix"),
    ("confusion_matrix_normalized.png","confusion_matrix_normalized"),
    ("BoxF1_curve.png",                "BoxF1_curve"),
    ("BoxP_curve.png",                 "BoxP_curve"),
    ("BoxPR_curve.png",                "BoxPR_curve"),
    ("BoxR_curve.png",                 "BoxR_curve"),
    ("labels.jpg",                     "labels"),
    ("train_batch0.jpg",               "train_batch0"),
    ("train_batch1.jpg",               "train_batch1"),
    ("train_batch2.jpg",               "train_batch2"),
    ("val_batch0_labels.jpg",          "val_batch0_labels"),
    ("val_batch0_pred.jpg",            "val_batch0_pred"),
    ("val_batch1_labels.jpg",          "val_batch1_labels"),
    ("val_batch1_pred.jpg",            "val_batch1_pred"),
    ("val_batch2_labels.jpg",          "val_batch2_labels"),
    ("val_batch2_pred.jpg",            "val_batch2_pred"),
]

COLS       = 3
CELL_W     = 3000
LABEL_H    = 120
PADDING    = 40
BG_COLOR   = (15, 15, 15)
TEXT_COLOR = (230, 230, 230)
FONT_SIZE  = 72
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path):
    return Image.open(path).convert("RGB")

def resize_to_width(img, width):
    ratio  = width / img.width
    height = int(img.height * ratio)
    return img.resize((width, height), Image.LANCZOS)

# Load and resize all images
cells = []
for fname, label in IMAGES:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        img = load_image(fpath)
        img = resize_to_width(img, CELL_W)
        cells.append((img, label))
    else:
        print(f"  SKIPPED (not found): {fname}")

if not cells:
    raise RuntimeError("No images were found — check that OUTPUT_DIR is correct.")

# Calculate grid dimensions
n      = len(cells)
cols   = min(COLS, n)
rows   = (n + cols - 1) // cols

cell_h  = max(c[0].height for c in cells)
total_w = cols * CELL_W + (cols + 1) * PADDING
total_h = rows * (cell_h + LABEL_H) + (rows + 1) * PADDING

# Build canvas
canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
draw   = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
except Exception:
    font = ImageFont.load_default()

for idx, (img, label) in enumerate(cells):
    row = idx // cols
    col = idx % cols
    x   = PADDING + col * (CELL_W + PADDING)
    y   = PADDING + row * (cell_h + LABEL_H + PADDING)

    y_img = y + (cell_h - img.height) // 2
    canvas.paste(img, (x, y_img))

    y_text = y + cell_h + 10
    bbox   = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x_text = x + (CELL_W - text_w) // 2
    draw.text((x_text, y_text), label, fill=TEXT_COLOR, font=font)

canvas.save(SAVE_PATH, "JPEG", quality=100, subsampling=0, dpi=(300, 300))
print(f"\nSaved -> {SAVE_PATH}")
print(f"   Size: {total_w} x {total_h} px")
