import os

LABELS_DIR = r"C:\Users\sands\Downloads\Artificial Intelligence\final project\train\labels"

# 6-class index → 15-class index
REMAP = {
    0: 1,   # Left Exit
    1: 2,   # Left/Right Exit
    2: 3,   # Right Exit
    3: 4,   # Straight Exit
    4: 7,   # emergency exit door
    5: 12,  # objects blocked emergency exit
}

fixed = 0
for fname in os.listdir(LABELS_DIR):
    if not fname.startswith("ML_"):
        continue
    fpath = os.path.join(LABELS_DIR, fname)
    lines = open(fpath).readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        old_cls = int(parts[0])
        new_cls = REMAP.get(old_cls, old_cls)
        parts[0] = str(new_cls)
        new_lines.append(" ".join(parts))
    open(fpath, "w").write("\n".join(new_lines))
    fixed += 1

print(f"Remapped {fixed} files.")