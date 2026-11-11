#!/usr/bin/env python3
"""IDD-style per-class instance count table for the cleaned datasets.

Rows = the 8 unified classes, columns = each dataset's train/val/test,
plus a Total row. Also renders a PNG table (like the IDD reference figure)
into <DRIVE>/object-detection/figures/.

Usage (Colab, Drive mounted):
    python data_cleaning/class_count_table.py
"""
import os
import matplotlib
matplotlib.use("Agg")          # Colab-safe; figures are saved, then shown if inline
import matplotlib.pyplot as plt

ROOT   = os.environ.get(
    "CLEANED_DATA_ROOT",
    "/content/drive/MyDrive/object-detection/cleaned_data")
FIGDIR = os.environ.get(
    "FIGDIR",
    "/content/drive/MyDrive/object-detection/figures")
os.makedirs(FIGDIR, exist_ok=True)

CLASSES = ["pedestrian", "bicycle", "car", "van", "truck",
           "bus", "tricycle", "motorcyclist"]
DATASETS = ["codrone", "visdrone", "dota"]
SPLITS   = ["train", "val", "test"]

# ---- count boxes per class, per dataset, per split ----
counts = {ds: {sp: {c: 0 for c in CLASSES} for sp in SPLITS} for ds in DATASETS}
for ds in DATASETS:
    for sp in SPLITS:
        lbl_d = os.path.join(ROOT, ds, sp, "labels")
        if not os.path.isdir(lbl_d):
            continue
        for f in os.listdir(lbl_d):
            if not f.endswith(".txt"):
                continue
            with open(os.path.join(lbl_d, f)) as fh:
                for ln in fh:
                    t = ln.split()
                    if t and t[0].isdigit():
                        i = int(t[0])
                        if i < len(CLASSES):
                            counts[ds][sp][CLASSES[i]] += 1

# ---- print text table ----
cols = [(f"{ds[:4].upper()} {sp[:3].title()}", counts[ds][sp])
        for ds in DATASETS for sp in SPLITS]
w = max(13, max(len(c) for c, _ in cols))
print(f"{'Class':<14}" + "".join(f"{c:>{w+1}}" for c, _ in cols))
for cl in CLASSES:
    print(f"{cl:<14}" + "".join(f"{d[cl]:>{w+1}}" for _, d in cols))
print(f"{'TOTAL':<14}" + "".join(f"{sum(d.values()):>{w+1}}" for _, d in cols))

# ---- render PNG table (IDD paper style) ----
rows = [[cl] + [d[cl] for _, d in cols] for cl in CLASSES]
rows.append(["Total"] + [sum(d.values()) for _, d in cols])
fig, ax = plt.subplots(figsize=(1.5 + (w/10)*len(cols), 0.55*(len(rows)+1)))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=["Class"] + [c for c, _ in cols],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("black")
    if r == 0 or c == 0 or r == len(rows):     # header row, class column, total row
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#d9e6f2" if r == 0 else "#f2f2f2")
plt.title("Instance counts per class (cleaned OBB datasets)", fontsize=12)
out = os.path.join(FIGDIR, "class_instance_counts.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved ->", out)
