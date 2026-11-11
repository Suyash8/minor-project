# =============================================================================
# show_annotation.py — display ONE image from a cleaned dataset with its
# OBB annotations drawn and labelled. Displays inline, saves nothing.
#
# Works on the cleaned YOLO-OBB datasets:
#   <root>/<dataset>/{train,val,test}/images  +  labels
#   label format: "class_id x1 y1 x2 y2 x3 y3 x4 y4"  (normalized 0-1)
#
# Colab usage:
#   show_annotation("codrone", "img_000123.jpg")
#   show_annotation("visdrone", "0000002_02315_d_0000012")   # stem works too
#   show_annotation("dota", "P0000__1.0__0___0.jpg", split="train")
#
# Optional: set CLEANED_DATA_ROOT env var if your Drive path differs.
# =============================================================================
import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.environ.get(
    "CLEANED_DATA_ROOT",
    "/content/drive/MyDrive/object-detection/cleaned_data")

CLASS_NAMES = ["pedestrian", "bicycle", "car", "van", "truck",
               "bus", "tricycle", "motorcyclist"]

# one fixed, easily distinguishable color per class (BGR for cv2)
CLASS_COLORS_BGR = [
    (60, 76, 231),    # 0 pedestrian  red
    (231, 119, 60),   # 1 bicycle     blue
    (80, 175, 76),    # 2 car         green
    (160, 101, 245),  # 3 van         purple
    (23, 190, 207),   # 4 truck       yellow-ish
    (0, 215, 255),    # 5 bus         amber
    (255, 0, 128),    # 6 tricycle    magenta
    (128, 240, 60),   # 7 motorcyclist lime
]


def _find_image(dataset, filename, split, root):
    """Return (split, image_path) — filename may be 'x.jpg' or just 'x'."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    base = os.path.join(root, dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"dataset folder not found: {base}  (known: "
            f"{[d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]})")
    splits = [split] if split else ["train", "val", "test"]
    for s in splits:
        img_dir = os.path.join(base, s, "images")
        if not os.path.isdir(img_dir):
            continue
        if os.path.splitext(filename)[1]:
            hits = [os.path.join(img_dir, filename)]
        else:
            hits = glob.glob(os.path.join(img_dir, stem + ".*"))
        hits = [h for h in hits if os.path.isfile(h)
                and os.path.splitext(h)[1].lower()
                in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")]
        if hits:
            return s, hits[0]
    raise FileNotFoundError(
        f"'{stem}' not found in {dataset} "
        f"({'/'.join(splits)}) under {root}")


def _read_obb_labels(label_path):
    """YOLO-OBB txt -> list of (class_idx, [(x,y) x4 in normalized coords])."""
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    for line in open(label_path):
        t = line.split()
        if len(t) != 9:
            continue
        try:
            uid = int(t[0])
            c = [float(v) for v in t[1:9]]
        except ValueError:
            continue
        boxes.append((uid, list(zip(c[0::2], c[1::2]))))
    return boxes


def show_annotation(dataset, filename, split=None, root=ROOT):
    """
    Display one image with its OBB annotations drawn and labelled.
    dataset  : 'codrone' | 'visdrone' | 'dota'
    filename : image file name or stem, e.g. 'img_0001.jpg' or 'img_0001'
    split    : 'train' | 'val' | 'test'  (None = search all three)
    Returns dict with what was drawn (no file is written anywhere).
    """
    s, img_path = _find_image(dataset, filename, split, root)
    label_path = os.path.join(root, dataset, s, "labels",
                              os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    boxes = _read_obb_labels(label_path)

    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"could not read image: {img_path}")
    H, W = img.shape[:2]

    # scale line/text size with image size so labels stay readable on 1280px
    th = max(1, round(min(H, W) / 400))          # polygon thickness
    fs = max(0.35, min(H, W) / 1200)             # font scale
    pad = max(1, round(min(H, W) / 350))         # label bg padding
    n_drawn, skipped = 0, 0
    for uid, pts_norm in boxes:
        if not (0 <= uid < len(CLASS_NAMES)):
            skipped += 1
            continue
        pts = [(int(min(max(x, 0.0), 1.0) * (W - 1)),
                int(min(max(y, 0.0), 1.0) * (H - 1))) for x, y in pts_norm]
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        color = CLASS_COLORS_BGR[uid]
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=th)
        tx, ty = pts[0]
        ty = max(ty, int(24 * fs * 2))           # keep label inside the frame
        label = f"{uid} {CLASS_NAMES[uid]}"
        (tw, thh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       fs, max(1, round(fs * 2)))
        cv2.rectangle(img, (tx, ty - thh - 2 * pad), (tx + tw + 2 * pad, ty),
                      color, -1)
        cv2.putText(img, label, (tx + pad, ty - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    max(1, round(fs * 2)), cv2.LINE_AA)
        n_drawn += 1

    plt.figure(figsize=(min(14, W / 100), min(14, W / 100) * H / W))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"{dataset} / {s} / {os.path.basename(img_path)}   "
              f"({n_drawn} boxes" + (f", {skipped} unknown class" if skipped else "") + ")",
              fontsize=11)
    plt.show()          # display only — nothing is saved

    return dict(dataset=dataset, split=s, image=img_path,
                boxes_drawn=n_drawn, boxes_skipped=skipped,
                unknown_class_ids=skipped)
