# Data Cleaning Pipeline (v4.1)

Converts the three raw sources — **CODrone**, **VisDrone**, **DOTA** — into
three **separated but fully cross-compatible** YOLO-OBB datasets with an
identical structure and identical 8-class space:

```
cleaned_data/
├── codrone/{train,val,test}/{images,labels}   9,894 imgs · 416,727 boxes
├── visdrone/{train,val,test}/{images,labels}  8,633 imgs · 446,585 boxes
├── dota/{train,val,test}/{images,labels}      1,233 imgs · 193,073 boxes
├── dataset_all.yaml        <- merge all three (Ultralytics accepts lists)
└── cleaning_report.json    <- full machine-readable audit
```

Unified classes (identical everywhere):
`0 pedestrian · 1 bicycle · 2 car · 3 van · 4 truck · 5 bus · 6 tricycle · 7 motorcyclist`

Label format (YOLO-OBB, normalized 0–1): `class x1 y1 x2 y2 x3 y3 x4 y4`

## What the cleaner does (per image)

1. **Recursive discovery** — walks the whole source tree; matches image ↔ label
   purely by filename stem. No hardcoded paths.
2. **Label priority** — `labels-v1.5`/`labelTxt` > `labels` > `annfile` >
   `annotations`; Google-Docs placeholder shells and empty files bypassed.
3. **Format auto-detection** — VisDrone raw CSV (1-based, `x,y,w,h,score,cat`),
   YOLO-OBB (9 tokens, normalized), DOTA class-first pixels, original DOTA
   `x1..y4 class difficult` (class-last, 10 tokens, with `imagesource:`/`gsd:`
   headers).
4. **Class unification** — synonym table (`people→pedestrian`,
   `cyclist→bicycle`, `small-vehicle→car`, `large-vehicle→truck`,
   `awning-tricycle→tricycle`, `motor→motorcyclist`); anything unmapped is
   dropped and counted, never silently kept.
5. **Geometry hygiene** — EXIF transpose, pixel→normalized conversion by the
   true image size, clip to [0,1], drop boxes < 4 px, dedup exact box repeats.
6. **Deduplication** — MD5 within source AND across sources (no train/test
   leakage when merging).
7. **Resize** — long side → 1280 px (Lanczos), JPEG quality 90.
8. **Stratified split (v4.1)** — schedule-anchored 82/9/9 train/val/test,
   rarest-class-first assignment; holds image proportions within ~1 pp and
   per-class shares within ~2 pp on heavy-tailed data.
9. **Audit** — every input image accounted for by a counted status; every
   dropped box counted by reason (`cleaning_report.json`).

## Files

| File | Purpose |
|---|---|
| `clean_build_v4.py` | The cleaner (v4.1). TEST mode by default (60 imgs/source). |
| `intensity_histogram.py` | Mean-intensity histograms per dataset (train/val/test overlay), IDD style. |
| `class_count_table.py` | IDD-style per-class instance count table (text + PNG). |
| `count_images_boxes.py` | Quick images/boxes per dataset per split table. |
| `show_annotation.py` | Display one image with its OBB labels drawn (saves nothing). |
| `CLEANING_FULL_RUN_REPORT.md` | Report of the full v4 run + bug history. |

## Run it (Colab, Drive mounted)

```python
# 1) one-time: extract the DOTA labels next to the raw data
import zipfile
z = "/content/drive/MyDrive/object-detection/dota_labels_all.zip"   # = assets/dota_labels_all.zip in this repo
dst = "/content/drive/MyDrive/object-detection/data/dota"
with zipfile.ZipFile(z) as f:
    f.extractall(dst)
# expect: labels-v1.5/{train,val} = 1411/458 files, labels-v1.0/{train,val} = 1411/458

# 2) smoke test (60 images/source, writes /content/cleaned_data_TEST)
%run data_cleaning/clean_build_v4.py

# 3) FULL build (~24 min, replaces Drive cleaned_data)
import os; os.environ["CLEAN_TEST_RUN"] = "0"
%run data_cleaning/clean_build_v4.py
```

Env knobs: `CLEAN_DATA_DIR`, `CLEAN_OUT_ROOT`, `CLEAN_TEST_RUN`, `CLEAN_TEST_N`,
`CLEAN_WORKERS`, `CLEAN_MIN_BOX_PX`, `CLEAN_CROSS_DEDUP`, `CLEAN_LOCAL_ROOT`.

**Finish flag:** the run is complete only after the line
`copied N files to Drive` and the final `DIGEST` JSON are printed. If a Colab
disconnect interrupts the copy, Drive may be left with only `train` — re-run.

## Verify after a build

```python
%run data_cleaning/count_images_boxes.py     # splits ≈ 82/9/9, all 8 classes present in val/test
%run data_cleaning/class_count_table.py      # per-class table like the IDD paper
%run data_cleaning/intensity_histogram.py    # train/val/test brightness curves should overlap
```

`show_annotation.py` usage:

```python
from data_cleaning.show_annotation import show_annotation
show_annotation("codrone", "img_000123.jpg")            # auto-finds the split
show_annotation("visdrone", "0000002_02315_d_0000012")  # stem alone works
show_annotation("dota", "P0000__1.0__0___0", split="train")
```

See `CLEANING_FULL_RUN_REPORT.md` for the full audit trail of the last run,
the two historical bugs (v3 pixel-format normalization; v4 split proportions)
and how they were found and fixed.
