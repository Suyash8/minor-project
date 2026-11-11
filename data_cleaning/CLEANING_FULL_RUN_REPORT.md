# Cleaned Dataset Build Report (v4 full run)

Generated: 2026-11-05 · Script: `clean_build_v4.py` · Source: Google Drive `/object-detection/data`
Output: `/object-detection/cleaned_data/{codrone,visdrone,dota}/{train,val,test}/{images,labels}`

## 1. Headline numbers

| Dataset | Kept images | Boxes kept / parsed | Notes |
|---|---|---|---|
| CODrone | 9,894 / 10,011 | 416,727 / 416,821 | 94 no-label-row images, 22 duplicates, 1 empty label |
| VisDrone | 8,633 / 8,636 | 446,585 / 457,074 | 1,580 test-challenge images unlabeled (no GT exists), 3 duplicates |
| DOTA | 1,233 / 1,876 | 193,073 / 197,202 | 642 label files contain zero vehicle rows (non-vehicle classes only) |

Combined: **19,760 images / 1,056,385 boxes** in one cross-compatible 8-class space.

## 2. Why CODrone box counts differ from the v3 run (289,558 → 416,727)

The v3 script had a normalization bug that affected **pixel-coordinate label
formats only** (CODrone's DOTA-style annfiles, VisDrone raw CSVs): those boxes
were mis-scaled and then almost all of them failed the 4 px minimum-size check
and were silently discarded as "degenerate". Evidence from the v3 TEST digest:
CODrone had 808 DOTA-format boxes and 768 of them were dropped (95%), while
normalized YOLO-format boxes passed normally. v4 normalizes pixel coordinates
by the true original image size before the size check, so those **~127,000
boxes are recovered, not invented**. v4's numbers are the correct ones.

## 3. Split-proportion fix (v4.1)

The first v4 full run produced distorted split sizes for CODrone
(67.9 / 16.1 / 15.2) and DOTA (84.6 / 5.3 / 10.1) because the greedy
instance-deficit splitter breaks when image box counts are extremely
heavy-tailed (DOTA has images with 1 to 10,066 boxes; one image can carry 6%
of an entire class). The splitter was redesigned: image counts are anchored to
the cumulative 82/9/9 targets with a small slack, and the split choice inside
the slack window is driven by the image's rarest-class quota. Validated on
real DOTA per-image distributions: image proportions within ~1 pp and
per-class shares within ~2 pp of target. **Re-run the full build with v4.1 to
apply it** (only split membership changes; kept images and boxes are
identical).

## 4. DOTA labels: what was wrong and what is used now

- The original `data/dota/labels/*.txt` files on Drive were empty web-upload
  shells (0 bytes) — the real annotations never made it to Drive. That is why
  earlier runs found only 89 boxes; the parser was never at fault.
- Real labels were provided as the official zips (`DOTA-train.zip`,
  `DOTA-val.zip`), each containing **both v1.0 and v1.5** annotation sets.
- **Only v1.5 is used** (packaged as `dota_labels_all.zip` → extracted to
  `data/dota/labels-v1.5/{train,val}`). v1.5 re-annotated small objects, giving
  3.4× more vehicle boxes than v1.0 (148,719 vs 43,095 train vehicle rows;
  48,476 vs 9,825 val). v1.0 is intentionally not used.
- DOTA format is the original oriented format: `x1 y1 x2 y2 x3 y3 x4 y4 class
  difficult` (class-last, absolute pixels, `imagesource:`/`gsd:` headers),
  mapped `small-vehicle→car`, `large-vehicle→truck`; all other classes dropped.
  `difficult=1` boxes are kept.
- The flat `data/dota/images/` folder (2,813 files named `visdrone_*.jpg`) has
  no matching labels and is skipped as unlabeled — expected.

## 5. Class space (identical in all three datasets)

`0 pedestrian · 1 bicycle · 2 car · 3 van · 4 truck · 5 bus · 6 tricycle · 7 motorcyclist`

- CODrone's own `van` class has almost no real instances in the source data.
- Class mappings are yaml-driven with a synonym table
  (people→pedestrian, cyclist→bicycle, motor→motorcyclist,
  awning-tricycle→tricycle, small-vehicle→car, large-vehicle→truck).
- Each dataset ships its own `dataset.yaml` (same 8 names), plus a merged
  `dataset_all.yaml` at the root. Any dataset folder can replace another, and
  all three can be merged, without code changes.

## 6. Known non-issues

- `gdoc_placeholders` NOTEs on CODrone `annfile` and VisDrone `annotations`
  dirs: Drive web-upload shells, correctly bypassed by label priority.
- `yaml conflict` lines for synonyms (cyclist/people/motor/small-vehicle) are
  informational; v4.1 only warns when the unified class actually differs.
- Cross-source duplicates: none detected (CODrone and VisDrone imagery are
  disjoint; merging them introduces no train/test leakage).

## 7. Adding DOTA v1.0 labels (optional)

- A single package `dota_labels_all.zip` contains BOTH label versions:
  `labels-v1.5/{train,val}` (1,411 + 458 files, used by the cleaner) and
  `labels-v1.0/{train,val}` (1,411 + 458 files, fallback only).
- Upload it to `/object-detection/` and extract once:
  ```python
  import zipfile, os, glob
  z = "/content/drive/MyDrive/object-detection/dota_labels_all.zip"
  dst = "/content/drive/MyDrive/object-detection/data/dota"
  with zipfile.ZipFile(z) as f:
      f.extractall(dst)
  ```
- Important expectation: v1.0 and v1.5 annotate the SAME images. The cleaner
  picks the best label file per image and `labels-v1.5` has top priority, so
  v1.0 never changes the output — v1.5 has 3.4x more vehicle boxes. It is
  included for completeness and reproducibility only.

## 8. DOTA-only rebuild (optional, for future top-ups)

Three edits to `clean_build_v4.py`:

1. Replace `sources = ["codrone", "visdrone", "dota"]` with:
   ```python
   sources = os.environ.get("CLEAN_SOURCES", "codrone,visdrone,dota").split(",")
   ```
   then set `os.environ["CLEAN_SOURCES"] = "dota"` before running.
2. Replace the whole-output Drive cleanup
   (`if os.path.exists(out_drive): shutil.rmtree(out_drive)`) with a
   per-dataset cleanup:
   ```python
   for s in sources:
       shutil.rmtree(os.path.join(out_drive, s), ignore_errors=True)
   ```
   Without this, a dota-only run would DELETE the CODrone and VisDrone outputs
   on Drive.
3. In the `dataset_all.yaml` writer, iterate over the hardcoded full list
   `["codrone", "visdrone", "dota"]` so a dota-only run keeps the merged yaml
   intact.

## 9. Bug status and required actions

| Bug | Status | Action |
|---|---|---|
| v3 pixel-format normalization (silently dropped ~127k CODrone boxes) | Fixed inside v4 | None — current box counts are correct |
| Split proportions (CODrone 68/16/15, DOTA 85/5/10 instead of 82/9/9) | Fix shipped in v4.1, but the last full run used the older copy | One FULL re-run with the updated `clean_build_v4.py` |

Verification after the re-run: every dataset's `split_sizes` should be close
to 82/9/9 and val/test must contain all 8 classes (including bicycle,
tricycle, van in CODrone). The updated script contains the string
`Schedule-anchored stratified split (v4.1)` inside `stratified_assign` — if
your copy does not have it, re-copy the file.

## 10. Other digest notes (all harmless)

- `read_error` counters: Google Drive web-upload shell files (`.gdoc`) that
  fail to read; the converted labels always take priority, so nothing is lost.
- `difficult_kept: 124,411` (DOTA): 64% of DOTA vehicle boxes are flagged
  `difficult=1` and are kept on purpose; recorded in each
  `cleaning_report.json`.
- `gdoc_placeholders` NOTEs on `labels-v1.5/...`: the extracted files parse
  correctly (193,073 boxes), the note is informational only.

## 11. Recommended next steps

1. Extract `dota_labels_all.zip` (section 7) and run the full build once with
   v4.1 (`CLEAN_TEST_RUN=0`, ~24 min). This is the final dataset build.
2. Generate the mean-intensity histograms with `intensity_histogram.py`
   (one figure per dataset, same style as the IDD reference: Train blue /
   Validation orange / Test green overlays).
3. Train per-dataset with each `dataset.yaml`; optionally compare with
   `dataset_all.yaml`.
4. DOTA is aerial imagery from higher altitude with a different box-size
   distribution — treat cross-dataset evaluation results accordingly.
