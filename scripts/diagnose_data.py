#!/usr/bin/env python3
# =============================================================================
# DATA / HARNESS / CHECKPOINT DIAGNOSTIC for the Aerial OBB Benchmark Suite
# =============================================================================
# Answers, without training or GPU-hours, the three questions that decide
# whether evaluation numbers are real:
#
#   1. DATA     - Are val labels found, parseable, and in-class?
#                 -> per-class GT box counts, unrecognized class names,
#                    label-vs-image resolution sanity, unlabeled images.
#
#   2. HARNESS  - Is the cross-dataset evaluation machinery sound?
#                 -> feeds GROUND TRUTH as if it came from a perfect model
#                    through the exact remap + metric path used by
#                    scripts/cross_dataset_eval.py. Every (source, eval)
#                    pair must score mAP50 ~= 1.0. If it does, any near-zero
#                    result in a real run is the MODEL, not the harness.
#
#   3. CHECKPOINTS - Did the saved weights actually load, and did training
#                 converge?
#                 -> loads every checkpoint into its architecture with
#                    strict=False and reports missing/unexpected keys
#                    (silent random-init exposure), plus the epoch count and
#                    first-vs-last val loss stored inside the checkpoint.
#
# Usage (Colab, CPU is enough):
#   python scripts/diagnose_data.py \
#       --data-dir    /content/drive/MyDrive/object-detection/data \
#       --weights-dir /content/drive/MyDrive/object-detection/weights
#
#   Options:
#     --datasets codrone visdrone dota   restrict datasets (default: all)
#     --skip-checkpoints                only run data + harness checks
#     --skip-harness                    only run data + checkpoint checks
#     --harness-samples 40              val images used by the harness self-test
#     --bounds-sample 200               image headers opened for bounds check
#
# Exit code 0 = everything passed; 1 = at least one check failed (details in
# the output and in diagnose_report.json written next to --data-dir/../ or cwd).
# =============================================================================

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_CLASSES, SUPPORTED_DATASETS          # noqa: E402


# ---------------------------------------------------------------------------
# Label-side class matching, mirroring AerialOBBDataset.get_ground_truth()
# ---------------------------------------------------------------------------

_LABEL_SYNONYMS = {
    "people": "pedestrian",
    "motor": "motorcyclist",
    "bicycle": "cyclist",
    "traffic_signs": "traffic_sign",
    "traffic_lights": "traffic_light",
    "traffic-signs": "traffic-sign",
    "traffic-lights": "traffic-light",
}

# Explicit ignore-region categories present in some benchmark label sets
# (CODrone annfiles contain 'ignored'). The parser skips these by design, so
# the diagnostic must not report them as unrecognized classes.
_IGNORED_NAMES = {"ignored", "ignore", "ignored-region", "others", "background"}


def match_class_name(raw_name: str, class_to_idx: dict):
    """Return class index for a label class name, or None if unrecognized.
    Mirrors the candidate order used in src/data/dataset.py."""
    raw = raw_name.strip().lower()
    cands = [raw, raw.replace("-", "_"), raw.replace("_", "-"),
             _LABEL_SYNONYMS.get(raw, raw),
             _LABEL_SYNONYMS.get(raw.replace("-", "_"), raw.replace("-", "_"))]
    for c in cands:
        if c in class_to_idx:
            return class_to_idx[c]
    return None


def scan_label_file(lbl_path: Path, class_names):
    """Parse one DOTA/CODrone (8-coord + name) or YOLO-OBB (9-part) label file.
    Returns (n_boxes_by_class_idx, Counter(unrecognized_names), n_yolo_lines,
    n_skipped_lines, n_ignored_boxes)."""
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    counts = np.zeros(len(class_names), dtype=np.int64)
    unknown = Counter()
    n_yolo, n_skipped, n_ignored = 0, 0, 0

    try:
        lines = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return counts, unknown, 0, 1, 0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        # VisDrone CSV: left, top, width, height, score, category, trunc, occl
        # (mirrors dataset.py Format 1: category is 1-indexed, score==1 kept)
        if "," in line:
            subparts = [p.strip() for p in line.split(",")]
            if len(subparts) >= 6:
                try:
                    score = int(float(subparts[4]))
                    cat = int(float(subparts[5]))
                    cls_idx = cat - 1
                    if score == 1 and 0 <= cls_idx < len(class_names):
                        counts[cls_idx] += 1
                    elif cat == 0:
                        n_ignored += 1
                    continue
                except (ValueError, IndexError):
                    pass
            n_skipped += 1
            continue
        # YOLO-OBB: cls x1 y1 x2 y2 x3 y3 x4 y4 (all numeric) — mirrors the
        # parser's gate: 9 parts, parts[8] contains no letter and no hyphen.
        if len(parts) == 9 and not any(ch.isalpha() or ch == "-" for ch in parts[8]):
            try:
                ci = int(parts[0])
                if 0 <= ci < len(class_names):
                    counts[ci] += 1
                    n_yolo += 1
                    continue
            except ValueError:
                pass
            n_skipped += 1
            continue
        # DOTA / CODrone: x1 y1 x2 y2 x3 y3 x4 y4 name [difficult] — 8 leading
        # floats, class name at parts[8] (may contain hyphens/underscores).
        if len(parts) >= 9:
            try:
                [float(p) for p in parts[:8]]
            except ValueError:
                n_skipped += 1
                continue
            if parts[8].lower() in _IGNORED_NAMES:
                n_ignored += 1
                continue
            ci = match_class_name(parts[8], class_to_idx)
            if ci is None:
                unknown[parts[8]] += 1
            else:
                counts[ci] += 1
            continue
        # DOTA header lines (imagesource/gsd) and anything else
        n_skipped += 1
    return counts, unknown, n_yolo, n_skipped, n_ignored


# ---------------------------------------------------------------------------
# Section 1: dataset / label sanity
# ---------------------------------------------------------------------------

def check_dataset(dataset_name: str, data_dir: Path, bounds_sample: int):
    """Report label discovery, per-class GT counts, unknown names, bounds."""
    from src.data.dataset import AerialOBBDataset

    findings = {"dataset": dataset_name, "status": "ok", "problems": []}

    try:
        ds = AerialOBBDataset(dataset_name=dataset_name, data_dir=data_dir,
                              split="val")
    except Exception as e:
        findings["status"] = "error"
        findings["problems"].append(f"dataset construction failed: {e}")
        return findings

    n_imgs = len(ds)
    labeled = [s for s in ds.samples if s["label_path"] is not None]
    unlabeled = n_imgs - len(labeled)
    findings["val_images"] = n_imgs
    findings["labeled_images"] = len(labeled)

    label_dirs = Counter(str(Path(s["label_path"]).parent) for s in labeled)
    findings["label_directories"] = dict(label_dirs)
    findings["active_label_dir"] = (label_dirs.most_common(1)[0][0]
                                    if label_dirs else None)

    if n_imgs == 0:
        findings["status"] = "fail"
        findings["problems"].append("0 validation images discovered - check "
                                   "DATASET_STATIC_PATHS / data layout")
        return findings
    if len(labeled) == 0:
        findings["status"] = "fail"
        findings["problems"].append("0 label files matched the val images")
        return findings
    if unlabeled > 0.2 * n_imgs:
        findings["status"] = "warn"
        findings["problems"].append(
            f"{unlabeled}/{n_imgs} val images have NO label file "
            f"({100.0 * unlabeled / n_imgs:.1f}%)")

    class_names = ds.class_names
    total_counts = np.zeros(len(class_names), dtype=np.int64)
    unknown_names = Counter()
    yolo_lines = 0
    skipped_lines = 0
    ignored_boxes = 0

    for s in labeled:
        c, u, y, sk, ig = scan_label_file(Path(s["label_path"]), class_names)
        total_counts += c
        unknown_names.update(u)
        yolo_lines += y
        skipped_lines += sk
        ignored_boxes += ig

    findings["gt_boxes_per_class"] = {n: int(total_counts[i])
                                      for i, n in enumerate(class_names)}
    findings["total_gt_boxes"] = int(total_counts.sum())
    findings["label_style"] = "yolo_obb" if yolo_lines else "polygon_dota_style"
    findings["unparsed_lines"] = int(skipped_lines)
    findings["ignored_boxes"] = int(ignored_boxes)

    empty_classes = [n for n in class_names if total_counts[class_names.index(n)] == 0]
    if len(empty_classes) == len(class_names):
        findings["status"] = "fail"
        findings["problems"].append(
            "0 GT boxes parsed for EVERY class - labels exist but nothing "
            "parses (format or class-name mismatch). cross-eval will report 0 "
            "for every model.")
    elif empty_classes:
        findings["status"] = "warn"
        findings["problems"].append(
            f"classes with 0 GT boxes in val: {', '.join(empty_classes)}")
    if unknown_names:
        findings["status"] = "warn" if findings["status"] == "ok" else findings["status"]
        top = ", ".join(f"{k}({v})" for k, v in unknown_names.most_common(12))
        findings["problems"].append(
            f"annotation class names NOT in {dataset_name} class list "
            f"(boxes ignored by the parser): {top}")

    # -- bounds sanity on a sample: boxes must live inside the real image ----
    out_of_bounds, checked_boxes, opened = 0, 0, 0
    from PIL import Image
    sample_idx_list = [ds.samples.index(s) for s in labeled[:bounds_sample]]
    for s_i in sample_idx_list:
        try:
            with Image.open(ds.samples[s_i]["image_path"]) as im:
                W, H = im.size
            opened += 1
        except Exception:
            continue
        gt = ds.get_ground_truth(s_i, img_width=W, img_height=H)
        for c in range(len(class_names)):
            boxes = gt[c].get("boxes", np.zeros((0, 5)))
            for b in boxes:
                checked_boxes += 1
                cx, cy, w, h = b[0], b[1], b[2], b[3]
                if not (0 <= cx <= W and 0 <= cy <= H and w < 2 * W and h < 2 * H):
                    out_of_bounds += 1
    if opened and checked_boxes:
        oob_pct = 100.0 * out_of_bounds / checked_boxes
        findings["bounds_checked_boxes"] = checked_boxes
        findings["bounds_outside_image_pct"] = round(oob_pct, 2)
        if oob_pct > 5.0:
            findings["status"] = "fail"
            findings["problems"].append(
                f"{oob_pct:.1f}% of GT boxes fall outside the actual image "
                f"bounds -> label/image resolution mismatch (labels written "
                f"for a different image size). Every IoU is destroyed and all "
                f"models will score ~0.")

    return findings


def _index_of(ds, sample):
    try:
        return ds.samples.index(sample)
    except ValueError:
        return 0


def _n_boxes(entry):
    if entry is None:
        return 0
    return len(np.asarray(entry.get("boxes", np.zeros((0, 5))),
                           dtype=np.float32).reshape(-1, 5))


# ---------------------------------------------------------------------------
# Section 2: harness self-test (ground truth as a perfect model)
# ---------------------------------------------------------------------------

def harness_selftest(eval_ds_name: str, data_dir: Path, n_samples: int):
    """Run the exact cross_dataset_eval remap + metrics path with ground truth
    playing the role of a perfect model, for every source class space.
    A sound harness must return mAP50 ~= 1.0 in common mode."""
    from src.data.dataset import AerialOBBDataset
    import importlib
    cde = importlib.import_module("scripts.cross_dataset_eval")

    ds = AerialOBBDataset(dataset_name=eval_ds_name, data_dir=data_dir,
                          split="val", max_samples=n_samples)
    results = []
    if len(ds) == 0:
        return [{"pair": f"?->{eval_ds_name}", "status": "fail",
                 "detail": "no val images"}]

    eval_names = list(ds.class_names)
    all_gt = []
    for i in range(len(ds)):
        im = ds.get_image(i)
        g = ds.get_ground_truth(i, img_width=im.size[0], img_height=im.size[1])
        all_gt.append({c: v for c, v in enumerate(g)} if isinstance(g, list) else g)

    class _Args:
        conf_thresh = 0.25
        iou_thresh = 0.50
        batch_size = 8
        match_mode = "common"

    for src_name in SUPPORTED_DATASETS:
        src_names = list(DATASET_CLASSES[src_name])
        src_to_eval = cde.build_class_map(src_names, eval_names)

        # Perfect-model simulation: each eval class's GT is predicted by the
        # FIRST source class that maps to it (canonical inverse). Feeding GT
        # through the real remap must then reproduce the same boxes -> mAP 1.
        inverse = {}
        for s_idx, e_idx in src_to_eval.items():
            if e_idx is not None and e_idx not in inverse:
                inverse[e_idx] = s_idx

        perfect_preds = []
        for img_gt in all_gt:
            pred_src = cde.empty_preds(len(src_names))
            for e_idx, s_idx in inverse.items():
                entry = img_gt[e_idx] if e_idx in img_gt else None
                boxes = (np.asarray(entry.get("boxes", np.zeros((0, 5))),
                                    dtype=np.float32).reshape(-1, 5)
                         if entry is not None else np.zeros((0, 5), dtype=np.float32))
                if len(boxes):
                    pred_src[s_idx]["boxes"] = boxes
                    pred_src[s_idx]["scores"] = np.full(len(boxes), 0.9, dtype=np.float32)
            perfect_preds.append(pred_src)

        mapped_targets = sorted({v for v in src_to_eval.values() if v is not None})
        eval_index_map = {old: new for new, old in enumerate(mapped_targets)}
        n_sub = len(mapped_targets)

        all_gt_sub, all_pred_sub = [], []
        for img_gt, preds in zip(all_gt, perfect_preds):
            gt_sub = [None] * n_sub
            for c in range(len(eval_names)):
                v = img_gt.get(c)
                if v is not None and c in eval_index_map:
                    gt_sub[eval_index_map[c]] = v
            all_gt_sub.append(gt_sub)
            all_pred_sub.append(cde.remap_predictions(
                preds, src_to_eval, eval_index_map, n_target=n_sub))
        from src.metrics import compute_map_metrics
        names = [eval_names[i] for i in mapped_targets]
        res = compute_map_metrics(all_gt_sub, all_pred_sub, names)

        # expected ceiling: fraction of GT boxes that lie in mapped classes
        total_gt_boxes = sum(_n_boxes(v) for img_gt in all_gt for v in
                             (img_gt.values() if isinstance(img_gt, dict) else img_gt))
        covered = sum(_n_boxes(img_gt[ei]) for img_gt in all_gt for ei in inverse)
        ceiling = (covered / total_gt_boxes) if total_gt_boxes else 0.0

        map50 = float(res["map50"])
        pair = f"{src_name}->{eval_ds_name}"
        if total_gt_boxes == 0:
            status, detail = "skip", "no GT boxes in sample"
        elif map50 < 0.98:
            status = "fail"
            detail = (f"perfect-model mAP50={map50:.3f} (expected ~=1.0) — "
                      f"the harness remap/metric path corrupts this pair")
        else:
            status, detail = "ok", f"perfect-model mAP50={map50:.3f} (harness sound)"
        results.append({"pair": pair, "status": status, "map50": round(map50, 4),
                        "gt_coverage": round(ceiling, 3), "detail": detail})
    return results


# ---------------------------------------------------------------------------
# Section 3: checkpoint integrity
# ---------------------------------------------------------------------------

def check_checkpoints(weights_dir: Path):
    """Load every checkpoint, verify state-dict compatibility and training
    convergence history."""
    import torch
    from src.models import get_model
    from src.config import DATASET_CLASSES as CLS

    findings = []
    # discovery mirrors scripts/cross_dataset_eval.find_trained_weights
    for src in SUPPORTED_DATASETS:
        for model in ["std", "rvsa", "ars-detr", "oriented-former", "rio-detr",
                      "rhino", "ao2-detr", "swin-obb", "lsknet", "yolo11-obb",
                      "custom-obb", "yolov8n-obb", "yolo11n-obb"]:
            cands = [weights_dir / f"{model}_{src}_best.pt",
                     weights_dir / f"{model}_{src}" / f"{model}_best.pt",
                     weights_dir / f"{model}_{src}" / "best.pt",
                     weights_dir / f"{model}_{src}" / "weights" / "best.pt"]
            path = next((c for c in cands if c.exists()), None)
            if path is None:
                continue

            f = {"model": model, "source": src, "path": str(path),
                 "status": "ok", "problems": []}
            try:
                ck = torch.load(path, map_location="cpu")
            except Exception as e:
                f["status"] = "fail"
                f["problems"].append(f"torch.load failed: {e}")
                findings.append(f)
                continue

            if isinstance(ck, dict):
                f["epoch"] = ck.get("epoch")
                f["num_classes_in_ckpt"] = ck.get("num_classes")
                f["best_val_loss"] = (round(float(ck["best_val_loss"]), 5)
                                      if isinstance(ck.get("best_val_loss"), (int, float))
                                      else ck.get("best_val_loss"))
                hist = ck.get("history") or []
                if hist:
                    def _vl(h):
                        for key in ("val_loss", "val", "loss"):
                            if key in h:
                                v = h[key]
                                try:
                                    return float(v)
                                except Exception:
                                    return None
                        return None
                    first, last = _vl(hist[0]), _vl(hist[-1])
                    f["val_loss_first"] = first
                    f["val_loss_last"] = last
                    if first is not None and last is not None:
                        improved = last < first
                        f["val_loss_improved"] = bool(improved)
                        if not improved:
                            f["status"] = "warn"
                            f["problems"].append(
                                "val loss did NOT decrease between first and "
                                "last epoch - this checkpoint likely learned "
                                "nothing; near-zero eval results are genuine")

                state = ck.get("model_state")
                if state is None and "model" in ck:
                    # Ultralytics-style checkpoint (whole nn.Module)
                    f["checkpoint_type"] = "ultralytics"
                elif state is not None and hasattr(state, "items"):
                    f["checkpoint_type"] = "state_dict"
                    n_cls = ck.get("num_classes") or len(CLS.get(src, []))
                    try:
                        det = get_model(model, device="cpu", num_classes=int(n_cls))
                        det.load()  # random init, architecture only
                        res = det.net.load_state_dict(state, strict=False)
                        missing, unexpected = len(res.missing_keys), len(res.unexpected_keys)
                        f["state_missing_keys"] = missing
                        f["state_unexpected_keys"] = unexpected
                        if missing or unexpected:
                            f["status"] = "fail"
                            f["problems"].append(
                                f"state-dict MISMATCH: {missing} missing / "
                                f"{unexpected} unexpected keys — load_state_dict("
                                f"strict=False) silently ran this model with "
                                f"{missing} random-init layers")
                    except Exception as e:
                        f["status"] = "warn"
                        f["problems"].append(f"architecture check failed: {e}")
            else:
                f["checkpoint_type"] = "raw_state_dict"

            findings.append(f)
    return findings


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Dataset / harness / checkpoint diagnostics")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--datasets", nargs="+", default=None,
                    choices=SUPPORTED_DATASETS)
    ap.add_argument("--skip-checkpoints", action="store_true")
    ap.add_argument("--skip-harness", action="store_true")
    ap.add_argument("--harness-samples", type=int, default=40)
    ap.add_argument("--bounds-sample", type=int, default=200)
    args = ap.parse_args()

    if not args.data_dir and not args.weights_dir:
        ap.error("provide --data-dir and/or --weights-dir")

    datasets = args.datasets or list(SUPPORTED_DATASETS)
    report = {"data": [], "harness": [], "checkpoints": []}
    failures = 0
    t0 = time.time()

    print("=" * 78)
    print("DIAGNOSTIC 1/3: dataset & label sanity")
    print("=" * 78)
    if args.data_dir:
        for name in datasets:
            print(f"\n--- {name} ---")
            f = check_dataset(name, Path(args.data_dir), args.bounds_sample)
            report["data"].append(f)
            print(json.dumps(f, indent=2, ensure_ascii=False))
            if f["status"] == "fail":
                failures += 1
    else:
        print("skipped (--data-dir not given)")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC 2/3: harness self-test (ground truth as perfect model)")
    print("=" * 78)
    if args.data_dir and not args.skip_harness:
        for name in datasets:
            print(f"\n--- eval on {name} (all source class spaces) ---")
            try:
                res = harness_selftest(name, Path(args.data_dir),
                                       args.harness_samples)
            except Exception as e:
                import traceback
                traceback.print_exc()
                res = [{"pair": f"?->{name}", "status": "fail",
                        "detail": f"self-test crashed: {e}"}]
            report["harness"].extend(res)
            for r in res:
                mark = {"ok": "[PASS]", "fail": "[FAIL]", "skip": "[SKIP]"}[r["status"]]
                print(f"  {mark} {r['pair']:24s} mAP50={r.get('map50')}  {r['detail']}")
                if r["status"] == "fail":
                    failures += 1
    else:
        print("skipped")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC 3/3: checkpoint integrity & training history")
    print("=" * 78)
    if args.weights_dir and not args.skip_checkpoints:
        try:
            for f in check_checkpoints(Path(args.weights_dir)):
                report["checkpoints"].append(f)
                mark = {"ok": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}[f["status"]]
                print(f"  {mark} {f['model']:14s} ({f['source']}) epoch="
                      f"{f.get('epoch')} best_val_loss={f.get('best_val_loss')}"
                      f" missing_keys={f.get('state_missing_keys')} "
                      f"unexpected={f.get('state_unexpected_keys')}")
                for p in f["problems"]:
                    print(f"        - {p}")
                if f["status"] == "fail":
                    failures += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            report["checkpoints"] = [{"status": "fail", "problems": [str(e)]}]
            failures += 1
    else:
        print("skipped")

    out_path = Path(args.data_dir or ".")
    try:
        with open(out_path.parent / "diagnose_report.json", "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[*] full report -> {out_path.parent / 'diagnose_report.json'}")
    except Exception:
        pass

    print("=" * 78)
    if failures:
        print(f"VERDICT: {failures} check(s) FAILED - see [FAIL] entries above.")
        print("Interpretation: data [FAIL] = fix labels before trusting any "
              "numbers; harness [FAIL] = the eval code corrupts results; "
              "checkpoint [FAIL]/[WARN] = the model weights themselves.")
        sys.exit(1)
    print("VERDICT: data, harness and checkpoints all passed. Any near-zero "
          "evaluation result is genuine model performance, not a bug.")
    print(f"(completed in {time.time()-t0:.1f}s)")
    sys.exit(0)


if __name__ == "__main__":
    main()
