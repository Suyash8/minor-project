#!/usr/bin/env python3
"""
evaluate.py — ONE evaluator for every family (v4, W6).

Batched AMP inference, shared rotated NMS, the same confidence/IoU protocol
for every model, computed exclusively through `src/metrics/`:
  - mAP50, mAP50-95, mAP75, per-class AP / precision / recall
  - small/medium/large breakdown (COCO-style area bands)
  - confusion matrix (rows = GT, cols = predicted, last row/col = background)
  - speed (imgs/s, latency ms), parameter count

Results are saved next to the run as eval_<dataset>_<split>.json and printed.
Evaluating the same weights through `scripts/run_evaluation.py` (a shim) or
this CLI gives byte-identical metrics.

Usage:
  python training/evaluate.py --run runs/train/visdrone_yolo26s
  python training/evaluate.py --run runs/train/codrone_lsknet --datasets codrone
  python training/evaluate.py --weights runs/.../best.pt --model lsknet \
      --datasets codrone visdrone dota
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402


# ---------------------------------------------------------------------------
# Ultralytics YOLO-OBB path (unchanged protocol; Ultralytics computes the
# metrics for its own models with its native val()).
# ---------------------------------------------------------------------------
def evaluate_weights(weights: str, dataset: str, split: str = "test",
                     data_root: str | None = None, imgsz: int = 1280,
                     batch: int = 16, device: str | None = None,
                     out_dir: str | None = None, stage_local: bool = False,
                     save: bool = True, verbose: bool = True) -> dict:
    """Run Ultralytics val() on one dataset/split; return (and optionally
    save + print) a structured result dict."""
    from ultralytics import YOLO

    data_root = common.resolve_data_root(data_root)
    if dataset == "mock":
        # CI dataset: use the YAML written by create_mock_dataset verbatim
        cand = os.path.join(data_root, "mock", "mock.yaml")
        if not os.path.isfile(cand):
            raise FileNotFoundError(f"mock dataset yaml not found: {cand}")
        ds_yaml = cand
    elif stage_local:
        data_root, ds_yaml = common.stage_local(dataset, data_root)
    else:
        ds_yaml = common.ensure_dataset_yaml(dataset, data_root)

    model = YOLO(weights)
    val_name = f"eval_{dataset}_{split}"
    m = model.val(data=ds_yaml, split=split, imgsz=imgsz, batch=batch,
                  device=device, workers=0 if device == "cpu" else 8,
                  plots=True, verbose=False,
                  project=out_dir or os.path.dirname(os.path.abspath(weights)),
                  name=val_name, exist_ok=True)

    box = m.box
    per_class = []
    for i, cidx in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        c = int(cidx)
        per_class.append(dict(class_id=c,
                              name=common.CLASSES[c] if c < len(common.CLASSES) else f"cls{c}",
                              precision=round(float(p), 4),
                              recall=round(float(r), 4),
                              ap50=round(float(ap50), 4),
                              ap50_95=round(float(ap), 4)))
    per_class.sort(key=lambda d: d["class_id"])

    n_params = sum(p.numel() for p in model.model.parameters())
    res = dict(
        weights=os.path.abspath(weights), weights_fingerprint=common.weights_fingerprint(weights),
        dataset=dataset, split=split, imgsz=imgsz,
        overall={k: round(float(v), 5) for k, v in m.results_dict.items()
                 if isinstance(v, (int, float))},
        map50=round(float(box.map50), 5),
        map50_95=round(float(box.map), 5),
        map75=round(float(box.map75), 5),
        speed_ms={k: round(float(v), 2) for k, v in (m.speed or {}).items()},
        parameters=int(n_params),
        per_class=per_class,
        gt_instances=common.count_instances(dataset, data_root, split),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if save:
        out_dir = out_dir or os.path.dirname(os.path.abspath(weights))
        out = os.path.join(out_dir, f"eval_{dataset}_{split}.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        res["json_path"] = out

    if verbose:
        print(f"\n=== {os.path.basename(weights)} on {dataset}/{split} "
              f"(imgsz={imgsz}) ===")
        print(f"mAP50: {res['map50']:.4f}   mAP50-95: {res['map50_95']:.4f}   "
              f"mAP75: {res['map75']:.4f}   params: {n_params/1e6:.2f}M")
        if res["speed_ms"]:
            sp = res["speed_ms"]
            print("speed(ms): " + "  ".join(f"{k}={v}" for k, v in sp.items()))
        print(f"{'class':<14}{'GT':>7}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>9}")
        for d in per_class:
            print(f"{d['name']:<14}{res['gt_instances'].get(d['name'], 0):>7}"
                  f"{d['precision']:>8.3f}{d['recall']:>8.3f}"
                  f"{d['ap50']:>8.3f}{d['ap50_95']:>9.3f}")
        missing = [c for c in common.CLASSES
                   if c not in {d["name"] for d in per_class}]
        if missing:
            print("classes with no predictions/GT in this split:", missing)
        if save:
            print("saved ->", res.get("json_path"))
    return res


def evaluate_yolo_unified(
    weights: str,
    dataset: str,
    split: str = "test",
    data_root: Optional[str] = None,
    imgsz: int = 1280,
    batch: int = 8,
    device: Optional[str] = None,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.45,
    max_batches: Optional[int] = None,
    out_dir: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """YOLO-family evaluation through the SHARED src/metrics protocol (W6.3).

    Ultralytics runs inference only; mAP, per-class AP, sm/md/lg and the
    confusion matrix are computed by src/metrics exactly as for dense/DETR.
    GT is taken in ORIGINAL image coordinates (identical at every imgsz,
    B1); predictions are inverse-letterboxed into the same frame (R3).
    Angle convention: Ultralytics OBB emits xywhr with angle in radians;
    converted to degrees mod 180 (the OpenCV-style convention the metrics
    expect).
    """
    import torch
    from ultralytics import YOLO

    from src.data.augment import letterbox
    from src.data.dataset import AerialOBBDataset
    from src.metrics.detection_map import compute_map_metrics, evaluate_class_detection
    from src.metrics.geometry import ultralytics_xywhr_to_suite

    if device in (None, "auto"):
        device = "0" if torch.cuda.is_available() else "cpu"

    dev = torch.device("cuda:0" if str(device) in ("0", "cuda") else str(device))
    raw = AerialOBBDataset(dataset_name=dataset, data_dir=data_root, split=split)
    if len(raw) == 0:
        raise FileNotFoundError(f"No {dataset}/{split} images under {data_root}")
    class_names = list(raw.class_names)
    num_classes = len(class_names)
    model = YOLO(weights)

    indices = list(range(len(raw)))
    if max_batches is not None:
        indices = indices[: max_batches * max(batch, 1)]

    def _prep(idx):
        """Untimed data prep: image into letterboxed input space; GT stays
        in ORIGINAL coordinates (B1: identical GT at every eval imgsz)."""
        img = raw.get_image_array(idx)  # RGB uint8, original resolution
        orig_size = (img.shape[1], img.shape[0])
        # B1: GT straight from the dataset in original px - no letterbox,
        # no MIN_BOX_PX drop, no canvas clipping.
        gt_img = _original_gt_per_class(raw, idx, img.shape[1],
                                        img.shape[0], num_classes)
        lb_img, _ = letterbox(img, np.zeros((0, 4, 2), dtype=np.float32),
                              imgsz)
        return lb_img, gt_img, orig_size

    all_gt: List[List[Dict[str, Any]]] = []
    all_pred: List[List[Dict[str, Any]]] = []
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    n_images = 0
    timed_images = 0
    t_infer = 0.0
    t_start = time.time()

    # Batched inference (review B); only forward+NMS is timed, after one
    # warmup batch, so the number is comparable with dense/DETR latency.
    for start in range(0, len(indices), max(1, batch)):
        chunk = indices[start: start + max(1, batch)]
        prepped = [_prep(i) for i in chunk]
        lb_imgs = [p[0] for p in prepped]
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        results = model.predict(lb_imgs, imgsz=imgsz, conf=conf_thresh,
                                iou=iou_thresh, max_det=1000, device=device,
                                verbose=False)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        if start > 0:  # first batch is the warmup
            t_infer += time.time() - t0
            timed_images += len(chunk)

        for (lb_img, gt_img, orig_size), r in zip(prepped, results):
            pred_lists: List[list] = [[] for _ in range(num_classes)]
            score_lists: List[list] = [[] for _ in range(num_classes)]
            obb = getattr(r, "obb", None)
            if obb is not None and len(obb) > 0:
                xywhr = obb.xywhr.cpu().numpy()  # (N,5) cx,cy,w,h,angle_rad
                conf = obb.conf.cpu().numpy()
                cls = obb.cls.cpu().numpy().astype(int)
                for (cx, cy, w, h, ar), s, c in zip(xywhr, conf, cls):
                    if 0 <= c < num_classes:
                        # Verified conversion (synthetic round-trip test in
                        # tests/test_v4_units.py): same rectangle as the
                        # suite's own polygon fitting for every case.
                        pred_lists[c].append(
                            ultralytics_xywhr_to_suite(cx, cy, w, h, ar))
                        score_lists[c].append(float(s))
            ow, oh = orig_size
            # R3+B1: predictions are inverse-letterboxed into the ORIGINAL
            # image frame where the (unfiltered) GT already lives.
            pred_img = [
                {"boxes": (_unletterbox_boxes(
                    np.asarray(pred_lists[c], dtype=np.float32).reshape(-1, 5),
                    ow, oh, imgsz) if pred_lists[c]
                    else np.zeros((0, 5), dtype=np.float32)),
                 "scores": np.asarray(score_lists[c], dtype=np.float32)}
                for c in range(num_classes)]
            all_gt.append(gt_img)
            all_pred.append(pred_img)
            n_images += 1

    for gt_img, pred_img in zip(all_gt, all_pred):
        _confusion_update(cm, gt_img, pred_img)

    wall_s = time.time() - t_start
    metrics = compute_map_metrics(all_gt, all_pred, class_names)

    per_class = []
    for c, name in enumerate(class_names):
        gt_cls = [all_gt[i][c] for i in range(len(all_gt))]
        pred_cls = [all_pred[i][c] for i in range(len(all_pred))]
        r50 = evaluate_class_detection(gt_cls, pred_cls, iou_threshold=0.50)
        r95 = float(np.mean([
            evaluate_class_detection(gt_cls, pred_cls, iou_threshold=t)["ap"]
            for t in [round(x, 2) for x in
                      [float(i) / 100.0 for i in range(50, 100, 5)]]]))
        per_class.append(dict(class_id=c, name=name,
                              precision=round(float(r50["precision"]), 4),
                              recall=round(float(r50["recall"]), 4),
                              ap50=round(float(r50["ap"]), 4),
                              ap50_95=round(r95, 4)))

    n_params = sum(p.numel() for p in model.model.parameters())
    # Forward+NMS only, after warmup (review B); falls back to wall time
    # when the timed window is empty (single-batch eval).
    if timed_images > 0 and t_infer > 0:
        imgs_per_s = timed_images / t_infer
        timing_basis = "forward+nms (post-warmup)"
    else:
        imgs_per_s = n_images / max(wall_s, 1e-6)
        timing_basis = "wall-clock (single batch; no warmup window)"
    res: Dict[str, Any] = dict(
        comparable=True,
        weights=os.path.abspath(weights), weights_fingerprint=common.weights_fingerprint(weights),
        model=(model_name or os.path.basename(
            os.path.dirname(os.path.abspath(weights)))),
        family="yolo", protocol="src/metrics (unified)", metric_version="v4.2-greedy-unmatched",
        dataset=dataset, split=split, imgsz=imgsz,
        coord_space="original image px (preds inverse-letterbox, GT raw, R3+B1)",
        timing_basis=timing_basis,
        device=str(dev), hardware=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"),
        max_det=1000,
        map50=round(float(metrics["map50"]), 5),
        map50_95=round(float(metrics["map50_95"]), 5),
        map75=round(float(metrics["map75"]), 5),
        ap_small=round(float(metrics["ap_small"]), 5),
        ap_medium=round(float(metrics["ap_medium"]), 5),
        ap_large=round(float(metrics["ap_large"]), 5),
        speed_ms={"inference": round(1000.0 / max(imgs_per_s, 1e-6), 2)},
        imgs_per_sec=round(imgs_per_s, 2),
        parameters=int(n_params),
        per_class=per_class,
        per_class_ap50={k: round(float(v), 5)
                        for k, v in metrics["per_class_ap50"].items()},
        confusion_matrix=cm.tolist(),
        class_names=class_names,
        gt_instances=(common.count_instances(dataset, data_root, split)
                      if dataset in common.REAL_DATASETS else {}),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if save:
        out_dir = out_dir or os.path.dirname(os.path.abspath(weights))
        out = os.path.join(out_dir, f"eval_{dataset}_{split}.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        res["json_path"] = out

    if verbose:
        print(f"\n=== YOLO (unified src/metrics protocol) on "
              f"{dataset}/{split} (imgsz={imgsz}) ===")
        print(f"mAP50: {res['map50']:.4f}   mAP50-95: {res['map50_95']:.4f}   "
              f"mAP75: {res['map75']:.4f}   params: {n_params / 1e6:.2f}M   "
              f"{res['imgs_per_sec']} imgs/s")
        print(f"scale AP50: small {res['ap_small']:.4f} | "
              f"medium {res['ap_medium']:.4f} | large {res['ap_large']:.4f}")
        if save:
            print("saved ->", res.get("json_path"))
    return res


# ---------------------------------------------------------------------------
# Generic dense/DETR path — same protocol for both PyTorch families (W6.1).
# ---------------------------------------------------------------------------
def _original_gt_per_class(raw, idx: int, orig_w: int, orig_h: int,
                           num_classes: int) -> List[Dict[str, Any]]:
    """Evaluation GT in ORIGINAL image coordinates, straight from the
    dataset (review B1): NO letterbox, NO MIN_BOX_PX filter, NO canvas
    clipping. The GT handed to metrics for a given split is identical at
    every eval imgsz and for every family, which is what makes the
    small/medium/large area bands (and mAP) comparable.
    """
    gt = raw.get_ground_truth(idx, orig_w, orig_h)
    out: List[Dict[str, Any]] = []
    for c in range(num_classes):
        boxes = np.asarray(gt[c].get("boxes", np.zeros((0, 5))),
                           dtype=np.float32).reshape(-1, 5)
        out.append({"boxes": boxes})
    return out


def _inverse_letterbox_params(orig_w: int, orig_h: int, imgsz: int):
    """(inverse scale, pad_x, pad_y) mapping letterbox px -> original px.

    Mirrors src.data.augment.letterbox exactly (uniform scale on the long
    side, centered padding), so it is the exact inverse of the transform
    applied to GT corners.
    """
    scale = imgsz / float(max(orig_h, orig_w))
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    return 1.0 / scale, pad_x, pad_y


def _unletterbox_boxes(boxes: np.ndarray, orig_w: int, orig_h: int,
                       imgsz: int) -> np.ndarray:
    """xywhr (degrees) boxes from the letterboxed imgsz frame -> original
    image frame (review R3: area bands must be measured in original-image
    px so small/medium/large are comparable across per-family imgsz).
    Uniform scale + translation preserves angle.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    if len(boxes) == 0:
        return boxes
    inv, pad_x, pad_y = _inverse_letterbox_params(orig_w, orig_h, imgsz)
    out = boxes.copy()
    out[:, 0] = (out[:, 0] - pad_x) * inv
    out[:, 1] = (out[:, 1] - pad_y) * inv
    out[:, 2] *= inv
    out[:, 3] *= inv
    return out


def _confusion_update(cm: np.ndarray,
                      gt: List[Dict[str, Any]],
                      pred: List[Dict[str, Any]],
                      iou_thresh: float = 0.5) -> None:
    """Greedy score-ordered matching; cm[gt, pred], last index = background."""
    from src.metrics.geometry import compute_obb_iou_matrix

    nc1 = cm.shape[0] - 1
    gt_boxes, gt_labels = [], []
    for c, entry in enumerate(gt):
        for b in entry["boxes"]:
            gt_boxes.append(b)
            gt_labels.append(c)
    pred_boxes, pred_scores, pred_labels = [], [], []
    for c, entry in enumerate(pred):
        for b, s in zip(entry["boxes"], entry["scores"]):
            pred_boxes.append(b)
            pred_scores.append(float(s))
            pred_labels.append(c)

    if not gt_boxes and not pred_boxes:
        return
    if not gt_boxes:
        for pl in pred_labels:
            cm[nc1, pl] += 1
        return
    if not pred_boxes:
        for gl in gt_labels:
            cm[gl, nc1] += 1
        return

    gt_arr = np.stack(gt_boxes).astype(np.float32)
    pred_arr = np.stack(pred_boxes).astype(np.float32)
    iou = compute_obb_iou_matrix(gt_arr, pred_arr)  # (G, P)
    order = np.argsort(-np.asarray(pred_scores))
    matched_gt: set = set()
    for p_idx in order:
        col = iou[:, p_idx]
        g_idx = int(np.argmax(col))
        if col[g_idx] >= iou_thresh and g_idx not in matched_gt:
            matched_gt.add(g_idx)
            cm[gt_labels[g_idx], pred_labels[p_idx]] += 1
        else:
            cm[nc1, pred_labels[p_idx]] += 1
    for g_idx, gl in enumerate(gt_labels):
        if g_idx not in matched_gt:
            cm[gl, nc1] += 1


def evaluate_generic(
    model_name: str,
    weights: str,
    dataset: str,
    split: str = "test",
    data_root: Optional[str] = None,
    imgsz: int = 1280,
    batch: int = 8,
    device: Optional[str] = None,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.45,
    max_batches: Optional[int] = None,
    out_dir: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Unified evaluation for the dense and DETR PyTorch families.

    One protocol: letterboxed inputs, shared decode (W5.3) + shared rotated
    NMS (W5.4), metrics exclusively from `src/metrics/`.
    """
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import AerialOBBDataset
    from src.metrics.detection_map import compute_map_metrics, evaluate_class_detection
    from src.models import get_model
    from src.training.dataset import UnifiedOBBDataset, obb_collate_fn
    from src.training.decode import decode_dense_grid, decode_detr, to_metric_format
    from src.training.dispatcher import get_model_family
    from src.training.engine_common import AmpContext
    from src.training.train_dense import _normalize_on_gpu, forward_dense_network
    from src.training.train_detr import extract_detr_predictions

    family = get_model_family(model_name)
    if family == "yolo":
        return evaluate_weights(weights, dataset, split=split,
                                data_root=data_root, imgsz=imgsz, batch=batch,
                                device=device, out_dir=out_dir, save=save,
                                verbose=verbose)

    if device in (None, "auto"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    raw = AerialOBBDataset(dataset_name=dataset, data_dir=data_root, split=split)
    if len(raw) == 0:
        raise FileNotFoundError(f"No {dataset}/{split} images under {data_root}")
    class_names = list(raw.class_names)
    num_classes = len(class_names)
    ds = UnifiedOBBDataset(raw, img_size=imgsz, augment=False,
                           resize_mode="letterbox", normalize_on_gpu=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        collate_fn=obb_collate_fn, num_workers=0,
                        pin_memory=(dev.type == "cuda"))

    detector = get_model(model_name, device=str(dev), num_classes=num_classes)
    detector.load(weights_path=weights)
    net = detector.net.to(dev)
    net.eval()
    amp = AmpContext(dev)

    all_gt: List[List[Dict[str, Any]]] = []
    all_pred: List[List[Dict[str, Any]]] = []
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    n_images = 0
    timed_images = 0
    t_infer = 0.0
    t_start = time.time()

    with torch.no_grad():
        for b_idx, (imgs, targets) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            imgs = _normalize_on_gpu(imgs, dev)
            # Forward+decode only is timed, after a one-batch warmup, so the
            # latency is comparable across families (review B).
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with amp.autocast():
                if family == "detr":
                    raw_out = net(imgs)
                    pred_logits, pred_boxes, pred_angles = \
                        extract_detr_predictions(raw_out, model_name=model_name)
                    decoded = decode_detr(
                        pred_logits.float(), pred_boxes.float(),
                        pred_angles.float(), img_size=imgsz,
                        num_classes=num_classes, conf_thresh=conf_thresh,
                        nms_iou=iou_thresh)
                else:
                    cls_logits, reg_preds, _fh, _fw = forward_dense_network(
                        model_name, net, imgs, num_classes)
                    decoded = decode_dense_grid(
                        cls_logits.float(), reg_preds.float(), img_size=imgsz,
                        conf_thresh=conf_thresh, nms_iou=iou_thresh)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            if b_idx > 0:  # first batch is the warmup
                t_infer += time.time() - t0
                timed_images += imgs.shape[0]

            # R3+B1: inverse-letterbox preds into the ORIGINAL image
            # frame; GT comes straight from the dataset in original
            # coordinates (identical at every imgsz, never min-box
            # filtered), so area bands and mAP compare across families.
            batch_pred = to_metric_format(decoded, num_classes)
            for pred_img, tgt in zip(batch_pred, targets):
                ow, oh = (int(v) for v in tgt["orig_size"])
                for c in range(num_classes):
                    pred_img[c]["boxes"] = _unletterbox_boxes(
                        pred_img[c]["boxes"], ow, oh, imgsz)
                gt_img = _original_gt_per_class(raw, int(tgt["index"]),
                                                ow, oh, num_classes)
                all_gt.append(gt_img)
                all_pred.append(pred_img)
            n_images += imgs.shape[0]

    for gt_img, pred_img in zip(all_gt, all_pred):
        _confusion_update(cm, gt_img, pred_img)

    wall_s = time.time() - t_start
    metrics = compute_map_metrics(all_gt, all_pred, class_names)

    per_class = []
    for c, name in enumerate(class_names):
        gt_cls = [all_gt[i][c] for i in range(len(all_gt))]
        pred_cls = [all_pred[i][c] for i in range(len(all_pred))]
        r50 = evaluate_class_detection(gt_cls, pred_cls, iou_threshold=0.50)
        r95 = float(np.mean([
            evaluate_class_detection(gt_cls, pred_cls, iou_threshold=t)["ap"]
            for t in [round(x, 2) for x in
                      [float(i) / 100.0 for i in range(50, 100, 5)]]]))
        per_class.append(dict(class_id=c, name=name,
                              precision=round(float(r50["precision"]), 4),
                              recall=round(float(r50["recall"]), 4),
                              ap50=round(float(r50["ap"]), 4),
                              ap50_95=round(r95, 4)))

    n_params = int(getattr(detector, "num_params", 0) or
                   sum(p.numel() for p in net.parameters()))
    if timed_images > 0 and t_infer > 0:
        imgs_per_s = timed_images / t_infer
        timing_basis = "forward+decode (post-warmup)"
    else:
        imgs_per_s = n_images / max(wall_s, 1e-6)
        timing_basis = "wall-clock (single batch; no warmup window)"
    res: Dict[str, Any] = dict(
        weights=os.path.abspath(weights), weights_fingerprint=common.weights_fingerprint(weights),
        model=model_name, family=family,
        protocol="src/metrics (unified)", comparable=True, metric_version="v4.2-greedy-unmatched",
        dataset=dataset, split=split, imgsz=imgsz,
        coord_space="original image px (preds inverse-letterbox, GT raw, R3+B1)",
        timing_basis=timing_basis,
        device=str(dev), hardware=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"),
        max_det=1000,
        map50=round(float(metrics["map50"]), 5),
        map50_95=round(float(metrics["map50_95"]), 5),
        map75=round(float(metrics["map75"]), 5),
        ap_small=round(float(metrics["ap_small"]), 5),
        ap_medium=round(float(metrics["ap_medium"]), 5),
        ap_large=round(float(metrics["ap_large"]), 5),
        speed_ms={"inference": round(1000.0 / max(imgs_per_s, 1e-6), 2)},
        imgs_per_sec=round(imgs_per_s, 2),
        parameters=n_params,
        per_class=per_class,
        per_class_ap50={k: round(float(v), 5)
                        for k, v in metrics["per_class_ap50"].items()},
        confusion_matrix=cm.tolist(),
        class_names=class_names,
        gt_instances=(common.count_instances(dataset, data_root, split)
                      if dataset in common.REAL_DATASETS else {}),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if save:
        out_dir = out_dir or os.path.dirname(os.path.abspath(weights))
        out = os.path.join(out_dir, f"eval_{dataset}_{split}.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        res["json_path"] = out

    if verbose:
        print(f"\n=== {model_name} ({family}) on {dataset}/{split} "
              f"(imgsz={imgsz}) ===")
        print(f"mAP50: {res['map50']:.4f}   mAP50-95: {res['map50_95']:.4f}   "
              f"mAP75: {res['map75']:.4f}   params: {n_params / 1e6:.2f}M   "
              f"{res['imgs_per_sec']} imgs/s")
        print(f"scale AP50: small {res['ap_small']:.4f} | "
              f"medium {res['ap_medium']:.4f} | large {res['ap_large']:.4f}")
        print(f"{'class':<14}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>9}")
        for d in per_class:
            print(f"{d['name']:<14}{d['precision']:>8.3f}{d['recall']:>8.3f}"
                  f"{d['ap50']:>8.3f}{d['ap50_95']:>9.3f}")
        if save:
            print("saved ->", res.get("json_path"))
    return res


# ---------------------------------------------------------------------------
# Run-level entry (used by training/run.py and the CLI)
# ---------------------------------------------------------------------------
def _model_for_run(run_dir: str, weights: str,
                   explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    man = os.path.join(run_dir, "run_manifest.json")
    if os.path.isfile(man):
        try:
            with open(man) as f:
                payload = json.load(f)
            name = payload.get("model") or payload.get("model_name")
            if name:
                return name
        except Exception:
            pass
    # runs/train/<dataset>_<model>: dataset names carry no underscore
    base = os.path.basename(os.path.normpath(run_dir))
    if "_" in base:
        return base.split("_", 1)[1]
    return base


def evaluate_run(run_dir: str, dataset: str, split: str = "test",
                 imgsz: Optional[int] = None, batch: int = 8,
                 device: Optional[str] = None,
                 data_root: Optional[str] = None,
                 model_name: Optional[str] = None,
                 test_mode: bool = False,
                 save: bool = True,
                 allow_native_fallback: bool = False) -> Dict[str, Any]:
    """Evaluate one training run directory on one dataset/split.

    Every family reports through the shared src/metrics protocol (W6);
    Ultralytics performs YOLO inference only. The Ultralytics-native val
    protocol is reachable ONLY through an explicit opt-in
    (--allow-native-fallback / V4_ALLOW_NATIVE_YOLO_EVAL=1) because its
    numbers are NOT comparable with the unified protocol; when opted in, the
    result JSON is marked protocol/comparable so downstream summaries and
    the cross-eval matrix can refuse to mix it in.
    """
    from src.training.dispatcher import get_model_family

    if dataset == "mock" and split == "test":
        split = "val"  # the generated mock set ships train/val only
    weights = common.find_weights(run_dir)
    model_name = _model_for_run(run_dir, weights, model_name)
    family = get_model_family(model_name)

    # A5 (review): evaluate at the resolution the model was TRAINED at by
    # default - dense trains at 1024, DETR at 768; forcing 1280 changes
    # strides/priors/object scale and makes cross-model numbers
    # non-comparable. The trained resolution is read from the run manifest;
    # an explicit --imgsz always wins.
    if imgsz is None:
        man = os.path.join(run_dir, "run_manifest.json")
        if os.path.isfile(man):
            try:
                imgsz = int(json.load(open(man)).get(
                    "resolved_tuning", {}).get("img_size")) or None
            except (OSError, ValueError, TypeError):
                imgsz = None
        if imgsz is None:
            imgsz = 1280
        print(f"[eval] imgsz={imgsz} (from run manifest / default)",
              flush=True)

    if family == "yolo":
        try:
            # W6.3: one metric protocol for every family - Ultralytics runs
            # inference only; src/metrics computes all metrics.
            return evaluate_yolo_unified(weights, dataset, split=split,
                                         data_root=data_root, imgsz=imgsz,
                                         batch=batch, device=device,
                                         out_dir=run_dir,
                                         max_batches=(2 if test_mode else None),
                                         save=save, model_name=model_name)
        except Exception as e:
            allowed = (allow_native_fallback or
                       os.environ.get("V4_ALLOW_NATIVE_YOLO_EVAL") == "1")
            if not allowed:
                raise RuntimeError(
                    "unified YOLO evaluation failed and the Ultralytics-"
                    "native fallback is DISABLED by default because its "
                    "metrics are not comparable with the suite protocol "
                    f"(hard constraint: one evaluation protocol). Cause: {e}. "
                    "Fix the cause, or rerun with --allow-native-fallback / "
                    "V4_ALLOW_NATIVE_YOLO_EVAL=1 to opt in explicitly.") from e
            print("=" * 70, flush=True)
            print("[eval][WARNING] unified YOLO eval failed "
                  f"({e}); PRODUCING NON-COMPARABLE METRICS via the "
                  "Ultralytics-native val protocol because the fallback was "
                  "explicitly opted into. This result is excluded from the "
                  "cross-eval matrix and master summary.", flush=True)
            print("=" * 70, flush=True)
            res = evaluate_weights(weights, dataset, split=split,
                                   data_root=data_root, imgsz=imgsz,
                                   batch=batch, device=device, out_dir=run_dir,
                                   save=save)
            res["protocol"] = "ultralytics-native (fallback)"
            res["comparable"] = False
            res["protocol_fallback_reason"] = str(e)
            return res
    return evaluate_generic(model_name, weights, dataset, split=split,
                            data_root=data_root, imgsz=imgsz, batch=batch,
                            device=device, out_dir=run_dir,
                            max_batches=(2 if test_mode else None),
                            save=save)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a trained OBB run "
                                            "(all 23 models, one protocol)")
    p.add_argument("--run", help="run directory containing best.pt")
    p.add_argument("--weights", help="explicit weights path (or use --run)")
    p.add_argument("--model", help="model name when it cannot be inferred "
                                   "from the run manifest / directory name")
    p.add_argument("--datasets", nargs="+",
                   help="default: the run's training dataset (from manifest), "
                        "else all three")
    p.add_argument("--split", default="test", choices=common.SPLITS)
    p.add_argument("--imgsz", type=int, default=None,
                   help="eval resolution; default = the run's trained "
                        "img_size from run_manifest.json (1280 fallback)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--allow-native-fallback", action="store_true",
                   help="explicitly opt into the Ultralytics-native val "
                        "protocol if the unified YOLO evaluator fails; "
                        "produces NON-COMPARABLE metrics that cross_eval and "
                        "summarize_runs exclude")
    aa = p.parse_args(argv)
    if not aa.run and not aa.weights:
        p.error("need --run or --weights")
    return aa


def main(argv=None):
    a = parse_args(argv)
    weights = common.find_weights(a.run or a.weights)
    run_dir = a.run or os.path.dirname(os.path.abspath(weights))
    model_name = _model_for_run(run_dir, weights, a.model)

    datasets = a.datasets
    if not datasets:
        man = os.path.join(run_dir, "run_manifest.json")
        if os.path.isfile(man):
            with open(man) as f:
                td = json.load(f).get("dataset")
            if td in common.DATASETS:
                datasets = [td]
        if not datasets:
            datasets = common.REAL_DATASETS

    data_root = common.resolve_data_root(a.data_root)
    for ds in datasets:
        evaluate_run(run_dir, ds, split=a.split, imgsz=(a.imgsz or None), batch=a.batch,
                     device=a.device, data_root=data_root,
                     model_name=model_name, save=not a.no_save,
                     allow_native_fallback=a.allow_native_fallback)


if __name__ == "__main__":
    main()
