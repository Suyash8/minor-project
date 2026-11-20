#!/usr/bin/env python3
"""
test_v43_accounting.py — Verification suite for v4.3 full-accounting metrics.

Tests:
1. Exact hand-verified 2-image / 3-class accounting scenario (TP/FP/FN, P/R/F1,
   iou_mean, accounting totals, macro_f1 over classes with GT, mAP50 mean).
2. Early-return edge cases in evaluate_class_detection / evaluate_class_multi_iou.
3. Regression identity against pre-v4.3 detection_map.py loaded from git history:
   map50, map75, map50_95, ap_small, ap_medium, ap_large, per_class_ap50,
   per_class_ap75 must be bit-identical.
4. End-to-end eval_all_v43.py verification on temp mock runs (best.pt + last.pt
   fallback, --quick non-destructive smoke, .bak-v42 backup, full eval, resume
   idempotency, error isolation, and exact 4-file bundle/CSV schemas).
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.metrics import detection_map as new_map  # noqa: E402


def _build_spec_scenario():
    """Exact 2-image / 3-class scenario from the v4.3 verification spec."""
    class_names = ["car", "bus", "van"]
    box_a = np.array([[50.0, 50.0, 40.0, 30.0, 0.0]], dtype=np.float32)
    box_b = np.array([[100.0, 100.0, 50.0, 40.0, 0.0]], dtype=np.float32)
    empty_boxes = np.zeros((0, 5), dtype=np.float32)
    empty_scores = np.zeros((0,), dtype=np.float32)

    all_gt = [
        # Image 0: car box A
        [
            {"boxes": box_a.copy()},
            {"boxes": empty_boxes.copy()},
            {"boxes": empty_boxes.copy()},
        ],
        # Image 1: car box B and van box B
        [
            {"boxes": box_b.copy()},
            {"boxes": empty_boxes.copy()},
            {"boxes": box_b.copy()},
        ],
    ]

    p1 = [50.0, 50.0, 40.0, 30.0, 0.0]   # IoU(A, P1) = 1.0
    p2 = [55.0, 50.0, 40.0, 30.0, 0.0]   # IoU(A, P2) = 1050 / 1350 = 0.77778
    p3 = [115.0, 100.0, 50.0, 40.0, 0.0] # IoU(B, P3) = 1400 / 2600 = 0.53846
    bus_p = [[10.0, 10.0, 20.0, 20.0, 0.0]]

    all_preds = [
        # Image 0 predictions
        [
            {"boxes": np.array([p1, p2], dtype=np.float32), "scores": np.array([0.9, 0.8], dtype=np.float32)},
            {"boxes": np.array(bus_p, dtype=np.float32), "scores": np.array([0.5], dtype=np.float32)},
            {"boxes": empty_boxes.copy(), "scores": empty_scores.copy()},
        ],
        # Image 1 predictions
        [
            {"boxes": np.array([p3], dtype=np.float32), "scores": np.array([0.7], dtype=np.float32)},
            {"boxes": empty_boxes.copy(), "scores": empty_scores.copy()},
            {"boxes": empty_boxes.copy(), "scores": empty_scores.copy()},
        ],
    ]
    return all_gt, all_preds, class_names


def test_exact_accounting_scenario() -> None:
    all_gt, all_preds, class_names = _build_spec_scenario()
    res = new_map.compute_map_metrics(all_gt, all_preds, class_names)

    by_name = {d["name"]: d for d in res["per_class"]}
    car = by_name["car"]
    bus = by_name["bus"]
    van = by_name["van"]

    assert car["class_id"] == 0
    assert car["instances"] == 2
    assert car["tp"] == 2
    assert car["fp"] == 1
    assert car["fn"] == 0
    assert math.isclose(car["precision"], 2.0 / 3.0, abs_tol=1e-5), car
    assert math.isclose(car["recall"], 1.0, abs_tol=1e-5), car
    assert math.isclose(car["f1"], 0.8, abs_tol=1e-5), car
    expected_iou_mean = (1.0 + 1400.0 / 2600.0) / 2.0
    assert math.isclose(car["iou_mean"], expected_iou_mean, abs_tol=1e-5), car

    assert bus["class_id"] == 1
    assert bus["instances"] == 0
    assert bus["tp"] == 0
    assert bus["fp"] == 1
    assert bus["fn"] == 0
    assert bus["iou_mean"] is None, bus

    assert van["class_id"] == 2
    assert van["instances"] == 1
    assert van["tp"] == 0
    assert van["fp"] == 0
    assert van["fn"] == 1
    assert van["iou_mean"] is None, van
    assert van["f1"] == 0.0, van

    acc = res["accounting"]
    assert acc["instances"] == 3, acc
    assert acc["predictions"] == 4, acc
    assert acc["tp"] == 2, acc
    assert acc["fp"] == 2, acc
    assert acc["fn"] == 1, acc
    assert acc["iou_pairs"] == 2, acc
    assert math.isclose(acc["iou_mean"], expected_iou_mean, abs_tol=1e-5), acc
    assert math.isclose(acc["macro_f1"], 0.4, abs_tol=1e-5), acc
    assert acc["classes_with_gt"] == 2, acc

    mean_gt_ap50 = (res["per_class_ap50"]["car"] + res["per_class_ap50"]["van"]) / 2.0
    assert res["map50"] == mean_gt_ap50, (res["map50"], mean_gt_ap50)
    print("[ok] test_exact_accounting_scenario PASS")


def test_early_returns_and_single_class() -> None:
    empty_gt = [{"boxes": np.zeros((0, 5), dtype=np.float32)}]
    one_pred = [{"boxes": np.array([[10, 10, 20, 20, 0]], dtype=np.float32), "scores": np.array([0.5], dtype=np.float32)}]
    r_no_gt = new_map.evaluate_class_detection(empty_gt, one_pred, iou_threshold=0.50)
    assert r_no_gt == {
        "ap": 0.0, "precision": 0.0, "recall": 0.0,
        "tp": 0, "fp": 1, "fn": 0,
        "n_gt": 0, "iou_sum": 0.0, "iou_n": 0, "n_preds": 1,
    }, r_no_gt

    one_gt = [{"boxes": np.array([[10, 10, 20, 20, 0]], dtype=np.float32)}]
    empty_pred = [{"boxes": np.zeros((0, 5), dtype=np.float32), "scores": np.zeros(0, dtype=np.float32)}]
    r_no_pred = new_map.evaluate_class_detection(one_gt, empty_pred, iou_threshold=0.50)
    assert r_no_pred == {
        "ap": 0.0, "precision": 0.0, "recall": 0.0,
        "tp": 0, "fp": 0, "fn": 1,
        "n_gt": 1, "iou_sum": 0.0, "iou_n": 0, "n_preds": 0,
    }, r_no_pred
    print("[ok] test_early_returns_and_single_class PASS")


def _load_pre_v43_detection_map() -> types.ModuleType:
    """Load the original pre-v4.3 src/metrics/detection_map.py from git history."""
    revs = subprocess.check_output(
        ["git", "rev-list", "--max-count=20", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
    ).splitlines()
    for rev in revs:
        try:
            src = subprocess.check_output(
                ["git", "show", f"{rev}:src/metrics/detection_map.py"],
                cwd=str(REPO_ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            continue
        if "V43_FULL_ACCOUNTING" not in src:
            mod = types.ModuleType("old_detection_map")
            mod.__file__ = f"<git:{rev}:src/metrics/detection_map.py>"
            exec(compile(src, mod.__file__, "exec"), mod.__dict__)
            assert not hasattr(mod, "V43_FULL_ACCOUNTING")
            return mod
    raise RuntimeError("Could not find pre-v4.3 src/metrics/detection_map.py in git history")


def test_regression_identity_vs_pre_v43() -> None:
    old_map = _load_pre_v43_detection_map()

    scenarios = [_build_spec_scenario()]

    # Second randomized multi-scale, rotated, tie/empty-class scenario
    rng = np.random.default_rng(2026)
    cls_names = ["pedestrian", "car", "van", "bus"]
    rand_gt, rand_pred = [], []
    for _ in range(5):
        g_img, p_img = [], []
        for c_idx in range(len(cls_names)):
            if c_idx == 3:  # empty GT class
                g_boxes = np.zeros((0, 5), dtype=np.float32)
            else:
                n_g = int(rng.integers(0, 4))
                g_boxes = np.column_stack([
                    rng.uniform(40, 400, n_g),
                    rng.uniform(40, 400, n_g),
                    rng.uniform(10, 140, n_g),
                    rng.uniform(10, 140, n_g),
                    rng.uniform(0, 180, n_g),
                ]).astype(np.float32) if n_g else np.zeros((0, 5), dtype=np.float32)
            g_img.append({"boxes": g_boxes})

            n_p = int(rng.integers(0, 5))
            if n_p and len(g_boxes):
                base = g_boxes[rng.integers(0, len(g_boxes), n_p)].copy()
                base[:, :2] += rng.normal(0, 8, (n_p, 2))
                base[:, 4] = (base[:, 4] + rng.normal(0, 5, n_p)) % 180.0
                scores = rng.uniform(0.1, 0.99, n_p).astype(np.float32)
            elif n_p:
                base = np.column_stack([
                    rng.uniform(40, 400, n_p), rng.uniform(40, 400, n_p),
                    rng.uniform(15, 90, n_p), rng.uniform(15, 90, n_p),
                    rng.uniform(0, 180, n_p),
                ]).astype(np.float32)
                scores = rng.uniform(0.1, 0.99, n_p).astype(np.float32)
            else:
                base = np.zeros((0, 5), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
            p_img.append({"boxes": base, "scores": scores})
        rand_gt.append(g_img)
        rand_pred.append(p_img)
    scenarios.append((rand_gt, rand_pred, cls_names))

    identical_keys = (
        "map50",
        "map75",
        "map50_95",
        "ap_small",
        "ap_medium",
        "ap_large",
        "per_class_ap50",
        "per_class_ap75",
    )
    for idx, (gt, pred, names) in enumerate(scenarios):
        old_res = old_map.compute_map_metrics(gt, pred, names)
        new_res = new_map.compute_map_metrics(gt, pred, names)
        for k in identical_keys:
            assert old_res[k] == new_res[k], f"scenario {idx} mismatch on {k}: {old_res[k]} != {new_res[k]}"
        assert "per_class" not in old_res and "per_class" in new_res
        assert "accounting" not in old_res and "accounting" in new_res

    print("[ok] test_regression_identity_vs_pre_v43 PASS (bit-identical across all existing metrics)")


def test_eval_all_v43_end_to_end() -> None:
    import torch
    from src.models import get_model
    from tests.mock_data import create_mock_dataset
    from scripts import eval_all_v43
    from training import common

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        runs_dir = root / "runs"
        out_dir = root / "out"
        create_mock_dataset("mock", data_root, num_train=4, num_val=4, img_size=(320, 320))

        # 1) Dense run with best.pt and an existing v4.2 eval JSON to test backup
        r1 = runs_dir / "train" / "mock_lsknet"
        (r1 / "weights").mkdir(parents=True)
        det1 = get_model("lsknet", device="cpu", num_classes=5)
        det1.load()
        torch.save({"model_state": det1.net.state_dict(), "epoch": 1, "num_classes": 5}, r1 / "weights" / "best.pt")
        (r1 / "run_manifest.json").write_text(json.dumps({
            "model": "lsknet", "dataset": "mock", "resolved_tuning": {"img_size": 160}
        }))
        v42_sentinel = {"metric_version": "v4.2-greedy-unmatched", "map50": 0.123, "sentinel": "v42"}
        (r1 / "eval_mock_val.json").write_text(json.dumps(v42_sentinel))

        # 2) DETR run with last.pt ONLY (tests last.pt fallback & flat dir layout)
        r2 = runs_dir / "mock_ao2-detr"
        r2.mkdir(parents=True)
        det2 = get_model("ao2-detr", device="cpu", num_classes=5)
        det2.load()
        torch.save({"model_state": det2.net.state_dict(), "epoch": 1, "num_classes": 5}, r2 / "last.pt")
        # Deliberately omit run_manifest.json to test dirname fallback parsing

        # 3) YOLO run (exercises evaluate_yolo_unified path too if yolov8n-obb.pt exists)
        yolo_src = REPO_ROOT / "yolov8n-obb.pt"
        expected_ok = 2
        if yolo_src.is_file():
            import shutil
            r_yolo = runs_dir / "train" / "mock_yolov8n-obb"
            (r_yolo / "weights").mkdir(parents=True)
            shutil.copy2(yolo_src, r_yolo / "weights" / "best.pt")
            (r_yolo / "run_manifest.json").write_text(json.dumps({
                "model": "yolov8n-obb", "dataset": "mock", "resolved_tuning": {"img_size": 160}
            }))
            expected_ok = 3

        # 4) Broken run to verify error isolation & continuation
        r3 = runs_dir / "train" / "mock_std"
        (r3 / "weights").mkdir(parents=True)
        (r3 / "weights" / "best.pt").write_bytes(b"corrupt-checkpoint")

        # Pass 1: --quick smoke (must NOT overwrite v4.2 eval_mock_val.json)
        rc = eval_all_v43.main([
            "--runs-dir", str(runs_dir),
            "--data-root", str(data_root),
            "--out-dir", str(out_dir),
            "--split", "test",
            "--batch", "2",
            "--device", "cpu",
            "--quick",
        ])
        assert rc == 0
        assert json.loads((r1 / "eval_mock_val.json").read_text()) == v42_sentinel
        assert not (r1 / "eval_mock_val.json.bak-v42").exists()
        quick_json = json.loads((out_dir / "runs_v4_full_metrics.json").read_text())
        assert quick_json["n_runs"] == expected_ok and quick_json["n_errors"] == 1

        # Pass 2: Full evaluation
        rc = eval_all_v43.main([
            "--runs-dir", str(runs_dir),
            "--data-root", str(data_root),
            "--out-dir", str(out_dir),
            "--split", "test",
            "--batch", "2",
            "--device", "cpu",
        ])
        assert rc == 0

        # Backup preserved original v4.2 JSON
        bak = r1 / "eval_mock_val.json.bak-v42"
        assert bak.is_file() and json.loads(bak.read_text()) == v42_sentinel

        # Updated per-run JSONs contain both newer fields and v4.3 fields
        ev1 = json.loads((r1 / "eval_mock_val.json").read_text())
        assert ev1["metric_version"] == "v4.3-full-accounting"
        assert ev1["evaluated_from"] == "best.pt"
        assert ev1["comparable"] is True and "weights_fingerprint" in ev1 and "coord_space" in ev1
        assert common.protocol_current(ev1)
        assert len(ev1["per_class"]) == 5 and "tp" in ev1["per_class"][0]
        assert ev1["accounting"]["tp"] + ev1["accounting"]["fn"] == ev1["accounting"]["instances"]
        assert ev1["accounting"]["tp"] + ev1["accounting"]["fp"] == ev1["accounting"]["predictions"]

        ev2 = json.loads((r2 / "eval_mock_val.json").read_text())
        assert ev2["evaluated_from"] == "last.pt"
        assert ev2["model"] == "ao2-detr" and ev2["dataset"] == "mock"

        if expected_ok == 3:
            ev_y = json.loads((r_yolo / "eval_mock_val.json").read_text())
            assert ev_y["family"] == "yolo" and ev_y["metric_version"] == "v4.3-full-accounting"
            assert len(ev_y["per_class"]) == 5 and "tp" in ev_y["per_class"][0]
            assert ev_y["accounting"]["tp"] + ev_y["accounting"]["fn"] == ev_y["accounting"]["instances"]

        # Verify all 4 consolidated outputs + exact CSV columns
        full_json = json.loads((out_dir / "runs_v4_full_metrics.json").read_text())
        assert full_json["n_runs"] == expected_ok and full_json["n_errors"] == 1
        assert full_json["errors"][0]["run"] == "mock_std"

        with open(out_dir / "runs_v4_full_metrics_overall.csv", encoding="utf-8") as f:
            ov_reader = list(csv.reader(f))
        assert ov_reader[0] == eval_all_v43.OVERALL_COLS
        assert len(ov_reader) == 1 + expected_ok

        with open(out_dir / "runs_v4_full_metrics_perclass.csv", encoding="utf-8") as f:
            pc_reader = list(csv.reader(f))
        assert pc_reader[0] == eval_all_v43.PERCLASS_COLS
        assert len(pc_reader) == 1 + expected_ok * 5

        with zipfile.ZipFile(out_dir / "runs_v4_full_metrics_bundle.zip") as zf:
            names = sorted(zf.namelist())
        assert names == [
            "runs_v4_full_metrics.json",
            "runs_v4_full_metrics_overall.csv",
            "runs_v4_full_metrics_perclass.csv",
        ], names

        # Pass 3: Idempotent RESUME (completed v4.3 runs must NOT re-evaluate)
        (r3 / "weights" / "best.pt").unlink()  # remove broken run so Pass 3 checks resume cleanly
        import unittest.mock
        from training import evaluate as eval_mod
        with unittest.mock.patch.object(eval_mod, "evaluate_run", side_effect=AssertionError("must not re-evaluate")):
            rc3 = eval_all_v43.main([
                "--runs-dir", str(runs_dir),
                "--data-root", str(data_root),
                "--out-dir", str(out_dir),
                "--split", "test",
                "--batch", "2",
                "--device", "cpu",
            ])
        assert rc3 == 0
        resumed_json = json.loads((out_dir / "runs_v4_full_metrics.json").read_text())
        assert resumed_json["n_runs"] == expected_ok and resumed_json["n_errors"] == 0

    print("[ok] test_eval_all_v43_end_to_end PASS")


def test_per_dataset_data_root_and_oom_retry() -> None:
    import unittest.mock
    from scripts import eval_all_v43

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nvme = root / "nvme"
        drive = root / "drive"
        (nvme / "codrone" / "test" / "images").mkdir(parents=True)
        (nvme / "codrone" / "test" / "images" / "a.jpg").write_bytes(b"x")
        (drive / "dota" / "test" / "images").mkdir(parents=True)
        (drive / "dota" / "test" / "images" / "b.jpg").write_bytes(b"y")

        # Simulate /content/data having codrone only, Drive having dota
        orig_resolve = eval_all_v43._resolve_dataset_data_root

        # Test _dataset_has_split directly + explicit override preservation + env fallback
        assert eval_all_v43._dataset_has_split(nvme, "codrone", "test")
        assert not eval_all_v43._dataset_has_split(nvme, "dota", "test")
        assert eval_all_v43._dataset_has_split(drive, "dota", "test")
        assert orig_resolve("dota", "test", str(nvme)) == str(nvme.resolve())
        with unittest.mock.patch.dict(os.environ, {"VEHICLE_OBB_DATA": str(drive)}):
            assert orig_resolve("dota", "test", None) == str(drive)

        # Test OOM batch-halving retry in eval_all_v43.main
        runs_dir = root / "runs"
        out_dir = root / "out"
        rd = runs_dir / "train" / "codrone_std"
        (rd / "weights").mkdir(parents=True)
        (rd / "weights" / "best.pt").write_bytes(b"w")
        (rd / "run_manifest.json").write_text(json.dumps({"model": "std", "dataset": "codrone"}))

        seen_batches = []

        def fake_capture(**kw):
            b = kw["batch"]
            seen_batches.append(b)
            if b > 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 32.00 GiB")
            return {
                "model": "std", "family": "dense", "dataset": "codrone", "split": "test",
                "imgsz": 1024, "map50": 0.1, "map75": 0.05, "map50_95": 0.04,
                "ap_small": 0.01, "ap_medium": 0.02, "ap_large": 0.03,
                "parameters": 1000, "imgs_per_sec": 5.0, "speed_ms": {"inference": 200.0},
                "per_class": [{"class_id": 0, "name": "car", "instances": 1, "tp": 1, "fp": 0, "fn": 0,
                               "precision": 1.0, "recall": 1.0, "f1": 1.0, "iou_mean": 0.8,
                               "ap50": 1.0, "ap75": 1.0, "ap50_95": 0.9, "ap_small": 0.0,
                               "ap_medium": 1.0, "ap_large": 0.0}],
                "accounting": {"instances": 1, "predictions": 1, "tp": 1, "fp": 0, "fn": 0,
                               "macro_precision": 1.0, "macro_recall": 1.0, "macro_f1": 1.0,
                               "iou_mean": 0.8, "iou_pairs": 1, "classes_with_gt": 1},
            }

        with unittest.mock.patch.object(eval_all_v43, "_run_with_v43_capture", side_effect=fake_capture):
            rc = eval_all_v43.main([
                "--runs-dir", str(runs_dir),
                "--data-root", str(nvme),
                "--out-dir", str(out_dir),
                "--split", "test",
                "--batch", "8",
                "--device", "cpu",
            ])
        assert rc == 0
        assert seen_batches == [8, 4, 2], seen_batches
        out_json = json.loads((out_dir / "runs_v4_full_metrics.json").read_text())
        assert out_json["n_runs"] == 1 and out_json["n_errors"] == 0

    print("[ok] test_per_dataset_data_root_and_oom_retry PASS")


if __name__ == "__main__":
    test_exact_accounting_scenario()
    test_early_returns_and_single_class()
    test_regression_identity_vs_pre_v43()
    test_eval_all_v43_end_to_end()
    test_per_dataset_data_root_and_oom_retry()
    print("ALL v4.3 FULL-ACCOUNTING TESTS PASSED")
