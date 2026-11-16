"""Comprehensive regression & acceptance tests for audit findings F01-F25."""
from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
import unittest.mock
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))


def test_f01_f04_f05_mechanisms_spatial_budget_and_deterministic_eval():
    from src.models import get_model
    from src.models.rhino import RHINONet

    # F04: query budget >= 300 on all 5 DETR architectures; spatial sensitivity on two distinct spatial inputs
    x_tl = torch.zeros(1, 3, 128, 128)
    x_tl[:, :, :32, :32] = 1.0
    x_br = torch.zeros(1, 3, 128, 128)
    x_br[:, :, -32:, -32:] = 1.0

    for name in ("ao2-detr", "ars-detr", "rio-detr", "rhino", "oriented-former"):
        det = get_model(name, device="cpu", num_classes=8)
        det.load()
        net = det.net.eval()
        assert getattr(net, "num_queries", 0) >= 300, f"{name} query budget < 300"
        with torch.no_grad():
            o1 = net(x_tl)
            o2 = net(x_br)
        key = "box_deltas" if "box_deltas" in o1 else ("boxes" if "boxes" in o1 else ("boxes_norm" if "boxes_norm" in o1 else "box_norm"))
        box_diff = float((o1[key] - o2[key]).abs().max())
        cls_diff = float((o1["cls_logits"] - o2["cls_logits"]).abs().max())
        assert box_diff > 1e-5 and cls_diff > 1e-4, (
            f"{name} insensitive to spatial shift (box_diff={box_diff}, cls_diff={cls_diff})"
        )

    # F01: mechanism gradient checks
    # 1) AO2-DETR OPG grads
    ao2 = get_model("ao2-detr", device="cpu", num_classes=8)
    ao2.load()
    ao2.net.train()
    out = ao2.net(torch.randn(1, 3, 64, 64))
    (out["cls_logits"].sum() + out["box_deltas"].sum()).backward()
    assert ao2.net.opg.proj.weight.grad is not None and ao2.net.opg.proj.weight.grad.abs().sum() > 0
    assert ao2.net.opg.score.weight.grad is not None and ao2.net.opg.score.weight.grad.abs().sum() > 0

    # 2) RVSA head_angles and scale_factors grads
    rvsa = get_model("rvsa", device="cpu", num_classes=8)
    rvsa.load()
    rvsa.net.train()
    c, r, (fh, fw) = rvsa.net(torch.randn(1, 3, 64, 64))
    assert fh >= 8 and fw >= 8, (fh, fw)
    (c.sum() + r.sum()).backward()
    assert rvsa.net.blocks[0].attn.head_angles.grad is not None and rvsa.net.blocks[0].attn.head_angles.grad.abs().sum() > 0
    assert rvsa.net.blocks[0].attn.scale_factors.grad is not None and rvsa.net.blocks[0].attn.scale_factors.grad.abs().sum() > 0

    # 3) Swin-OBB stride <= 8 + shifted windows executed
    swin = get_model("swin-obb", device="cpu", num_classes=8)
    swin.load()
    assert any(b.shift_size > 0 for b in swin.net.stage1)
    c_sw, r_sw, (sh, sw) = swin.net(torch.randn(1, 3, 128, 128))
    assert sh == 16 and sw == 16, (sh, sw)  # 128 / 8 = 16 (not stride 64!)
    (c_sw.sum() + r_sw.sum()).backward()

    # F05: RHINO eval() determinism
    rhino = RHINONet(num_classes=8, num_queries=300, embed_dim=64).eval()
    img = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        r1 = rhino(img)
        r2 = rhino(img)
    assert float((r1["cls_logits"] - r2["cls_logits"]).abs().max()) == 0.0
    assert float((r1["box_norm"] - r2["box_norm"]).abs().max()) == 0.0


def test_f02_f03_small_box_representability_and_gradients():
    from src.training.train_detr import extract_detr_predictions

    for model_name, out_key in (("ao2-detr", "box_deltas"), ("oriented-former", "boxes")):
        raw = torch.tensor([[[-5.0, -5.0, -6.0, -6.0, 0.2]]], requires_grad=True)
        _, boxes_norm, _ = extract_detr_predictions(
            {"cls_logits": torch.zeros(1, 1, 9), out_key: raw},
            model_name=model_name,
        )
        # < 5px on 1024px image (< 0.005)
        assert float(boxes_norm[0, 0, 2]) < 0.005, (model_name, float(boxes_norm[0, 0, 2]))
        assert float(boxes_norm[0, 0, 3]) < 0.005, (model_name, float(boxes_norm[0, 0, 3]))
        boxes_norm.sum().backward()
        assert raw.grad is not None and torch.all(raw.grad[..., :4].abs() > 0), (model_name, raw.grad)


def test_f06_dense_target_collision_resolution():
    from src.training.train_dense import encode_dense_targets

    # Two boxes with identical center cell (32.2, 32.2) in a 10x10 grid on 80x80 image (stride=8)
    targets = [{
        "boxes": torch.tensor([
            [36.0, 36.0, 12.0, 10.0, 15.0],
            [36.2, 36.4, 24.0, 20.0, 45.0],
        ], dtype=torch.float32),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }]
    cls_t, reg_t = encode_dense_targets(
        targets, batch_size=1, num_classes=4, feat_h=10, feat_w=10, img_size=80, device=torch.device("cpu")
    )
    assert int(reg_t[0, 6].sum().item()) == 2, f"Collision dropped object: {reg_t[0, 6].sum()}"
    assert int(cls_t[0].sum().item()) == 2


def test_f07_persistent_workers_receive_set_epoch():
    from tests.test_v4_units import _mock_data_dir
    from src.data.augment import AugmentConfig
    from src.data.dataset import AerialOBBDataset
    from src.training.dataset import UnifiedOBBDataset, _worker_init, obb_collate_fn

    raw = AerialOBBDataset("mock", Path(_mock_data_dir()), split="train")
    ds = UnifiedOBBDataset(
        raw, img_size=160, augment=True, aug_cfg=AugmentConfig.dense_default(),
        resize_mode="letterbox", normalize_on_gpu=True, seed=123,
    )
    loader = DataLoader(
        ds, batch_size=2, shuffle=False, num_workers=2, persistent_workers=True,
        collate_fn=obb_collate_fn, worker_init_fn=_worker_init,
    )
    ds.set_epoch(1)
    imgs1, _ = next(iter(loader))
    ds.set_epoch(2)
    imgs2, _ = next(iter(loader))
    assert not torch.equal(imgs1, imgs2), "Persistent workers ignored set_epoch update"


def test_f08_f09_f18_factory_no_fakes_and_predict_contract():
    from src.models import get_model
    from src.models.yolo_obb import YoloOBBDetector

    # F18: Ultralytics variant must resolve to YoloOBBDetector, not local YOLO11OBBDetector
    u = get_model("yolo11n-obb", device="cpu", num_classes=8)
    assert isinstance(u, YoloOBBDetector), type(u)

    # F08: missing weights fails loudly instead of returning fake random boxes
    try:
        u.load("/nonexistent/path/missing.pt")
        raise AssertionError("Expected missing YOLO weights to fail loudly")
    except (FileNotFoundError, RuntimeError):
        pass

    c = get_model("custom-obb", device="cpu", num_classes=8)
    try:
        c.load("/nonexistent/path/missing.pt")
        raise AssertionError("Expected missing custom weights to fail loudly")
    except FileNotFoundError:
        pass

    # F08 + F09: public predict works on real PIL image & rejects GT hints
    img = Image.new("RGB", (320, 240), color=(40, 80, 120))
    preds = c.predict([img], conf_thresh=0.05)
    assert len(preds) == 1 and len(preds[0]) == 8
    try:
        c.predict([img], ground_truth_hints=[[{"boxes": np.zeros((0, 5))}]])
        raise AssertionError("Expected ground_truth_hints rejection")
    except ValueError:
        pass


def test_f10_f11_f23_summary_selection_and_split_guards():
    from training import common, summarize_runs

    with tempfile.TemporaryDirectory() as tmp:
        rd = Path(tmp) / "runs" / "train" / "visdrone_custom-obb"
        (rd / "weights").mkdir(parents=True)
        w = rd / "weights" / "last.pt"
        w.write_bytes(b"weights_v1")
        fp = common.weights_fingerprint(str(w))

        # F10: epoch 3 has higher mAP50 (0.40 vs 0.20), epoch 18 has higher mAP50-95 (0.25 vs 0.15)
        (rd / "results.csv").write_text(
            "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "3,0.40,0.15\n"
            "18,0.20,0.25\n"
        )
        (rd / "run_manifest.json").write_text(json.dumps({
            "model": "custom-obb", "dataset": "visdrone",
            "resolved_tuning": {"epochs": 20, "img_size": 1024},
            "final_metrics": {"epochs_trained": 20},
        }))

        # F11: validation-only eval JSON must NOT populate test columns
        (rd / "eval_visdrone_val.json").write_text(json.dumps({
            "comparable": True, "split": "val", "dataset": "visdrone",
            "coord_space": "original image px (preds inverse-letterbox, GT raw, R3+B1)",
            "metric_version": common.CURRENT_METRIC_VERSION,
            "map50": 0.90, "map50_95": 0.70,
            "weights_fingerprint": fp,
        }))
        row = summarize_runs.summarize_run(str(rd))
        assert str(row["best_epoch"]) == "3", row
        assert str(row["best_map5095_epoch"]) == "18", row
        assert row["map50_test"] == "" and row["eval_status"] == "missing", row

        # Stale weights eval_visdrone_test.json must be rejected
        (rd / "eval_visdrone_test.json").write_text(json.dumps({
            "comparable": True, "split": "test", "dataset": "visdrone",
            "coord_space": "original image px (preds inverse-letterbox, GT raw, R3+B1)",
            "metric_version": common.CURRENT_METRIC_VERSION,
            "map50": 0.88, "map50_95": 0.55,
            "weights_fingerprint": {"sha256": "0" * 64, "size": 1},
        }))
        row_stale = summarize_runs.summarize_run(str(rd))
        assert row_stale["map50_test"] == "" and row_stale["eval_status"] == "stale_weights", row_stale


def test_f12_confusion_matches_ap_on_crowded_boxes():
    from src.metrics.confusion_matrix import compute_detection_confusion_matrix
    from src.metrics.detection_map import evaluate_class_detection
    from training.evaluate import _confusion_update

    gt = [[{"boxes": np.array([[0, 0, 10, 10, 0], [2, 0, 10, 10, 0]], np.float32)}]]
    pred = [[{"boxes": np.array([[0, 0, 10, 10, 0], [0.5, 0, 10, 10, 0]], np.float32),
              "scores": np.array([0.9, 0.8], np.float32)}]]
    ap_res = evaluate_class_detection([gt[0][0]], [pred[0][0]], iou_threshold=0.50)
    assert ap_res["tp"] == 2 and ap_res["fp"] == 0, ap_res

    cm, _ = compute_detection_confusion_matrix(gt, pred, ["car"], iou_threshold=0.50)
    assert cm.tolist() == [[2, 0], [0, 0]], cm.tolist()

    cm2 = np.zeros((2, 2), dtype=np.int64)
    _confusion_update(cm2, gt[0], pred[0], iou_thresh=0.50)
    assert cm2.tolist() == [[2, 0], [0, 0]], cm2.tolist()


def test_f13_bundle_recovery_and_f15_compile_probe():
    from src.training.engine_common import maybe_compile, write_bundle_manifest
    from training.colab_recovery import recover

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mir = root / "mirror" / "visdrone_lsknet"
        runs = root / "runs"
        mir.mkdir(parents=True)
        (mir / "last.pt").write_bytes(b"last_bytes")
        (mir / "best.pt").write_bytes(b"best_bytes")
        write_bundle_manifest(
            mir,
            last_path=mir / "last.pt", last_epoch=12,
            best_path=mir / "best.pt", best_epoch=10,
            best_map50=0.42, is_ema=True,
        )
        rep = recover(str(root / "mirror"), str(runs), verbose=False)
        assert "best.pt" in rep["jobs"][0]["restored_files"]
        assert (runs / "train" / "visdrone_lsknet" / "weights" / "best.pt").read_bytes() == b"best_bytes"

    # F15: lazily failing compiled forward must trigger fallback in probe
    eager = torch.nn.Linear(4, 4)

    class FailingCompiled(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("lazy compile failure")

    with unittest.mock.patch.object(torch, "compile", return_value=FailingCompiled()):
        out_m = maybe_compile(eager, enabled=True, probe=lambda m: m(torch.zeros(1, 4)))
        assert out_m is eager


def test_f19_f20_f21_f25_cleaner_splits_caches_and_security():
    from data_cleaning.clean_build_v4 import _atomic_promote_dir
    from src.data.dataset import AerialOBBDataset
    from src.data.schema import audit_splits
    from src.utils.security import validate_tar_archive

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # F19: atomic promote preserves existing dataset as backup until swap succeeds
        live = root / "cleaned_data"
        live.mkdir()
        (live / "marker.txt").write_text("old")
        stage = root / "cleaned_data.staging"
        stage.mkdir()
        (stage / "marker.txt").write_text("new")
        bak = _atomic_promote_dir(str(stage), str(live), "t1")
        assert (live / "marker.txt").read_text() == "new"
        assert bak and (Path(bak) / "marker.txt").read_text() == "old"

        # F20: audit_splits detects sequence overlap & missing val/test class support
        ds_dir = root / "ds"
        for sp in ("train", "val", "test"):
            (ds_dir / sp / "images").mkdir(parents=True)
            (ds_dir / sp / "labels").mkdir(parents=True)
        Image.new("RGB", (32, 32), color=(1, 2, 3)).save(ds_dir / "train" / "images" / "uav00001_00001.jpg")
        Image.new("RGB", (32, 32), color=(4, 5, 6)).save(ds_dir / "val" / "images" / "uav00001_00002.jpg")
        Image.new("RGB", (32, 32), color=(7, 8, 9)).save(ds_dir / "test" / "images" / "other_00001.jpg")
        (ds_dir / "train" / "labels" / "uav00001_00001.txt").write_text("0 0.5 0.5 0.2 0.2 0\n5 0.5 0.5 0.2 0.2 0\n")
        (ds_dir / "val" / "labels" / "uav00001_00002.txt").write_text("0 0.5 0.5 0.2 0.2 0\n")
        (ds_dir / "test" / "labels" / "other_00001.txt").write_text("0 0.5 0.5 0.2 0.2 0\n")
        audit = audit_splits(str(ds_dir))
        assert not audit["passes"]
        assert audit["sequence_overlaps"], audit
        assert audit["coverage_gaps"], audit

        # F21 + F25: equal-size label edit invalidates JSON annotation cache (and no pickle!)
        (ds_dir / "visdrone" / "images" / "train").mkdir(parents=True)
        (ds_dir / "visdrone" / "labels" / "train").mkdir(parents=True)
        img_p = ds_dir / "visdrone" / "images" / "train" / "img0.jpg"
        lbl_p = ds_dir / "visdrone" / "labels" / "train" / "img0.txt"
        Image.new("RGB", (64, 64)).save(img_p)
        lbl_p.write_text("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
        d1 = AerialOBBDataset("visdrone", ds_dir, split="train", use_cache=True)
        gt1 = d1.get_ground_truth(0, 64, 64)
        assert len(gt1[0]["boxes"]) == 1
        # Equal-size edit with preserved mtime
        st = lbl_p.stat()
        lbl_p.write_text("1 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
        os.utime(lbl_p, ns=(st.st_atime_ns, st.st_mtime_ns))
        d2 = AerialOBBDataset("visdrone", ds_dir, split="train", use_cache=True)
        gt2 = d2.get_ground_truth(0, 64, 64)
        assert len(gt2[1]["boxes"]) == 1 and len(gt2[0]["boxes"]) == 0, gt2
        # Verify cache on disk is JSON, not pickle
        parsed = json.loads((ds_dir / "visdrone" / "labels" / "train" / "labels.cache").read_text())
        assert "fingerprint" in parsed

        # F25: tar archive traversal blocked
        bad_tar = root / "evil.tar"
        with tarfile.open(bad_tar, "w") as tf:
            info = tarfile.TarInfo("../escape.txt")
            info.size = 0
            tf.addfile(info)
        try:
            validate_tar_archive(bad_tar, root / "out")
            raise AssertionError("Expected traversal error")
        except ValueError:
            pass


def test_gate7_tiny_overfit_sanity():
    from src.models import get_model
    from src.training.detr_loss import DETROBBLoss
    from src.training.train_dense import OBBMaskLoss, encode_dense_targets, forward_dense_network
    from src.training.train_detr import extract_detr_predictions

    img = torch.randn(1, 3, 128, 128)
    targets = [{
        "boxes": torch.tensor([[64.0, 64.0, 20.0, 10.0, 30.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
    }]

    # Dense overfit check (swin-obb on tiny synthetic batch)
    swin = get_model("swin-obb", device="cpu", num_classes=4)
    swin.load()
    net_d = swin.net.train()
    opt_d = torch.optim.AdamW(net_d.parameters(), lr=5e-4)
    crit_d = OBBMaskLoss()
    losses_d = []
    for _ in range(10):
        opt_d.zero_grad()
        cls_l, reg_p, fh, fw = forward_dense_network("swin-obb", net_d, img, 4)
        ct, rt = encode_dense_targets(targets, 1, 4, fh, fw, 128, torch.device("cpu"))
        loss = crit_d(cls_l, reg_p, ct, rt)["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_d.parameters(), 10.0)
        opt_d.step()
        losses_d.append(float(loss.detach()))
    assert losses_d[-1] < losses_d[0], losses_d

    # DETR overfit check (ao2-detr on tiny synthetic batch)
    ao2 = get_model("ao2-detr", device="cpu", num_classes=4)
    ao2.load()
    net_q = ao2.net.train()
    opt_q = torch.optim.AdamW(net_q.parameters(), lr=5e-4)
    crit_q = DETROBBLoss(num_classes=4)
    losses_q = []
    for _ in range(10):
        opt_q.zero_grad()
        out = net_q(img)
        pl, pb, pa = extract_detr_predictions(out, "ao2-detr")
        loss = crit_q(pl, pb, pa, targets, img_size=128, aux_outputs=out["aux_outputs"], model_name="ao2-detr")["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_q.parameters(), 10.0)
        opt_q.step()
        losses_q.append(float(loss.detach()))
    assert losses_q[-1] < losses_q[0], losses_q


if __name__ == "__main__":
    torch.set_num_threads(2)
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"ALL {len(tests)} AUDIT REGRESSIONS PASSED")
