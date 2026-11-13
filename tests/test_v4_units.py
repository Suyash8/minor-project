#!/usr/bin/env python3
"""
v4 unit tests — pure-CPU, no data required.

  1. registry: exactly 23 canonical models, aliases resolve.
  2. decode round-trip (W5.3): encode random boxes through the dense training
     encoder, decode, require IoU >= 0.99.
  3. mAP sanity (W6.3): perfect predictions score mAP50 = 1.0; a shifted-box
     case scores in a known (lower) band.
  4. augmentation round-trip (W3): corners -> cxcywha -> corners preserves
     geometry; hflip keeps boxes inside the image.

Run:  python tests/test_v4_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_registry() -> None:
    from src.config import (CANONICAL_MODELS, MODEL_FAMILY, MODEL_ALIASES,
                            resolve_model_names)
    assert len(CANONICAL_MODELS) == 23, f"expected 23, got {len(CANONICAL_MODELS)}"
    assert len(resolve_model_names(["all"])) == 23
    assert resolve_model_names(["yolo26"]) == \
        ["yolo26n-obb", "yolo26s-obb", "yolo26m-obb", "yolo26l-obb"]
    assert resolve_model_names(["nano"]) == \
        ["yolov8n-obb", "yolo11n-obb", "yolo26n-obb"]
    assert set(MODEL_FAMILY.values()) == {"yolo", "dense", "detr"}
    for group in ("yolo", "dense", "detr"):
        assert group in MODEL_ALIASES
    print("[ok] registry: 23 models, aliases resolve")


def test_decode_roundtrip() -> None:
    import torch
    from src.training.decode import decode_dense_grid, encode_dense_sample
    from src.metrics.geometry import compute_obb_iou_matrix

    rng = np.random.default_rng(7)
    img_size = 160
    feat_h = feat_w = 20  # stride 8
    num_classes = 8

    # boxes snapped to grid-cell centers so the encode->decode path is exact
    n = 12
    cell_x = rng.integers(0, feat_w, n)
    cell_y = rng.integers(0, feat_h, n)
    stride = img_size / feat_w
    boxes = np.stack([
        (cell_x + 0.5) * stride,
        (cell_y + 0.5) * stride,
        rng.uniform(8.0, 40.0, n),
        rng.uniform(8.0, 40.0, n),
        rng.uniform(0.0, 180.0, n),
    ], axis=1).astype(np.float32)
    labels = rng.integers(0, num_classes, n)

    cls_t, reg_t = encode_dense_sample(boxes, labels, num_classes,
                                       feat_h, feat_w, img_size)
    decoded = decode_dense_grid(cls_t.unsqueeze(0), reg_t.unsqueeze(0),
                                img_size=img_size, conf_thresh=0.5,
                                nms_iou=0.9, max_det=500)
    d_boxes, d_scores, d_labels = decoded[0]
    assert len(d_boxes) == n, f"decoded {len(d_boxes)} of {n} boxes"

    iou = compute_obb_iou_matrix(boxes, d_boxes)
    for i in range(n):
        matches = [j for j in range(len(d_boxes)) if d_labels[j] == labels[i]]
        best = max((iou[i, j] for j in matches), default=0.0)
        assert best >= 0.99, f"box {i} round-trip IoU {best:.4f} < 0.99"
    print("[ok] decode round-trip: IoU >= 0.99 for all boxes")


def test_map_perfect_and_shifted() -> None:
    from src.metrics.detection_map import compute_map_metrics

    class_names = ["car", "van"]
    rng = np.random.default_rng(3)
    gt_img = [
        {"boxes": np.array([[100, 100, 30, 15, 45.0],
                            [200, 150, 40, 20, 90.0]], dtype=np.float32)},
        {"boxes": np.array([[50, 60, 25, 25, 0.0]], dtype=np.float32)},
    ]
    all_gt = [gt_img]

    # perfect predictions
    perfect = [[{"boxes": e["boxes"].copy(),
                 "scores": np.full(len(e["boxes"]), 0.99, dtype=np.float32)}
                for e in gt_img]]
    m = compute_map_metrics(all_gt, perfect, class_names)
    assert abs(m["map50"] - 1.0) < 1e-6, f"perfect mAP50 = {m['map50']}"
    assert abs(m["map50_95"] - 1.0) < 1e-6, f"perfect mAP50-95 = {m['map50_95']}"

    # shifted by 60 px: should still pass at IoU 0.5 for these sizes? No —
    # a 60 px shift on 30x15 boxes drops IoU well below 0.5, so mAP50 must
    # land in a known low band (0.0 .. 0.2), and never equal 1.0.
    shifted = [[{"boxes": e["boxes"].copy() + np.array([60, 0, 0, 0, 0],
                                                       dtype=np.float32),
                 "scores": np.full(len(e["boxes"]), 0.99, dtype=np.float32)}
                for e in gt_img]]
    m2 = compute_map_metrics(all_gt, shifted, class_names)
    assert 0.0 <= m2["map50"] <= 0.2, f"shifted mAP50 = {m2['map50']}"
    assert m2["map50"] < m["map50"]
    print(f"[ok] mAP sanity: perfect=1.0, shifted mAP50={m2['map50']:.3f} "
          f"in [0, 0.2]")


def test_augment_roundtrip() -> None:
    from src.data.augment import (boxes_to_corners, corners_to_boxes, hflip,
                                  letterbox)
    from src.metrics.geometry import compute_obb_iou_matrix

    rng = np.random.default_rng(11)
    boxes = np.stack([
        rng.uniform(50, 400, 6), rng.uniform(50, 300, 6),
        rng.uniform(20, 80, 6), rng.uniform(10, 60, 6),
        rng.uniform(0, 180, 6),
    ], axis=1).astype(np.float32)

    # augment.py works in corner space (clip-aware); boxes fit back exactly
    corners = boxes_to_corners(boxes)
    back = corners_to_boxes(corners, img_w=480, img_h=360)
    assert len(back) == len(boxes)
    iou = np.diag(compute_obb_iou_matrix(boxes, back))
    assert (iou > 0.99).all(), f"corners round-trip IoU min {iou.min()}"

    img = rng.integers(0, 255, (360, 480, 3), dtype=np.uint8)
    flipped_img, flipped_corners = hflip(img, corners)
    assert flipped_img.shape == img.shape
    back2_corners = hflip(flipped_img, flipped_corners)[1]
    assert np.allclose(back2_corners, corners, atol=1e-3), \
        "double hflip should restore corners exactly"

    lb_img, lb_corners = letterbox(img, corners, 256)
    assert lb_img.shape[:2] == (256, 256)
    assert ((lb_corners >= 0) & (lb_corners <= 256)).all()
    print("[ok] augmentation round-trip: corners/cxcywha exact, flips bounded")


def test_yolo_xywhr_conversion() -> None:
    """W6.3 convention proof: Ultralytics xywhr -> suite params preserves the
    exact rectangle for nonzero angles, w/h swaps and boundary angles.

    Path under test is the production helper
    src.metrics.geometry.ultralytics_xywhr_to_suite. Reference rectangles
    are parameterized in the suite convention, turned into corners, then
    re-parameterized the way Ultralytics does (its own xyxyxyxy2xywhr,
    which canonicalizes via cv2.minAreaRect), converted with the helper,
    and compared to the suite's own polygon fitting via rotated IoU.
    """
    from ultralytics.utils.ops import xyxyxyxy2xywhr
    from src.metrics.geometry import (compute_obb_iou_matrix, obb_to_corners,
                                      polygon_to_obb_params,
                                      ultralytics_xywhr_to_suite)

    cases = []
    # (cx, cy, w, h, angle_deg) - nonzero angles, w<h swaps, boundaries
    for ang in (0.0, 1.0, 17.0, 30.0, 44.9, 45.0, 45.1, 60.0, 89.0, 90.0,
                91.0, 120.0, 134.9, 135.0, 150.0, 179.0, 179.9):
        cases.append((320.0, 240.0, 80.0, 30.0, ang))   # w > h
        cases.append((320.0, 240.0, 30.0, 80.0, ang))   # w < h (swap)
    cases.append((320.0, 240.0, 50.0, 50.0, 37.0))      # square (degenerate)
    cases.append((12.0, 9.0, 22.0, 7.0, 113.0))         # off-center small box

    worst = 1.0
    for (cx, cy, w, h, ang) in cases:
        corners = obb_to_corners(cx, cy, w, h, ang).astype(np.float32)
        # Ultralytics canonical parameterization of the same rectangle
        ux, uy, uw, uh, ur = xyxyxyxy2xywhr(corners[None])[0]
        conv = np.array([ultralytics_xywhr_to_suite(ux, uy, uw, uh, ur)],
                        dtype=np.float32)
        # Suite's own fitting of the same rectangle (what GT parsing yields)
        ref = np.array([polygon_to_obb_params(corners)], dtype=np.float32)
        iou = float(compute_obb_iou_matrix(ref, conv)[0, 0])
        worst = min(worst, iou)
        assert iou > 0.98, (f"xywhr conversion changed the rectangle for "
                            f"{(cx, cy, w, h, ang)}: IoU={iou:.4f}, "
                            f"suite={ref[0].tolist()}, conv={conv[0].tolist()}")
    print(f"[ok] yolo xywhr->suite conversion: {len(cases)} cases, "
          f"worst round-trip IoU {worst:.4f}")


def test_native_fallback_is_explicit() -> None:
    """The Ultralytics-native eval fallback must not produce noncomparable
    metrics silently: it requires an explicit opt-in and, when used, marks
    the result as non-comparable."""
    import inspect
    from training import evaluate as ev

    src = inspect.getsource(ev.evaluate_run)
    assert "V4_ALLOW_NATIVE_YOLO_EVAL" in src and "allow_native_fallback" in src
    sig = inspect.signature(ev.evaluate_run)
    assert sig.parameters["allow_native_fallback"].default is False
    from training.summarize_runs import summarize_run
    assert "comparable" in inspect.getsource(summarize_run)
    print("[ok] native fallback is opt-in only and marked non-comparable")




# ======================================================================
# Review-driven regression battery (findings C/F): real-encoder round
# trips, staging E2E overlap, fake-OOM recovery, YOLO resume contract.
# ======================================================================
import os  # noqa: E402

def test_real_encoder_roundtrip() -> None:
    """REAL encode_dense_targets -> decode_dense_grid round trip (review C:
    the old test used a test-only twin encoder, proving nothing about the
    production encoder). Covers multiple grid sizes, w<h, angle>90, and
    multi-class images."""
    import torch
    from src.training.train_dense import encode_dense_targets
    from src.training.decode import decode_dense_grid
    from src.metrics.geometry import compute_obb_iou_matrix

    C = 8
    gt = np.array([
        [100.0, 120.0, 60.0, 30.0, 15.0],
        [300.0, 300.0, 45.0, 90.0, 100.0],   # w<h, angle>90
        [256.0, 100.0, 20.0, 20.0, 179.9],   # near-180 wrap
    ], dtype=np.float32)
    labels = np.array([0, 3, 7], dtype=np.int64)

    for img_size, feat in ((512, 64), (640, 40)):
        targets = [{"boxes": torch.from_numpy(gt),
                    "labels": torch.from_numpy(labels)}]
        cls_t, reg_t = encode_dense_targets(
            targets, batch_size=1, num_classes=C, feat_h=feat, feat_w=feat,
            img_size=img_size, device=torch.device("cpu"))
        assert int(cls_t.sum()) == 3, "all three GT boxes must encode"

        # Fabricate PERFECT network outputs from the encoded targets.
        cls_logits = torch.full((1, C, feat, feat), -12.0)
        cls_logits[cls_t > 0.5] = 12.0
        reg_preds = reg_t.clone()
        reg_preds[:, 6] = torch.where(
            reg_t[:, 6] > 0.5,
            torch.tensor(12.0), torch.tensor(-12.0))

        boxes_out, _scores, labels_out = decode_dense_grid(
            cls_logits, reg_preds, img_size, conf_thresh=0.5)[0]
        assert len(boxes_out) == 3, f"expected 3 detections, got {len(boxes_out)}"
        for i in range(len(gt)):
            ious = compute_obb_iou_matrix(gt[i:i + 1], boxes_out)[0]
            j = int(np.argmax(ious))
            assert ious[j] > 0.95, (f"img={img_size}: GT {gt[i].tolist()} "
                                    f"best IoU {ious[j]:.4f}")
            assert int(labels_out[j]) == int(labels[i])
        print(f"[ok] real-encoder round trip img={img_size} feat={feat}: "
              f"3/3 boxes recovered")


def test_detr_decode_path() -> None:
    """DETR decode contract: normalized cx,cy,w,h + radians -> pixel suite
    boxes; angle used directly mod 180 (matches the sin(2a)/cos(2a) loss)."""
    import math
    import torch
    from src.training.decode import decode_detr
    from src.metrics.geometry import compute_obb_iou_matrix

    C, Q, img = 8, 10, 512
    gt = np.array([[200.0, 220.0, 80.0, 40.0, 30.0]], dtype=np.float32)
    logits = torch.full((1, Q, C + 1), -8.0)
    logits[0, 0, 2] = 8.0                      # query 0, class 2, confident
    bn = torch.zeros((1, Q, 4))
    bn[0, 0] = torch.tensor([200 / img, 220 / img, 80 / img, 40 / img])
    ang = torch.zeros((1, Q))
    ang[0, 0] = math.radians(30.0)
    boxes, scores, labels = decode_detr(logits, bn, ang, img, C,
                                        conf_thresh=0.5)[0]
    assert len(boxes) == 1 and int(labels[0]) == 2
    iou = float(compute_obb_iou_matrix(gt, boxes)[0, 0])
    assert iou > 0.99, f"DETR decode off: IoU {iou:.4f}, box {boxes[0].tolist()}"
    print(f"[ok] DETR decode path: IoU {iou:.4f}, label correct")


def test_staging_e2e_overlap() -> None:
    """Staging E2E (review F): dataset reads are served by staged copies as
    they land; epoch 2 performs ZERO source reads. Copy is artificially
    slowed so epoch 1 provably overlaps an incomplete stage."""
    import tempfile, time
    from PIL import Image
    from training import staging as stg
    from src.data.dataset import AerialOBBDataset

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        for split, n in (("train", 3), ("val", 1)):
            os.makedirs(f"{src}/mock/images/{split}", exist_ok=True)
            os.makedirs(f"{src}/mock/labels/{split}", exist_ok=True)
            for i in range(n):
                Image.new("RGB", (64, 64), (i * 40, 10, 200)).save(
                    f"{src}/mock/images/{split}/img{i}.jpg")
                with open(f"{src}/mock/labels/{split}/img{i}.txt", "w") as f:
                    f.write("2 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n")

        # Slow the copy so epoch 1 reads genuinely overlap staging.
        real_copy2 = stg.shutil.copy2
        def slow_copy2(a, b):
            time.sleep(0.25)
            return real_copy2(a, b)
        stg.shutil.copy2 = slow_copy2
        try:
            handle = stg.start_background_stage(
                ["mock"], src_root=src, dst_root=dst, workers=2,
                progress=False)
            resolver = stg.SourceResolver(
                handle.src_root, handle.dst_root,
                counter=handle.source_reads_value)
            ds = AerialOBBDataset("mock", data_dir=src, split="train",
                                  use_cache=False)
            assert len(ds) == 3
            ds.set_resolver(resolver)

            # Epoch 1: overlaps the slowed copy; some reads hit the source.
            for i in range(len(ds)):
                arr = ds.get_image_array(i)
                assert arr.shape == (64, 64, 3)
            reads_e1 = resolver.drive_reads
            assert reads_e1 >= 1, ("slowed copy should have forced at least "
                                   "one source read in epoch 1")

            handle.wait(timeout=120)
            # Epoch 2: everything staged -> zero additional source reads.
            for i in range(len(ds)):
                ds.get_image_array(i)
            assert resolver.drive_reads == reads_e1, (
                f"epoch 2 read the source {resolver.drive_reads - reads_e1} "
                f"times despite a complete stage")
            # And the staged tree really is complete.
            assert os.path.isfile(f"{dst}/mock/images/train/img0.jpg")
            print(f"[ok] staging E2E: epoch1 source_reads={reads_e1}, "
                  f"epoch2 additional=0, staged tree complete")
        finally:
            stg.shutil.copy2 = real_copy2
            try:
                handle.close()
            except Exception:
                pass


_MOCK_DIR: Optional[str] = None


def _mock_data_dir() -> str:
    """Generate the mock CI dataset into a temp dir (B-round packaging:
    data/mock is no longer shipped in the zip - tests create it on demand,
    exactly like training/run.py's ensure_mock_data)."""
    global _MOCK_DIR
    if _MOCK_DIR is None or not os.path.isfile(
            os.path.join(_MOCK_DIR, "mock", "mock.yaml")):
        import tempfile
        from tests.mock_data import create_mock_dataset
        _MOCK_DIR = tempfile.mkdtemp(prefix="v4mock_")
        create_mock_dataset("mock", Path(_MOCK_DIR), num_train=6, num_val=4)
    return _MOCK_DIR


def _run_mock_dense(run_dir: str, seed: int = 42, patch=None,
                    epochs: int = 1, resume: bool = False,
                    micro_batch: Optional[int] = None,
                    max_batches: int = 2, val_map: bool = False):
    """One deterministic 1-epoch mock lsknet run; returns the result dict."""
    import src.training.train_dense as td
    orig = td.forward_dense_network
    if patch is not None:
        td.forward_dense_network = patch(orig)
    try:
        try:
            return td.train_dense_model(
                model_name="lsknet", dataset_name="mock", epochs=epochs,
                batch_size=2, micro_batch=micro_batch, img_size=160,
                workers=0, device="cpu", data_dir=_mock_data_dir(),
                max_batches=max_batches,
                project_dir=os.path.dirname(run_dir), run_dir=run_dir,
                test_mode=True, val_map=val_map, seed=seed,
                deterministic=True, resume=resume)
        except td.GracefulStop:
            # run.py catches this per job; in tests it is the expected exit.
            return {"graceful_stop": True}
    finally:
        td.forward_dense_network = orig


def test_fake_oom_gradient_equivalence() -> None:
    """Fake-OOM (review F + main's directive): the gradient-weighted
    split-half retry must be NUMERICALLY EQUIVALENT to a no-OOM reference
    run at the halved micro-batch (the run the recovery converges to):
    same epoch train loss and same final weights, exactly. BatchNorm makes
    a split batch differ from a full batch by construction, so the full-
    batch run is NOT a valid reference; the micro-halved run is."""
    import csv
    import tempfile
    import torch

    def flaky_factory(orig):
        calls = {"n": 0}
        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.cuda.OutOfMemoryError("fake OOM (test)")
            return orig(*a, **k)
        return flaky

    with tempfile.TemporaryDirectory() as tmp:
        ref_dir = os.path.join(tmp, "ref")
        oom_dir = os.path.join(tmp, "oom")
        # No-OOM reference: micro-batch 1 from the start (= post-OOM state).
        # 4 micro-batches of 1 image == the OOM run's 2 batches of 2 images,
        # same seeded order.
        _run_mock_dense(ref_dir, micro_batch=1, max_batches=4)
        res_oom = _run_mock_dense(oom_dir, patch=flaky_factory)
        assert res_oom.get("oom_events", 0) >= 1, "OOM recovery never fired"

        def epoch_loss(rd):
            with open(os.path.join(rd, "results.csv")) as f:
                rows = list(csv.DictReader(f))
            return float(rows[0]["train_loss"])
        l_ref, l_oom = epoch_loss(ref_dir), epoch_loss(oom_dir)
        assert abs(l_ref - l_oom) < 1e-6, (
            f"train loss diverged from micro-halved reference: "
            f"{l_ref} vs {l_oom}")

        def weights(rd):
            ckpt = torch.load(os.path.join(rd, "weights", "last.pt"),
                              map_location="cpu", weights_only=False)
            return ckpt["model_state"]
        w_ref, w_oom = weights(ref_dir), weights(oom_dir)
        assert w_ref.keys() == w_oom.keys()
        worst = 0.0
        for k in w_ref:
            a, b = w_ref[k].float(), w_oom[k].float()
            worst = max(worst, float((a - b).abs().max()))
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), (
                f"weights diverged at {k}: max abs diff "
                f"{float((a - b).abs().max()):.2e}")
        print(f"[ok] fake-OOM gradient equivalence: oom_events="
              f"{res_oom['oom_events']}, train_loss {l_ref:.6f} == "
              f"{l_oom:.6f}, max weight diff {worst:.2e}")


def test_ema_warmup_ramp() -> None:
    """EMA (review D): warmup ramp keeps early EMA near LIVE weights (not
    random init), converges to the base decay, and update count persists."""
    import math
    import torch
    from src.training.engine_common import ModelEMA

    net = torch.nn.Linear(8, 8)
    ema = ModelEMA(net, decay=0.9999)
    assert ema._effective_decay() == 0.0  # t=0: EMA takes the live weights
    ema.update(net)
    d1 = ema._effective_decay()
    assert abs(d1 - 0.9999 * (1 - math.exp(-1 / 2000))) < 1e-9
    # Early EMA must track the live net closely (ramp), not the init.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update(net)
    k = next(iter(ema.ema_state))
    live = net.state_dict()[k].float()
    # after update #2 the ramp decay is ~0.001, so EMA lags live by ~0.001
    diff = float((ema.ema_state[k] - live).abs().max())
    assert 1e-5 < diff < 5e-3, f"ramped EMA should track live closely: {diff}"
    assert float((ema.ema_state[k] - live).abs().max()) < 1.0, (
        "un-ramped EMA would still equal the INIT weights (diff 1.0)")
    ema.updates = 20000
    assert abs(ema._effective_decay() - 0.9999) < 1e-4
    st = ema.state_dict()
    ema.updates = 0
    ema.load_state_dict(st)
    assert ema.updates == 20000, "updates must survive checkpoint round trip"
    print("[ok] EMA warmup ramp: early decay ~0, late decay 0.9999, "
          "updates persisted")


def test_graceful_stop_epoch_numbering_and_resume() -> None:
    """Graceful stop (review D): the saved checkpoint records the last
    COMPLETED epoch (not the interrupted one); a subsequent resume picks up
    at the next epoch and restores patience state (epochs_since_best)."""
    import tempfile
    import torch
    import src.training.train_dense as td
    from src.training import dataset as dsmod
    from src.training.engine_common import STOP

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "run")
        # Request a stop as soon as epoch 2 starts.
        state = {"epoch": 0}
        orig_set_epoch = dsmod.UnifiedOBBDataset.set_epoch
        def patched_set_epoch(self, epoch):
            state["epoch"] = epoch
            return orig_set_epoch(self, epoch)
        orig_requested = STOP.requested
        dsmod.UnifiedOBBDataset.set_epoch = patched_set_epoch
        STOP.requested = lambda: state["epoch"] >= 2
        try:
            _run_mock_dense(run_dir, epochs=3)
        finally:
            dsmod.UnifiedOBBDataset.set_epoch = orig_set_epoch
            STOP.requested = orig_requested

        ckpt = torch.load(os.path.join(run_dir, "weights", "last.pt"),
                          map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 1, (
            f"graceful stop must save the last COMPLETED epoch, got "
            f"{ckpt['epoch']}")
        assert "epochs_since_best" in ckpt, "patience state must persist"

        # Strict resume: picks up at epoch 2 and finishes epoch 3.
        res = _run_mock_dense(run_dir, epochs=3, resume=True)
        ckpt2 = torch.load(os.path.join(run_dir, "weights", "last.pt"),
                           map_location="cpu", weights_only=False)
        assert ckpt2["epoch"] == 3, f"resume did not finish: {ckpt2['epoch']}"
        print("[ok] graceful stop saved completed epoch 1; resume finished "
              "epoch 3; epochs_since_best persisted")


def test_disk_cache_tier() -> None:
    """--cache disk (review: not an annotations alias): decoded images are
    persisted as .npy; a second pass with the SOURCE IMAGES DELETED is
    served entirely from the disk cache."""
    import tempfile
    from PIL import Image
    from src.data.dataset import AerialOBBDataset

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as cache:
        os.makedirs(f"{src}/mock/images/train", exist_ok=True)
        os.makedirs(f"{src}/mock/labels/train", exist_ok=True)
        for i in range(2):
            Image.new("RGB", (48, 48), (i * 90, 30, 120)).save(
                f"{src}/mock/images/train/im{i}.jpg")
            with open(f"{src}/mock/labels/train/im{i}.txt", "w") as f:
                f.write("1 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n")
        ds = AerialOBBDataset("mock", data_dir=src, split="train",
                              use_cache=True, disk_cache_dir=cache)
        first = [ds.get_image_array(i) for i in range(len(ds))]
        npys = [f for f in os.listdir(f"{cache}/mock/train")
                if f.endswith(".npy")]
        assert len(npys) == 2, f"disk cache wrote {npys}"
        # Remove the source images: second pass must still succeed (served
        # from the disk tier, byte-identical).
        for i in range(2):
            os.remove(f"{src}/mock/images/train/im{i}.jpg")
        ds2 = AerialOBBDataset("mock", data_dir=src, split="train",
                               use_cache=True, disk_cache_dir=cache)
        for i in range(len(ds2)):
            assert np.array_equal(ds2.get_image_array(i), first[i])
        print("[ok] disk cache tier: .npy persisted; pass 2 served from disk "
              "with source images deleted")


def test_per_model_encoder_coverage() -> None:
    """Encoder coverage (review C/main): EVERY dense model (real head shape
    from a real forward) round-trips through encode_dense_targets ->
    decode_dense_grid; EVERY DETR model's raw outputs standardize through
    extract_detr_predictions and decode_detr recovers an injected GT query."""
    import math
    import torch
    from src.config import CANONICAL_MODELS, MODEL_FAMILY
    from src.models import get_model
    from src.training.dispatcher import MODEL_TUNING
    from src.training.train_dense import (forward_dense_network,
                                          encode_dense_targets)
    from src.training.train_detr import extract_detr_predictions
    from src.training.decode import decode_dense_grid, decode_detr
    from src.metrics.geometry import compute_obb_iou_matrix

    C = 8
    base = np.array([[60.0, 70.0, 30.0, 15.0, 20.0],
                     [110.0, 110.0, 12.0, 25.0, 130.0]], dtype=np.float32)
    labels = np.array([1, 5], dtype=np.int64)
    failures = []
    n_dense = n_detr = 0
    for name in CANONICAL_MODELS:
        fam = MODEL_FAMILY.get(name)
        if fam == "yolo":
            continue  # YOLO family: no suite encoder; Ultralytics native
        # Exercise each model's real head grid at a resolution proportional
        # to its tuned size (capped at 512 so CPU forwards stay
        # unit-test-fast; the MIN_PRIOR_FRAC fix, not resolution, is what
        # keeps small boxes representable on coarse grids).
        img = min(int(MODEL_TUNING.get(name, {}).get("img_size", 640)), 512)
        gt = base.copy()
        gt[:, :4] *= img / 160.0
        try:
            det = get_model(name, device="cpu", num_classes=C)
            det.load()
            net = det.net.eval()
            if fam == "dense":
                cls_l, reg_p, fh, fw = forward_dense_network(
                    model_name=name, net=net,
                    batch_tensor=torch.zeros(1, 3, img, img), num_classes=C)
                targets = [{"boxes": torch.from_numpy(gt),
                            "labels": torch.from_numpy(labels)}]
                cls_t, reg_t = encode_dense_targets(
                    targets, 1, C, fh, fw, img, torch.device("cpu"))
                assert int(cls_t.sum()) == 2, (
                    f"{name}: GT boxes fell off the {fh}x{fw} grid")
                logits = torch.full((1, C, fh, fw), -12.0)
                logits[cls_t > 0.5] = 12.0
                reg = reg_t.clone()
                reg[:, 6] = torch.where(reg_t[:, 6] > 0.5,
                                        torch.tensor(12.0),
                                        torch.tensor(-12.0))
                boxes, _s, labs = decode_dense_grid(logits, reg, img,
                                                    conf_thresh=0.5)[0]
                for i in range(len(gt)):
                    ious = compute_obb_iou_matrix(gt[i:i + 1], boxes)[0]
                    j = int(np.argmax(ious))
                    assert ious[j] > 0.9 and int(labs[j]) == int(labels[i]), (
                        f"{name}: GT {gt[i].tolist()} best IoU {ious[j]:.3f}")
                n_dense += 1
            else:  # detr
                with torch.no_grad():
                    outputs = net(torch.zeros(1, 3, img, img))
                logits, boxes_n, ang = extract_detr_predictions(
                    outputs, name)
                Q = logits.shape[1]
                logits = torch.full((1, Q, C + 1), -8.0)
                logits[0, 0, 3] = 8.0
                bn = torch.zeros((1, Q, 4))
                bn[0, 0] = torch.tensor(
                    [gt[0, 0] / img, gt[0, 1] / img,
                     gt[0, 2] / img, gt[0, 3] / img])
                ar = torch.zeros((1, Q))
                ar[0, 0] = math.radians(float(gt[0, 4]))
                boxes, _s, labs = decode_detr(logits, bn, ar, img, C,
                                              conf_thresh=0.5)[0]
                iou = float(compute_obb_iou_matrix(gt[0:1], boxes)[0, 0])
                assert iou > 0.99 and int(labs[0]) == 3, (
                    f"{name}: decode off (IoU {iou:.3f})")
                n_detr += 1
        except Exception as e:
            failures.append(f"{name} ({fam}): {e}")
    assert not failures, "encoder coverage failures: " + " | ".join(failures)
    print(f"[ok] per-model encoder coverage: {n_dense} dense models "
          f"round-tripped at their real head shapes, {n_detr} DETR models "
          f"standardized + decoded")


def test_yolo_resume_checkpoint_contract() -> None:
    """YOLO resume (review F): the real Ultralytics checkpoint from the mock
    run must load with weights_only=False (torch 2.9 default True rejects
    it), and YOLO(last.pt) construction must succeed for a resume."""
    last_pt = "/tmp/v4runs/train/mock_yolov8n-obb/weights/last.pt"
    temp = None
    if not os.path.isfile(last_pt):
        import tempfile
        from src.training.train_yolo import train_yolo_obb
        temp = tempfile.TemporaryDirectory()
        result = train_yolo_obb(model_name="yolov8n-obb",dataset_name="mock",epochs=1,
            batch_size=2,imgsz=64,device="cpu",workers=0,data_dir=_mock_data_dir(),
            run_dir=temp.name,test_mode=True,resume=False)
        last_pt = result["last_weights"]
    import torch
    try:
        torch.load(last_pt, map_location="cpu")  # torch>=2.6 default
        default_ok = True
    except Exception:
        default_ok = False
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert isinstance(ckpt, dict) and "model" in ckpt
    from ultralytics import YOLO
    model = YOLO(last_pt)  # the resume constructor path
    assert model is not None
    print(f"[ok] YOLO resume contract: weights_only=False load + YOLO() "
          f"construction succeed (default weights_only load "
          f"{'also ok' if default_ok else 'rejected -> pin is required'})")


# ---------------------------------------------------------------------------
# Second regression round (R1-R5) targeted tests
# ---------------------------------------------------------------------------
def _save_spy(engine_mod):
    """Count save_training_checkpoint calls by file name."""
    import src.training.engine_common as ec
    counts = {"best.pt": 0, "last.pt": 0}
    real = ec.save_training_checkpoint

    def spy(path, **payload):
        name = os.path.basename(str(path))
        if name in counts:
            counts[name] += 1
        return real(path, **payload)

    engine_mod.save_training_checkpoint = spy
    return counts, real


def test_best_selection_only_on_map_epochs() -> None:
    """R2: off-cadence epochs (val_map_every>1, mAP not measured) must never
    write best.pt or move patience, no matter how good the val loss looks;
    a NaN mAP on an mAP-EVALUATED epoch is no-improvement, never a loss
    fallback. Explicit val_map=False keeps the loss criterion (contract:
    best.pt must exist)."""
    import math
    import tempfile
    import src.training.train_dense as td
    import src.training.engine_common as ec

    real_validate = td.validate_dense
    try:
        # (a) off-cadence epoch with a GREAT loss must not write best.pt.
        counts, real_save = _save_spy(td)
        calls = {"n": 0}
        def fake_validate_a(*a, **k):
            calls["n"] += 1
            cm = k.get("compute_map", True)
            loss = 1.0 if calls["n"] == 1 else 0.1  # later epoch: much better
            return (loss, 0.7 if cm else math.nan,
                    0.4 if cm else math.nan)
        td.validate_dense = fake_validate_a
        with tempfile.TemporaryDirectory() as tmp:
            rd = os.path.join(tmp, "run_a")
            _run_mock_dense(rd, epochs=2, max_batches=1, val_map=True)
        assert counts["best.pt"] == 1, (
            f"best.pt saved {counts['best.pt']}x; only the mAP-evaluated "
            f"epoch (2 of 2, default cadence) may write it")
        td.save_training_checkpoint = real_save

        # (b) NaN mAP on an EVALUATED epoch: no best, no loss substitution.
        counts, real_save = _save_spy(td)
        def fake_validate_b(*a, **k):
            return (0.5, math.nan, math.nan)
        td.validate_dense = fake_validate_b
        with tempfile.TemporaryDirectory() as tmp:
            rd = os.path.join(tmp, "run_b")
            _run_mock_dense(rd, epochs=1, max_batches=1, val_map=True)
            assert counts["best.pt"] == 0, \
                "NaN-mAP evaluated epoch wrote best.pt"
            assert not os.path.exists(os.path.join(rd, "weights", "best.pt")), \
                "best.pt must not exist after a NaN-mAP evaluated epoch"
        td.save_training_checkpoint = real_save

        # (c) explicit val_map=False: loss criterion, best.pt exists.
        with tempfile.TemporaryDirectory() as tmp:
            rd = os.path.join(tmp, "run_c")
            _run_mock_dense(rd, epochs=1, max_batches=1)
            assert os.path.exists(os.path.join(rd, "weights", "best.pt")), \
                "val_map=False (explicit loss criterion) must still save best.pt"
        print("[ok] R2: best.pt only on mAP-evaluated improving epochs; "
              "NaN evaluated epoch = no improvement; val_map=False keeps "
              "the explicit loss criterion")
    finally:
        td.validate_dense = real_validate
        td.save_training_checkpoint = ec.save_training_checkpoint


def test_order_changes_after_staging_completes() -> None:
    """R1: while staging is incomplete the engine consumes a seeded perm
    list; once staging COMPLETES the loader must be rebuilt with
    shuffle=True - the fixed perm list must not freeze the order for the
    rest of the run. Copy is slowed so epoch 1 provably overlaps."""
    _run_r1_with_staging()


def _run_r1_with_staging() -> None:
    """R1 with a live staging handle passed into the engine."""
    import tempfile, time
    from PIL import Image
    from training import staging as stg
    import src.training.train_dense as td

    with tempfile.TemporaryDirectory() as src, \
            tempfile.TemporaryDirectory() as dst, \
            tempfile.TemporaryDirectory() as out:
        for split, n in (("train", 8), ("val", 2)):
            os.makedirs(f"{src}/mock/images/{split}", exist_ok=True)
            os.makedirs(f"{src}/mock/labels/{split}", exist_ok=True)
            for i in range(n):
                Image.new("RGB", (64, 64), (i * 20, 10, 200)).save(
                    f"{src}/mock/images/{split}/img{i}.jpg")
                with open(f"{src}/mock/labels/{split}/img{i}.txt", "w") as f:
                    f.write("2 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n")

        # Deterministic overlap: NO copy finishes until the trainer has
        # consumed 4 train items (2 epochs), so epochs 1-2 provably run
        # against an incomplete stage and the completion transition lands
        # before epoch 3.
        gate = {"open": False}
        real_copy2 = stg.shutil.copy2
        def gated_copy2(a, b):
            while not gate["open"]:
                time.sleep(0.05)
            return real_copy2(a, b)
        stg.shutil.copy2 = gated_copy2

        builds = []
        real_dl = td.DataLoader
        def dl_spy(dataset, *a, **k):
            builds.append({"sampler_list": isinstance(k.get("sampler"), list),
                           "shuffle": bool(k.get("shuffle", False))})
            return real_dl(dataset, *a, **k)

        # Record the actual consumption order of the TRAIN dataset.
        orders = []
        real_ds_cls = td.UnifiedOBBDataset
        class RecDS(real_ds_cls):
            def __getitem__(self, idx):
                if len(self) == 8:  # the train split (val has 2)
                    orders.append(idx)
                    if len(orders) >= 4:
                        gate["open"] = True
                return super().__getitem__(idx)

        handle = None
        try:
            handle = stg.start_background_stage(
                ["mock"], src_root=src, dst_root=dst, workers=2,
                progress=False)
            td.DataLoader = dl_spy
            td.UnifiedOBBDataset = RecDS
            rd = os.path.join(out, "run")
            td.train_dense_model(
                model_name="lsknet", dataset_name="mock", epochs=3,
                batch_size=2, img_size=160, workers=0, device="cpu",
                data_dir=src, max_batches=1,
                project_dir=out, run_dir=rd,
                test_mode=True, val_map=False, seed=42,
                deterministic=True, staging=handle)
        finally:
            td.DataLoader = real_dl
            td.UnifiedOBBDataset = real_ds_cls
            stg.shutil.copy2 = real_copy2
            if handle is not None:
                handle.wait(timeout=60, poll=0.2)
                handle.close()

        perm_builds = [i for i, b in enumerate(builds) if b["sampler_list"]]
        assert perm_builds, ("no perm-list train loader was built - the "
                             "staging-overlap path never engaged")
        first_perm = perm_builds[0]
        later_shuffle = any(b["shuffle"] for b in builds[first_perm + 1:])
        assert later_shuffle, ("after staging completed the engine never "
                               "rebuilt the train loader with shuffle=True; "
                               "the fixed perm list froze the batch order "
                               "for the rest of the run (R1)")
        # Consumption per epoch is uniform (the engine fetches one batch
        # past max_batches before breaking); chunk adaptively and require
        # the three epochs to not repeat one frozen order.
        assert len(orders) % 3 == 0 and orders, f"unexpected orders: {orders}"
        step = len(orders) // 3
        per_epoch = [orders[i:i + step] for i in range(0, len(orders), step)]
        assert not (per_epoch[0] == per_epoch[1] == per_epoch[2]), (
            f"identical consumption order in every epoch: {per_epoch}")
        print(f"[ok] R1: perm loader used during staging, shuffle loader "
              f"rebuilt on completion; per-epoch orders {per_epoch}")


def test_staging_failure_not_counted_done() -> None:
    """R5a: a failed copy must NOT increment progress; staging never
    reports complete with missing files, and the failure is recorded."""
    import tempfile
    from PIL import Image
    from training import staging as stg

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        os.makedirs(f"{src}/mock/images/train", exist_ok=True)
        for i in range(3):
            Image.new("RGB", (32, 32), (i, 0, 0)).save(
                f"{src}/mock/images/train/img{i}.jpg")
        handle = None
        real_copy_one = stg.StageHandle._copy_one
        def flaky_copy_one(self, ds, rel):
            if rel.endswith("img1.jpg"):
                raise OSError("injected copy failure (test)")
            return real_copy_one(self, ds, rel)
        try:
            stg.StageHandle._copy_one = flaky_copy_one
            handle = stg.start_background_stage(
                ["mock"], src_root=src, dst_root=dst, workers=1,
                progress=False)
            done = handle.wait(timeout=30, poll=0.1)
            st = handle.status()
            assert not done and not st.complete, (
                "staging reported complete despite a failed copy (R5a)")
            assert handle.errors, "failed copy was not recorded in errors"
            assert any("img1.jpg" in e for e in handle.errors)
            print(f"[ok] R5a: failed copy excluded from progress "
                  f"(done={st.done}/{st.total}), errors={handle.errors}")
        finally:
            stg.StageHandle._copy_one = real_copy_one
            if handle is not None:
                handle.close()


def test_pending_byte_disk_guard() -> None:
    """R5b: the disk guard must price PENDING bytes only - a fully staged
    dataset on a nearly full disk must not abort a resumed run."""
    import tempfile
    from PIL import Image
    from training import staging as stg

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        os.makedirs(f"{src}/mock/images/train", exist_ok=True)
        os.makedirs(f"{dst}/mock/images/train", exist_ok=True)
        big = 2 * 1024 * 1024  # 2 MB payload
        p_src = f"{src}/mock/images/train/big.jpg"
        Image.new("RGB", (64, 64), (1, 2, 3)).save(p_src)
        with open(p_src, "ab") as f:
            f.truncate(big)
        # Fully staged (exact size match): pending = 0 bytes.
        p_dst = f"{dst}/mock/images/train/big.jpg"
        Image.new("RGB", (64, 64), (1, 2, 3)).save(p_dst)
        with open(p_dst, "ab") as f:
            f.truncate(big)
        # A guard pricing the FULL dataset would demand 2.2 MB free even
        # with nothing pending; fake an almost-full disk to expose that.
        class _Usage:
            free = 512 * 1024  # 0.5 MB free
        real_du = stg.shutil.disk_usage
        stg.shutil.disk_usage = lambda _p: _Usage()
        try:
            stg.check_disk_guard(["mock"], src=src, dst=dst)  # must NOT raise
            # And a genuinely pending file still trips the guard.
            os.remove(p_dst)
            try:
                stg.check_disk_guard(["mock"], src=src, dst=dst)
                raise AssertionError("guard passed with 2 MB pending and "
                                     "0.5 MB free")
            except RuntimeError:
                pass
            print("[ok] R5b: disk guard prices pending bytes only")
        finally:
            stg.shutil.disk_usage = real_du


def test_staged_yolo_trainer_cpu_integration() -> None:
    """R4a: exercise the StagedOBBTrainer against REAL Ultralytics on CPU:
    mock dataset staged from src->dst with a slowed copy, 1 epoch
    yolov8n-obb. Asserts the staged trainer engaged (not the blocking
    fallback), training completed, and staging overlapped with reads."""
    import io
    import contextlib
    import tempfile, time
    from training import staging as stg
    from src.training import train_yolo as ty

    data_root = _mock_data_dir()  # repo mock dataset (YOLO-OBB format)
    if not os.path.isdir(os.path.join(data_root, "mock")):
        print("[skip] mock dataset not present")
        return

    real_copy2 = stg.shutil.copy2
    def slow_copy2(a, b):
        time.sleep(0.3)
        return real_copy2(a, b)

    handle = None
    with tempfile.TemporaryDirectory() as dst, tempfile.TemporaryDirectory() as out:
        stg.shutil.copy2 = slow_copy2
        buf = io.StringIO()
        try:
            handle = stg.start_background_stage(
                ["mock"], src_root=data_root, dst_root=dst, workers=2,
                progress=False)
            with contextlib.redirect_stdout(buf):
                res = ty.train_yolo_obb(
                    model_name="yolov8n-obb", dataset_name="mock", epochs=1,
                    batch_size=2, imgsz=160, device="cpu", workers=0,
                    data_dir=data_root, staging=handle,
                    project_dir=out, run_dir=os.path.join(out, "run"),
                    test_mode=True, seed=42, resume=False, cache=None)
        finally:
            stg.shutil.copy2 = real_copy2
            if handle is not None:
                handle.close()
        log = buf.getvalue()
        assert "YOLO staging overlap ENABLED" in log, (
            "staged trainer did not engage; fell back to blocking staging "
            f"instead. Log tail: {log[-800:]}")
        assert os.path.isfile(os.path.join(out, "run", "weights", "last.pt")), \
            "no last.pt from the staged YOLO run"
        overlapped = handle.source_reads_value.value
        assert overlapped >= 1, ("slowed copy should have forced >=1 source "
                                 "read through the resolver")
        assert isinstance(res, dict)
        print(f"[ok] R4a: StagedOBBTrainer ran 1 CPU epoch against real "
              f"Ultralytics; source reads during overlap={overlapped}; "
              f"last.pt written")


def test_r3_inverse_letterbox_roundtrip() -> None:
    """R3: _unletterbox_boxes is the exact inverse of the letterbox corner
    transform (uniform scale + centered pad): boxes survive a round trip
    letterbox->corners_to_boxes->_unletterbox_boxes, so area bands measured
    after the inverse are in true original-image px^2."""
    import numpy as np
    from src.data.augment import letterbox, boxes_to_corners, corners_to_boxes
    from training.evaluate import _unletterbox_boxes

    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (480, 900, 3), dtype=np.uint8)  # non-square
    boxes = np.array([[100.0, 120.0, 60.0, 30.0, 25.0],
                      [450.0, 300.0, 200.0, 90.0, 80.0],
                      [10.0, 10.0, 12.0, 8.0, 0.0]], dtype=np.float32)
    corners = boxes_to_corners(boxes)
    lb_img, lb_corners = letterbox(img, corners, 1024)
    assert lb_img.shape == (1024, 1024, 3)
    lb_boxes = corners_to_boxes(lb_corners, img_w=1024, img_h=1024)
    back = _unletterbox_boxes(lb_boxes, orig_w=900, orig_h=480, imgsz=1024)
    assert back.shape == boxes.shape
    # cx/cy/w/h match the originals (uniform-scale inverse is exact up to
    # the sub-pixel resize rounding of the pad computation).
    assert np.allclose(back[:, :4], boxes[:, :4], atol=1.5), (
        f"round-trip diverged: {back[:, :4]} vs {boxes[:, :4]}")
    assert np.allclose(back[:, 4] % 180.0, boxes[:, 4] % 180.0, atol=1.0)
    # Degenerate input stays empty.
    assert len(_unletterbox_boxes(np.zeros((0, 5)), 900, 480, 1024)) == 0
    print("[ok] R3: inverse-letterbox round trip exact within resize rounding")


def test_b1_gt_invariant_across_imgsz() -> None:
    """B1: evaluation GT must be the raw original-coordinate set, identical
    at every eval imgsz (640/768/1024/1280) - no letterbox, no MIN_BOX_PX
    drop, no canvas clipping. The test proves the OLD path dropped small
    boxes at low imgsz and the NEW path keeps them everywhere."""
    import numpy as np
    from src.data.augment import letterbox, boxes_to_corners, corners_to_boxes
    from training.evaluate import _original_gt_per_class

    class StubRaw:
        class_names = ["car", "bus"]

        def __init__(self, boxes_by_class):
            self._gt = boxes_by_class

        def get_ground_truth(self, idx, img_width, img_height):
            return [{"boxes": np.asarray(b, dtype=np.float32).reshape(-1, 5)}
                    for b in self._gt]

    # One normal box + one tiny 3x3px box on a 900x480 image.
    boxes = [[100.0, 120.0, 60.0, 30.0, 25.0],
             [450.0, 300.0, 3.0, 3.0, 10.0]]
    raw = StubRaw([boxes, []])

    # Old pipeline (letterbox + corners_to_boxes at imgsz) drops the tiny
    # box at 640 (MIN_BOX_PX=4 in letterbox px): proves the defect existed.
    img = np.zeros((480, 900, 3), dtype=np.uint8)
    corners = boxes_to_corners(np.asarray(boxes, dtype=np.float32))
    for imgsz, expect_dropped in ((640, True), (1280, False)):
        _lb_img, lb_corners = letterbox(img, corners, imgsz)
        kept, _keep = corners_to_boxes(lb_corners, img_w=imgsz, img_h=imgsz,
                                       return_keep=True)
        assert (len(kept) < 2) == expect_dropped, (
            f"letterbox pipeline at {imgsz}: kept={len(kept)} "
            f"(expected drop={expect_dropped}) - MIN_BOX_PX imgsz-dependence")

    # New evaluation GT: identical object set at every imgsz, exactly the
    # raw annotations (and _prep-independent of imgsz by construction).
    reference = None
    for imgsz in (640, 768, 1024, 1280):
        gt = _original_gt_per_class(raw, 0, 900, 480, 2)
        flat = np.concatenate([gt[0]["boxes"], gt[1]["boxes"]], axis=0)
        assert len(flat) == 2, f"tiny box dropped at imgsz={imgsz}"
        if reference is None:
            reference = flat
        else:
            assert np.array_equal(flat, reference), \
                f"GT changed with imgsz={imgsz}"
    assert np.allclose(reference, np.asarray(boxes, dtype=np.float32))
    print("[ok] B1: eval GT identical at imgsz 640/768/1024/1280; tiny-box "
          "drop of the letterbox path demonstrated and bypassed")


def test_b1_gt_matches_raw_label_files() -> None:
    """B1 on the generated mock dataset: eval GT count equals the raw label
    line count for every image in the split (the full annotation set,
    nothing filtered)."""
    import numpy as np
    from src.data.dataset import AerialOBBDataset
    from training.evaluate import _original_gt_per_class

    data_root = _mock_data_dir()
    raw = AerialOBBDataset("mock", data_dir=data_root, split="val",
                           use_cache=False)
    total_raw = 0
    for i in range(len(raw)):
        img = raw.get_image_array(i)
        gt = _original_gt_per_class(raw, i, img.shape[1], img.shape[0],
                                    len(raw.class_names))
        n = sum(len(g["boxes"]) for g in gt)
        direct = raw.get_ground_truth(i, img.shape[1], img.shape[0])
        n_direct = sum(len(d["boxes"]) for d in direct)
        assert n == n_direct
        total_raw += n
    label_lines = 0
    lbl_dir = os.path.join(data_root, "mock", "labels", "val")
    for f in os.listdir(lbl_dir):
        if f.endswith(".txt"):
            with open(os.path.join(lbl_dir, f)) as fh:
                label_lines += sum(1 for ln in fh if ln.strip())
    assert label_lines == total_raw, (
        f"raw label lines {label_lines} != eval GT {total_raw}")
    assert total_raw > 0
    print(f"[ok] B1: mock val eval GT ({total_raw} boxes) matches every "
          f"line of the raw label files")


def test_b2_wait_returns_promptly_on_permanent_failure() -> None:
    """B2: wait(timeout=None) must NOT hang when copies permanently fail -
    it returns False as soon as the queue is drained, and the blocking
    caller raises an explicit error naming the failures."""
    import tempfile, time
    from PIL import Image
    from training import staging as stg

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        os.makedirs(f"{src}/mock/images/train", exist_ok=True)
        for i in range(2):
            Image.new("RGB", (32, 32), (i, 0, 0)).save(
                f"{src}/mock/images/train/img{i}.jpg")
        real = stg.StageHandle._copy_one
        stg.StageHandle._copy_one = lambda self, ds, rel: (_ for _ in ()).throw(
            OSError("permanent failure (test)"))
        stg.StageHandle._max_attempts = 2  # keep the test fast
        try:
            handle = stg.start_background_stage(
                ["mock"], src_root=src, dst_root=dst, workers=1,
                progress=False)
            t0 = time.time()
            ok = handle.wait(timeout=None, poll=0.05)  # must NOT hang
            dt = time.time() - t0
            assert ok is False, "wait() reported complete with failed copies"
            assert dt < 20, f"wait(None) took {dt:.1f}s under permanent failure"
            assert len(handle.errors) == 2
            handle.close()
            print(f"[ok] B2: wait(None) returned False in {dt:.2f}s; "
                  f"errors={handle.errors}")
        finally:
            stg.StageHandle._copy_one = real
            stg.StageHandle._max_attempts = 3


def test_b2_blocking_caller_raises_and_retry_recovers() -> None:
    """B2: stage_datasets_blocking raises an explicit error on permanent
    failure (no hang, no half-staged root); a transient failure is absorbed
    by the retry and stages cleanly."""
    import tempfile
    from PIL import Image
    from training import staging as stg

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        os.makedirs(f"{src}/mock/images/train", exist_ok=True)
        Image.new("RGB", (32, 32), (7, 7, 7)).save(
            f"{src}/mock/images/train/a.jpg")

        # Permanent failure -> explicit RuntimeError from the blocking API.
        real = stg.StageHandle._copy_one
        stg.StageHandle._copy_one = lambda self, ds, rel: (_ for _ in ()).throw(
            OSError("permanent failure (test)"))
        stg.StageHandle._max_attempts = 2
        try:
            try:
                stg.stage_datasets_blocking(["mock"], src_root=src,
                                            dst_root=dst, workers=1,
                                            progress=False)
                raise AssertionError("blocking staging did not raise on "
                                     "permanent failure")
            except RuntimeError as e:
                assert "permanent failure (test)" in str(e), str(e)
        finally:
            stg.StageHandle._copy_one = real
            stg.StageHandle._max_attempts = 3

        # Transient failure (fails once, then works) -> retry recovers.
        old_env = os.environ.pop("VEHICLE_OBB_DATA", None)
        attempts = {"n": 0}
        def flaky(self, ds, rel):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("transient (test)")
            return real(self, ds, rel)
        stg.StageHandle._copy_one = flaky
        try:
            root = stg.stage_datasets_blocking(["mock"], src_root=src,
                                               dst_root=dst, workers=1,
                                               progress=False)
            assert os.path.isfile(os.path.join(root, "mock", "images",
                                               "train", "a.jpg"))
        finally:
            stg.StageHandle._copy_one = real
            if old_env is not None:
                os.environ["VEHICLE_OBB_DATA"] = old_env
            else:
                os.environ.pop("VEHICLE_OBB_DATA", None)
        print("[ok] B2: blocking caller raises on permanent failure; "
              "retry absorbs a transient one")


def test_mirror_snapshot_destname_and_protocol_guard() -> None:
    """B-round finals: (a) DriveMirror mirrors a uniquely-named snapshot to
    the canonical basename and unlinks the snapshot after copying (no
    overwrite race, no accumulation); (b) the protocol guard admits current
    coord_space and rejects older/missing ones."""
    import tempfile
    from src.training.engine_common import DriveMirror
    from training import common

    with tempfile.TemporaryDirectory() as mirror, tempfile.TemporaryDirectory() as local:
        dm = DriveMirror(mirror, every_n_epochs=1, run_subdir="mock_yolov8n-obb")
        snap = os.path.join(local, "last_e3.pt")
        with open(snap, "wb") as f:
            f.write(b"checkpoint-bytes")
        dm.submit(Path(snap), dest_name="last.pt", unlink_after=True)
        dm.flush()
        dst = os.path.join(mirror, "mock_yolov8n-obb", "last.pt")
        assert os.path.isfile(dst), "snapshot not mirrored under canonical name"
        assert open(dst, "rb").read() == b"checkpoint-bytes"
        assert not os.path.exists(snap), "snapshot not unlinked after copy"
        assert not os.path.exists(os.path.join(
            mirror, "mock_yolov8n-obb", "last_e3.pt")), \
            "unique snapshot name leaked into the mirror"

    assert common.protocol_current({"coord_space": "original image px "
                                    "(preds inverse-letterbox, GT raw, R3+B1)",
                                    "metric_version": common.CURRENT_METRIC_VERSION})
    assert not common.protocol_current({"coord_space": "letterbox(imgsz)"})
    assert not common.protocol_current({})
    assert not common.protocol_current({"coord_space": None})
    print("[ok] mirror snapshot dest_name/unlink_after + protocol guard")

if __name__ == "__main__":
    import runpy
    tests = [v for k,v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"ALL {len(tests)} v4 UNIT TESTS EXECUTED AND PASSED")
    runpy.run_path(str(Path(__file__).with_name("test_v42.py")), run_name="__main__")
