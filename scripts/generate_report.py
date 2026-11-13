from __future__ import annotations
#!/usr/bin/env python3
"""
generate_report.py - C8 publication artifacts from SAVED, REAL results.

    python scripts/generate_report.py                      # runs under ./runs
    python scripts/generate_report.py --runs-dir /content/minor-project/runs \
        --results-dir results --hardware-label "NVIDIA L4"

Reads every current-protocol `eval_<dataset>_test.json` (written by
training/evaluate.py) under --runs-dir and writes, into --results-dir:

  benchmark_table.tex                         camera-ready booktabs table
  figures/pareto_speed_vs_accuracy_<ds>.{pdf,svg,png}   FPS vs mAP50
  figures/size_vs_accuracy_<ds>.{pdf,svg,png}           params (M) vs mAP50-95
  figures/radar_per_class_<ds>.{pdf,svg,png}            best YOLO/Dense/DETR
  figures/qualitative_<ds>.{pdf,svg,png}                GT vs predictions
  report_coverage.json                        what is in / missing / skipped

Honesty rules: no number is ever invented or defaulted. A model with no valid
result is absent and the figure titles say "n/23 models"; results from an
older/non-comparable protocol are skipped and listed. If there is no valid
result at all the script exits non-zero and writes nothing.

Legacy mode: `--run-dir <dir>` (a folder with benchmark_metrics.json) keeps the
old markdown report + comparison chart.
"""
import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import report_lib  # noqa: E402


# ---------------------------------------------------------------------------
# legacy single-run report (kept for backward compatibility)
# ---------------------------------------------------------------------------
def legacy_report(run_dir: Path) -> int:
    import pandas as pd
    from src.utils.reporter import format_metrics_table, generate_markdown_report
    from src.utils.visualizer import plot_benchmark_comparison

    json_path = run_dir / "benchmark_metrics.json"
    if not json_path.exists():
        print(f"[!] File not found: {json_path}", file=sys.stderr)
        return 1
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    best_row = df.loc[df["map50"].idxmax()]
    print(format_metrics_table(data))
    md = generate_markdown_report(data, best_model=best_row["model"],
                                  best_dataset=best_row["dataset"])
    (run_dir / "benchmark_report.md").write_text(md)
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_benchmark_comparison(data, plots / "model_benchmark_comparison.png")
    print(f"[ok] wrote {run_dir / 'benchmark_report.md'} and plots/")
    return 0


# ---------------------------------------------------------------------------
# qualitative panels (needs torch + the dataset + weights; best effort)
# ---------------------------------------------------------------------------
def _predict_samples(rec: dict, data_root: str, n: int, device: str,
                     conf: float = 0.25):
    """Real predictions of rec's weights on the first n test images.
    Returns (samples, class_names). Raises on any problem: the caller reports
    the reason instead of drawing something made up."""
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "training"))
    from src.data.dataset import AerialOBBDataset
    from src.metrics.geometry import ultralytics_xywhr_to_suite

    weights = rec["weights"]
    if not weights or not os.path.isfile(weights):
        from common import find_weights
        weights = find_weights(rec["run_dir"])
    raw = AerialOBBDataset(dataset_name=rec["eval_on"], data_dir=data_root,
                           split=rec["split"])
    if len(raw) == 0:
        raise FileNotFoundError(f"No {rec['eval_on']} {rec['split']} images under {data_root}")
    class_names = list(raw.class_names)
    nc = len(class_names)
    idxs = np.linspace(0, len(raw) - 1, num=min(n, len(raw)), dtype=int)
    imgsz = int(rec["imgsz"] or 1280)
    samples = []

    if rec["family"] == "yolo":
        from ultralytics import YOLO
        model = YOLO(weights)
        dev = "cpu" if device == "cpu" else (0 if device in ("auto", "cuda")
                                             else device)
        for i in idxs:
            img = raw.get_image_array(int(i))
            h, w = img.shape[:2]
            res = model.predict(img[:, :, ::-1], imgsz=imgsz, conf=conf,
                                device=dev, verbose=False)[0]
            pred = []
            if res.obb is not None and len(res.obb):
                xywhr = res.obb.xywhr.cpu().numpy()
                cls = res.obb.cls.cpu().numpy().astype(int)
                cf = res.obb.conf.cpu().numpy()
                suite = [ultralytics_xywhr_to_suite(*map(float, r)) for r in xywhr]
                pred = [(int(c), float(p), b) for c, p, b in zip(cls, cf, suite)]
            samples.append(dict(image=img, pred=pred,
                                gt=_gt_items(raw, int(i), w, h, nc),
                                name=f"#{int(i)}"))
        return samples, class_names

    import torch
    from training import evaluate as ev
    from src.models import get_model
    from src.training.dataset import UnifiedOBBDataset, obb_collate_fn
    from src.training.decode import decode_dense_grid, decode_detr, to_metric_format
    from src.training.train_dense import _normalize_on_gpu, forward_dense_network
    from src.training.train_detr import extract_detr_predictions

    dev = torch.device("cuda" if device in ("auto", "cuda")
                       and torch.cuda.is_available() else "cpu")
    det = get_model(rec["model"], device=str(dev), num_classes=nc)
    det.load(weights_path=weights)
    net = det.net.to(dev).eval()
    ds = UnifiedOBBDataset(raw, img_size=imgsz, augment=False,
                           resize_mode="letterbox", normalize_on_gpu=True)
    with torch.no_grad():
        for i in idxs:
            imgs, tg = obb_collate_fn([ds[int(i)]])
            x = _normalize_on_gpu(imgs, dev)
            if rec["family"] == "detr":
                out = net(x)
                lg, bx, an = extract_detr_predictions(out, model_name=rec["model"])
                dec = decode_detr(lg.float(), bx.float(), an.float(),
                                  img_size=imgsz, num_classes=nc,
                                  conf_thresh=conf, nms_iou=0.45)
            else:
                cl, rg, _a, _b = forward_dense_network(rec["model"], net, x, nc)
                dec = decode_dense_grid(cl.float(), rg.float(), img_size=imgsz,
                                        conf_thresh=conf, nms_iou=0.45)
            per = to_metric_format(dec, nc)[0]
            ow, oh = (int(v) for v in tg[0]["orig_size"])
            pred = []
            for c in range(nc):
                b = ev._unletterbox_boxes(per[c]["boxes"], ow, oh, imgsz)
                sc = per[c].get("scores", [None] * len(b))
                for k, box in enumerate(b):
                    pred.append((c, float(sc[k]) if sc[k] is not None else 1.0,
                                 box))
            img = raw.get_image_array(int(i))
            samples.append(dict(image=img, pred=pred,
                                gt=_gt_items(raw, int(i), ow, oh, nc),
                                name=f"#{int(i)}"))
    return samples, class_names


def _gt_items(raw, idx, w, h, nc):
    import numpy as np
    gt = raw.get_ground_truth(idx, w, h)
    out = []
    for c in range(nc):
        for b in np.asarray(gt[c].get("boxes", np.zeros((0, 5)))).reshape(-1, 5):
            out.append((c, b))
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Publication tables/figures from saved benchmark results.")
    ap.add_argument("--runs-dir", default=None,
                    help="where eval_*_test.json live (default: $VEHICLE_OBB_RUNS, "
                         "else <repo>/runs)")
    ap.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="+", default=report_lib.DATASETS)
    ap.add_argument("--hardware-label", default="",
                    help='e.g. "NVIDIA L4"; printed in the caption/axis only '
                         "when given (the eval JSONs do not record the GPU)")
    ap.add_argument("--data-root", default=None,
                    help="cleaned-data root for the qualitative panels")
    ap.add_argument("--qualitative-samples", type=int, default=4)
    ap.add_argument("--no-qualitative", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-dir", default=None,
                    help="LEGACY: folder with benchmark_metrics.json")
    a = ap.parse_args(argv)

    if a.run_dir:
        return legacy_report(Path(a.run_dir))

    runs_dir = (a.runs_dir or os.environ.get("VEHICLE_OBB_RUNS")
                or str(PROJECT_ROOT / "runs"))
    results = Path(a.results_dir)
    figs = results / "figures"
    recs, skipped = report_lib.load_eval_records([runs_dir], split=a.split)
    for s in skipped:
        print(f"[skip] {s['path']}: {s['reason']}")
    if not recs:
        print(f"[!] no valid current-protocol eval_*_{a.split}.json under "
              f"{runs_dir}; nothing written (run the eval step first).",
              file=sys.stderr)
        return 2

    if a.hardware_label:
        observed = {r.get("hardware", "") for r in recs}
        if any(not h or a.hardware_label.lower() not in h.lower() for h in observed):
            raise ValueError("Requested hardware label does not match every input result's measured hardware")
    registered = report_lib.registered_models()
    n_reg = len(registered)
    evaluated = sorted({r["model"] for r in recs})
    missing_models = [m for m in registered if m not in evaluated]
    print(f"[report] {len(recs)} results, {len(evaluated)}/{n_reg or '?'} "
          f"registered models evaluated")

    written = []
    groups = {}
    for record in recs:
        key=(record.get("device"),record.get("hardware"),record.get("timing_basis"))
        groups.setdefault(key,[]).append(record)
    table_recs=recs
    if len(groups)>1:
        print("[report] mixed speed provenance: combined table speed omitted; Pareto plots separated")
        table_recs=[dict(r,imgs_per_sec=float("nan"),device="omitted",hardware="omitted",timing_basis="omitted") for r in recs]
    tex = report_lib.latex_table(table_recs, datasets=a.datasets,
                                 hardware_label=a.hardware_label,
                                 split=a.split, n_registered=n_reg)
    if len(groups)>1:
        tex="% Speed omitted: mixed hardware/device/timing basis; see separate Pareto plots.\n"+tex
    results.mkdir(parents=True, exist_ok=True)
    (results / "benchmark_table.tex").write_text(tex)
    written.append(str(results / "benchmark_table.tex"))

    for ds in a.datasets:
        if not report_lib.in_domain(recs, ds):
            print(f"[report] {ds}: no in-domain results; its figures are "
                  f"not drawn")
            continue
        for group_index,(key,group) in enumerate(sorted(groups.items(),key=lambda item:str(item[0]))):
            group_figs=figs if len(groups)==1 else figs/f"speed_group_{group_index+1}"
            written += report_lib.plot_pareto(group, ds, group_figs, n_reg,
                                               " / ".join(str(k) for k in key))
        written += report_lib.plot_size_scatter(recs, ds, figs, n_reg)
        written += report_lib.plot_radar(recs, ds, figs)

    qual_status = {}
    if a.no_qualitative:
        qual_status["all"] = "disabled by --no-qualitative"
    else:
        data_root = a.data_root
        if data_root is None:
            from common import resolve_data_root
            data_root = resolve_data_root(None)
        for ds in a.datasets:
            top = report_lib.top_per_family(recs, ds)
            if not top:
                qual_status[ds] = "no in-domain result"
                continue
            samples, names = [], []
            try:
                for rec in top:
                    got, names = _predict_samples(
                        rec, data_root, a.qualitative_samples, a.device)
                    for s in got:
                        s["name"] = f"{rec['model']} {s['name']}"
                    samples += got
                paths = report_lib.draw_qualitative(
                    samples, names, figs / f"qualitative_{ds}",
                    title=f"Ground truth vs. prediction on {ds} ({a.split})")
                written += paths
                qual_status[ds] = f"{len(samples)} model predictions drawn"
            except Exception as e:  # honest failure, never fake boxes
                qual_status[ds] = f"NOT drawn: {type(e).__name__}: {e}"
                print(f"[report][warn] qualitative {ds}: {qual_status[ds]}")

    report_lib.write_coverage(
        results / "report_coverage.json",
        runs_dir=runs_dir, split=a.split, n_results=len(recs),
        models_evaluated=evaluated, registered_total=n_reg,
        models_without_results=missing_models, skipped_inputs=skipped,
        qualitative=qual_status, files=written,
        hardware_label=a.hardware_label or "not recorded in eval JSONs")
    print("[report] wrote:\n  " + "\n  ".join(written))
    if missing_models:
        print(f"[report][warn] {len(missing_models)} registered models have "
              f"no valid result and are absent from the artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
