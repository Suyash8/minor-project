#!/usr/bin/env python3
"""
C6-C8 unit tests - pure CPU, no torch, no real data.

The numbers below are SYNTHETIC fixtures that exist only to exercise the
plumbing (matrix algebra, protocol filtering, recovery, file outputs). They
are written to temp dirs and never reach results/.

Run:  python tests/test_c6_c8_units.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "scripts"))

CUR = "original image px (preds inverse-letterbox, GT raw, R3+B1)"
CLASSES = ["pedestrian", "bicycle", "car", "van", "truck", "bus",
           "tricycle", "motorcyclist"]


def _mk_run(runs: Path, model: str, trained_on: str, evals: dict,
            family: str = None, coord: str = CUR, comparable: bool = True,
            epochs: int = 3, mv: str = "v4.2-greedy-unmatched"):
    rd = runs / "train" / f"{trained_on}_{model}"
    (rd / "weights").mkdir(parents=True)
    (rd / "weights" / "best.pt").write_bytes(b"w")
    (rd / "run_manifest.json").write_text(json.dumps(dict(
        model=model, dataset=trained_on,
        resolved_tuning=dict(epochs=epochs, img_size=640),
        final_metrics=dict(epochs_trained=epochs))))
    time.sleep(0.01)
    for ds, (m50, m95) in evals.items():
        e = dict(model=model, dataset=ds, split="test", imgsz=640,
                 coord_space=coord, comparable=comparable, metric_version=mv,
                 protocol="src/metrics (unified)",
                 map50=m50, map50_95=m95, map75=m95, parameters=2_500_000,
                 imgs_per_sec=40.0 + len(model), class_names=CLASSES,
                 per_class=[dict(name=c, ap50=(m50 * (i + 1)) / 8)
                            for i, c in enumerate(CLASSES)],
                 weights=str(rd / "weights" / "best.pt"),
                 weights_fingerprint=__import__("common").weights_fingerprint(rd / "weights" / "best.pt"),
                 timestamp="2026-11-13T00:00:00+00:00")
        if family:
            e["family"] = family
        (rd / f"eval_{ds}_test.json").write_text(json.dumps(e))
    return rd


def _fixture(tmp: Path) -> Path:
    runs = tmp / "runs"
    _mk_run(runs, "custom-obb", "visdrone",
            {"visdrone": (0.50, 0.30), "codrone": (0.30, 0.15),
             "dota": (0.20, 0.10)})
    _mk_run(runs, "custom-obb", "codrone",
            {"codrone": (0.60, 0.35), "visdrone": (0.35, 0.18)})  # dota missing
    _mk_run(runs, "yolo11n-obb", "visdrone", {"visdrone": (0.55, 0.33)})
    _mk_run(runs, "ao2-detr", "visdrone", {"visdrone": (0.25, 0.12)})
    return runs


def test_load_and_protocol_filter() -> None:
    import report_lib
    with tempfile.TemporaryDirectory() as t:
        runs = _fixture(Path(t))
        _mk_run(runs, "lsknet", "visdrone", {"visdrone": (0.4, 0.2)},
                coord="letterbox(imgsz)")
        _mk_run(runs, "std", "visdrone", {"visdrone": (float("nan"), 0.2)})
        _mk_run(runs, "rvsa", "visdrone", {"visdrone": (0.4, 0.2)},
                comparable=False)
        _mk_run(runs, "swin-obb", "visdrone", {"visdrone": (0.4, 0.2)},
                mv="v4.1")                      # pre-matching-fix semantics
        recs, skipped = report_lib.load_eval_records([str(runs)])
        models = {r["model"] for r in recs}
        assert models == {"custom-obb", "yolo11n-obb", "ao2-detr"}, models
        reasons = " | ".join(s["reason"] for s in skipped)
        assert "coord_space" in reasons and "non-finite" in reasons \
            and "non-comparable" in reasons and "metric_version" in reasons, reasons
        assert sum("metric_version" in s["reason"] for s in skipped) == 2
        fam = {r["model"]: r["family"] for r in recs}
        assert fam == {"custom-obb": "dense", "yolo11n-obb": "yolo",
                       "ao2-detr": "detr"}
    print("[ok] eval loader: old protocol / NaN / non-comparable skipped with reasons")


def test_cross_matrix_algebra() -> None:
    import report_lib
    with tempfile.TemporaryDirectory() as t:
        recs, _ = report_lib.load_eval_records([str(_fixture(Path(t)))])
        df = report_lib.build_cross_matrix(recs)
        c = df[(df.model == "custom-obb") & (df.trained_on == "visdrone")
               & (df.eval_on == "codrone")].iloc[0]
        assert math.isclose(c.delta_map50, 0.30 - 0.50, abs_tol=1e-9)
        assert math.isclose(c.drop_pct_map50, 40.0, abs_tol=1e-6)
        ind = df[(df.model == "custom-obb") & df.in_domain]
        assert (ind.delta_map50 == 0).all()
        miss = report_lib.cross_coverage(df, ["custom-obb"])
        assert miss == ["custom-obb: codrone -> dota", "custom-obb: dota -> visdrone",
                        "custom-obb: dota -> codrone", "custom-obb: dota -> dota"], miss
        md = report_lib.cross_markdown(df)
        assert "--" in md and "(-0.200)" in md
    print("[ok] cross matrix: dmAP, drop %, missing cells reported (never filled)")


def test_cross_eval_end_to_end_reuse() -> None:
    """cross_eval.main with valid cached evals: zero re-inference, no torch."""
    import cross_eval
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        runs = _fixture(tmp)
        res = tmp / "results"
        assert "torch" not in sys.modules
        cross_eval.main(["--runs-dir", str(runs), "--results-dir", str(res),
                         "--data-root", str(tmp / "nodata"), "--models",
                         "custom-obb", "--datasets", "visdrone", "codrone"])
        assert "evaluate" not in sys.modules, "cached cells must not re-evaluate"
        csv = (res / "cross_dataset_matrix.csv").read_text().splitlines()
        assert csv[0].startswith("model,family,trained_on,eval_on")
        assert len(csv) - 1 == 4                  # 2 runs x 2 cached cells
        # full 3-dataset pass: uncached cells need torch (absent here) ->
        # recorded as failures, the matrix still holds only real cells
        cross_eval.main(["--runs-dir", str(runs), "--results-dir", str(res),
                         "--data-root", str(tmp / "nodata")])
        js = json.loads((runs / "cross_eval.json").read_text())
        assert js["failures"], "uncached cell errors must be recorded"
        csv = (res / "cross_dataset_matrix.csv").read_text().splitlines()
        assert len(csv) - 1 == 3 + 2 + 1 + 1      # cells with evidence only
        assert (res / "cross_dataset_summary.md").is_file()
        for stem in ("cross_dataset_heatmap_map50", "cross_dataset_heatmap_delta_map50"):
            for ext in ("pdf", "svg", "png"):
                p = res / "figures" / f"{stem}.{ext}"
                assert p.is_file() and p.stat().st_size > 500, p
        assert "Missing cells" in (res / "cross_dataset_summary.md").read_text()
        # stale weights (newer than the eval json) must NOT be reused
        rd = runs / "train" / "visdrone_custom-obb"
        old = time.time() - 100
        os.utime(rd / "eval_codrone_test.json", (old, old))
        assert cross_eval.reusable_eval(
            str(rd), "codrone", "test", str(rd / "weights" / "best.pt"), None) is None
        assert cross_eval.reusable_eval(
            str(rd), "visdrone", "test", str(rd / "weights" / "best.pt"), None)
    print("[ok] cross_eval: cached cells reused (no torch), matrix+heatmaps written, stale weights re-evaluated")


def test_latex_table() -> None:
    import report_lib
    with tempfile.TemporaryDirectory() as t:
        recs, _ = report_lib.load_eval_records([str(_fixture(Path(t)))])
        tex = report_lib.latex_table(recs, hardware_label="NVIDIA L4",
                                     n_registered=23)
        assert r"\begin{table*}" in tex and r"\bottomrule" in tex
        assert r"custom-obb".replace("-", "-") in tex
        assert r"\textbf{0.550}" in tex          # best visdrone mAP50 (yolo)
        assert r"\textbf{0.600}" in tex          # best codrone mAP50
        assert "3 of 23 registered" in tex and "NVIDIA L4" in tex
        assert "& --" in tex                     # models absent on a dataset
        assert tex.count(r"\begin{tabular}") == 1
        open_ = tex.count("{") - tex.count(r"\{")
        assert tex.count("{") == tex.count("}"), "unbalanced braces"
        try:
            report_lib.latex_table([])
            raise AssertionError("empty must raise")
        except ValueError:
            pass
    print("[ok] latex: balanced, best bolded, missing '--', caption honest about n/23")


def test_generate_report_outputs_and_honesty() -> None:
    import generate_report
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        runs, res = _fixture(tmp), tmp / "results"
        rc = generate_report.main(["--runs-dir", str(runs), "--results-dir",
                                   str(res), "--no-qualitative"])
        assert rc == 0
        assert (res / "benchmark_table.tex").stat().st_size > 400
        figs = res / "figures"
        for stem in ("pareto_speed_vs_accuracy_visdrone",
                     "size_vs_accuracy_visdrone", "radar_per_class_visdrone",
                     "pareto_speed_vs_accuracy_codrone"):
            for ext in ("pdf", "svg", "png"):
                p = figs / f"{stem}.{ext}"
                assert p.is_file() and p.stat().st_size > 500, p
        assert not (figs / "pareto_speed_vs_accuracy_dota.pdf").exists()
        cov = json.loads((res / "report_coverage.json").read_text())
        assert cov["registered_total"] == 23 and len(cov["models_evaluated"]) == 3
        assert len(cov["models_without_results"]) == 20
        # with nothing valid -> non-zero exit and nothing written
        empty = tmp / "empty"
        empty.mkdir()
        rc2 = generate_report.main(["--runs-dir", str(empty), "--results-dir",
                                    str(tmp / "r2"), "--no-qualitative"])
        assert rc2 == 2 and not (tmp / "r2").exists()
    print("[ok] generate_report: 4 figure kinds x pdf/svg/png + tex; empty input fails loudly")


def test_qualitative_drawing_and_corners() -> None:
    import report_lib
    from src.metrics.geometry import obb_to_corners
    for box in [(50, 40, 30, 10, 0), (20, 20, 8, 4, 33.0), (5, 9, 3, 7, 170)]:
        assert np.allclose(report_lib.xywhr_to_corners(box), obb_to_corners(*box))
    img = np.zeros((120, 160, 3), dtype=np.uint8) + 90
    s = [dict(image=img, name=f"s{i}", gt=[(2, (80, 60, 40, 16, 25.0))],
              pred=[(2, 0.9, (82, 61, 40, 15, 24.0)),
                    (0, 0.4, (20, 20, 10, 6, 90.0))]) for i in range(5)]
    with tempfile.TemporaryDirectory() as t:
        paths = report_lib.draw_qualitative(s, CLASSES, Path(t) / "q", title="x")
        assert len(paths) == 3 and all(os.path.getsize(p) > 1000 for p in paths)
    assert report_lib.draw_qualitative([], CLASSES, Path("/nonexistent/q")) == []
    print("[ok] qualitative: corners identical to src.metrics.geometry, panels drawn")


def test_colab_recovery() -> None:
    import colab_recovery as cr
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        mir, runs = tmp / "mirror", tmp / "runs"
        for name, n_ep in (("visdrone_custom-obb", 7), ("dota_ao2-detr", 3),
                           ("codrone_yolo11n-obb", 5)):
            d = mir / name
            d.mkdir(parents=True)
            (d / "last.pt").write_bytes(b"L" + name.encode())
            (d / "best.pt").write_bytes(b"B")
            (d / "epochs.jsonl").write_text("{}\n" * n_ep)
        (mir / "_suite").mkdir()
        (mir / "_suite" / "master_summary.csv").write_text("x")
        state = dict(models=["custom-obb", "ao2-detr", "yolo11n-obb"],
                     datasets=["visdrone", "dota", "codrone"],
                     steps=["train", "eval"],
                     jobs=[dict(id="visdrone:custom-obb", status="done", run_dir="/gone"),
                           dict(id="dota:ao2-detr", status="running", run_dir="/gone"),
                           dict(id="codrone:yolo11n-obb", status="pending", run_dir=None)])
        (mir / "pipeline_state.json").write_text(json.dumps(state))
        # a surviving LOCAL checkpoint must never be overwritten
        keep = runs / "train" / "codrone_yolo11n-obb" / "weights"
        keep.mkdir(parents=True)
        (keep / "last.pt").write_bytes(b"LOCAL-NEWER")

        rep = cr.recover(str(mir), str(runs), verbose=False)
        assert rep["mirror_found"] and rep["state_restored"]
        assert (keep / "last.pt").read_bytes() == b"LOCAL-NEWER"
        assert (runs / "train" / "dota_ao2-detr" / "weights" / "last.pt").read_bytes() \
            == b"Ldota_ao2-detr"
        assert not list(runs.rglob("*.part"))
        st = json.loads((runs / "pipeline_state.json").read_text())
        jobs = {j["id"]: j for j in st["jobs"]}
        assert jobs["visdrone:custom-obb"]["status"] == "pending"  # unverified legacy epoch cannot prove done
        assert jobs["dota:ao2-detr"]["status"] == "pending"      # was running
        assert jobs["visdrone:custom-obb"]["run_dir"] == str(
            runs / "train" / "visdrone_custom-obb")
        ep = {j["id"]: j["epochs_logged"] for j in rep["jobs"]}
        assert ep == {"visdrone:custom-obb": 7, "dota:ao2-detr": 3,
                      "codrone:yolo11n-obb": 5}, ep
        assert rep["resume_models"] == state["models"]
        assert rep["resume_datasets"] == state["datasets"]
        assert not any("_suite" in j["id"] for j in rep["jobs"])
        # idempotent
        rep2 = cr.recover(str(mir), str(runs), verbose=False)
        assert all(not j["restored_files"] for j in rep2["jobs"])
        # no mirror / missing mirror dir: clean fresh start, no crash
        assert cr.recover(str(tmp / "nope"), str(tmp / "r2"), verbose=False)["mirror_found"] is False
        assert cr.recover(None, str(tmp / "r3"), verbose=False)["jobs"] == []
    assert cr.split_job_dir("visdrone_yolo11n-obb") == ("visdrone", "yolo11n-obb")
    assert cr.split_job_dir("_suite") is None
    print("[ok] colab recovery: state+checkpoints restored, local never clobbered, running->pending, idempotent")


def test_quickstart_relaunches_restored_queue() -> None:
    import colab_quickstart as cq
    calls = []
    orig_sh, orig_ensure = cq.sh, cq.ensure_drive
    cq.sh = lambda cmd: calls.append(cmd)
    cq.ensure_drive = lambda: False
    try:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            mir, runs = tmp / "m", tmp / "r"
            (mir / "visdrone_custom-obb").mkdir(parents=True)
            (mir / "visdrone_custom-obb" / "last.pt").write_bytes(b"x")
            (mir / "pipeline_state.json").write_text(json.dumps(dict(
                models=["custom-obb"], datasets=["visdrone"],
                steps=["train", "eval", "summary"], jobs=[])))
            cq.main(["--mirror-dir", str(mir), "--runs-dir", str(runs)])
    finally:
        cq.sh, cq.ensure_drive = orig_sh, orig_ensure
    run_cmd = next(c for c in calls if c[1].endswith("run.py") and "--env" not in c)
    j = " ".join(run_cmd)
    assert "--models custom-obb" in j and "--datasets visdrone" in j
    assert "--steps train,eval,summary" in j and "--mirror-every 5" in j
    assert f"--mirror-dir {mir}" in j and "--cache annotations" in j
    print("[ok] quickstart: relaunches the restored queue with mirror flags")


def test_summarize_guards() -> None:
    import summarize_runs
    with tempfile.TemporaryDirectory() as t:
        runs = _fixture(Path(t))
        _mk_run(runs, "lsknet", "visdrone", {"visdrone": (0.4, 0.2)},
                coord="letterbox(imgsz)")
        old = summarize_runs.summarize_run(str(runs / "train" / "visdrone_lsknet"))
        assert old["map50_test"] == "", "older protocol must not be summarized"
        ok = summarize_runs.summarize_run(str(runs / "train" / "visdrone_custom-obb"))
        assert ok["map50_test"] == 0.50 and ok["status"] == "complete"
        # truncated manifest/eval json must not crash the summary
        (runs / "train" / "visdrone_custom-obb" / "eval_visdrone_test.json").write_text("{")
        (runs / "train" / "visdrone_ao2-detr" / "run_manifest.json").write_text("{")
        assert summarize_runs.summarize_run(str(runs / "train" / "visdrone_custom-obb"))
        assert summarize_runs.summarize_run(str(runs / "train" / "visdrone_ao2-detr"))
    print("[ok] summarize_runs: protocol guard, partial/complete status, corrupt json tolerated")


TESTS = [test_load_and_protocol_filter, test_cross_matrix_algebra,
         test_cross_eval_end_to_end_reuse, test_latex_table,
         test_generate_report_outputs_and_honesty,
         test_qualitative_drawing_and_corners, test_colab_recovery,
         test_quickstart_relaunches_restored_queue, test_summarize_guards]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
