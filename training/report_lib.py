#!/usr/bin/env python3
"""
report_lib.py - evidence-only publication artifacts (C6 + C8).

Everything here is computed from REAL evaluation results on disk
(`eval_<dataset>_<split>.json` written by training/evaluate.py plus the run's
`run_manifest.json`). Nothing is interpolated, defaulted or invented:

  * a result enters a table/plot only if it is from the CURRENT unified
    protocol (coord_space "original image px ..."), is marked comparable and
    has finite mAP50 / mAP50-95;
  * everything else is listed in the coverage report with the reason;
  * a model without a result is simply absent and the figures say how many
    of the registered models they actually contain.

Pure numpy / pandas / matplotlib (seaborn optional). No torch import, so it
runs on a machine with only the result files.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as _common  # noqa: E402  (stdlib-only protocol definition)

CURRENT_COORD_SPACE_PREFIX = _common.CURRENT_COORD_SPACE_PREFIX
CURRENT_METRIC_VERSION = _common.CURRENT_METRIC_VERSION
DATASETS = ["visdrone", "codrone", "dota"]
FAMILY_ORDER = ["yolo", "dense", "detr"]
FAMILY_LABEL = {"yolo": "YOLO-OBB", "dense": "Dense", "detr": "DETR"}
FAMILY_STYLE = {"yolo": dict(color="#1f77b4", marker="o"),
                "dense": dict(color="#2ca02c", marker="s"),
                "detr": dict(color="#d62728", marker="^")}
CLASS_PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                 "#42d4f4", "#f032e6", "#9A6324", "#808000", "#469990"]

_DETR = {"ao2-detr", "ars-detr", "rio-detr", "rhino", "oriented-former"}
_YOLO_RE = re.compile(r"^yolo(v8|11|26)[nsmlx]-obb$")


def model_family(model: str, hint: Optional[str] = None) -> str:
    if hint in FAMILY_ORDER:
        return hint
    if _YOLO_RE.match(model or ""):
        return "yolo"
    if model in _DETR:
        return "detr"
    return "dense"


def registered_models() -> List[str]:
    """The canonical registry when importable (it is torch-free), else []."""
    try:
        import sys
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.config import CANONICAL_MODELS
        return list(CANONICAL_MODELS)
    except Exception:
        return []


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# evidence loading
# ---------------------------------------------------------------------------
def load_eval_records(roots: Iterable[str], split: str = "test",
                      max_depth: int = 4) -> Tuple[List[dict], List[dict]]:
    """Scan roots for eval_<ds>_<split>.json. Returns (records, skipped).

    One record per (model, trained_on, eval_on); the newest timestamp wins.
    `skipped` rows carry {path, reason} so the report can state what was left
    out and why.
    """
    records: Dict[Tuple[str, str, str], dict] = {}
    skipped: List[dict] = []
    seen: set = set()
    for root in roots:
        root = str(root)
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for cur, dirs, files in os.walk(root):
            if cur.count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            for fn in files:
                m = re.fullmatch(r"eval_(.+)_(train|val|test)\.json", fn)
                if not m or m.group(2) != split:
                    continue
                path = os.path.join(cur, fn)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                e = _read_json(Path(path))
                if e is None:
                    skipped.append(dict(path=path, reason="unreadable json"))
                    continue
                if e.get("comparable", True) is False:
                    skipped.append(dict(path=path, reason="non-comparable "
                                        f"protocol '{e.get('protocol')}'"))
                    continue
                if not _common.protocol_current(e):
                    skipped.append(dict(
                        path=path, reason="older/missing coord_space or "
                        f"metric_version (got '{e.get('metric_version')}', "
                        f"need '{CURRENT_METRIC_VERSION}'): re-run evaluation"))
                    continue
                if not (_finite(e.get("map50")) and _finite(e.get("map50_95"))):
                    skipped.append(dict(path=path,
                                        reason="non-finite mAP50/mAP50-95"))
                    continue
                try:
                    selected_weights=_common.find_weights(cur)
                except FileNotFoundError:
                    skipped.append(dict(path=path,reason="no selected checkpoint for eval binding"));continue
                if not _common.eval_weights_match(e,selected_weights):
                    skipped.append(dict(path=path,reason="evaluation weights hash mismatch/missing; re-run eval"));continue
                man = _read_json(Path(cur) / "run_manifest.json") or {}
                model = e.get("model") or man.get("model") or "?"
                trained_on = man.get("dataset") or "?"
                eval_on = e.get("dataset") or m.group(1)
                class_names = list(e.get("class_names") or
                                   [d.get("name") for d in e.get("per_class", [])])
                pc = {}
                for d in e.get("per_class") or []:
                    if d.get("name") is not None and _finite(d.get("ap50")):
                        pc[d["name"]] = float(d["ap50"])
                if not pc and isinstance(e.get("per_class_ap50"), dict):
                    pc = {k: float(v) for k, v in e["per_class_ap50"].items()
                          if _finite(v)}
                rec = dict(
                    model=model, family=model_family(model, e.get("family")),
                    trained_on=trained_on, eval_on=eval_on, split=split,
                    map50=float(e["map50"]), map50_95=float(e["map50_95"]),
                    map75=(float(e["map75"]) if _finite(e.get("map75"))
                           else float("nan")),
                    params_m=(float(e["parameters"]) / 1e6
                              if _finite(e.get("parameters"))
                              and float(e["parameters"]) > 0 else float("nan")),
                    imgs_per_sec=(float(e["imgs_per_sec"])
                                  if _finite(e.get("imgs_per_sec"))
                                  else float("nan")),
                    timing_basis=e.get("timing_basis", ""),
                    hardware=e.get("hardware", "not recorded"), device=e.get("device", "not recorded"),
                    imgsz=e.get("imgsz"), per_class_ap50=pc,
                    class_names=class_names,
                    timestamp=e.get("timestamp", ""), path=path,
                    run_dir=cur, weights=e.get("weights", ""))
                key = (model, trained_on, eval_on)
                if key not in records or rec["timestamp"] >= records[key]["timestamp"]:
                    records[key] = rec
    return sorted(records.values(),
                  key=lambda r: (r["model"], r["trained_on"], r["eval_on"])), skipped


def validate_speed_provenance(records):
    observed = {(r.get("device", "not recorded"), r.get("hardware", "not recorded"), r.get("timing_basis", "not recorded")) for r in records}
    if len(observed) > 1:
        raise ValueError(f"Mixed speed hardware/device/timing_basis evidence: {sorted(observed)}. Generate separate reports per hardware.")


def in_domain(records: Sequence[dict], dataset: str) -> List[dict]:
    return [r for r in records
            if r["trained_on"] == dataset and r["eval_on"] == dataset]


# ---------------------------------------------------------------------------
# C6: cross-dataset matrix
# ---------------------------------------------------------------------------
def build_cross_matrix(records: Sequence[dict],
                       datasets: Sequence[str] = DATASETS):
    """Long-form pandas DataFrame, one row per (model, trained_on, eval_on).

    delta_map50 = cross mAP50 - in-domain mAP50 of the SAME model trained on
    that dataset (negative = drop under domain shift); NaN when the in-domain
    cell is not available. drop_pct is the same as a percentage of in-domain.
    """
    import pandas as pd
    base = {(r["model"], r["trained_on"]): r for r in records
            if r["trained_on"] == r["eval_on"]}
    rows = []
    for r in records:
        if r["trained_on"] not in datasets or r["eval_on"] not in datasets:
            continue
        ind = base.get((r["model"], r["trained_on"]))
        i50 = ind["map50"] if ind else float("nan")
        i95 = ind["map50_95"] if ind else float("nan")
        rows.append(dict(
            model=r["model"], family=r["family"], trained_on=r["trained_on"],
            eval_on=r["eval_on"], in_domain=(r["trained_on"] == r["eval_on"]),
            map50=r["map50"], map50_95=r["map50_95"],
            in_domain_map50=i50, in_domain_map50_95=i95,
            delta_map50=r["map50"] - i50, delta_map50_95=r["map50_95"] - i95,
            drop_pct_map50=(100.0 * (i50 - r["map50"]) / i50
                            if _finite(i50) and i50 > 0 else float("nan"))))
    cols = ["model", "family", "trained_on", "eval_on", "in_domain", "map50",
            "map50_95", "in_domain_map50", "in_domain_map50_95",
            "delta_map50", "delta_map50_95", "drop_pct_map50"]
    return pd.DataFrame(rows, columns=cols)


def cross_coverage(df, models: Sequence[str],
                   datasets: Sequence[str] = DATASETS) -> List[str]:
    """Missing (model, trained_on, eval_on) cells of the 3x3 design."""
    have = set(zip(df["model"], df["trained_on"], df["eval_on"])) \
        if len(df) else set()
    return [f"{m}: {t} -> {e}" for m in models for t in datasets
            for e in datasets if (m, t, e) not in have]


def cross_markdown(df, datasets: Sequence[str] = DATASETS) -> str:
    """Per-model 3x3 matrix of 'mAP50 / mAP50-95 (dmAP50)'; '--' = no result."""
    if len(df) == 0:
        return "# Cross-dataset transfer\n\nNo evaluated cells.\n"
    out = ["# Cross-dataset transfer matrix (test split)", "",
           "Cell = mAP50 / mAP50-95. In brackets: dmAP50 = cross mAP50 minus "
           "the in-domain mAP50 of the same model (negative = drop). "
           "`--` = not evaluated (never filled in).", ""]
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        out += [f"## {model}", "",
                "| trained on \\ tested on | " + " | ".join(datasets) + " |",
                "|---|" + "---|" * len(datasets)]
        for t in datasets:
            cells = []
            for e in datasets:
                c = sub[(sub["trained_on"] == t) & (sub["eval_on"] == e)]
                if len(c) == 0:
                    cells.append("--")
                    continue
                c = c.iloc[0]
                txt = f"{c['map50']:.3f} / {c['map50_95']:.3f}"
                if not c["in_domain"] and _finite(c["delta_map50"]):
                    txt += f" ({c['delta_map50']:+.3f})"
                elif c["in_domain"]:
                    txt = f"**{txt}**"
                cells.append(txt)
            out.append(f"| {t} | " + " | ".join(cells) + " |")
        out.append("")
    cross = df[~df["in_domain"] & df["delta_map50"].notna()]
    if len(cross):
        g = cross.groupby("model")["delta_map50"].mean().sort_values()
        out += ["## Mean domain-shift change in mAP50 (cross cells only)", "",
                "| model | mean dmAP50 | cells |", "|---|---|---|"]
        for m, v in g.items():
            out.append(f"| {m} | {v:+.4f} | "
                       f"{int((cross['model'] == m).sum())} |")
        out.append("")
    return "\n".join(out)


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False})
    return plt


def _save(fig, stem: Path, formats=("pdf", "svg", "png")) -> List[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = stem.with_name(stem.name + "." + ext)
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
        out.append(str(p))
    return out


def plot_cross_heatmaps(df, out_dir: Path, datasets: Sequence[str] = DATASETS,
                        max_models_per_fig: int = 6) -> List[str]:
    """Seaborn heatmaps (rows trained-on, cols tested-on): mAP50 and
    dmAP50. Missing cells stay blank (masked). Returns written paths."""
    if len(df) == 0:
        return []
    plt = _setup_mpl()
    try:
        import seaborn as sns
    except Exception:
        sns = None
    written: List[str] = []
    models = sorted(df["model"].unique())
    for metric, title, cmap, center in (
            ("map50", "mAP50", "viridis", None),
            ("delta_map50", "$\\Delta$mAP50 vs in-domain", "RdBu", 0.0)):
        for k in range(0, len(models), max_models_per_fig):
            chunk = models[k:k + max_models_per_fig]
            ncol = min(3, len(chunk))
            nrow = int(math.ceil(len(chunk) / ncol))
            fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                     figsize=(3.4 * ncol, 3.0 * nrow + 0.4))
            for ax in axes.ravel():
                ax.axis("off")
            for ax, model in zip(axes.ravel(), chunk):
                ax.axis("on")
                sub = df[df["model"] == model]
                mat = np.full((len(datasets), len(datasets)), np.nan)
                for i, t in enumerate(datasets):
                    for j, e in enumerate(datasets):
                        c = sub[(sub["trained_on"] == t) & (sub["eval_on"] == e)]
                        if len(c):
                            mat[i, j] = float(c.iloc[0][metric])
                if np.isnan(mat).all():
                    ax.set_title(f"{model}\n(no result)", fontsize=8)
                    continue
                mask = np.isnan(mat)
                kw = dict(annot=True, fmt=".3f", mask=mask, cmap=cmap,
                          linewidths=0.8, linecolor="white", cbar=False,
                          ax=ax, xticklabels=list(datasets),
                          yticklabels=list(datasets),
                          annot_kws={"fontsize": 8})
                if center is not None:
                    lim = max(0.05, float(np.nanmax(np.abs(mat))))
                    kw.update(center=center, vmin=-lim, vmax=lim)
                else:
                    kw.update(vmin=0.0, vmax=max(0.3, float(np.nanmax(mat))))
                if sns is not None:
                    sns.heatmap(mat, **kw)
                else:  # seaborn missing: same picture with matplotlib
                    ax.imshow(np.ma.masked_invalid(mat), cmap=cmap,
                              vmin=kw["vmin"], vmax=kw["vmax"])
                    ax.set_xticks(range(len(datasets)), datasets)
                    ax.set_yticks(range(len(datasets)), datasets)
                    for (i, j), v in np.ndenumerate(mat):
                        if not np.isnan(v):
                            ax.text(j, i, f"{v:.3f}", ha="center",
                                    va="center", fontsize=8)
                ax.set_title(model, fontsize=9)
                ax.set_xlabel("tested on")
                ax.set_ylabel("trained on")
            fig.suptitle(f"Cross-dataset transfer: {title} (test split)",
                         fontsize=10)
            fig.tight_layout()
            written += _save(fig, out_dir / (f"cross_dataset_heatmap_{metric}"
                             + (f"_{k // max_models_per_fig + 1}"
                                if len(models) > max_models_per_fig else "")))
            plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# C8: LaTeX table
# ---------------------------------------------------------------------------
def _tex(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def latex_table(records: Sequence[dict], datasets: Sequence[str] = DATASETS,
                hardware_label: str = "", split: str = "test",
                n_registered: int = 0) -> str:
    """booktabs table: model, family, params, imgs/s, then in-domain
    mAP50 / mAP50-95 per dataset. Best value per column is bold; missing
    values print '--'. Needs \\usepackage{booktabs}."""
    validate_speed_provenance(records)
    use = [d for d in datasets if in_domain(records, d)]
    by_model: Dict[str, dict] = {}
    for r in records:
        if r["trained_on"] == r["eval_on"] and r["eval_on"] in use:
            by_model.setdefault(r["model"], {})[r["eval_on"]] = r
    models = sorted(by_model, key=lambda m: (
        FAMILY_ORDER.index(model_family(m)), m))
    if not models:
        raise ValueError("no in-domain evaluation results to tabulate")

    def pick(m, key):
        vals = [v[key] for v in by_model[m].values() if _finite(v[key])]
        return vals[0] if vals else float("nan")

    cols: Dict[str, List[float]] = {}
    for d in use:
        cols[f"{d}_50"] = [by_model[m][d]["map50"] if d in by_model[m]
                           else float("nan") for m in models]
        cols[f"{d}_95"] = [by_model[m][d]["map50_95"] if d in by_model[m]
                           else float("nan") for m in models]
    best = {k: (np.nanmax(v) if np.isfinite(v).any() else None)
            for k, v in cols.items()}

    def cell(k, i):
        v = cols[k][i]
        if not _finite(v):
            return "--"
        s = f"{v:.3f}"
        return rf"\textbf{{{s}}}" if best[k] is not None and v == best[k] else s

    cap = (f"Oriented-box detection benchmark ({split} split, in-domain: "
           "trained and tested on the same dataset). mAP50 and mAP50-95 are "
           "computed by one shared protocol in original-image pixels. ")
    if hardware_label:
        cap += f"Speed measured on {_tex(hardware_label)}. "
    cap += (f"{len(models)} of {n_registered} registered models shown; "
            "missing entries (--) were not evaluated." if n_registered
            else "Missing entries (--) were not evaluated.")
    head1 = r"Model & Family & Params (M) & imgs/s"
    for d in use:
        head1 += rf" & \multicolumn{{2}}{{c}}{{{_tex(d)}}}"
    head2 = r" & & & "
    for _ in use:
        head2 += r" & mAP50 & mAP50-95"
    cmid = "".join(rf"\cmidrule(lr){{{5 + 2 * i}-{6 + 2 * i}}}"
                   for i in range(len(use)))
    lines = [r"% requires \usepackage{booktabs}",
             r"\begin{table*}[t]", r"\centering", r"\small",
             rf"\caption{{{cap.strip()}}}", r"\label{tab:obb-benchmark}",
             r"\begin{tabular}{llrr" + "rr" * len(use) + "}", r"\toprule",
             head1 + r" \\", cmid, head2 + r" \\", r"\midrule"]
    last_fam = None
    for i, m in enumerate(models):
        fam = model_family(m)
        if last_fam is not None and fam != last_fam:
            lines.append(r"\midrule")
        last_fam = fam
        p, f = pick(m, "params_m"), pick(m, "imgs_per_sec")
        row = f"{_tex(m)} & {FAMILY_LABEL[fam]}"
        row += f" & {p:.2f}" if _finite(p) else " & --"
        row += f" & {f:.1f}" if _finite(f) else " & --"
        for d in use:
            row += f" & {cell(f'{d}_50', i)} & {cell(f'{d}_95', i)}"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# C8: figures
# ---------------------------------------------------------------------------
def _pareto_front(xs: np.ndarray, ys: np.ndarray) -> List[int]:
    """Indices of non-dominated points (higher x AND higher y is better)."""
    idx = sorted(range(len(xs)), key=lambda i: -xs[i])
    front, best_y = [], -np.inf
    for i in idx:
        if ys[i] > best_y:
            front.append(i)
            best_y = ys[i]
    return sorted(front, key=lambda i: xs[i])


def _plain_log_x(ax):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _annotate(ax, xs, ys, labels):
    for x, y, t in zip(xs, ys, labels):
        ax.annotate(t, (x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize=6)


def plot_pareto(records: Sequence[dict], dataset: str, out_dir: Path,
                n_registered: int = 0, hardware_label: str = "") -> List[str]:
    rows = [r for r in in_domain(records, dataset)
            if _finite(r["imgs_per_sec"]) and r["imgs_per_sec"] > 0]
    if not rows:
        return []
    validate_speed_provenance(rows)
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    xs = np.array([r["imgs_per_sec"] for r in rows])
    ys = np.array([r["map50"] for r in rows])
    for fam in FAMILY_ORDER:
        sel = [i for i, r in enumerate(rows) if r["family"] == fam]
        if sel:
            ax.scatter(xs[sel], ys[sel], s=28, label=FAMILY_LABEL[fam],
                       **FAMILY_STYLE[fam])
    front = _pareto_front(xs, ys)
    if len(front) > 1:
        ax.step(xs[front], ys[front], where="post", color="0.4", lw=1,
                ls="--", label="Pareto front")
    _annotate(ax, xs, ys, [r["model"]+(" (native dense)" if r["model"] == "yolo11-obb" else "") for r in rows])
    _plain_log_x(ax)
    measured = hardware_label or ", ".join(sorted({r.get("hardware", "not recorded") for r in rows}))
    ax.set_xlabel("Throughput (images/s), " + measured)
    ax.set_ylabel("mAP50")
    ax.set_title(f"Speed vs. accuracy on {dataset} ({rows[0].get('split', 'test')}): {len(rows)}"
                 + (f"/{n_registered}" if n_registered else "") + " models")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = _save(fig, out_dir / f"pareto_speed_vs_accuracy_{dataset}")
    plt.close(fig)
    return paths


def plot_size_scatter(records: Sequence[dict], dataset: str, out_dir: Path,
                      n_registered: int = 0) -> List[str]:
    rows = [r for r in in_domain(records, dataset) if _finite(r["params_m"])]
    if not rows:
        return []
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    xs = np.array([r["params_m"] for r in rows])
    ys = np.array([r["map50_95"] for r in rows])
    for fam in FAMILY_ORDER:
        sel = [i for i, r in enumerate(rows) if r["family"] == fam]
        if sel:
            ax.scatter(xs[sel], ys[sel], s=28, label=FAMILY_LABEL[fam],
                       **FAMILY_STYLE[fam])
    _annotate(ax, xs, ys, [r["model"]+(" (native dense)" if r["model"] == "yolo11-obb" else "") for r in rows])
    _plain_log_x(ax)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("mAP50-95")
    ax.set_title(f"Model size vs. accuracy on {dataset} ({rows[0].get('split', 'test')}): {len(rows)}"
                 + (f"/{n_registered}" if n_registered else "") + " models")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = _save(fig, out_dir / f"size_vs_accuracy_{dataset}")
    plt.close(fig)
    return paths


def top_per_family(records: Sequence[dict], dataset: str) -> List[dict]:
    best: Dict[str, dict] = {}
    for r in in_domain(records, dataset):
        if not r["per_class_ap50"]:
            continue
        if r["family"] not in best or r["map50"] > best[r["family"]]["map50"]:
            best[r["family"]] = r
    return [best[f] for f in FAMILY_ORDER if f in best]


def plot_radar(records: Sequence[dict], dataset: str, out_dir: Path) -> List[str]:
    """Per-class AP50 of the best YOLO / Dense / DETR model on `dataset`."""
    top = top_per_family(records, dataset)
    if not top:
        return []
    names = list(top[0]["class_names"]) or sorted(top[0]["per_class_ap50"])
    names = [n for n in names if n in top[0]["per_class_ap50"]]
    if len(names) < 3:
        return []
    plt = _setup_mpl()
    ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(5.4, 5.4))
    ax = fig.add_subplot(111, projection="polar")
    for r in top:
        # a class a model has no value for stays NaN -> gap, never 0-filled
        vals = np.array([r["per_class_ap50"].get(n, np.nan) for n in names])
        v = np.concatenate([vals, vals[:1]])
        st = FAMILY_STYLE[r["family"]]
        ax.plot(ang_c, v, color=st["color"], lw=1.6,
                label=f"{r['model']} ({FAMILY_LABEL[r['family']]}, "
                      f"mAP50 {r['map50']:.3f})")
        if np.isfinite(v).all():
            ax.fill(ang_c, v, color=st["color"], alpha=0.12)
    ax.set_xticks(ang, names)
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class AP50 on {dataset} ({top[0].get('split', 'test')})", y=1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False,
              fontsize=7)
    paths = _save(fig, out_dir / f"radar_per_class_{dataset}")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# C8: qualitative GT vs prediction panels
# ---------------------------------------------------------------------------
def xywhr_to_corners(box: Sequence[float]) -> np.ndarray:
    """(cx, cy, w, h, angle_deg clockwise from +x) -> (4, 2); identical to
    src.metrics.geometry.obb_to_corners."""
    cx, cy, w, h, a = (float(v) for v in box)
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
    return np.stack([cx + dx * c - dy * s, cy + dx * s + dy * c], axis=-1)


def draw_qualitative(samples: Sequence[dict], class_names: Sequence[str],
                     out_stem: Path, title: str = "",
                     max_cols: int = 4) -> List[str]:
    """samples: [{image: HxWx3 uint8 RGB, gt: [(cls, xywhr)], pred:
    [(cls, conf, xywhr)], name}]. Ground truth above prediction for each
    image; box colour is fixed by class index so colours match across
    figures."""
    if not samples:
        return []
    plt = _setup_mpl()
    from matplotlib.patches import Polygon as MplPoly, Patch
    n = len(samples)
    ncol = min(max_cols, n)
    groups = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(2 * groups, ncol, squeeze=False,
                             figsize=(3.4 * ncol, 3.0 * 2 * groups))
    for ax in axes.ravel():
        ax.axis("off")
    for k, s in enumerate(samples):
        g, c = divmod(k, ncol)
        for row, kind in ((2 * g, "gt"), (2 * g + 1, "pred")):
            ax = axes[row, c]
            ax.axis("on")
            ax.imshow(s["image"])
            items = s["gt"] if kind == "gt" else s["pred"]
            total_count = len(items)
            if kind == "pred":
                items = sorted(items, key=lambda item: item[1], reverse=True)[:20]
            for it in items:
                col = CLASS_PALETTE[int(it[0]) % len(CLASS_PALETTE)]
                ax.add_patch(MplPoly(xywhr_to_corners(it[-1]), closed=True,
                                     fill=False, edgecolor=col, lw=1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(("Ground truth" if kind == "gt" else "Prediction")
                         + f" - {s.get('name', '')} ({len(items)}"
                         + (f"/{total_count} shown, top-20 visual only)" if kind == "pred" else ")"),
                         fontsize=7)
    used = sorted({int(it[0]) for s in samples for it in s["gt"] + s["pred"]})
    fig.legend(handles=[Patch(edgecolor=CLASS_PALETTE[i % len(CLASS_PALETTE)],
                              facecolor="none",
                              label=class_names[i] if i < len(class_names)
                              else str(i)) for i in used],
               loc="lower center", ncol=max(1, min(len(used), 8)),
               frameon=False, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    paths = _save(fig, out_stem)
    plt.close(fig)
    return paths


def write_coverage(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fields, f, indent=2, default=str)
