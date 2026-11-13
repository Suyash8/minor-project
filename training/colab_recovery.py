#!/usr/bin/env python3
"""
colab_recovery.py - C7: restore a preempted Colab session from the Drive mirror.

Pure standard library (no torch), so it runs before dependencies are installed.

What the pipeline mirrors to Drive (src/training/engine_common.DriveMirror, one
sub-folder per job: <mirror_dir>/<dataset>_<model>/):
    last.pt  best.pt  results.csv  epochs.jsonl  train.log      (+ _suite/)
What this module restores into a wiped local runs dir (<runs_dir>/train/<ds>_<model>/):
    weights/last.pt, weights/best.pt, results.csv, epochs.jsonl, train.log,
    and - when they were mirrored - run_manifest.json and eval_*.json,
plus <runs_dir>/pipeline_state.json (from <mirror_dir>/pipeline_state.json or
<mirror_dir>/_suite/pipeline_state.json).

Rules: verified checkpoint epochs decide checkpoint replacement; unknown
local checkpoints are kept conservatively. State mtimes are only queue hints;
completion is rechecked against selected checkpoint epoch. Metadata mtimes
select copies, but evals are invalidated and manifest epoch is reconciled; copies are atomic (.part + os.replace); a
finished job (state says done AND epochs_done >= the recorded target) is left
"done" for the target-aware pipeline skip check; an interrupted job keeps its last.pt so the
trainers resume at the first unfinished epoch (the schedule/LR continuity
itself is the trainers' job, C3/C4).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MIRROR_DIR = "/content/drive/MyDrive/object-detection/runs_v4"
STATE_FILENAME = "pipeline_state.json"
KNOWN_DATASETS = ("visdrone", "codrone", "dota", "mock", "all")
_FILES = ("last.pt", "best.pt", "results.csv", "epochs.jsonl", "train.log",
          "run_manifest.json")


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def split_job_dir(name: str) -> Optional[tuple]:
    """'<dataset>_<model>' -> (dataset, model); None for non-job folders."""
    if name.startswith("_") or name.startswith("."):
        return None
    for ds in KNOWN_DATASETS:
        if name.startswith(ds + "_") and len(name) > len(ds) + 1:
            return ds, name[len(ds) + 1:]
    return None


def epochs_logged(run_dir: Path) -> int:
    """Completed epochs recorded by the mirrored log (epochs.jsonl lines, else
    results.csv data rows). 0 when neither exists."""
    j = run_dir / "epochs.jsonl"
    if j.is_file():
        try:
            return sum(1 for ln in j.read_text().splitlines() if ln.strip())
        except OSError:
            return 0
    c = run_dir / "results.csv"
    if c.is_file():
        try:
            return max(0, sum(1 for ln in c.read_text().splitlines()
                              if ln.strip()) - 1)
        except OSError:
            return 0
    return 0


def _read_state(path: Path) -> Optional[dict]:
    try:
        d = json.loads(path.read_text())
        return d if isinstance(d, dict) and isinstance(d.get("jobs"), list) else None
    except (OSError, ValueError):
        return None


def checkpoint_epoch(path):
    """Completed epoch only if sidecar is cryptographically tied to bytes."""
    import hashlib
    path=Path(path)
    try:
        side=json.loads(path.with_name(path.name+".progress.json").read_text())
        if path.stat().st_size != side["checkpoint_size"]:
            return None
        hasher=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                hasher.update(block)
        return int(side["epoch"]) if hasher.hexdigest()==side["sha256"] else None
    except (OSError,ValueError,KeyError,TypeError):
        return None


def backup(path):
    import uuid
    target=path.with_name(path.name+".bak-"+uuid.uuid4().hex)
    shutil.copy2(path,target)
    return target


def restore_checkpoint(src,dst):
    """Never regress a verified local epoch because Drive rewrote mtimes.

    Without verified epochs on both sides, preserve any existing local bytes.
    A missing local checkpoint may safely be restored from legacy mirrors.
    """
    src,dst=Path(src),Path(dst)
    if dst.exists():
        se,de=checkpoint_epoch(src),checkpoint_epoch(dst)
        if se is None or de is None or se <= de:
            return False
        backup(dst)
    _atomic_copy(src,dst)
    side=src.with_name(src.name+".progress.json")
    if side.is_file():
        _atomic_copy(side,dst.with_name(dst.name+".progress.json"))
    return True


def reconcile_run(src_dir,run_dir,last_advanced,restored_files=()):
    """Keep best from the winning mirror bundle; invalidate unbound evals.

    Existing evals lack a cryptographic weight binding, so no restored eval is
    safe to reuse solely because its mtime is newer. Keep bytes as a backup
    and mark them non-comparable until evaluation is rerun.
    """
    src_dir,run_dir=Path(src_dir),Path(run_dir)
    best=run_dir/"weights/best.pt"
    if last_advanced and best.exists():
        # Even a verified older best cannot prove the latest best upload landed.
        # No transactional bundle manifest exists: use selected last safely.
        backup(best);best.unlink()
    for ev in run_dir.glob("eval_*.json"):
        if not last_advanced and ev.name not in restored_files:continue
        try:
            data=json.loads(ev.read_text())
            if data.get("recovery_invalidated"):continue
            backup(ev)
            data.update(comparable=False,recovery_invalidated=True,
                        recovery_reason="Evaluation is not hash-bound to selected checkpoint; rerun eval")
            tmp=ev.with_name(ev.name+".part")
            tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,ev)
        except (ValueError,OSError):
            continue
    manifest=run_dir/"run_manifest.json"
    if manifest.is_file():
        try:
            data=json.loads(manifest.read_text())
            epoch=checkpoint_epoch(run_dir/"weights/last.pt")
            if epoch is not None:
                data.setdefault("final_metrics",{})["epochs_trained"]=epoch
            data["recovered_checkpoint_epoch"]=epoch
            manifest.write_text(json.dumps(data,indent=2))
        except (ValueError,OSError):pass


def recover(mirror_dir: Optional[str], runs_dir: str,
            verbose: bool = True) -> Dict[str, Any]:
    """Restore state + surviving checkpoints. Returns a report dict:
    {mirror_found, state_restored, state_source, jobs: [{id, status,
    epochs_mirrored, restored_files}], resume_models, resume_datasets,
    resume_steps}. Never raises on a missing/partial mirror."""
    rep: Dict[str, Any] = dict(
        mirror_dir=mirror_dir, mirror_found=False, state_restored=False,
        state_source=None, jobs=[], resume_models=[], resume_datasets=[],
        resume_steps=[])
    say = print if verbose else (lambda *a, **k: None)
    if not mirror_dir or not os.path.isdir(mirror_dir):
        say(f"[recover] no mirror at {mirror_dir}: fresh start")
        return rep
    rep["mirror_found"] = True
    mirror, runs = Path(mirror_dir), Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)

    # ---- pipeline_state.json -------------------------------------------
    local_state = runs / STATE_FILENAME
    cands = [p for p in (mirror / STATE_FILENAME,
                         mirror / "_suite" / STATE_FILENAME)
             if p.is_file() and _read_state(p) is not None]
    if cands:
        best = max(cands, key=lambda p: p.stat().st_mtime)
        if (not local_state.exists()
                or local_state.stat().st_mtime < best.stat().st_mtime
                ):
            if local_state.exists():
                backup(local_state)
            _atomic_copy(best, local_state)
            rep["state_restored"], rep["state_source"] = True, str(best)
            say(f"[recover] restored {STATE_FILENAME} from {best}")
    state = _read_state(local_state) if local_state.exists() else None

    # ---- per-job checkpoints --------------------------------------------
    state_jobs = {j.get("id"): j for j in (state or {}).get("jobs", [])}
    for sub in sorted(p for p in mirror.iterdir() if p.is_dir()):
        parsed = split_job_dir(sub.name)
        if not parsed:
            continue
        ds, model = parsed
        run_dir = runs / "train" / sub.name
        got: List[str] = []
        for fn in _FILES + tuple(x.name for x in sub.glob("eval_*.json")) + tuple(x.name for x in sub.glob("*.progress.json")):
            src = sub / fn
            if not src.is_file():
                continue
            if fn.endswith(".progress.json"):
                continue # restored atomically with the selected checkpoint
            dst = (run_dir / "weights" / fn if fn.endswith(".pt") else run_dir / fn)
            if fn=="best.pt":
                continue  # async legacy best has no trustworthy provenance
            if fn.endswith(".pt"):
                if restore_checkpoint(src,dst):
                    got.append(fn)
                continue
            if dst.exists():
                if dst.stat().st_mtime >= src.stat().st_mtime:
                    continue
                backup(dst)
            _atomic_copy(src,dst)
            got.append(fn)
        reconcile_run(sub,run_dir,"last.pt" in got or "best.pt" in got,got)
        n_logs = epochs_logged(sub)
        n_ep = checkpoint_epoch(run_dir/"weights/last.pt")
        jid = f"{ds}:{model}"
        job = state_jobs.get(jid)
        status = job.get("status") if job else "no-state"
        if job is not None:
            job["run_dir"] = str(run_dir)      # old session's path is stale
            # A job left "running" died mid-epoch: queue it again (its
            # last.pt resumes it); a "done" job stays done.
            if job.get("status") == "running" or n_ep is None:
                job["status"] = "pending"
                status = "pending (was running at preemption)"
        rep["jobs"].append(dict(id=jid, status=status, epochs_mirrored=n_ep, epochs_logged=n_logs,
                                restored_files=got,
                                has_last=(run_dir / "weights" / "last.pt").is_file()))
    if state is not None:
        tmp = local_state.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, local_state)
        rep["resume_models"] = list(state.get("models") or [])
        rep["resume_datasets"] = list(state.get("datasets") or [])
        rep["resume_steps"] = list(state.get("steps") or [])
        rep["resume_invocation"] = state.get("invocation") or {}

    if verbose:
        if rep["jobs"]:
            say(f"[recover] {len(rep['jobs'])} job folder(s) found in the mirror:")
            for j in rep["jobs"]:
                say(f"  {j['id']:<34} {str(j['status']):<10} "
                    f"selected local checkpoint epoch: {str(j['epochs_mirrored'])} (logs {j['epochs_logged']}) "
                    f"restored: {', '.join(j['restored_files']) or '-'}")
        else:
            say("[recover] mirror has no job folders yet")
        if state is None:
            say("[recover] no pipeline_state.json in the mirror: the queue "
                "restarts, but every job resumes from its own restored "
                "last.pt")
    return rep


if __name__ == "__main__":  # manual use: python colab_recovery.py MIRROR RUNS
    import sys
    recover(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MIRROR_DIR,
            sys.argv[2] if len(sys.argv) > 2 else "runs")
