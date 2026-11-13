#!/usr/bin/env python3
"""
[DEPRECATED SHIM] scripts/run_pipeline.py -> training/run.py (v4)

The old pre/post-eval pipeline orchestrator is superseded by the unified
pipeline `training/run.py` (W1), which runs stage/train/eval/cross/summary/
export/report in one process with job-queue auto-resume and per-model failure
isolation. All arguments are passed straight through.

New canonical usage:
    python training/run.py --models all --datasets visdrone \
        --steps stage,train,eval,summary
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NOTICE = """\
[deprecation] scripts/run_pipeline.py is a shim in v4. The unified pipeline is:
    python training/run.py --models all --datasets all --mode full
Old step names map as: pre_eval/train/post_eval -> --steps eval,train,eval;
--mode full|train|eval selects presets. Extra v3-only flags are accepted via
--set-style pass-through only where training/run.py supports them.
"""


def main() -> None:
    print(NOTICE, file=sys.stderr)
    from training import run
    raise SystemExit(run.main(sys.argv[1:]))


if __name__ == "__main__":
    main()
