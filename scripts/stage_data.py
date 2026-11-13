#!/usr/bin/env python3
"""
[DEPRECATED SHIM] scripts/stage_data.py -> training/staging.py (v4)

The staging implementation now lives in `training/staging.py` (W2: atomic
.part copies, byte-exact verification, .stage_manifest.json receipts,
background overlap with training). This shim preserves the old CLI
(--datasets/--src/--dst/--workers/--force) so existing commands keep working.

New canonical usage:
    python -m training.staging --datasets visdrone dota
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.staging import main as _staging_main

NOTICE = (
    "[deprecation] scripts/stage_data.py is a shim in v4; use "
    "`python -m training.staging --datasets ...` (same flags).\n"
    "              staging is now atomic + byte-verified, and the unified "
    "pipeline (`training/run.py`) stages automatically in the background."
)


def main() -> None:
    print(NOTICE, file=sys.stderr)
    _staging_main(sys.argv[1:])


if __name__ == "__main__":
    main()
