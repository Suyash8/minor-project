"""Dataset class-order and split-integrity checks, without changing labels."""
from pathlib import Path
from typing import Optional
import hashlib
import json


def validate_names(dataset_dir, expected):
    import yaml
    root=Path(dataset_dir)
    for file in list(root.glob('*.yaml'))+list(root.glob('*.yml')):
        obj=yaml.safe_load(file.read_text()) or {}
        names=obj.get('names')
        if names is None:
            continue
        if isinstance(names,dict):
            names=[v for k,v in sorted(names.items(),key=lambda kv:int(kv[0]))]
        if list(names)!=list(expected):
            raise ValueError(f"Class schema mismatch in {file}: {names}; expected existing cleaned class order {list(expected)}. Refusing silent relabeling. See docs/CLASS_SCHEMA_CONFLICT.md")


def _extract_sequence_group(stem: str) -> Optional[str]:
    """Heuristic scene/video sequence prefix (e.g., VisDrone `uav0000123_00001_s` -> `uav0000123`)."""
    import re
    m = re.match(r"^([A-Za-z]*\d{4,})[_-]\d+", stem)
    return m.group(1) if m else None


def audit_splits(dataset_dir, hash_images: bool = True, check_coverage: bool = True):
    """Audit train/val/test splits for content duplicates, sequence leakage, and class coverage (F20).

    - `hash_images=True` (default): SHA-256 content hashes detect renamed/copied identical images.
    - `check_coverage=True`: checks per-class label counts across splits for missing val/test classes.
    - Also reports sequence/scene prefix overlaps across splits when detected.
    """
    root = Path(dataset_dir)
    seen = {}
    duplicates = []
    seq_seen = {}
    sequence_overlaps = []
    counts = {}
    missing = []
    class_counts = {s: {} for s in ("train", "val", "test")}

    for split in ("train", "val", "test"):
        candidates = [root / split / "images", root / "images" / split]
        folder = next((p for p in candidates if p.is_dir()), None)
        if folder is None:
            missing.append(split)
            continue
        images = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") and p.is_file()
        )
        counts[split] = len(images)
        for path in images:
            if hash_images:
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                key = h.hexdigest()
            else:
                key = str(path.resolve())
            if key in seen and seen[key][0] != split:
                duplicates.append(dict(first=seen[key][1], second=str(path)))
            else:
                seen[key] = (split, str(path))

            grp = _extract_sequence_group(path.stem)
            if grp:
                if grp in seq_seen and seq_seen[grp][0] != split:
                    sequence_overlaps.append(dict(group=grp, first_split=seq_seen[grp][0], second_split=split))
                else:
                    seq_seen[grp] = (split, str(path))

        lbl_candidates = [root / split / "labels", root / "labels" / split]
        lbl_dir = next((p for p in lbl_candidates if p.is_dir()), None)
        if check_coverage and lbl_dir is not None:
            for lf in lbl_dir.glob("*.txt"):
                try:
                    for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.strip().split()
                        if parts and parts[0].isdigit():
                            cid = parts[0]
                            class_counts[split][cid] = class_counts[split].get(cid, 0) + 1
                except OSError:
                    continue

    coverage_gaps = []
    if check_coverage and class_counts.get("train"):
        for cid, tr_n in sorted(class_counts["train"].items()):
            if tr_n <= 0:
                continue
            for sp in ("val", "test"):
                if sp not in missing and class_counts[sp].get(cid, 0) == 0:
                    coverage_gaps.append(dict(class_id=cid, split=sp, train_count=tr_n))

    passes = (not duplicates) and (not coverage_gaps) and (not sequence_overlaps)
    return dict(
        dataset=str(root),
        counts=counts,
        missing_splits=missing,
        duplicates=duplicates,
        sequence_overlaps=sequence_overlaps,
        class_counts=class_counts,
        coverage_gaps=coverage_gaps,
        hash_images=hash_images,
        passes=passes,
    )
