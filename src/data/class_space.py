"""Dataset-specific foreground heads and non-destructive compact label views.

Only TRAIN annotations choose a head. Validation/test never create output classes.
The source dataset is never relabelled in place. Views live on local scratch disk.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml


def _inventory(dataset, data_root):
    from src.data.dataset import AerialOBBDataset
    return {sp: AerialOBBDataset(dataset, Path(data_root), split=sp, use_cache=False)
            for sp in ('train', 'val', 'test')}


def _parsed(raw, sample):
    path = sample.get('label_path')
    return raw._parse_label_file(path) if path else {'fmt': 'corners_norm', 'items': []}


def prepare_class_view(dataset, data_root):
    """Return (local root, schema) for the classes in TRAIN annotations only."""
    root = Path(data_root).resolve()
    marker = root / dataset / 'class_schema.json'
    if marker.exists():
        return str(root), json.loads(marker.read_text())
    inventories = _inventory(dataset, root)
    for sp, raw in inventories.items():
        if not raw.samples or not any(sample.get('label_path') for sample in raw.samples):
            raise ValueError(f'{dataset}/{sp}: missing images or ground-truth labels; no split fallback is allowed')
    train = inventories['train']
    source_names = list(train.class_names)
    rows = {}
    digest = hashlib.sha256()
    digest.update(str(root).encode())
    for sp, raw in inventories.items():
        rows[sp] = []
        for sample in raw.samples:
            parsed = _parsed(raw, sample)
            rows[sp].append((sample, parsed))
            digest.update(str(sample['image_path']).encode())
            st = sample['image_path'].stat()
            digest.update(str((st.st_size, st.st_mtime_ns)).encode())
            digest.update(json.dumps(parsed, sort_keys=True).encode())
    train_fp = hashlib.sha256(json.dumps([
        [sample['image_path'].name, parsed] for sample, parsed in rows['train']
    ], sort_keys=True).encode()).hexdigest()
    present = sorted({int(it[0]) for _, p in rows['train'] for it in p['items']})
    if not present:
        raise ValueError(f'{dataset}: no foreground training labels; refusing an empty head')
    names = [source_names[i] for i in present]
    schema = dict(version=1, dataset=dataset, source_names=source_names,
                  source_ids=present, head_to_canonical=present,
                  canonical_to_head={str(i): present.index(i) if i in present else -1
                                     for i in range(len(source_names))},
                  class_names=names, num_classes=len(names),
                  selected_from='train', train_fingerprint=train_fp, source_root=str(root))
    head_names = list(names)
    observed = {source_names[int(it[0])] for sp in rows for _, p in rows[sp] for it in p['items']}
    extras = sorted(observed - set(names))
    if extras:
        raise ValueError(f'{dataset}: validation/test contains classes absent from training: {extras}')
    digest.update(json.dumps(schema, sort_keys=True).encode())
    base = Path(os.environ.get('OBB_CLASS_VIEW_CACHE', str(Path(tempfile.gettempdir()) / 'obb_class_views')))
    target_root = base / digest.hexdigest()[:24]
    target = target_root / dataset
    if (target / 'class_schema.json').exists():
        return str(target_root), json.loads((target / 'class_schema.json').read_text())
    # Build under a separate name. A failed build cannot be mistaken for complete.
    base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix='building-', dir=base))
    dest = tmp_root / dataset
    from PIL import Image
    from src.data.augment import boxes_to_corners
    import numpy as np
    for sp, entries in rows.items():
        imdir, lbdir = dest / sp / 'images', dest / sp / 'labels'
        imdir.mkdir(parents=True)
        lbdir.mkdir(parents=True)
        for sample, parsed in entries:
            image = sample['image_path']
            (imdir / image.name).symlink_to(image.resolve())
            lines = []
            size = None
            for item in parsed['items']:
                old = int(item[0])
                name = source_names[old]
                if name not in names:
                    raise ValueError(f'Unmapped label {name}')
                cid = names.index(name)
                if parsed['fmt'] == 'corners_norm':
                    coords = item[1:]
                else:
                    if size is None:
                        with Image.open(image) as im:
                            size = im.size
                    coords = (boxes_to_corners(np.asarray([item[1:]], dtype=np.float32))[0]
                              / np.asarray(size)).reshape(-1).tolist()
                lines.append(str(cid) + ' ' + ' '.join(f'{float(v):.9g}' for v in coords))
            (lbdir / (image.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
    out_schema = dict(schema, class_names=names, num_classes=len(names), head_class_names=head_names)
    (dest / 'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(target), train='train/images',
        val='val/images', test='test/images', nc=len(names), names=names), sort_keys=False))
    (dest / 'class_schema.json').write_text(json.dumps(out_schema, indent=2))
    try:
        tmp_root.rename(target_root)
    except FileExistsError:
        import shutil
        shutil.rmtree(tmp_root)
    return str(target_root), out_schema


def save_schema(run_dir, schema, resume=False):
    """Bind heads/checkpoint resumes to a label mapping, not just head size."""
    path = Path(run_dir) / 'class_schema.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and any((Path(run_dir) / 'weights' / name).exists() for name in ('last.pt', 'best.pt')):
        if not path.exists():
            raise ValueError('Legacy checkpoint has no class mapping; use a fresh runs directory and --no-auto-resume')
        old = json.loads(path.read_text())
        if old['class_names'] != schema['class_names'] or old.get('source_ids') != schema.get('source_ids') or old.get('train_fingerprint') != schema.get('train_fingerprint'):
            raise ValueError('Checkpoint class mapping changed; retrain in a fresh runs directory')
    path.write_text(json.dumps(schema, indent=2))
    weights = path.parent / 'weights'
    weights.mkdir(exist_ok=True)
    (weights / path.name).write_text(path.read_text())


def schema_for_weights(weights):
    p = Path(weights)
    for folder in (p.parent, p.parent.parent):
        f = folder / 'class_schema.json'
        if f.is_file():
            return json.loads(f.read_text())
    try:
        import torch
        checkpoint = torch.load(p, map_location='cpu', weights_only=True)
        return checkpoint.get('class_schema') if isinstance(checkpoint, dict) else None
    except Exception:
        return None
