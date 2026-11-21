"""Regression tests for compact foreground heads. CPU, no Drive required."""
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from src.data.class_space import prepare_class_view, save_schema
from src.data.dataset import AerialOBBDataset
from src.config import CLEANED_DATASET_CLASSES


def fixture_data(root, ids=(2, 7), val_ids=None):
    import yaml
    for si, sp in enumerate(('train', 'val', 'test')):
        d = root / 'codrone' / sp
        (d / 'images').mkdir(parents=True)
        (d / 'labels').mkdir()
        labels = ids if sp != 'val' or val_ids is None else val_ids
        for i in range(4):
            rng = np.random.default_rng(si * 10 + i)
            Image.fromarray(rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)).save(d / 'images' / f'{sp}_{i}.jpg')
            (d / 'labels' / f'{sp}_{i}.txt').write_text('\n'.join(
                f'{cid} .2 .2 .5 .2 .5 .4 .2 .4' for cid in labels) + '\n')
    (root / 'codrone' / 'dataset.yaml').write_text(yaml.safe_dump(dict(names=CLEANED_DATASET_CLASSES['codrone'])))
    return root


def test_nonprefix_compact_and_source_unchanged(tmp_path):
    root = fixture_data(tmp_path / 'data')
    src = root / 'codrone/train/labels/train_0.txt'
    before = src.read_bytes()
    view, schema = prepare_class_view('codrone', root)
    assert schema['source_ids'] == [2, 7]
    assert schema['class_names'] == ['car', 'motorcyclist']
    assert src.read_bytes() == before
    assert not list(root.rglob('*.cache'))
    raw = AerialOBBDataset('codrone', Path(view), split='val')
    assert raw.class_names == ['car', 'motorcyclist']
    gt = raw.get_ground_truth(0, 160, 160)
    assert len(gt) == 2 and all(len(g['boxes']) == 1 for g in gt)
    assert prepare_class_view('codrone', view)[0] == view


def test_val_only_class_cannot_expand_head(tmp_path):
    root = fixture_data(tmp_path / 'data', (2,), (2, 7))
    with pytest.raises(ValueError, match='absent from training'):
        prepare_class_view('codrone', root)


def test_single_and_identity(tmp_path):
    for label, ids in [('single', (7,)), ('identity', tuple(range(8)))]:
        root = fixture_data(tmp_path / label, ids)
        _, schema = prepare_class_view('codrone', root)
        assert schema['source_ids'] == list(ids)
        assert schema['num_classes'] == len(ids)


def test_resume_rejects_same_size_different_mapping(tmp_path):
    run = tmp_path / 'run'
    schema = dict(class_names=['car'], source_ids=[2])
    save_schema(run, schema)
    (run / 'weights/last.pt').write_bytes(b'fake')
    with pytest.raises(ValueError, match='mapping changed'):
        save_schema(run, dict(class_names=['bus'], source_ids=[5]), resume=True)
    (run / 'class_schema.json').unlink()
    with pytest.raises(ValueError, match='Legacy'):
        save_schema(run, schema, resume=True)


def test_empty_train_refused(tmp_path):
    root = fixture_data(tmp_path / 'empty', ())
    with pytest.raises(ValueError, match='empty head'):
        prepare_class_view('codrone', root)


def test_missing_test_not_validation_fallback(tmp_path):
    root = fixture_data(tmp_path / 'data')
    import shutil
    shutil.rmtree(root / 'codrone/test')
    with pytest.raises(ValueError, match='no split fallback'):
        prepare_class_view('codrone', root)


def test_train_label_change_invalidates_resume(tmp_path):
    root = fixture_data(tmp_path / 'data')
    _, old = prepare_class_view('codrone', root)
    run = tmp_path / 'run'
    save_schema(run, old)
    (run / 'weights/last.pt').write_bytes(b'fake')
    label = root / 'codrone/train/labels/train_0.txt'
    label.write_text(label.read_text().replace('.5', '.6'))
    _, new = prepare_class_view('codrone', root)
    assert old['class_names'] == new['class_names']
    assert old['train_fingerprint'] != new['train_fingerprint']
    with pytest.raises(ValueError, match='mapping changed'):
        save_schema(run, new, resume=True)


def test_current_run_excel_export_has_per_class_rows(tmp_path):
    import subprocess
    import sys
    import csv
    run = tmp_path / 'runs/train/codrone_custom-obb'
    run.mkdir(parents=True)
    (run / 'run_manifest.json').write_text(json.dumps(dict(model='custom-obb', dataset='codrone')))
    (run / 'eval_codrone_test.json').write_text(json.dumps(dict(model='custom-obb',dataset='codrone',
        split='test',map50=0.2,map50_95=0.1,per_class=[dict(name='car',instances=1,tp=1,fp=0,fn=0,
        precision=1.0,recall=1.0,f1=1.0,ap50=1.0,ap75=1.0,ap50_95=1.0)])))
    out = tmp_path / 'export'
    proc = subprocess.run([sys.executable, 'scripts/export_results.py', '--results-root', str(tmp_path/'runs'),
                           '--output-dir', str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with (out/'class_wise_all.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['class'] == 'car'
