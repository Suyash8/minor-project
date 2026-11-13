"""Dataset class-order and split-integrity checks, without changing labels."""
from pathlib import Path
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


def audit_splits(dataset_dir, hash_images=False):
    """Audit all available train/val/test images. Hash mode detects renamed copies.

    No images are removed or moved. Missing splits are reported, not fabricated.
    """
    root=Path(dataset_dir);seen={};duplicates=[];counts={};missing=[]
    for split in ['train','val','test']:
        candidates=[root/split/'images',root/'images'/split]
        folder=next((p for p in candidates if p.is_dir()),None)
        if folder is None:
            missing.append(split);continue
        images=sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp') and p.is_file())
        counts[split]=len(images)
        for path in images:
            if hash_images:
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
                key=h.hexdigest()
            else:key=str(path.resolve())
            if key in seen and seen[key][0]!=split:
                duplicates.append(dict(first=seen[key][1],second=str(path)))
            else:seen[key]=(split,str(path))
    return dict(dataset=str(root),counts=counts,missing_splits=missing,duplicates=duplicates,
                hash_images=hash_images,passes=not duplicates)
