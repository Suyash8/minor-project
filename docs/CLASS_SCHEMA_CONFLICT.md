# Class schema: unresolved specification conflict

Do not rename or remap existing numeric labels or checkpoints.

The supplied cleaner (`data_cleaning/clean_build_v4.py`), its full-run report,
`src/config.py::CLEANED_DATASET_CLASSES`, and `training/common.py::CLASSES`
agree on the existing cleaned order:

| ID | Existing cleaned meaning | New PROMPT.md meaning |
|---:|---|---|
| 0 | pedestrian | pedestrian |
| 1 | bicycle | people |
| 2 | car | bicycle |
| 3 | van | car |
| 4 | truck | van |
| 5 | bus | truck |
| 6 | tricycle | tricycle |
| 7 | motorcyclist | awning-tricycle |

The new prompt's order is not a permutation of the old one. The cleaner
merged people into pedestrian and awning-tricycle into tricycle. Those
original subclasses cannot be recovered from the cleaned class IDs alone.
Rebuilding from the raw annotation sources and retraining would be required
to adopt the new order safely. No such rebuild has been performed.

The owner was asked to confirm the intended schema. Until answered, this
revision preserves existing label and checkpoint semantics, and does not
certify the new prompt's frozen-class-order acceptance condition. YAML class
names that disagree with the expected existing cleaned schema cause a clear
error before cleaned dataset use. This guard cannot authenticate numeric
labels without a class-name sidecar, so inspect the dataset's provenance too.

Read-only verification:

```bash
python scripts/audit_dataset.py --data-root /path/to/cleaned_data --hash-images
```

The hash audit detects identical image content across train/val/test,
including differently named copies; it reports missing splits and never
moves, edits or invents data. No real dataset split audit was possible here.
