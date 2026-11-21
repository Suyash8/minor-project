# Dataset-specific foreground heads

The training dispatcher discovers accepted classes from the complete training split only.
A dataset with canonical IDs 2 and 7 trains a 2-class foreground head: 0 car, 1 motorcyclist.
DETR keeps its required internal no-object/background channel, which is not a predicted foreground label.
The original cleaned class order and labels are unchanged.

Images are symlinked into a local scratch view. Only copied labels are compacted.
Ultralytics uses that view's smaller names/NC YAML, not a prediction filter.
All engines use the same mapping. With staging enabled the view is built only after
local staging completes, rather than training from a partial inventory.

Each run and weights directory contains class_schema.json with the original IDs,
active class names and training-label fingerprint. Dense/DETR checkpoints embed it too.
Resume rejects changed mappings or training-label fingerprints and legacy checkpoints
without a mapping. For the newly resplit data use a NEW runs directory and
--no-auto-resume. Never mix earlier split scores with the new benchmark.
Keep the mapping beside copied YOLO checkpoints; YOLO also embeds active names.

Validation and test cannot enlarge the head. A class found only in a held-out split
is an explicit training preflight error, not a silently dropped label. Correct the
split/coverage before training. Missing test images or labels are errors; there is
no fallback from test to validation, including in mock smoke runs.

After training, the selected best checkpoint is reloaded for full validation and then
a distinct final test. Shared evaluators scatter compact outputs back into canonical
class slots so confusion matrices, per-class metrics and cross-dataset reports retain
their original meaning. Inactive foreground classes have no model output channels.
The report can still contain their empty canonical rows for comparable accounting.

Full mode retains cross-evaluation, summary, report tables/figures and Excel CSVs.
Reports and exports use the current runs directory, not unrelated legacy saves.

Example fresh retraining (real compute, not run by smoke tests):

    python training/run.py --models custom-obb --datasets codrone \
        --data-dir /path/to/cleaned_data --runs-dir /path/to/fresh_runs \
        --mode full --no-auto-resume

Tests: tests/test_class_space.py uses distinct synthetic train/val/test images,
non-prefix IDs, single/identity mappings, empty-head rejection and unsafe-resume rejection.
A 1-epoch synthetic smoke checks plumbing, not convergence or real-data quality.
