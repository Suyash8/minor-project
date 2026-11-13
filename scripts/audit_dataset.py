"""Read-only schema and optional image-hash leakage audit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.schema import validate_names,audit_splits
from src.config import CLEANED_DATASET_CLASSES
p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--datasets',nargs='+',default=['visdrone','codrone','dota']);p.add_argument('--hash-images',action='store_true');p.add_argument('--output');a=p.parse_args()
reports=[]
for ds in a.datasets:
 root=Path(a.data_root)/ds;validate_names(root,CLEANED_DATASET_CLASSES[ds]);reports.append(audit_splits(root,a.hash_images))
text=json.dumps(reports,indent=2);print(text)
if a.output:Path(a.output).write_text(text)
sys.exit(0 if all(r['passes'] and not r['missing_splits'] for r in reports) else 1)
