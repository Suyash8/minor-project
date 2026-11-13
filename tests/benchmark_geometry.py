"""Reproducible C1 kernel latency/parity gate, no manufactured GPU results.

python tests/benchmark_geometry.py --device cuda --pairs 10000
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import torch
from src.metrics.tensor_obb import aligned_iou
from src.metrics.geometry import compute_obb_iou_matrix


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--pairs',type=int,default=10000);p.add_argument('--output',default=None)
    args=p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('NOT RUN: CUDA is unavailable');return 2
    dev=torch.device(args.device);g=torch.Generator(device=dev).manual_seed(41)
    a=torch.rand(args.pairs,5,device=dev,generator=g);b=torch.rand(args.pairs,5,device=dev,generator=g)
    a[:,:2]*=1280;b[:,:2]=a[:,:2]+(b[:,:2]-.5)*50
    a[:,2:4]=a[:,2:4]*100+1;b[:,2:4]=b[:,2:4]*100+1
    a[:,4]*=180;b[:,4]*=180
    for _ in range(10):aligned_iou(a,b)
    if dev.type=='cuda':torch.cuda.synchronize()
    timings=[]
    for _ in range(50):
        if dev.type=='cuda':
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();actual=aligned_iou(a,b);end.record();end.synchronize();timings.append(start.elapsed_time(end))
        else:
            start=time.perf_counter();actual=aligned_iou(a,b);timings.append((time.perf_counter()-start)*1000)
    # Exact diagonal reference in bounded chunks, no giant NxN allocation.
    an,bn=a.cpu().numpy(),b.cpu().numpy();refs=[]
    for i in range(0,args.pairs,100):refs.extend(compute_obb_iou_matrix(an[i:i+100],bn[i:i+100],device='cpu').diagonal())
    error=float(np.max(abs(actual.cpu().numpy()-np.asarray(refs))))
    result=dict(device=str(dev),hardware=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'CPU',pairs=args.pairs,
                median_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),max_abs_iou_error=error,
                cuda_latency_pass=bool(np.percentile(timings,95)<25) if dev.type=='cuda' else None,parity_pass=error<1e-4)
    print(json.dumps(result,indent=2))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2))
    return 0 if error<1e-4 and (dev.type!='cuda' or result['cuda_latency_pass']) else 1

if __name__=='__main__':raise SystemExit(main())
