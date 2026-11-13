"""Real CPU optimizer/checkpoint integration: 2 epochs, resume to 5.

Optional --native exercises Ultralytics full-state and stripped continuation.
No real-data convergence or no-jump loss-curve claim follows from this test.
"""
import sys, tempfile, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tests.test_v4_units import _mock_data_dir
from src.training.train_dense import train_dense_model
from src.training.train_detr import train_detr_model

def custom():
 for name,engine in [('lsknet',train_dense_model),('ao2-detr',train_detr_model)]:
  with tempfile.TemporaryDirectory() as temp:
   kw=dict(model_name=name,dataset_name='mock',data_dir=_mock_data_dir(),run_dir=temp,device='cpu',workers=0,
           img_size=160,batch_size=2,micro_batch=1,max_batches=2,test_mode=True,augment=False,ema=False,val_map=False,patience=100,lr=.001)
   if name=='ao2-detr':kw['warmup_epochs']=1
   engine(epochs=2,**kw)
   ckpt=torch.load(Path(temp)/'weights/last.pt',map_location='cpu',weights_only=False)
   oldstep=max(float(v['step']) for v in ckpt['optimizer_state']['state'].values() if 'step' in v)
   result=engine(epochs=5,**kw)
   ckpt=torch.load(Path(temp)/'weights/last.pt',map_location='cpu',weights_only=False)
   assert result['epochs_trained']==5 and ckpt['epoch']==5
   assert ckpt['scheduler_state']['T_max']==5
   assert max(float(v['step']) for v in ckpt['optimizer_state']['state'].values() if 'step' in v)>oldstep
   assert all(math.isfinite(h['train_loss']) for h in ckpt['history'])
   assert abs(ckpt['optimizer_state']['param_groups'][0]['lr']-ckpt['scheduler_state']['eta_min'])<1e-10
   print('PASS 2->5',name,[(h['epoch'],h['lr'],h['train_loss']) for h in ckpt['history']])

def native():
 from src.training.train_yolo import train_yolo_obb
 from ultralytics.utils.torch_utils import strip_optimizer
 with tempfile.TemporaryDirectory() as temp:
  kw=dict(model_name='yolo11n-obb',dataset_name='mock',data_dir=_mock_data_dir(),run_dir=temp,device='cpu',workers=0,
          imgsz=64,batch_size=2,test_mode=True,patience=100,close_mosaic=0)
  for epochs in [2,5]:
   result=train_yolo_obb(epochs=epochs,**kw)
   ckpt=torch.load(result['last_weights'],map_location='cpu',weights_only=False)
   assert result['epochs_trained']==epochs and ckpt['epoch']==epochs-1 and ckpt.get('optimizer') is not None
  strip_optimizer(result['last_weights'])
  result=train_yolo_obb(epochs=6,**kw)
  ckpt=torch.load(result['last_weights'],map_location='cpu',weights_only=False)
  assert ckpt['epoch']==5 and result['epochs_trained']==6 and ckpt.get('optimizer') is not None
  print('PASS native 2->5 full-state; stripped continuation to global epoch 6')

if __name__=='__main__':
 torch.set_num_threads(2);custom()
 if '--native' in sys.argv:native()
