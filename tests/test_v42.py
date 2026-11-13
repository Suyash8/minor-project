"""v4.2 numerical, resume and contract regressions (CPU, no real data)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import tempfile
import numpy as np
import torch
from src.metrics.tensor_obb import aligned_iou, pairwise_iou, rotated_nms_tensor
from src.metrics.geometry import compute_obb_iou_matrix
from src.training.detr_loss import DETROBBLoss, gaussian_kld, sigmoid_focal
from src.training.schedule import ContinuousCosineLR


def test_tensor_geometry():
    rng = np.random.default_rng(121)
    a = rng.uniform(-200,200,(100,5)); b = rng.uniform(-200,200,(100,5))
    a[:,2:4] = rng.uniform(.01,100,(100,2)); b[:,2:4] = rng.uniform(.01,100,(100,2))
    a[:8] = [[0,0,20,10,0],[0,0,20,10,90],[0,0,0,10,0],[0,0,20,10,180],
             [10000,10000,20,10,30],[0,0,20,10,45],[0,0,.001,100,89],[10,0,20,10,0]]
    b[:8] = a[:8]
    ta,tb = torch.tensor(a,dtype=torch.float32),torch.tensor(b,dtype=torch.float32)
    actual = pairwise_iou(ta,tb).numpy()
    ref = compute_obb_iou_matrix(a,b,device='cpu')
    assert np.max(abs(actual-ref)) < 1e-4, np.max(abs(actual-ref))
    assert torch.isfinite(pairwise_iou(ta[:0],tb)).all()
    assert torch.allclose(pairwise_iou(ta,tb),pairwise_iou(tb,ta).T,atol=2e-5)
    assert (actual >= 0).all() and (actual <= 1).all()
    assert torch.equal(rotated_nms_tensor(ta[:2],torch.tensor([.9,.8]),.1),torch.tensor([0]))
    print('geometry: 10,000 pairs, degeneracy, edge-touch, symmetry, empty and NMS PASS; max error',np.max(abs(actual-ref)))


def test_metric_unmatched_gt():
    from src.metrics.detection_map import evaluate_class_detection
    gt=[dict(boxes=np.array([[0,0,10,10,0],[2,0,10,10,0]],np.float32))]
    pred=[dict(boxes=np.array([[0,0,10,10,0],[.5,0,10,10,0]],np.float32),scores=np.array([.9,.8]))]
    r=evaluate_class_detection(gt,pred,.5)
    assert r['tp']==2 and r['fp']==0, r


def test_focal_and_kl():
    logits=torch.tensor([[-40.,40.],[40.,-40.]],requires_grad=True)
    target=torch.tensor([[0.,1.],[1.,0.]])
    loss=sigmoid_focal(logits,target).sum();loss.backward()
    assert torch.isfinite(logits.grad).all()
    b=torch.tensor([[.5,.5,.1,.2]],requires_grad=True);a=torch.tensor([[.3]],requires_grad=True)
    same=gaussian_kld(b,a,b,a)
    assert float(same)<1e-5
    swapped=torch.cat([b[:,:2],b[:,3:4],b[:,2:3]],-1)
    assert float(gaussian_kld(b,a,swapped,a+math.pi/2))<1e-5
    assert float(gaussian_kld(b,a,b,a+math.pi))<1e-5
    for targets in [[dict(boxes=torch.empty(0,5),labels=torch.empty(0,dtype=torch.long))],
                    [dict(boxes=torch.tensor([[20.,20.,8.,4.,179.]]),labels=torch.tensor([2]))]]:
        l=torch.randn(1,8,9,requires_grad=True);bx=torch.rand(1,8,4,requires_grad=True);an=torch.rand(1,8,1,requires_grad=True)
        out=DETROBBLoss(8)(l,bx,an,targets,64);out['total_loss'].backward()
        assert torch.isfinite(l.grad).all() and torch.isfinite(bx.grad).all()
        assert l.grad[...,8].abs().sum()==0 # legacy background row unused


def test_schedule_extension():
    p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=.01)
    old=ContinuousCosineLR(opt,20,1e-5)
    for _ in range(20):
        opt.zero_grad();p.square().backward();opt.step();old.step()
    state=opt.state_dict();mom=state['state'][0]['exp_avg'].clone()
    newer=ContinuousCosineLR(opt,80,1e-5)
    opt.load_state_dict(state);newer.load_state_dict(old.state_dict());newer.resume_at(20)
    expected=1e-5+.5*(.01-1e-5)*(1+math.cos(math.pi*20/80))
    assert abs(opt.param_groups[0]['lr']-expected)<1e-12
    assert torch.equal(opt.state[p]['exp_avg'],mom)
    values=[]
    for _ in range(60):newer.step();values.append(opt.param_groups[0]['lr'])
    assert all(a>=b for a,b in zip(values,values[1:])) and abs(values[-1]-1e-5)<1e-12
    newer.resume_at(20,warm_restart=True)
    assert abs(opt.param_groups[0]['lr']-.01)<1e-12


def test_all_detr_backbone_gradients():
    from src.models import get_model
    from src.training.train_detr import extract_detr_predictions
    for name in ['ao2-detr','ars-detr','rio-detr','rhino','oriented-former']:
        d=get_model(name,device='cpu',num_classes=8);d.load();n=d.net.train()
        out=n(torch.rand(2,3,64,64));l,b,a=extract_detr_predictions(out,name)
        assert out.get('aux_outputs'),name
        t=[dict(boxes=torch.tensor([[30.,30.,12.,6.,175.]]),labels=torch.tensor([3])) for _ in range(2)]
        loss=DETROBBLoss(8)(l,b,a,t,64,aux_outputs=out['aux_outputs'],model_name=name)
        loss['total_loss'].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in n.parameters())
        assert sum(float(p.grad.abs().sum()) for p in n.backbone.parameters() if p.grad is not None)>0,name


def test_no_label_leak():
    from src.models import get_model
    d=get_model('ao2-detr',device='cpu',num_classes=8)
    try:d._simulate_if_hinted(0,10,10,8,.1,[])
    except ValueError:pass
    else:raise AssertionError('GT-conditioned inference allowed')


def test_disk_cache_written():
    from tests.test_v4_units import _mock_data_dir
    from src.data.dataset import AerialOBBDataset
    with tempfile.TemporaryDirectory() as temp:
        ds=AerialOBBDataset('mock',Path(_mock_data_dir()),split='train',disk_cache_dir=temp)
        image=ds.get_image_array(0)
        files=list(Path(temp).rglob('*.npy'))
        assert files and np.array_equal(np.load(files[0]),image)
        assert not list(Path(temp).rglob('*.part*'))


def test_adversarial_geometry():
    base=torch.tensor([[1280.,1073.,97.,31.,27.],[10000.,10000.,20.,10.,47.],
                       [1000.,1000.,1200.,.1,89.],[0.,0.,20.,10.,0.]])
    for shift in [0.,.001,1.,10.]:
        other=base.clone();other[:,0]+=shift
        actual=aligned_iou(base,other).numpy()
        ref=np.diag(compute_obb_iou_matrix(base.numpy(),other.numpy(),device='cpu'))
        assert np.max(abs(actual-ref))<1e-4,(shift,actual,ref)
    swapped=base.clone();swapped[:,2:4]=base[:,[3,2]];swapped[:,4]+=90
    assert torch.allclose(aligned_iou(base,swapped),torch.ones(4),atol=1e-4)
    edge=torch.tensor([[20.,0.,20.,10.,0.],[20.,10.,20.,10.,0.]])
    assert aligned_iou(base[-1:].expand(2,-1),edge).max()<1e-6
    assert aligned_iou(torch.tensor([[0.,0.,20.,10.,37.]]),torch.tensor([[0.,0.,10.,5.,37.]])).item()>.24999


def test_kl_autocast_degenerate():
    for dtype in [torch.bfloat16,torch.float16]:
        b=torch.tensor([[.5,.5,0.,0.],[.5,.5,1.,1e-6]],requires_grad=True)
        a=torch.tensor([[0.],[math.pi/2]],requires_grad=True)
        t=torch.tensor([[.4,.4,.01,.01],[.4,.4,.001,1.]])
        with torch.autocast('cpu',dtype=dtype):
            loss=gaussian_kld(b,a,t,a+.3).sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(b.grad).all() and torch.isfinite(a.grad).all()


def test_nonfinite_resilience():
    l=torch.zeros(1,4,9,requires_grad=True)
    with torch.no_grad():l[0,0,0]=float('nan')
    b=torch.rand(1,4,4,requires_grad=True);a=torch.rand(1,4,1,requires_grad=True)
    targets=[dict(boxes=torch.tensor([[20.,20.,8.,4.,10.]]),labels=torch.tensor([0]))]
    out=DETROBBLoss(8)(l,b,a,targets,64)
    out['total_loss'].backward()
    assert torch.isfinite(out['total_loss']) and torch.isfinite(l.grad).all()


def test_mirror_immutable():
    from src.training.engine_common import DriveMirror
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);src=root/'last.pt';src.write_bytes(b'old')
        mirror=DriveMirror(str(root/'mirror'))
        # Delay queue thread by holding its lock while submit snapshots.
        import unittest.mock
        with unittest.mock.patch.object(mirror,'_drain',return_value=None):
            mirror.submit(src)
            if mirror._thread:mirror._thread.join()
        src.write_bytes(b'new')
        mirror._drain();mirror.flush()
        assert (root/'mirror/last.pt').read_bytes()==b'old'
        assert not list(root.rglob('.mirror_snapshot/*.pt'))


def test_aux_divergence_bound():
    criterion=DETROBBLoss(8)
    logits=torch.zeros(1,4,9);boxes=torch.rand(1,4,4);angle=torch.zeros(1,4,1)
    aux=dict(cls_logits=logits.clone(),boxes_norm=boxes,angle_vec=torch.ones(1,4,2))
    aux['cls_logits'][0,0,0]=float('nan')
    target=[dict(boxes=torch.empty(0,5),labels=torch.empty(0,dtype=torch.long))]
    for step in range(3):
        try:criterion(logits,boxes,angle,target,64,aux_outputs=[aux],model_name='ars-detr')
        except FloatingPointError:
            assert step==2;break
    else:raise AssertionError('aux divergence did not abort')


def test_odd_letterbox_inverse():
    from src.data.augment import letterbox,boxes_to_corners,corners_to_boxes
    from training.evaluate import _unletterbox_boxes
    from src.training.metric_protocol import original_metric_batch
    import types
    box=np.array([[100.,80.,30.,12.,37.]],np.float32)
    for w,h in [(901,477),(477,901),(1001,633),(319,321)]:
        for target in [160,641,1024]:
            image=np.zeros((h,w,3),np.uint8)
            _,c=letterbox(image,boxes_to_corners(box),target)
            from src.metrics.geometry import polygon_to_obb_params
            lb=np.array([polygon_to_obb_params(poly) for poly in c],np.float32)
            back=_unletterbox_boxes(lb,w,h,target)
            assert np.allclose(back[:,:4],box[:,:4],atol=2e-4),(w,h,target,back)
            raw=types.SimpleNamespace(get_ground_truth=lambda *args:[dict(boxes=box)])
            ds=types.SimpleNamespace(raw=raw,resize_mode='letterbox')
            gt,pred=original_metric_batch([(lb.copy(),np.array([.9]),np.array([0]))],
                [dict(orig_size=(w,h),index=0)],ds,target,1)
            assert np.allclose(pred[0][0]['boxes'][:,:4],box[:,:4],atol=2e-4)
            assert np.array_equal(gt[0][0]['boxes'],box)


def test_newer_recovery_and_speed_guard():
    import json,os
    from training.colab_recovery import recover
    from training.report_lib import validate_speed_provenance
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);mir=root/'m';run=root/'r';(mir/'visdrone_lsknet').mkdir(parents=True);run.mkdir()
        state={'jobs':[],'models':['lsknet']}
        (mir/'pipeline_state.json').write_text(json.dumps(state));(run/'pipeline_state.json').write_text(json.dumps({'jobs':[],'models':['old']}))
        os.utime(run/'pipeline_state.json',(1,1))
        local=run/'train/visdrone_lsknet/weights/last.pt';local.parent.mkdir(parents=True);local.write_bytes(b'old');os.utime(local,(1,1))
        (mir/'visdrone_lsknet/last.pt').write_bytes(b'new')
        rep=recover(str(mir),str(run),False)
        assert rep['state_restored'] and local.read_bytes()==b'old'
        assert rep['jobs'][0]['epochs_mirrored'] is None
    try:validate_speed_provenance([{'device':'cpu','hardware':'CPU'},{'device':'cuda','hardware':'L4'}])
    except ValueError:pass
    else:raise AssertionError('mixed CPU/CUDA speed evidence allowed')


def test_checkpoint_authority_selection():
    import hashlib,json,os
    from training.colab_recovery import restore_checkpoint,checkpoint_epoch
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);src=root/'mirror.pt';dst=root/'local.pt'
        def save(path,epoch,body):
            path.write_bytes(body)
            path.with_name(path.name+'.progress.json').write_text(json.dumps(dict(epoch=epoch,checkpoint_size=len(body),sha256=hashlib.sha256(body).hexdigest())))
        save(src,4,b'old-mirror');save(dst,5,b'new-local');os.utime(dst,(1,1))
        assert not restore_checkpoint(src,dst) and checkpoint_epoch(dst)==5
        save(src,6,b'new-mirror')
        assert restore_checkpoint(src,dst) and checkpoint_epoch(dst)==6
        assert list(root.glob('local.pt.bak-*'))
        save(src,7,b'next-mirror');assert restore_checkpoint(src,dst)
        assert len(list(root.glob('local.pt.bak-*')))==2


def test_recovered_artifact_consistency():
    import hashlib,json
    from training.colab_recovery import recover
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);source=root/'mirror/mock_yolo11n-obb';run=root/'runs/train/mock_yolo11n-obb'
        source.mkdir(parents=True);(run/'weights').mkdir(parents=True)
        for path,epoch,body in [(source/'last.pt',7,b'mirror7'),(run/'weights/last.pt',5,b'local5')]:
            path.write_bytes(body)
            path.with_name(path.name+'.progress.json').write_text(json.dumps(dict(epoch=epoch,checkpoint_size=len(body),sha256=hashlib.sha256(body).hexdigest())))
        (source/'best.pt').write_bytes(b'mirror-best')
        (run/'weights/best.pt').write_bytes(b'stale-local-best')
        (source/'eval_mock_val.json').write_text(json.dumps(dict(comparable=True,map50=.9)))
        (source/'run_manifest.json').write_text(json.dumps(dict(final_metrics=dict(epochs_trained=99))))
        report=recover(str(root/'mirror'),str(root/'runs'),False)
        assert report['jobs'][0]['epochs_mirrored']==7
        assert not (run/'weights/best.pt').exists()  # unverified async best cannot win
        assert json.loads((run/'eval_mock_val.json').read_text())['comparable'] is False
        assert json.loads((run/'run_manifest.json').read_text())['final_metrics']['epochs_trained']==7


def test_mixed_timing_basis_rejected():
    from training.report_lib import validate_speed_provenance
    try:
        validate_speed_provenance([dict(device='cpu',hardware='CPU',timing_basis='a'),dict(device='cpu',hardware='CPU',timing_basis='b')])
    except ValueError:return
    raise AssertionError('mixed timing basis must not be combined')


def test_noop_recovery_preserves_eval_and_legacy_manifest():
    import json
    from training.colab_recovery import recover
    from training.common import weights_fingerprint,eval_weights_match
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);src=root/'mirror/mock_lsknet';run=root/'runs/train/mock_lsknet'
        src.mkdir(parents=True);(run/'weights').mkdir(parents=True)
        (src/'last.pt').write_bytes(b'legacy');(run/'weights/last.pt').write_bytes(b'legacy')
        manifest=run/'run_manifest.json';manifest.write_text(json.dumps(dict(final_metrics=dict(epochs_trained=12))))
        ev=run/'eval_mock_val.json';ev.write_text(json.dumps(dict(comparable=True,weights_fingerprint=weights_fingerprint(run/'weights/last.pt'))))
        for _ in range(2):recover(str(root/'mirror'),str(root/'runs'),False)
        data=json.loads(ev.read_text());assert data['comparable']
        assert eval_weights_match(data,run/'weights/last.pt')
        assert json.loads(manifest.read_text())['final_metrics']['epochs_trained']==12
        (run/'weights/last.pt').write_bytes(b'changed');assert not eval_weights_match(data,run/'weights/last.pt')


if __name__=='__main__':
    torch.set_num_threads(2)
    tests=[v for k,v in list(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test();print('PASS',test.__name__)
    print(f'ALL {len(tests)} v4.2 REGRESSIONS PASSED')
