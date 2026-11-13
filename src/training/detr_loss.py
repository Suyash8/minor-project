"""Sigmoid focal DETR supervision and rotation-aware Gaussian matching."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def covariance(boxes, angles):
    wh = boxes[..., 2:4].float().clamp_min(1e-4).square()/12
    c, s = angles.float().squeeze(-1).cos(), angles.float().squeeze(-1).sin()
    u, v = wh.unbind(-1)
    return torch.stack((u*c*c+v*s*s, (u-v)*c*s,
                        (u-v)*c*s, u*s*s+v*c*c), -1).reshape(*wh.shape[:-1], 2, 2)


def gaussian_kld(box_a, angle_a, box_b, angle_b):
    """Log1p symmetric Gaussian KL, supports pairwise broadcasting.

    Smooth and rotation-aware, invariant under theta+pi and width/height swap
    plus pi/2. An epsilon diagonal makes collapsed boxes finite under AMP.
    """
    ca, cb = covariance(box_a, angle_a), covariance(box_b, angle_b)
    # Closed-form 2x2 inverse, elementwise FP32 even under autocast. No
    # cuSOLVER calls/syncs or autocast matmuls on nearly singular matrices.
    floor = 1e-6/12
    def regularized(c):
        x, y, z = c[...,0,0]+floor, c[...,0,1], c[...,1,1]+floor
        det = (x*z-y*y).clamp_min(floor*floor)
        return x, y, z, z/det, -y/det, x/det
    ax, ay, az, aix, aiy, aiz = regularized(ca)
    bx, by, bz, bix, biy, biz = regularized(cb)
    tr = aix*bx+2*aiy*by+aiz*bz+bix*ax+2*biy*ay+biz*az
    d = (box_a[..., :2]-box_b[..., :2]).float()
    dx,dy=d.unbind(-1)
    loc = (aix+bix)*dx.square()+2*(aiy+biy)*dx*dy+(aiz+biz)*dy.square()
    return ((tr+loc-4)*.25).clamp_min(0).log1p()


def sigmoid_focal(logits, targets, alpha=.25, gamma=2):
    logits = logits.float()
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = prob*targets+(1-prob)*(1-targets)
    return ce*(1-pt).pow(gamma)*(alpha*targets+(1-alpha)*(1-targets))


class HungarianOBBMatcher(nn.Module):
    def __init__(self, cost_class=2., cost_bbox=5., cost_angle=1., cost_kld=2.):
        super().__init__()
        self.cost_class, self.cost_bbox = cost_class, cost_bbox
        self.cost_angle, self.cost_kld = cost_angle, cost_kld

    @torch.no_grad()
    def forward(self, logits, boxes, angles, targets, img_size):
        indices = []
        for b, target in enumerate(targets):
            labels = target["labels"].to(logits.device)
            raw = target["boxes"].to(boxes.device).float()
            if len(labels) == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                indices.append((empty, empty)); continue
            p = logits[b].float().sigmoid().clamp(1e-7, 1-1e-7)
            # Focal positive minus negative assignment cost, not softmax.
            neg = .75*p.square()*(-(1-p).log())
            pos = .25*(1-p).square()*(-p.log())
            cls = (pos-neg)[:, labels]
            tgt = raw[:, :4]/float(img_size)
            ta = torch.deg2rad(raw[:, 4:5])
            l1 = torch.cdist(boxes[b].float(), tgt, p=1)
            kld = gaussian_kld(boxes[b,:,None].float(), angles[b,:,None].float(), tgt[None], ta[None])
            diff = 2*(angles[b].float()-ta.T)
            ang = 1-diff.cos()
            cost = self.cost_class*cls+self.cost_bbox*l1+self.cost_kld*kld+self.cost_angle*ang
            if not torch.isfinite(cost).all():
                import warnings
                warnings.warn("Non-finite Hungarian cost entries excluded; inspect data/model numerics", RuntimeWarning)
                cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
            qi, ti = linear_sum_assignment(cost.cpu().numpy())
            indices.append((torch.as_tensor(qi,device=logits.device), torch.as_tensor(ti,device=logits.device)))
        return indices


class DETROBBLoss(nn.Module):
    def __init__(self, num_classes, loss_class=2., loss_bbox=5., loss_angle=1., loss_kld=2.):
        super().__init__()
        self.num_classes = num_classes
        self.loss_class, self.loss_bbox = loss_class, loss_bbox
        self.loss_angle, self.loss_kld = loss_angle, loss_kld
        self.matcher = HungarianOBBMatcher()
        self.degraded_batches = 0
        self.consecutive_degraded = 0
        self.max_consecutive_degraded = 3

    def forward(self, pred_logits, pred_boxes, pred_angles, targets, img_size, aux_outputs=None, model_name="", _count_batch=True):
        # Keep the legacy K+1 physical head for checkpoint/momentum compatibility.
        # Its last row is unused; the K foreground channels are independent.
        tensors = [pred_logits, pred_boxes, pred_angles]
        if aux_outputs:
            tensors += [v for out in aux_outputs for v in out.values() if isinstance(v, torch.Tensor)]
        degraded = not all(bool(torch.isfinite(t).all()) for t in tensors)
        if _count_batch:
            if degraded:
                import warnings
                self.degraded_batches += 1
                self.consecutive_degraded += 1
                warnings.warn(f"Non-finite DETR main/aux predictions sanitized; degraded batches={self.degraded_batches}, consecutive={self.consecutive_degraded}", RuntimeWarning)
                if self.consecutive_degraded >= self.max_consecutive_degraded:
                    raise FloatingPointError("DETR diverged: three consecutive degraded batches; refusing false convergence")
            else:
                self.consecutive_degraded = 0
        logits = torch.nan_to_num(pred_logits[..., :self.num_classes].float(), nan=0., posinf=30., neginf=-30.)
        boxes = torch.nan_to_num(pred_boxes.float(), nan=.5, posinf=1., neginf=0.)
        angles = torch.nan_to_num(pred_angles.float(), nan=0., posinf=0., neginf=0.)
        indices = self.matcher(logits, boxes, angles, targets, img_size)
        onehot = torch.zeros_like(logits)
        num_matched = sum(len(q) for q,t in indices)
        denom = max(num_matched, 1)
        box_loss, angle_loss, kld_loss = boxes.sum()*0, angles.sum()*0, boxes.sum()*0
        matched = []
        for b, (qi, ti) in enumerate(indices):
            if not len(qi): continue
            labels = targets[b]["labels"].to(logits.device)[ti]
            onehot[b, qi, labels] = 1
            raw = targets[b]["boxes"].to(boxes.device).float()[ti]
            tgt, ta = raw[:, :4]/float(img_size), torch.deg2rad(raw[:, 4:5])
            box_loss = box_loss+F.l1_loss(boxes[b,qi], tgt, reduction="sum")/denom
            pa = angles[b,qi]
            angle_loss = angle_loss+(F.smooth_l1_loss((2*pa).sin(), (2*ta).sin(), reduction="sum")+
                         F.smooth_l1_loss((2*pa).cos(), (2*ta).cos(), reduction="sum"))/denom
            kld_loss = kld_loss+gaussian_kld(boxes[b,qi], pa, tgt, ta).sum()/denom
            matched.append(logits[b,qi,labels].sigmoid().detach())
        cls_loss = sigmoid_focal(logits, onehot).sum()/denom
        total = self.loss_class*cls_loss+self.loss_bbox*box_loss+self.loss_angle*angle_loss+self.loss_kld*kld_loss
        aux_loss = total*0
        if aux_outputs:
            from src.training.train_detr import extract_detr_predictions
            for aux in aux_outputs:
                al, ab, aa = extract_detr_predictions(aux, model_name)
                aux_loss = aux_loss+self(al, ab, aa, targets, img_size, _count_batch=False)["total_loss"]
            aux_loss = aux_loss/len(aux_outputs)
            total = total+aux_loss
        with torch.no_grad():
            bg_fraction = float((logits.sigmoid().amax(-1) < .05).float().mean())
            matched_fg_prob = float(torch.cat(matched).mean()) if matched else 0.
        return dict(total_loss=total, cls_loss=cls_loss.detach(), box_loss=box_loss.detach(),
                    angle_loss=angle_loss.detach(), kld_loss=kld_loss.detach(), aux_loss=aux_loss.detach(),
                    matched_queries=num_matched, bg_fraction=bg_fraction, matched_fg_prob=matched_fg_prob)
