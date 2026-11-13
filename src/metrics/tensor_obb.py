"""Exact convex rectangle intersection using device-local PyTorch operations.

Intersection vertices are contained corners plus edge crossings (the vectorized
convex polygon clipping construction). No rasterization, CPU copies or Shapely.
The topology masks are piecewise differentiable, as with polygon clipping.
"""
from __future__ import annotations
import torch


def corners(boxes):
    boxes = boxes.float() if boxes.dtype in (torch.float16, torch.bfloat16) else boxes
    signs = boxes.new_tensor([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    local = boxes[..., None, 2:4].clamp_min(0) * signs
    a = torch.deg2rad(boxes[..., 4])
    c, s = a.cos()[..., None], a.sin()[..., None]
    x, y = local.unbind(-1)
    return torch.stack((x*c-y*s, x*s+y*c), -1) + boxes[..., None, :2]


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def aligned_iou(a, b):
    """IoU for corresponding (..., 5) boxes, angles in degrees."""
    # Float64 clipping prevents cancellation on aerial slivers/high aspect
    # ratios. Output retains input dtype; CUDA latency must be measured.
    out_dtype = a.dtype if a.dtype not in (torch.float16, torch.bfloat16) else torch.float32
    a, b = a.double(), b.double()
    # Shift each pair near the origin to avoid large-coordinate shoelace loss.
    origin = (a[..., :2] + b[..., :2])*.5
    local_a = torch.cat((a[..., :2]-origin, a[..., 2:]), -1)
    local_b = torch.cat((b[..., :2]-origin, b[..., 2:]), -1)
    pa, pb = corners(local_a), corners(local_b)
    ea, eb = pa.roll(-1, -2)-pa, pb.roll(-1, -2)-pb
    # Coordinates are CCW in the algebraic plane (clockwise on image pixels).
    eps = torch.finfo(pa.dtype).eps
    da = pa[..., :, None, :]-pb[..., None, :, :]
    db = pb[..., :, None, :]-pa[..., None, :, :]
    tol_a = eps*8*(eb[..., None, :, :].norm(dim=-1)*da.norm(dim=-1)).clamp_min(1e-10)
    tol_b = eps*8*(ea[..., None, :, :].norm(dim=-1)*db.norm(dim=-1)).clamp_min(1e-10)
    ina = (cross(eb[..., None, :, :], da) >= -tol_a).all(-1)
    inb = (cross(ea[..., None, :, :], db) >= -tol_b).all(-1)
    r, s = ea[..., :, None, :], eb[..., None, :, :]
    delta = pb[..., None, :, :]-pa[..., :, None, :]
    den = cross(r, s)
    tol = eps*8*(r.norm(dim=-1)*s.norm(dim=-1)).clamp_min(1e-10)
    safe = torch.where(den.abs() > tol, den, torch.ones_like(den))
    t, u = cross(delta, s)/safe, cross(delta, r)/safe
    valid = (den.abs() > tol) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    pts = pa[..., :, None, :] + t[..., None]*r
    pts = torch.cat((pa, pb, pts.flatten(-3, -2)), -2)
    mask = torch.cat((ina, inb, valid.flatten(-2, -1)), -1)
    count = mask.sum(-1)
    center = (pts*mask[..., None]).sum(-2)/count.clamp_min(1)[..., None]
    rel = pts-center[..., None, :]
    angle = torch.atan2(rel[..., 1], rel[..., 0])
    angle = torch.where(mask, angle, torch.full_like(angle, 10))
    order = angle.argsort(dim=-1, stable=True)
    pts = pts.gather(-2, order[..., None].expand(*order.shape, 2))
    mask = mask.gather(-1, order)
    # Padding repeats the first vertex, preserving the closing polygon edge.
    pts = torch.where(mask[..., None], pts, pts[..., :1, :])
    area = cross(pts, pts.roll(-1, -2)).sum(-1).abs()*.5
    aa = a[..., 2:4].clamp_min(0).prod(-1)
    ab = b[..., 2:4].clamp_min(0).prod(-1)
    inter = torch.minimum(area, torch.minimum(aa, ab))
    inter = torch.where(count >= 3, inter, torch.zeros_like(inter))
    iou = (inter/(aa+ab-inter).clamp_min(torch.finfo(pa.dtype).tiny)).clamp(0, 1)
    identical = (a == b).all(-1) & (aa > 0)
    return torch.where(identical, torch.ones_like(iou), iou).to(out_dtype)


def pairwise_iou(a, b, chunk_size=16384):
    """Chunked exact NxM IoU; O(chunk_size) temporary polygon storage."""
    n, m = len(a), len(b)
    result = a.new_zeros((n*m,))
    ac, bc = corners(a), corners(b)
    amin, amax, bmin, bmax = ac.amin(-2), ac.amax(-2), bc.amin(-2), bc.amax(-2)
    for start in range(0, n*m, chunk_size):
        ids = torch.arange(start, min(start+chunk_size, n*m), device=a.device)
        ia, ib = ids//m, ids % m
        candidate = ((amax[ia] >= bmin[ib]).all(-1) & (bmax[ib] >= amin[ia]).all(-1))
        selected = ids[candidate]
        result[selected] = aligned_iou(a[selected//m], b[selected % m])
    return result.reshape(n, m)


def rotated_nms_tensor(boxes, scores, iou_thresh=.45, max_det=300):
    """Stable greedy NMS, exact overlap matrix computed once on the device."""
    order = scores.argsort(descending=True, stable=True)
    if len(order) == 0 or max_det <= 0:
        return order[:0]
    overlaps = pairwise_iou(boxes[order], boxes[order]) > iou_thresh
    suppressed = torch.zeros(len(order), dtype=torch.bool, device=boxes.device)
    keep = torch.zeros_like(suppressed)
    positions = torch.arange(len(order), device=boxes.device)
    # Fixed device loop avoids a bool(cuda_tensor) host sync for every box.
    for i in range(len(order)):
        keep[i] = ~suppressed[i]
        suppressed |= overlaps[i] & keep[i] & (positions > i)
    keep &= keep.long().cumsum(0) <= max_det
    return order[keep]


def nms_per_class_tensor(boxes, scores, labels, iou_thresh=.45, max_det=1000):
    kept = []
    for c in labels.unique():
        ids = torch.where(labels == c)[0]
        kept.append(ids[rotated_nms_tensor(boxes[ids], scores[ids], iou_thresh, max_det)])
    if not kept:
        return boxes[:0], scores[:0], labels[:0]
    ids = torch.cat(kept)
    ids = ids[scores[ids].argsort(descending=True, stable=True)][:max_det]
    return boxes[ids], scores[ids], labels[ids]
