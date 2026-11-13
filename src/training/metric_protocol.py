"""Shared original-coordinate training-validation metric adapter."""
import numpy as np
from src.training.decode import to_metric_format


def original_metric_batch(decoded, targets, dataset, img_size, num_classes):
    from src.data.augment import boxes_to_corners, corners_to_boxes
    from src.data.dataset import polygon_to_obb_params
    preds = to_metric_format(decoded, num_classes)
    gt = []
    for pred, target in zip(preds, targets):
        ow, oh = target['orig_size']
        scale = img_size/float(max(ow,oh))
        px,py=(img_size-round(ow*scale))//2,(img_size-round(oh*scale))//2
        for entry in pred:
            boxes=entry['boxes']
            if dataset.resize_mode=='letterbox':
                boxes[:,0]=(boxes[:,0]-px)/scale
                boxes[:,1]=(boxes[:,1]-py)/scale
                boxes[:,2:4]/=scale
            else:
                # Legacy stretch requires transforming polygon corners, not
                # pretending anisotropic scaling preserves rotated angles.
                corners=boxes_to_corners(boxes)*np.array([ow/img_size,oh/img_size])
                entry['boxes']=np.asarray([polygon_to_obb_params(c) for c in corners],np.float32).reshape(-1,5)
        raw=dataset.raw.get_ground_truth(target['index'],ow,oh)
        gt.append([{'boxes':np.asarray(raw[c]['boxes'],np.float32).reshape(-1,5)} for c in range(num_classes)])
    return gt,preds
