"""Segmentation metrics for binary landslide masks.

Reported set: precision / recall / F1 (micro, accumulated over all pixels) and
mIoU (mean of the per-image IoU). The two IoU definitions differ a lot on landslide
data, because small landslides drag the per-image average down.

`score` is only used for checkpoint selection during training.
"""

import numpy as np
import torch


def micro_counts(pred_binary, gt):
    tp = (pred_binary * gt).sum().item()
    fp = (pred_binary * (1 - gt)).sum().item()
    fn = ((1 - pred_binary) * gt).sum().item()
    return tp, fp, fn


def per_image_iou(pred_binary, gt):
    """Per-sample IoU, averaged upstream into mIoU."""
    out = []
    for i in range(pred_binary.shape[0]):
        p, g = pred_binary[i, 0], gt[i, 0]
        tp = (p * g).sum().item()
        fp = (p * (1 - g)).sum().item()
        fn = ((1 - p) * g).sum().item()
        d = tp + fp + fn
        out.append(tp / d if d > 0 else 0.0)
    return out


def summarize(tp, fp, fn, ious):
    """Build the reported metric dict from accumulated counts and per-image IoUs.

    mIoU is also the criterion used for checkpoint selection in `train.py`.
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    miou = float(np.mean(ious)) if len(ious) else 0.0
    return dict(precision=precision, recall=recall, f1=f1, mIoU=miou)
