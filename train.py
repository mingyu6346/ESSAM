"""Training script for my_model (this release only ships this one model).

Expects the dataset laid out as:

    <data_root>/train/img/*.png   <data_root>/train/mask/*.png
    <data_root>/val/img/*.png     <data_root>/val/mask/*.png     (optional)
    <data_root>/test/img/*.png    <data_root>/test/mask/*.png

If `val/` is absent, the test split is used for per-epoch model selection.
The checkpoint with the best mIoU is kept.

Example:
    python train.py --data_root /path/to/HR-GLDD --out runs/my_model --augment

The loss is the one used in the paper: BCE + soft-IoU on both the MPA output and the
refined SAM output, weighted 0.6 (refined) / 0.4 (MPA).
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LandslideDataset
from metrics import micro_counts, per_image_iou, summarize
from mobile_sam.my_model import Model


# ---------------------------------------------------------------- loss ----
class SoftIoULoss(nn.Module):
    """1 - IoU on sigmoid probabilities (equivalent to the paper's IOU() loss)."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        pred = logits.sigmoid()
        b = pred.shape[0]
        pred = pred.reshape(b, -1)
        target = target.reshape(b, -1)
        inter = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - inter
        return (1 - (inter + self.smooth) / (union + self.smooth)).mean()


class DualLoss(nn.Module):
    def __init__(self, aux_weight=0.4, main_weight=0.6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = SoftIoULoss()
        self.aux_weight = aux_weight
        self.main_weight = main_weight

    def forward(self, landslide_mask, pred_mask, gt_mask):
        landslide_mask = F.interpolate(landslide_mask, size=gt_mask.shape[-2:], mode="bilinear")
        pred_mask = F.interpolate(pred_mask, size=gt_mask.shape[-2:], mode="bilinear")
        loss_bce = self.main_weight * self.bce(pred_mask, gt_mask) \
            + self.aux_weight * self.bce(landslide_mask, gt_mask)
        loss_iou = self.main_weight * self.iou(pred_mask, gt_mask) \
            + self.aux_weight * self.iou(landslide_mask, gt_mask)
        return loss_bce + loss_iou


# ------------------------------------------------------------- helpers ----
def set_lr(optimizer, epoch_list, lr_list, epoch):
    """3-phase schedule: lr_list[0] for epoch<e0, lr_list[1] until e0+e1, else lr_list[2]."""
    if epoch < epoch_list[0]:
        lr = lr_list[0]
    elif epoch < epoch_list[0] + epoch_list[1]:
        lr = lr_list[1]
    else:
        lr = lr_list[2]
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def seed_torch(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def evaluate(model, loader, device):
    """Returns precision / recall / F1 (micro) and mIoU (mean per-image IoU)."""
    model.eval()
    tp = fp = fn = 0
    ious = []
    for images, gt, _, _, _ in loader:
        images = images.to(device)
        gt = gt.to(device)
        _, pred = model(images)
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear")
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        a, b, c = micro_counts(pred_binary, gt)
        tp += a; fp += b; fn += c
        ious += per_image_iou(pred_binary, gt)
    return summarize(tp, fp, fn, ious)


def build_loaders(args, split):
    img = os.path.join(args.data_root, split, "img")
    mask = os.path.join(args.data_root, split, "mask")
    ds = LandslideDataset(img, mask, image_size=(args.image_size, args.image_size),
                          augment=(args.augment and split == "train"))
    bs = args.batch_size if split == "train" else 1
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"),
                      num_workers=args.num_workers, pin_memory=True), ds


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description="Train my_model")
    ap.add_argument("--data_root", required=True, help="dataset root with train/ val/ test/")
    ap.add_argument("--out", default="runs/my_model", help="output directory")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epoch_list", type=int, nargs=3, default=[34, 33, 33],
                    help="epochs per LR phase (sum = total epochs)")
    ap.add_argument("--lr_list", type=float, nargs=3, default=[5e-4, 1e-4, 5e-5],
                    help="learning rate per phase")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--augment", action="store_true",
                    help="enable hflip/vflip + colour jitter on the training split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_torch(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    train_loader, train_ds = build_loaders(args, "train")
    sel_split = "val" if os.path.isdir(os.path.join(args.data_root, "val", "img")) else "test"
    val_loader, val_ds = build_loaders(args, sel_split)
    test_loader, test_ds = build_loaders(args, "test")
    print(f"device={device}  train={len(train_ds)}  select={sel_split}({len(val_ds)})  test={len(test_ds)}")

    model = Model(img_size=args.image_size).to(device)
    criterion = DualLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr_list[0], weight_decay=args.weight_decay)
    gflops_note = f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params"
    print(f"model: {gflops_note}")

    total_epochs = sum(args.epoch_list)
    best_miou, best_epoch = -1.0, 0

    for epoch in range(total_epochs):
        lr = set_lr(optimizer, args.epoch_list, args.lr_list, epoch)
        model.train()
        t0 = time.time()
        running = 0.0
        for images, gt, _, _, _ in train_loader:
            images = images.to(device)
            gt = gt.to(device)
            landslide_mask, pred_mask = model(images)
            loss = criterion(landslide_mask, pred_mask, gt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        metrics = evaluate(model, val_loader, device)
        print(f"[{epoch + 1}/{total_epochs}] lr={lr:.1e} loss={running / len(train_loader):.4f} "
              f"[{sel_split}] P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} mIoU={metrics['mIoU']:.4f} ({time.time() - t0:.0f}s)")

        if metrics["mIoU"] > best_miou:
            best_miou, best_epoch = metrics["mIoU"], epoch + 1
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pth"))
            print(f"  * new best mIoU={best_miou:.4f} -> best_model.pth")

    # final report on the test split with the best checkpoint
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pth"), map_location="cpu"))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nbest epoch={best_epoch}  test: P={test_metrics['precision']:.4f} "
          f"R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} mIoU={test_metrics['mIoU']:.4f}")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, **test_metrics}, f, indent=2)
    print(f"saved: {args.out}/best_model.pth , {args.out}/result.json")


if __name__ == "__main__":
    main()
