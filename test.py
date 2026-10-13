"""Evaluate a trained my_model checkpoint.

Example:
    python test.py --data_root /path/to/HR-GLDD --ckpt runs/my_model/best_model.pth

Reports both micro scores (pixel-accumulated, what `train.py` prints) and
per-image averages (mIoU / mean F1), and can dump the binary prediction maps.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataset import LandslideDataset
from metrics import micro_counts, per_image_iou, summarize
from mobile_sam.my_model import Model


def main():
    ap = argparse.ArgumentParser(description="Test my_model")
    ap.add_argument("--data_root", required=True, help="dataset root (with <split>/img|mask)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", required=True, help="path to best_model.pth")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_maps", default=None, help="optional dir for binary prediction maps")
    ap.add_argument("--out", default=None, help="optional path to write metrics as json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LandslideDataset(os.path.join(args.data_root, args.split, "img"),
                          os.path.join(args.data_root, args.split, "mask"),
                          image_size=(args.image_size, args.image_size))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = Model(img_size=args.image_size).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    print(f"device={device}  loaded {args.ckpt}  ({sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params)")
    if args.save_maps:
        os.makedirs(args.save_maps, exist_ok=True)

    tp = fp = fn = 0
    ious = []
    with torch.inference_mode():
        for images, gt, name, w, h in loader:
            images = images.to(device)
            gt = gt.to(device)
            _, pred = model(images)
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear")
            pred_bin = (torch.sigmoid(pred) > 0.5).float()

            a, b, c = micro_counts(pred_bin, gt)
            tp += a; fp += b; fn += c
            ious += per_image_iou(pred_bin, gt)

            if args.save_maps:
                arr = (pred_bin[0, 0].cpu().numpy() * 255).astype("uint8")
                Image.fromarray(arr).resize((w.item(), h.item()), Image.NEAREST).save(
                    os.path.join(args.save_maps, os.path.splitext(name[0])[0] + ".png"))

    m = summarize(tp, fp, fn, ious)
    m["n"] = len(ds)

    print(f"[{args.split}] n={m['n']}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  mIoU={m['mIoU']:.4f}")
    if args.save_maps:
        print(f"saved prediction maps -> {args.save_maps}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"saved metrics -> {args.out}")


if __name__ == "__main__":
    main()
