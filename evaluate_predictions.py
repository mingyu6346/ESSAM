"""Evaluate binary prediction PNGs using explicitly specified GT/pred paths.

Specify PRED_ROOT with --pred_root_dir and GT_ROOT with --gt_root_dir.
Both paths are required; there are no built-in data paths.

Prediction directory layout (--pred_root_dir PRED_ROOT):
PRED_ROOT/
|-- ESSAM/                     <- method name
|   |-- Bijie/                 <- dataset name
|   |   |-- df010.png
|   |   `-- df015.png
|   `-- CLD/
|       `-- sample.png
`-- LightFormer/
    |-- Bijie/
    |   |-- df010.png
    |   `-- df015.png
    `-- CLD/
        `-- sample.png

GT layout A (--gt_root_dir GT_ROOT, no --gt_subdir):
GT_ROOT/
|-- Bijie/
|   |-- df010.png
|   `-- df015.png
`-- CLD/
    `-- sample.png

GT layout B (--gt_root_dir GT_ROOT --gt_subdir test/mask):
GT_ROOT/
|-- Bijie/
|   `-- test/
|       `-- mask/
|           |-- df010.png
|           `-- df015.png
`-- CLD/
    `-- test/
        `-- mask/
            `-- sample.png

Dataset and PNG filenames must match exactly between GT and predictions.
GT_ROOT is the parent of dataset folders, not the mask folder itself.
PRED_ROOT is the parent of method folders, not a single method folder.

Example with explicitly specified paths (replace these paths with your own):
python evaluate_predictions.py --pred_root_dir "D:/data/pred" --gt_root_dir "D:/data/gt" --save_dir "D:/results/evaluation"
For GT layout B, append: --gt_subdir test/mask
To select methods/datasets, append: --methods ESSAM --datasets Bijie CLD

Output: SAVE_DIR/metrics.csv; an existing SAVE_DIR is never overwritten.
Requires NumPy and Pillow. Metrics pool all pixels in each dataset;
mIoU is the mean of foreground/background IoU, not per-image IoU.
Prediction images are bilinearly resized to GT dimensions when needed.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


COUNT_COLUMNS = ("TP", "FP", "FN", "TN")
METRIC_COLUMNS = ("precision", "recall", "f1", "iou", "miou")


def png_files(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = {path.name: path for path in directory.iterdir()
             if path.is_file() and path.suffix.lower() == ".png"}
    if not files:
        raise FileNotFoundError(f"No prediction/GT PNGs directly under {directory}")
    return files


def binary_image(path, size=None, threshold=128):
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8) >= threshold


def scores(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    background_iou = tn / (tn + fp + fn) if tn + fp + fn else 0.0
    return precision, recall, f1, iou, (iou + background_iou) / 2


def run(cfg):
    pred_root, gt_root, output = Path(cfg.pred_root_dir), Path(cfg.gt_root_dir), Path(cfg.save_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    for root in (pred_root, gt_root):
        if not root.is_dir():
            raise FileNotFoundError(root)
    if not 1 <= cfg.threshold <= 255:
        raise ValueError("Threshold must be an integer from 1 to 255")
    gt_subdir = Path(cfg.gt_subdir)
    if gt_subdir.is_absolute() or gt_subdir.drive or ".." in gt_subdir.parts:
        raise ValueError("GT subdirectory must be a relative path inside each dataset")
    methods = cfg.methods or sorted(path.name for path in pred_root.iterdir() if path.is_dir())
    for method in methods:
        if method in (".", "..") or Path(method).name != method or not (pred_root / method).is_dir():
            raise ValueError(f"Unknown or invalid method: {method}")
    datasets = cfg.datasets or sorted({path.name for method in methods
                                      for path in (pred_root / method).iterdir() if path.is_dir()})
    for label, selected in (("method", methods), ("dataset", datasets)):
        if not selected or len(selected) != len(set(selected)):
            raise ValueError(f"Select at least one {label}, without duplicates")
        if any(name in (".", "..") or Path(name).name != name for name in selected):
            raise ValueError(f"Invalid {label} name")

    ground_truth = {dataset: png_files(gt_root / dataset / gt_subdir) for dataset in datasets}
    plan = []
    for method in methods:
        for dataset in datasets:
            predictions = png_files(pred_root / method / dataset)
            gt_files = ground_truth[dataset]
            if predictions.keys() != gt_files.keys():
                missing = sorted(gt_files.keys() - predictions.keys())[:5]
                extra = sorted(predictions.keys() - gt_files.keys())[:5]
                raise ValueError(f"{method}/{dataset}: missing={missing}, extra={extra}")
            plan.append((method, dataset, predictions, gt_files))
    print(f"Prediction root: {pred_root.resolve()}\nGT root: {gt_root.resolve()}\n"
          f"GT subdirectory: {cfg.gt_subdir or '(none)'}", flush=True)
    print(f"Preflight passed: {len(methods)} methods, {len(datasets)} datasets; threshold={cfg.threshold}", flush=True)

    rows = []
    for method, dataset, predictions, gt_files in plan:
        tp = fp = fn = tn = 0
        for name in sorted(gt_files):
            gt = binary_image(gt_files[name], threshold=cfg.threshold)
            pred = binary_image(predictions[name], size=(gt.shape[1], gt.shape[0]), threshold=cfg.threshold)
            gt_positive = int(np.count_nonzero(gt))
            pred_positive = int(np.count_nonzero(pred))
            image_tp = int(np.count_nonzero(pred & gt))
            image_fp = pred_positive - image_tp
            image_fn = gt_positive - image_tp
            tp += image_tp
            fp += image_fp
            fn += image_fn
            tn += gt.size - image_tp - image_fp - image_fn
        values = scores(tp, fp, fn, tn)
        rows.append({"method": method, "dataset": dataset, "image_count": len(gt_files),
                     **dict(zip(COUNT_COLUMNS, (tp, fp, fn, tn))), **dict(zip(METRIC_COLUMNS, values))})
        print(f"[{method}/{dataset}] images={len(gt_files)}; "
              + ", ".join(f"{column}={value:.8f}" for column, value in zip(METRIC_COLUMNS, values)), flush=True)

    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluated {sum(row['image_count'] for row in rows)} method-image pairs; "
          f"wrote {len(rows)} rows to {output / 'metrics.csv'}", flush=True)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_root_dir", required=True, help="Your prediction path: parent of METHOD/DATASET/*.png (required)")
    parser.add_argument("--gt_root_dir", required=True, help="Your GT path: parent of DATASET folders (required)")
    parser.add_argument("--gt_subdir", default="", help="Relative GT subfolder within each dataset, e.g. test/mask")
    parser.add_argument("--methods", nargs="+", help="Selected methods; default: all prediction methods")
    parser.add_argument("--datasets", nargs="+", help="Selected datasets; default: all prediction datasets")
    parser.add_argument("--threshold", type=int, default=128, help="Grayscale binarization threshold, 1-255 (default: 128)")
    parser.add_argument("--save_dir", default="evaluation_results", help="New output directory; existing directories are never overwritten")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
