## An Efficient Self-prompting Segment Anything Model for Landslide Detection

## Requirements

```
pip install -r requirements.txt
```

## Training

The dataset is expected as plain image/mask folders:

```
<data_root>/train/img/*.png   <data_root>/train/mask/*.png
<data_root>/val/img/*.png     <data_root>/val/mask/*.png
<data_root>/test/img/*.png    <data_root>/test/mask/*.png
```

## Evaluation

```bash
python test.py --data_root /path/to/dataset --ckpt runs/model/best_model.pth \
               --save_maps preds --out metrics.json
```

## Results

The released prediction maps are available on Baidu Netdisk:

| Datasets                        | Download link                                                        | Extraction code |
| ------------------------------ | -------------------------------------------------------------------- | --------------- |
| `15 methods`                   | [Download](https://pan.baidu.com/s/15U2IipY5A81k-t6ENsB-kQ?pwd=gvee) | `gvee`          |
| `15 methods with regular/hard` | [Download](https://pan.baidu.com/s/1A1pRosbq4mOySuuYesuKuQ?pwd=u5f1) | `u5f1`          |

Download and extract the desired package to access the provided prediction maps.

## Licence and acknowledgements

Released under the **Apache License 2.0** (see `LICENSE`).

This release builds on:

* **[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)**

* **[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)**

* **[TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT)** 

