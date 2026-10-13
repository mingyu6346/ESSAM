"""Minimal, self-contained checkpoint loading (torch only).

Kept here so the released package does not depend on the training repo's
`utils.py` (which pulls in cv2 / fvcore / ptflops / thop / matplotlib).
The loading logic is identical to the original `load_param_2`.
"""

import torch


def load_param_2(pth_path, model, mode=None):
    """Load a checkpoint with key/shape filtering, then `strict=False`.

    Args:
        pth_path: path to the checkpoint file.
        model: the model to load into.
        mode: None            -> checkpoint is a plain state_dict
              "rgb"/"smt"     -> checkpoint is a dict under "state_dict"/"model"

    Returns:
        load ratio (percentage of checkpoint entries that were used).
    """
    if mode == "rgb":
        ckpt = torch.load(pth_path, weights_only=False)["state_dict"]
    elif mode == "smt":
        ckpt = torch.load(pth_path, weights_only=False)["model"]
    else:
        ckpt = torch.load(pth_path, weights_only=False)

    model_dict = model.state_dict()

    filtered_dict = {}
    removed = []

    for k, v in ckpt.items():
        if k not in model_dict:
            removed.append((k, "not found in model"))
            continue
        if model_dict[k].shape != v.shape:
            removed.append((
                k,
                f"shape mismatch: checkpoint {tuple(v.shape)} vs model {tuple(model_dict[k].shape)}"
            ))
            continue
        filtered_dict[k] = v

    model.load_state_dict(filtered_dict, strict=False)

    total = len(ckpt)
    loaded = len(filtered_dict)
    ratio = loaded / total * 100 if total > 0 else 100

    print("\n========== Parameter Loading Report ==========")
    print(f"Total checkpoint params : {total}")
    print(f"Loaded into model       : {loaded}")
    print(f"Skipped                 : {len(removed)}")
    print(f"Load ratio              : {ratio:.2f}%")
    print("==============================================\n")

    if removed:
        print("[Skipped Parameters]")
        for k, reason in removed:
            print(f"  - {k} ({reason})")
        print()

    return ratio
