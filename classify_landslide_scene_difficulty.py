"""Classify test RGB/GT pairs as regular or hard landslide segmentation scenes.

Requires the OpenAI Python package and DEEPSEEK_API_KEY in conda base.
Results are copied to output/<dataset>/<regular|hard>/<img|mask>; reruns resume.
"""

import argparse
import base64
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
DATASETS = ("Bijie", "CLD", "GVLM", "Landslide4sense", "HR-GLDD")
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = """你是遥感滑坡语义分割数据集的质量评估员。请判断：如果分割模型只看到 RGB 图像，准确分出 GT 中的滑坡区域，相对于本类数据集的典型场景是常规还是困难。

第一张图是 RGB，第二张图是对应的二值 GT（白色为滑坡，黑色为背景）。GT 仅用于定位目标和了解形状；判断难度时必须考虑 RGB 中目标与背景的实际可辨性，不能因为 GT 已给出就认为容易。

常规场景：滑坡在 RGB 中较清晰，与周围地物有可辨的颜色或纹理差异，边界大体可追踪，干扰较少。
困难场景：目标与裸土、道路、河床等背景相似；边界模糊或被植被、阴影、云遮挡；目标很小、细碎或分布复杂且难以在 RGB 中定位；或者 GT 为空但 RGB 有明显易误判的滑坡样干扰物。

根据最显著的视觉证据综合判断。不要仅凭滑坡面积、数量、图像文件名或数据集名称下结论。只返回一个 JSON 对象，不要 Markdown。label 必须是 "regular" 或 "hard"；reason 是一句简短中文理由，指出具体视觉依据。

返回格式示例（仅示范格式，实际内容必须依据当前图像）：
{"label":"regular","reason":"滑坡区域与周围植被对比清晰，边界较易辨认。"}
{"label":"hard","reason":"滑坡与邻近裸土颜色相近，局部边界难以区分。"}
每次只返回其中一个 JSON 对象。"""


def data_url(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_pairs(dataset):
    base = DATA_ROOT / dataset / "test"
    image_dir, mask_dir = base / "img", base / "mask"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing img or mask directory under {base}")
    images = {p.name: p for p in image_dir.glob("*.png")}
    masks = {p.name: p for p in mask_dir.glob("*.png")}
    if images.keys() != masks.keys():
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        raise ValueError(
            f"{dataset}: missing masks={missing_masks[:5]}, "
            f"missing images={missing_images[:5]}"
        )
    if not images:
        raise ValueError(f"No PNG pairs under {base}")
    return [(images[name], masks[name]) for name in sorted(images)]


def load_results(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return results


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def copy_pair(image_path, mask_path, dataset_dir, label):
    for source, kind in ((image_path, "img"), (mask_path, "mask")):
        target = dataset_dir / label / kind / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        other_label = "hard" if label == "regular" else "regular"
        (dataset_dir / other_label / kind / source.name).unlink(missing_ok=True)


def classify(client, image_path, mask_path, model, max_retries):
    rgb, gt = data_url(image_path), data_url(mask_path)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": "RGB 图像："},
                        {"type": "image_url", "image_url": {"url": rgb}},
                        {"type": "text", "text": "对应的二值 GT："},
                        {"type": "image_url", "image_url": {"url": gt}},
                    ]},
                ],
                temperature=0,
                max_tokens=160,
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("Truncated response")
            result = json.loads(choice.message.content)
            if (
                not isinstance(result, dict)
                or result.get("label") not in ("regular", "hard")
                or not isinstance(result.get("reason"), str)
                or not result["reason"].strip()
            ):
                raise ValueError(f"Invalid classification: {result!r}")
            return {"label": result["label"], "reason": result["reason"].strip()}
        except Exception as exc:
            if attempt + 1 == max_retries:
                raise RuntimeError(f"{image_path.name}: {exc}") from exc
            delay = min(2 ** attempt, 30)
            print(f"[retry] {image_path.name}: {exc}; wait {delay}s", flush=True)
            time.sleep(delay)


def process_dataset(client, dataset, args):
    pairs = image_pairs(dataset)
    dataset_dir = args.output_dir / dataset
    output_path = dataset_dir / "results.json"
    record_dir = dataset_dir / "records"
    results = load_results(output_path)
    for path in sorted(record_dir.glob("*.json")):
        results.update(load_results(path))
    pending = [pair for pair in pairs if args.overwrite or pair[0].stem not in results]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"[{dataset}] pairs={len(pairs)} existing={len(results)} pending={len(pending)}", flush=True)
    if args.dry_run:
        return
    for label in ("regular", "hard"):
        for kind in ("img", "mask"):
            (dataset_dir / label / kind).mkdir(parents=True, exist_ok=True)
    for rgb, gt in pairs:
        if rgb.stem in results:
            label = results[rgb.stem].get("label")
            if label not in ("regular", "hard"):
                raise ValueError(f"Invalid saved label for {dataset}/{rgb.name}: {label!r}")
            copy_pair(rgb, gt, dataset_dir, label)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify, client, rgb, gt, args.model, args.max_retries): (rgb, gt)
                   for rgb, gt in pending}
        for future in as_completed(futures):
            rgb, gt = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append(rgb.name)
                print(f"[failed] {dataset}/{rgb.name}: {exc}", flush=True)
                continue
            results[rgb.stem] = result
            save_results(record_dir / f"{rgb.stem}.json", {rgb.stem: result})
            copy_pair(rgb, gt, dataset_dir, result["label"])
            print(f"[{dataset}] {rgb.name}: {result['label']} — {result['reason']}", flush=True)

    save_results(output_path, dict(sorted(results.items())))
    print(f"[{dataset}] saved={len(results)} failed={len(failures)} -> {output_path}", flush=True)
    if failures:
        raise RuntimeError(f"{dataset}: {len(failures)} requests failed; rerun to retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=("all", *DATASETS), default=["HR-GLDD"])
    parser.add_argument("--model", default=MODEL, help="Vision-capable model; defaults to the reference script model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int, help="Maximum new pairs per dataset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Count pairs without API calls or writes")
    args = parser.parse_args()
    if args.workers < 1 or args.max_retries < 1 or args.timeout <= 0:
        parser.error("workers, max-retries and timeout must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")

    client = None
    if not args.dry_run:
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            parser.error("DEEPSEEK_API_KEY is not set")
        client = OpenAI(api_key=api_key, base_url=BASE_URL,
                        timeout=args.timeout, max_retries=0)
    if "all" in args.dataset and len(args.dataset) > 1:
        parser.error("'all' cannot be combined with individual dataset names")
    datasets = DATASETS if args.dataset == ["all"] else tuple(dict.fromkeys(args.dataset))
    for dataset in datasets:
        process_dataset(client, dataset, args)


if __name__ == "__main__":
    main()
