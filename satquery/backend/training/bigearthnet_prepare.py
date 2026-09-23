#!/usr/bin/env python3
"""
SatQuery AI — BigEarthNet.txt → GeoChat instruction dataset builder.

This prepares a REAL BigEarthNet.txt image/text subset for GeoChat's native
multimodal SFT format. It does not fabricate labels or synthetic images.

Supported text source:
  * BigEarthNet.txt.parquet (recommended)
  * JSONL with patch_id/input/output/split
  * CSV with the same fields

Image source:
  BigEarthNet v2 Sentinel-2 image root. The script resolves patch_id to an
  optical patch directory and creates RGB B04/B03/B02 JPEGs for GeoChat.

Output:
  output/
    images/*.jpg
    train.json
    val.json
    manifest.jsonl
    dataset_summary.json

Example:
  python -m satquery.backend.training.bigearthnet_prepare \
    --annotations data/bigearthnet/BigEarthNet.txt.parquet \
    --s2-root data/bigearthnet/S2 \
    --output-dir data/bigearthnet/geochat_sft \
    --max-samples 1000

The produced JSON follows the format expected by the embedded GeoChat
LazySupervisedDataset: {"image": "...", "conversations": [...]}.

BigEarthNet.txt is the primary adaptation source named by SIH PS 26167.
The dataset contains paired Sentinel-1/Sentinel-2 observations and rich
text annotations; this preparation path intentionally uses the Sentinel-2
optical image for the GeoChat visual-language adaptation stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger("bigearthnet_prepare")

QUESTIONS = (
    "Describe the land-cover and major objects visible in this Sentinel-2 image.",
    "What land-use or land-cover information can be identified from this remote-sensing image?",
    "Answer the following remote-sensing question using only the visual evidence: {input}",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True,
                   help="BigEarthNet.txt.parquet, JSONL, or CSV")
    p.add_argument("--s2-root", required=True,
                   help="Root containing BigEarthNet Sentinel-2 patch directories")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=0,
                   help="Maximum annotation rows; 0 = all rows")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=26167)
    p.add_argument("--max-annotations-per-patch", type=int, default=2)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--overwrite-images", action="store_true")
    return p.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas + pyarrow to read BigEarthNet.txt.parquet") from exc
        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")

    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, list) else obj.get("data", [])

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    raise ValueError(f"Unsupported annotation format: {path.suffix}")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _patch_id(row: dict[str, Any]) -> str:
    for key in ("patch_id", "patch", "id", "s2_name"):
        value = _text(row.get(key))
        if value:
            return value
    raise ValueError("Annotation row has no patch_id/patch/id/s2_name field")


def _answer(row: dict[str, Any]) -> str:
    for key in ("output", "answer", "caption", "text", "target"):
        value = _text(row.get(key))
        if value:
            return value
    raise ValueError("Annotation row has no output/answer/caption/text/target field")


def _question(row: dict[str, Any]) -> str:
    for key in ("input", "question", "query", "prompt"):
        value = _text(row.get(key))
        if value:
            return value
    return QUESTIONS[0]


def _split(row: dict[str, Any], patch_id: str) -> str:
    value = _text(row.get("split")).lower()
    if value in {"train", "val", "validation", "test"}:
        return "val" if value == "validation" else value
    digest = int(hashlib.sha256(patch_id.encode("utf-8")).hexdigest()[:8], 16)
    return "val" if digest % 100 < 5 else "train"


def _index_patch_dirs(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    # BigEarthNet v2 stores Sentinel-2 acquisitions in patch-named folders.
    for child in root.rglob("*"):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("S2") or name.startswith("patch"):
            index.setdefault(name, child)
    LOGGER.info("Indexed %d candidate Sentinel-2 patch directories", len(index))
    return index


def _find_band(patch_dir: Path, band: str) -> Path | None:
    candidates = sorted(patch_dir.glob(f"*{band}*.tif")) + sorted(patch_dir.glob(f"*{band}*.jp*g"))
    return candidates[0] if candidates else None


def _read_band(path: Path) -> np.ndarray:
    try:
        import tifffile
        arr = tifffile.imread(path)
    except Exception:
        from PIL import Image
        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr.squeeze()
    return arr.astype(np.float32)


def _stretch(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr[np.isfinite(arr)], [2, 98])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _make_rgb(patch_dir: Path) -> np.ndarray:
    b4, b3, b2 = (_find_band(patch_dir, x) for x in ("B04", "B03", "B02"))
    if not all((b4, b3, b2)):
        raise FileNotFoundError(f"Missing B04/B03/B02 in {patch_dir}")
    channels = [_stretch(_read_band(p)) for p in (b4, b3, b2)]
    shape = channels[0].shape
    if any(c.shape != shape for c in channels):
        raise ValueError(f"RGB band shapes do not match in {patch_dir}")
    return (np.stack(channels, axis=-1) * 255).round().astype(np.uint8)


def _save_rgb(rgb: np.ndarray, path: Path, quality: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="JPEG", quality=quality)


def _sample_id(patch_id: str, question: str) -> str:
    raw = f"{patch_id}|{question}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    annotations = Path(args.annotations).resolve()
    s2_root = Path(args.s2_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not annotations.exists():
        raise FileNotFoundError(annotations)
    if not s2_root.exists():
        raise FileNotFoundError(s2_root)

    rows = _read_rows(annotations)
    LOGGER.info("Loaded %d annotation rows", len(rows))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_samples:
        rows = rows[:args.max_samples]

    patch_index = _index_patch_dirs(s2_root)
    image_cache: dict[str, str] = {}
    per_patch_count: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for row in rows:
        try:
            patch_id = _patch_id(row)
            if per_patch_count.get(patch_id, 0) >= args.max_annotations_per_patch:
                continue

            patch_dir = patch_index.get(patch_id)
            if patch_dir is None:
                # tolerate directory names containing the exact patch id
                matches = [p for k, p in patch_index.items() if patch_id in k or k in patch_id]
                patch_dir = matches[0] if len(matches) == 1 else None
            if patch_dir is None:
                raise FileNotFoundError(f"Sentinel-2 patch not found for {patch_id}")

            if patch_id not in image_cache:
                rgb = _make_rgb(patch_dir)
                image_name = f"{patch_id}.jpg"
                image_path = out / "images" / image_name
                _save_rgb(rgb, image_path, args.jpeg_quality, args.overwrite_images)
                image_cache[patch_id] = f"images/{image_name}"

            question = _question(row)
            answer = _answer(row)
            if question.startswith("Answer the following"):
                question = question.format(input=question)

            samples.append({
                "id": _sample_id(patch_id, question),
                "image": image_cache[patch_id],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}",
                    },
                    {
                        "from": "gpt",
                        "value": answer,
                    },
                ],
                "metadata": {
                    "patch_id": patch_id,
                    "source": "BigEarthNet.txt",
                    "source_split": _split(row, patch_id),
                },
            })
            per_patch_count[patch_id] = per_patch_count.get(patch_id, 0) + 1

        except Exception as exc:
            skipped.append({"patch_id": _text(row.get("patch_id")), "reason": str(exc)})

    # Preserve dataset-provided split when available; otherwise deterministic split.
    train, val = [], []
    for sample in samples:
        split = sample["metadata"]["source_split"]
        if split == "test":
            continue
        (val if split == "val" else train).append(sample)

    # If the source has no usable val rows, deterministically hold out from train.
    if not val and train:
        holdout = max(1, int(len(train) * args.val_fraction))
        val = train[-holdout:]
        train = train[:-holdout]

    for sample in samples:
        sample.pop("metadata", None)

    out.joinpath("train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    out.joinpath("val.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    with out.joinpath("manifest.jsonl").open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "source_annotations": str(annotations),
        "source_image_root": str(s2_root),
        "seed": args.seed,
        "rows_seen": len(rows),
        "samples_created": len(samples),
        "train_samples": len(train),
        "val_samples": len(val),
        "unique_patches": len(image_cache),
        "skipped_rows": len(skipped),
        "skipped_examples": skipped[:100],
        "format": "GeoChat LazySupervisedDataset JSON",
    }
    out.joinpath("dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Prepared %d samples: train=%d val=%d skipped=%d", len(samples), len(train), len(val), len(skipped))


if __name__ == "__main__":
    main()
