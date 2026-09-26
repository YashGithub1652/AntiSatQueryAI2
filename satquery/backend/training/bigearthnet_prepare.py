# ============================================================
# FILE:
# satquery/backend/training/bigearthnet_prepare.py
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import tifffile
except ImportError:
    tifffile = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

PATCH_KEYS = (
    "patch_id",
    "patch",
    "id",
    "s2_name",
    "name",
)

QUESTION_KEYS = (
    "input",
    "question",
    "query",
    "prompt",
)

ANSWER_KEYS = (
    "output",
    "answer",
    "caption",
    "text",
    "target",
)

SPLIT_KEYS = (
    "split",
    "subset",
)


def first_value(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key not in row:
            continue

        value = row[key]

        if value is None:
            continue

        if isinstance(value, float) and math.isnan(value):
            continue

        value = str(value).strip()

        if value:
            return value

    return None


def normalize_patch_id(value: str) -> str:
    value = value.strip()

    if value.endswith(".SAFE"):
        value = value[:-5]

    return value


def deterministic_split(
    patch_id: str,
    val_fraction: float,
    seed: int,
) -> str:
    digest = hashlib.sha256(
        f"{seed}:{patch_id}".encode("utf-8")
    ).hexdigest()

    value = int(digest[:8], 16) / 0xFFFFFFFF

    return "val" if value < val_fraction else "train"


def read_annotations(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        if pd is None:
            raise RuntimeError(
                "pandas + pyarrow are required for parquet files."
            )

        df = pd.read_parquet(path)

        return df.to_dict(orient="records")

    if suffix == ".csv":
        if pd is None:
            raise RuntimeError(
                "pandas is required for CSV files."
            )

        df = pd.read_csv(path)

        return df.to_dict(orient="records")

    if suffix in {".json", ".jsonl"}:
        rows: List[Dict[str, Any]] = []

        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    rows.append(json.loads(line))

            return rows

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]

            return [data]

        raise ValueError(
            f"Unsupported JSON structure in {path}"
        )

    raise ValueError(
        f"Unsupported annotation format: {path.suffix}"
    )


# ------------------------------------------------------------
# Sentinel-2 discovery
# ------------------------------------------------------------

def build_patch_index(
    s2_root: Path,
) -> Dict[str, Path]:
    """
    Build:

        patch_id -> patch directory

    Supports layouts such as:

        S2/
            patch_001/
                *_B02.tif
                *_B03.tif
                *_B04.tif

    and nested layouts.
    """

    index: Dict[str, Path] = {}

    # Support Kaggle BigEarthNet-14k single multi-band TIFF patches.
    for tif in s2_root.rglob('*.tif'):
        if tif.is_file() and tif.stem.startswith(('S2A_', 'S2B_')):
            index[tif.stem] = tif
            index[tif.name] = tif

    if index:
        return index


    for directory in s2_root.rglob("*"):

        if not directory.is_dir():
            continue

        files = list(directory.glob("*"))

        band_names = [
            p.name.upper()
            for p in files
            if p.is_file()
        ]

        has_b02 = any("B02" in name for name in band_names)
        has_b03 = any("B03" in name for name in band_names)
        has_b04 = any("B04" in name for name in band_names)

        if not (has_b02 and has_b03 and has_b04):
            continue

        index[directory.name] = directory

    return index


def resolve_patch_directory(
    patch_id: str,
    index: Dict[str, Path],
) -> Optional[Path]:

    normalized = normalize_patch_id(patch_id)

    if normalized in index:
        return index[normalized]
    tif_key = f"{normalized}.tif"
    if tif_key in index:
        return index[tif_key]

    for key, path in index.items():

        if normalize_patch_id(key) == normalized:
            return path

    normalized_lower = normalized.lower()

    for key, path in index.items():

        key_lower = normalize_patch_id(key).lower()

        if normalized_lower in key_lower:
            return path

    return None


def find_band(
    patch_dir: Path,
    band: str,
) -> Optional[Path]:

    candidates = []

    for path in patch_dir.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.upper()

        if f"_{band.upper()}." in name:
            candidates.append(path)
        elif f"_{band.upper()}_" in name:
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            0 if p.suffix.lower() in {".tif", ".tiff"} else 1,
            len(p.name),
        )
    )

    return candidates[0]


# ------------------------------------------------------------
# Image conversion
# ------------------------------------------------------------

def read_raster(path: Path) -> np.ndarray:

    if tifffile is None:
        raise RuntimeError(
            "tifffile is required to read Sentinel-2 TIFF files."
        )

    array = tifffile.imread(str(path))

    array = np.asarray(array)

    while array.ndim > 2:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D raster for {path}, got {array.shape}"
        )

    return array.astype(np.float32)


def percentile_stretch(
    array: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    lo = np.percentile(array, low)
    hi = np.percentile(array, high)

    if hi <= lo:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = (
        (array - lo)
        / (hi - lo)
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return (
        normalized * 255.0
    ).astype(np.uint8)


def create_rgb_image(
    patch_dir: Path,
    output_path: Path,
) -> None:

    # Kaggle BigEarthNet-14k stores each patch as one 10-band TIFF.
    if patch_dir.is_file() and patch_dir.suffix.lower() == ".tif":
        import rasterio
        with rasterio.open(patch_dir) as src:
            if src.count < 4:
                raise FileNotFoundError(f"Insufficient bands in {patch_dir}")
            r = percentile_stretch(src.read(4))
            g = percentile_stretch(src.read(3))
            b = percentile_stretch(src.read(2))
        rgb = np.stack([r, g, b], axis=-1)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(output_path, quality=95)
        return

    red = find_band(patch_dir, "B04")
    green = find_band(patch_dir, "B03")
    blue = find_band(patch_dir, "B02")

    if not red or not green or not blue:
        raise FileNotFoundError(
            f"Missing RGB Sentinel-2 bands in {patch_dir}"
        )

    r = percentile_stretch(
        read_raster(red)
    )

    g = percentile_stretch(
        read_raster(green)
    )

    b = percentile_stretch(
        read_raster(blue)
    )

    shape = r.shape

    if g.shape != shape or b.shape != shape:
        raise ValueError(
            f"Band shape mismatch in {patch_dir}: "
            f"R={r.shape}, G={g.shape}, B={b.shape}"
        )

    rgb = np.stack(
        [r, g, b],
        axis=-1,
    )

    image = Image.fromarray(
        rgb,
        mode="RGB",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output_path,
        format="JPEG",
        quality=95,
    )


# ------------------------------------------------------------
# GeoChat conversion
# ------------------------------------------------------------

def make_geochat_sample(
    sample_id: str,
    image_relative_path: str,
    question: str,
    answer: str,
) -> Dict[str, Any]:

    return {
        "id": sample_id,
        "image": image_relative_path,
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\n"
                    f"{question}"
                ),
            },
            {
                "from": "gpt",
                "value": answer,
            },
        ],
    }


def normalize_question(question: str) -> str:

    question = question.strip()

    if not question:
        return (
            "Describe the remote sensing scene "
            "and identify the main land-cover characteristics."
        )

    return question


def normalize_answer(answer: str) -> str:

    answer = answer.strip()

    if not answer:
        return (
            "No valid annotation answer was provided."
        )

    return answer


# ------------------------------------------------------------
# Main preparation
# ------------------------------------------------------------

def prepare_dataset(
    annotations_path: Path,
    s2_root: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    max_annotations_per_patch: Optional[int] = None,
    overwrite_images: bool = False,
) -> Dict[str, Any]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_dir = output_dir / "images"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_annotations(
        annotations_path
    )

    patch_index = build_patch_index(
        s2_root
    )

    # Restrict processing to locally available S2 patches.
    available_patch_ids = {normalize_patch_id(k) for k in patch_index.keys()}
    rows = [
        r for r in rows
        if normalize_patch_id(str(first_value(r, PATCH_KEYS))) in available_patch_ids
    ]
    print(f'[INFO] Local S2 patches: {len(available_patch_ids)}')
    print(f'[INFO] Matched annotation rows: {len(rows)}')

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    manifest_path = (
        output_dir / "manifest.jsonl"
    )

    stats = {
        "annotations_total": len(rows),
        "annotations_processed": 0,
        "annotations_skipped": 0,
        "missing_patch": 0,
        "missing_question": 0,
        "missing_answer": 0,
        "image_errors": 0,
        "train": 0,
        "val": 0,
    }

    patch_annotation_counts: Dict[str, int] = {}

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as manifest:

        for row_index, row in enumerate(rows):

            if (
                max_samples is not None
                and stats["annotations_processed"]
                >= max_samples
            ):
                break

            patch_id = first_value(
                row,
                PATCH_KEYS,
            )

            question = first_value(
                row,
                QUESTION_KEYS,
            )

            answer = first_value(
                row,
                ANSWER_KEYS,
            )

            if not patch_id:
                stats["annotations_skipped"] += 1
                stats["missing_patch"] += 1
                continue

            if not question:
                stats["annotations_skipped"] += 1
                stats["missing_question"] += 1
                continue

            if not answer:
                stats["annotations_skipped"] += 1
                stats["missing_answer"] += 1
                continue

            patch_id = normalize_patch_id(
                patch_id
            )

            if (
                max_annotations_per_patch is not None
                and patch_annotation_counts.get(
                    patch_id,
                    0,
                )
                >= max_annotations_per_patch
            ):
                continue

            patch_dir = resolve_patch_directory(
                patch_id,
                patch_index,
            )

            if patch_dir is None:

                stats["annotations_skipped"] += 1
                stats["missing_patch"] += 1

                continue

            sample_id = (
                f"bigearthnet_{row_index:08d}"
            )

            image_filename = (
                f"{sample_id}.jpg"
            )

            image_path = (
                image_dir / image_filename
            )

            try:

                if (
                    not image_path.exists()
                    or overwrite_images
                ):
                    create_rgb_image(
                        patch_dir,
                        image_path,
                    )

            except Exception as exc:

                stats["annotations_skipped"] += 1
                stats["image_errors"] += 1

                print(
                    f"[WARN] image conversion failed "
                    f"for {patch_id}: {exc}"
                )

                continue

            split = first_value(
                row,
                SPLIT_KEYS,
            )

            if split:
                split_lower = split.lower()

                if "test" in split_lower:
                    split = "val"
                elif "valid" in split_lower:
                    split = "val"
                elif "train" in split_lower:
                    split = "train"
                else:
                    split = deterministic_split(
                        patch_id,
                        val_fraction,
                        seed,
                    )
            else:
                split = deterministic_split(
                    patch_id,
                    val_fraction,
                    seed,
                )

            sample = make_geochat_sample(
                sample_id=sample_id,
                image_relative_path=(
                    f"images/{image_filename}"
                ),
                question=normalize_question(
                    question
                ),
                answer=normalize_answer(
                    answer
                ),
            )

            metadata = {
                "sample_id": sample_id,
                "patch_id": patch_id,
                "split": split,
                "source_row": row_index,
            }

            manifest.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                )
                + "\n"
            )

            if split == "val":
                val_samples.append(sample)
            else:
                train_samples.append(sample)

            patch_annotation_counts[
                patch_id
            ] = (
                patch_annotation_counts.get(
                    patch_id,
                    0,
                )
                + 1
            )

            stats["annotations_processed"] += 1

            if split == "val":
                stats["val"] += 1
            else:
                stats["train"] += 1

    train_path = (
        output_dir / "train.json"
    )

    val_path = (
        output_dir / "val.json"
    )

    with train_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            train_samples,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with val_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            val_samples,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "dataset": "BigEarthNet.txt",
        "format": "GeoChat multimodal SFT",
        "annotations": str(
            annotations_path
        ),
        "sentinel2_root": str(
            s2_root
        ),
        "output_dir": str(
            output_dir
        ),
        "stats": stats,
        "train_samples": len(
            train_samples
        ),
        "validation_samples": len(
            val_samples
        ),
    }

    with (
        output_dir
        / "dataset_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    return summary


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare BigEarthNet.txt "
            "for GeoChat SFT."
        )
    )

    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--s2-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-annotations-per-patch",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite-images",
        action="store_true",
    )

    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError(
            "--val-fraction must be between 0 and 1."
        )

    summary = prepare_dataset(
        annotations_path=args.annotations,
        s2_root=args.s2_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_annotations_per_patch=(
            args.max_annotations_per_patch
        ),
        overwrite_images=args.overwrite_images,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

