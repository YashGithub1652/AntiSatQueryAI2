# ============================================================
# FILE:
# satquery/backend/training/test_adaptation_pipeline.py
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

from .bigearthnet_prepare import (
    prepare_dataset,
)


def create_fake_patch(
    root: Path,
    patch_id: str,
) -> None:

    patch = root / patch_id

    patch.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(42)

    for band in (
        "B02",
        "B03",
        "B04",
    ):

        data = (
            rng.random(
                (64, 64)
            )
            * 10000
        ).astype(
            np.uint16
        )

        tifffile.imwrite(
            patch
            / f"{patch_id}_{band}.tif",
            data,
        )


def test_prepare_pipeline() -> None:

    with TemporaryDirectory() as temp:

        root = Path(temp)

        s2_root = (
            root / "S2"
        )

        output = (
            root / "geochat_sft"
        )

        patch_id = "TEST_PATCH_001"

        create_fake_patch(
            s2_root,
            patch_id,
        )

        annotations = (
            root
            / "annotations.jsonl"
        )

        with annotations.open(
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    {
                        "patch_id": patch_id,
                        "input": (
                            "Describe the "
                            "land cover in "
                            "this scene."
                        ),
                        "output": (
                            "The scene contains "
                            "vegetation and "
                            "built-up areas."
                        ),
                        "split": "train",
                    }
                )
                + "\n"
            )

        summary = prepare_dataset(
            annotations_path=annotations,
            s2_root=s2_root,
            output_dir=output,
        )

        assert (
            summary["train_samples"]
            == 1
        )

        train_path = (
            output / "train.json"
        )

        assert train_path.exists()

        with train_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        assert len(data) == 1

        sample = data[0]

        assert (
            sample["image"]
            == "images/bigearthnet_00000000.jpg"
        )

        assert (
            sample["conversations"][0][
                "from"
            ]
            == "human"
        )

        assert (
            sample["conversations"][1][
                "from"
            ]
            == "gpt"
        )

        image_path = (
            output
            / sample["image"]
        )

        assert image_path.exists()


if __name__ == "__main__":
    test_prepare_pipeline()

    print(
        "[PASS] adaptation pipeline test"
    )