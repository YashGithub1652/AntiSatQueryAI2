"""
SatQuery AI — Scientific Model Status
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelStatus:
    name: str
    available: bool
    scientific: bool
    checkpoint: Optional[str] = None
    message: str = ""


def evaluate_checkpoint(
    name: str,
    checkpoint: Optional[Path],
    *,
    required: bool = True,
) -> ModelStatus:

    if checkpoint is None:

        return ModelStatus(
            name=name,
            available=False,
            scientific=False,
            checkpoint=None,
            message="No checkpoint configured.",
        )

    if not checkpoint.exists():

        return ModelStatus(
            name=name,
            available=False,
            scientific=False,
            checkpoint=str(checkpoint),
            message="Checkpoint file does not exist.",
        )

    if checkpoint.stat().st_size < 1024:

        return ModelStatus(
            name=name,
            available=False,
            scientific=False,
            checkpoint=str(checkpoint),
            message="Checkpoint file is suspiciously small.",
        )

    return ModelStatus(
        name=name,
        available=True,
        scientific=True,
        checkpoint=str(checkpoint),
        message="Checkpoint file exists.",
    )