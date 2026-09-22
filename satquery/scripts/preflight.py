"""
SatQuery AI — Project Preflight
===============================

Run from the project root:

    python -m satquery.scripts.preflight

This script checks:

1. Project structure
2. Core source files
3. Scientific model checkpoints
4. Python dependencies
5. CUDA availability
6. Model/checkpoint presence

IMPORTANT:
Checkpoint presence does NOT mean the model is scientifically
validated or benchmarked. This script only reports filesystem
and environment state.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# preflight.py:
#
# C:\SIH\satquery\scripts\preflight.py
#
# parents[0] = scripts
# parents[1] = satquery
# parents[2] = C:\SIH

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
SATQUERY_DIR = PROJECT_ROOT / "satquery"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# HELPERS
# ============================================================

def status_line(
    label: str,
    path: Path,
    exists: bool,
) -> None:

    if exists:
        print(
            f"[OK]   {label:<30}"
        )

    else:
        print(
            f"[MISS] {label:<30} {path}"
        )


def checkpoint_status(
    label: str,
    path: Path,
) -> bool:

    if path.exists() and path.is_file():

        size_mb = path.stat().st_size / (
            1024 * 1024
        )

        print(
            f"[OK]   {label:<30} "
            f"{size_mb:.2f} MB"
        )

        return True

    print(
        f"[MISS] {label:<30} {path}"
    )

    return False


def package_status(
    package_name: str,
) -> bool:

    available = (
        importlib.util.find_spec(
            package_name
        )
        is not None
    )

    if available:

        print(
            f"[OK]   {package_name}"
        )

    else:

        print(
            f"[MISS] {package_name}"
        )

    return available


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 72)
print(" SATQUERY AI — PREFLIGHT")
print("=" * 72)


# ============================================================
# PROJECT
# ============================================================

print()
print("PROJECT")
print("-" * 72)

print(
    f"Root: {PROJECT_ROOT}"
)

status_line(
    "configs/",
    CONFIG_DIR,
    CONFIG_DIR.is_dir(),
)

status_line(
    "models/",
    MODELS_DIR,
    MODELS_DIR.is_dir(),
)

status_line(
    "models/checkpoints/",
    CHECKPOINT_DIR,
    CHECKPOINT_DIR.is_dir(),
)

status_line(
    "satquery/data/",
    SATQUERY_DIR / "data",
    (SATQUERY_DIR / "data").is_dir(),
)

status_line(
    "results/",
    RESULTS_DIR,
    RESULTS_DIR.is_dir(),
)


# ============================================================
# CORE FILES
# ============================================================

print()
print("CORE FILES")
print("-" * 72)

core_files = [
    (
        "configs/model_registry.yaml",
        CONFIG_DIR
        / "model_registry.yaml",
    ),

    (
        "satquery/backend/core/agent.py",
        SATQUERY_DIR
        / "backend"
        / "core"
        / "agent.py",
    ),

    (
        "satquery/backend/core/confidence.py",
        SATQUERY_DIR
        / "backend"
        / "core"
        / "confidence.py",
    ),

    (
        "satquery/backend/geospatial/coregistration.py",
        SATQUERY_DIR
        / "backend"
        / "geospatial"
        / "coregistration.py",
    ),

    (
        "satquery/backend/models/model_loader.py",
        SATQUERY_DIR
        / "backend"
        / "models"
        / "model_loader.py",
    ),

    (
        "satquery/backend/models/sar_fusion_engine.py",
        SATQUERY_DIR
        / "backend"
        / "models"
        / "sar_fusion_engine.py",
    ),
]

for label, path in core_files:

    status_line(
        label,
        path,
        path.is_file(),
    )


# ============================================================
# SCIENTIFIC CHECKPOINTS
# ============================================================

print()
print("SCIENTIFIC CHECKPOINTS")
print("-" * 72)

checkpoint_results = {}


# ------------------------------------------------------------
# GeoChat LoRA
# ------------------------------------------------------------

geochat_dir = (
    MODELS_DIR
    / "geochat_lora_bigearthnet"
)

adapter_config = (
    geochat_dir
    / "adapter_config.json"
)

adapter_safe = (
    geochat_dir
    / "adapter_model.safetensors"
)

adapter_bin = (
    geochat_dir
    / "adapter_model.bin"
)

if (
    adapter_config.exists()
    and (
        adapter_safe.exists()
        or adapter_bin.exists()
    )
):

    # Report actual weight size.
    adapter_weights = (
        adapter_safe
        if adapter_safe.exists()
        else adapter_bin
    )

    size_mb = (
        adapter_weights.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"[OK]   {'GeoChat LoRA':<30} "
        f"{size_mb:.2f} MB"
    )

    checkpoint_results[
        "GeoChat LoRA"
    ] = True

else:

    print(
        f"[MISS] {'GeoChat LoRA':<30} "
        f"adapter weights not found"
    )

    checkpoint_results[
        "GeoChat LoRA"
    ] = False


# ------------------------------------------------------------
# ChangeFormer
# ------------------------------------------------------------

checkpoint_results[
    "ChangeFormer"
] = checkpoint_status(
    "ChangeFormer",
    MODELS_DIR
    / "ChangeFormer_LEVIR.pth",
)


# ------------------------------------------------------------
# RemoteCLIP
# ------------------------------------------------------------

checkpoint_results[
    "RemoteCLIP"
] = checkpoint_status(
    "RemoteCLIP",
    MODELS_DIR
    / "RemoteCLIP-ViT-B-32.pt",
)


# ------------------------------------------------------------
# RSVG
# ------------------------------------------------------------

checkpoint_results[
    "RSVG"
] = checkpoint_status(
    "RSVG",
    MODELS_DIR
    / "rsvg_best.pth",
)


# ------------------------------------------------------------
# SAM
# ------------------------------------------------------------

checkpoint_results[
    "SAM"
] = checkpoint_status(
    "SAM",
    MODELS_DIR
    / "sam_vit_b_01ec64.pth",
)


# ------------------------------------------------------------
# SAR-Optical
# ------------------------------------------------------------

sar_optical_checkpoint = (
    CHECKPOINT_DIR
    / "sar_optical_cross_attention_best.pth"
)

checkpoint_results[
    "SAR-Optical"
] = checkpoint_status(
    "SAR-Optical",
    sar_optical_checkpoint,
)


# ============================================================
# PYTHON PACKAGES
# ============================================================

print()
print("PYTHON PACKAGES")
print("-" * 72)

packages = [
    "numpy",
    "PIL",
    "yaml",
    "scipy",
    "skimage",
    "rasterio",
    "torch",
    "fastapi",
]

package_results = {}

for package in packages:

    package_results[
        package
    ] = package_status(
        package
    )


# ============================================================
# PYTHON / TORCH INFORMATION
# ============================================================

print()
print("RUNTIME")
print("-" * 72)

print(
    f"Python: {sys.version.split()[0]}"
)

try:

    import torch

    print(
        f"PyTorch: {torch.__version__}"
    )

    cuda_available = (
        torch.cuda.is_available()
    )

    if cuda_available:

        print(
            "[OK]   CUDA available"
        )

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        try:

            vram_gb = (
                torch.cuda.get_device_properties(0)
                .total_memory
                / (1024 ** 3)
            )

            print(
                f"VRAM: {vram_gb:.2f} GB"
            )

        except Exception:

            print(
                "VRAM: unavailable"
            )

    else:

        print(
            "[INFO] CUDA unavailable — CPU execution"
        )

except Exception as e:

    print(
        f"[MISS] PyTorch runtime check: {e}"
    )


# ============================================================
# CHECKPOINT SUMMARY
# ============================================================

present_count = sum(
    1
    for value in checkpoint_results.values()
    if value
)

total_count = len(
    checkpoint_results
)

print()
print(
    f"Checkpoint files present: "
    f"{present_count}/{total_count}"
)


# ============================================================
# SCIENTIFIC WARNING
# ============================================================

print()
print("IMPORTANT:")
print(
    "Missing scientific checkpoints must NOT be "
    "presented as achieved model capability."
)

print(
    "Checkpoint presence alone does NOT establish "
    "benchmark performance or scientific validation."
)

print(
    "REFERENCE_ONLY benchmark values must remain "
    "separate from locally achieved results."
)


# ============================================================
# CURRENT PROJECT ASSESSMENT
# ============================================================

print()
print("SCIENTIFIC READINESS SUMMARY")
print("-" * 72)

if checkpoint_results.get(
    "GeoChat LoRA",
    False,
):

    print(
        "[OK]   GeoChat LoRA weights present"
    )

else:

    print(
        "[MISS] GeoChat LoRA weights"
    )


if checkpoint_results.get(
    "ChangeFormer",
    False,
):

    print(
        "[OK]   ChangeFormer checkpoint present"
    )

else:

    print(
        "[MISS] ChangeFormer checkpoint"
    )


if checkpoint_results.get(
    "RemoteCLIP",
    False,
):

    print(
        "[OK]   RemoteCLIP checkpoint present"
    )

else:

    print(
        "[MISS] RemoteCLIP checkpoint"
    )


if checkpoint_results.get(
    "RSVG",
    False,
):

    print(
        "[OK]   RSVG checkpoint present"
    )

else:

    print(
        "[MISS] RSVG checkpoint"
    )


if checkpoint_results.get(
    "SAM",
    False,
):

    print(
        "[OK]   SAM checkpoint present"
    )

else:

    print(
        "[MISS] SAM checkpoint"
    )


if checkpoint_results.get(
    "SAR-Optical",
    False,
):

    print(
        "[OK]   SAR-Optical checkpoint file present"
    )

else:

    print(
        "[MISS] SAR-Optical checkpoint"
    )


print()
print("=" * 72)
print(" PREFLIGHT COMPLETE")
print("=" * 72)
print()