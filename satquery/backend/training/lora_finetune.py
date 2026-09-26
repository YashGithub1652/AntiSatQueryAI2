# ============================================================
# FILE:
# satquery/backend/training/lora_finetune.py
# ============================================================

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

GEOCHAT_ROOT = (
    REPO_ROOT
    / "satquery"
    / "external"
    / "GeoChat"
)

TRAIN_SCRIPT = (
    GEOCHAT_ROOT
    / "geochat"
    / "train"
    / "train_mem.py"
)


def build_command(args: argparse.Namespace) -> list[str]:

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"GeoChat training script not found:\n"
            f"{TRAIN_SCRIPT}"
        )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),

        "--model_name_or_path",
        args.base_model,

        "--version",
        args.version,

        "--data_path",
        str(
            args.data_dir
            / "train.json"
        ),

        "--image_folder",
        str(args.data_dir),

        "--vision_tower",
        args.vision_tower,

        "--mm_projector_type",
        "mlp2x_gelu",

        "--mm_vision_select_layer",
        "-2",

        "--mm_use_im_start_end",
        "False",

        "--mm_use_im_patch_token",
        "False",

        "--image_aspect_ratio",
        "pad",

        "--group_by_modality_length",
        "True",

        "--bf16",
        "True",

        "--output_dir",
        str(args.output_dir),

        "--num_train_epochs",
        str(args.epochs),

        "--per_device_train_batch_size",
        str(args.batch_size),

        "--per_device_eval_batch_size",
        "1",

        "--gradient_accumulation_steps",
        str(
            args.gradient_accumulation_steps
        ),

        "--evaluation_strategy",
        "no",

        "--save_strategy",
        "steps",

        "--save_steps",
        str(args.save_steps),

        "--save_total_limit",
        "2",

        "--learning_rate",
        str(args.learning_rate),

        "--weight_decay",
        "0.",

        "--warmup_ratio",
        "0.03",

        "--lr_scheduler_type",
        "cosine",

        "--logging_steps",
        "1",

        "--model_max_length",
        str(args.max_length),

        "--gradient_checkpointing",
        "True",

        "--dataloader_num_workers",
        str(args.workers),

        "--lazy_preprocess",
        "True",

        "--report_to",
        "none",

        "--lora_enable",
        "True",

        "--lora_r",
        str(args.lora_r),

        "--lora_alpha",
        str(args.lora_alpha),

        "--lora_dropout",
        str(args.lora_dropout),

        "--lora_bias",
        "none",

        "--bits",
        str(args.bits),

        "--double_quant",
        "True",

        "--quant_type",
        "nf4",
    ]

    if args.smoke_test:
        command.extend(
            [
                "--max_steps",
                "1",
            ]
        )

    return command


def verify_output(
    output_dir: Path,
) -> None:

    adapter_config = (
        output_dir
        / "adapter_config.json"
    )

    safetensors = (
        output_dir
        / "adapter_model.safetensors"
    )

    pytorch_bin = (
        output_dir
        / "adapter_model.bin"
    )

    if not adapter_config.exists():
        raise RuntimeError(
            "Training finished without "
            "adapter_config.json"
        )

    if (
        not safetensors.exists()
        and not pytorch_bin.exists()
    ):
        raise RuntimeError(
            "Training finished without "
            "adapter_model.safetensors "
            "or adapter_model.bin"
        )

    print(
        "\n"
        "[OK] Real LoRA adapter detected:"
    )

    print(
        f"     {output_dir}"
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "data"
            / "bigearthnet"
            / "geochat_sft"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "models"
            / "geochat_lora_bigearthnet"
        ),
    )

    parser.add_argument(
        "--base-model",
        default="MBZUAI/geochat-7B",
    )

    parser.add_argument(
        "--vision-tower",
        default=(
            "openai/"
            "clip-vit-large-patch14-336"
        ),
    )

    parser.add_argument(
        "--version",
        default="v1",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{args.data_dir}"
        )

    train_json = (
        args.data_dir / "train.json"
    )

    if not train_json.exists():
        raise FileNotFoundError(
            f"Missing:\n{train_json}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = build_command(args)

    print(
        "\n"
        "============================================================\n"
        "SATQUERY — GEoCHAT QLORA TRAINING\n"
        "============================================================\n"
    )

    print(
        " ".join(
            f'"{x}"' if " " in x else x
            for x in command
        )
    )

    if args.dry_run:
        return

    env = os.environ.copy()

    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = "expandable_segments:True"

    env[
        "TOKENIZERS_PARALLELISM"
    ] = "false"

    process = subprocess.run(
        command,
        cwd=str(GEOCHAT_ROOT),
        env=env,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "GeoChat training failed with "
            f"exit code {process.returncode}"
        )

    verify_output(
        args.output_dir
    )


if __name__ == "__main__":
    main()