"""
SatQuery AI — Native GeoChat QLoRA training launcher.

This replaces the previous generic AutoModelForCausalLM training loop with
GeoChat's own multimodal training implementation. The official GeoChat
training code uses its GeoChatLlamaForCausalLM architecture, multimodal
image preprocessing and PEFT LoRA path; using that native path avoids
silently training a text-only/incorrect model.

Workflow:
  1. Prepare BigEarthNet.txt:
       python -m satquery.backend.training.bigearthnet_prepare ...
  2. Run this launcher:
       python -m satquery.backend.training.lora_finetune ...
  3. Validate the resulting adapter:
       python -m satquery.backend.training.validate_adaptation ...

A 7B QLoRA run is GPU-intensive. The launcher therefore exposes a smoke
test mode for one/few batches and a full mode for a real run. It never
creates fake adapter weights or reports synthetic training as completion.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEOCHAT_ROOT = PROJECT_ROOT / "satquery" / "external" / "GeoChat"
TRAIN_SCRIPT = GEOCHAT_ROOT / "geochat" / "train" / "train_mem.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/bigearthnet/geochat_sft")
    p.add_argument("--image-folder", default=None)
    p.add_argument("--output-dir", default="models/geochat_lora_bigearthnet")
    p.add_argument("--base-model", default="MBZUAI/geochat-7B")
    p.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a tiny one-step training job to validate the stack.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)


def _build_command(args: argparse.Namespace) -> list[str]:
    data_dir = Path(args.data_dir).resolve()
    image_folder = Path(args.image_folder or data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    train_json = data_dir / "train.json"
    _require_file(train_json)
    _require_file(TRAIN_SCRIPT)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model_name_or_path", args.base_model,
        "--data_path", str(train_json),
        "--image_folder", str(image_folder),
        "--vision_tower", args.vision_tower,
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--bits", "4",
        "--double_quant", "True",
        "--quant_type", "nf4",
        "--lora_enable", "True",
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--lora_bias", "none",
        "--output_dir", str(output_dir),
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--evaluation_strategy", "no",
        "--save_strategy", "epoch",
        "--save_total_limit", "2",
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", "0.0",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--model_max_length", str(args.max_length),
        "--gradient_checkpointing", "True",
        "--lazy_preprocess", "True",
        "--dataloader_num_workers", str(args.num_workers),
        "--report_to", "none",
    ]

    if args.bf16:
        cmd += ["--bf16", "True"]
    else:
        cmd += ["--fp16", "True"]

    if args.smoke_test:
        cmd += [
            "--max_steps", "1",
            "--save_strategy", "no",
            "--logging_steps", "1",
        ]

    return cmd


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "dataset_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("samples_created", 0) < 1:
            raise RuntimeError("Prepared dataset contains zero usable samples.")

    command = _build_command(args)
    print("\n=== SatQuery GeoChat QLoRA ===")
    print("GeoChat root:", GEOCHAT_ROOT)
    print("Training script:", TRAIN_SCRIPT)
    print("Command:")
    print(" ".join(command))
    print()

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    result = subprocess.run(command, cwd=str(GEOCHAT_ROOT), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    adapter_config = output_dir / "adapter_config.json"
    adapter_weights = [output_dir / "adapter_model.safetensors",
                       output_dir / "adapter_model.bin"]
    if not adapter_config.exists() or not any(p.exists() for p in adapter_weights):
        raise RuntimeError(
            "Training process exited successfully but a complete PEFT adapter "
            "was not produced. Refusing to mark BigEarthNet adaptation complete."
        )

    print("✓ Complete GeoChat LoRA adapter produced:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
