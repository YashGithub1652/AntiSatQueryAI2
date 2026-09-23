"""
Validate a BigEarthNet GeoChat PEFT adapter before SatQuery can label the
primary VLM as RS-adapted.

Checks:
  * adapter_config.json exists
  * adapter weights exist and are non-trivial
  * PEFT metadata points to the expected LoRA task
  * optional adapter load against the configured base model
  * writes a machine-readable validation report

This script deliberately does not infer scientific quality from file size.
A valid adapter proves that training produced weights; benchmark evaluation
is still required to claim improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", default="models/geochat_lora_bigearthnet")
    p.add_argument("--base-model", default="MBZUAI/geochat-7B")
    p.add_argument("--load-test", action="store_true",
                   help="Actually attach the adapter to the base model. Requires GPU/PEFT deps.")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    adapter = Path(args.adapter_dir).resolve()
    config_path = adapter / "adapter_config.json"
    weight_paths = [adapter / "adapter_model.safetensors", adapter / "adapter_model.bin"]

    errors: list[str] = []
    warnings: list[str] = []

    if not config_path.exists():
        errors.append("adapter_config.json is missing")
        config = {}
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))

    weights = next((p for p in weight_paths if p.exists()), None)
    if weights is None:
        errors.append("adapter_model.safetensors or adapter_model.bin is missing")
    elif weights.stat().st_size < 1024 * 1024:
        errors.append(f"adapter weight file is suspiciously small: {weights.stat().st_size} bytes")

    if config:
        if config.get("peft_type") != "LORA":
            errors.append(f"Expected PEFT LoRA adapter, got {config.get('peft_type')!r}")
        target_modules = config.get("target_modules", [])
        if target_modules and not {"q_proj", "v_proj"}.issubset(set(target_modules)):
            warnings.append(f"Adapter target_modules={target_modules}; expected at least q_proj and v_proj")

    load_result = "not_requested"
    if args.load_test and not errors:
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for the GeoChat 7B load test.")

            q = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                quantization_config=q,
                device_map="auto",
                trust_remote_code=True,
            )
            PeftModel.from_pretrained(base, str(adapter))
            load_result = "passed"
            del base
        except Exception as exc:
            errors.append(f"adapter load test failed: {exc}")
            load_result = "failed"

    report = {
        "adapter_dir": str(adapter),
        "base_model": args.base_model,
        "adapter_present": weights is not None,
        "adapter_weight": str(weights) if weights else None,
        "load_test": load_result,
        "errors": errors,
        "warnings": warnings,
        "valid_artifact": not errors,
        "scientific_quality_claim": "requires benchmark evaluation",
    }

    output = Path(args.output) if args.output else adapter / "validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
