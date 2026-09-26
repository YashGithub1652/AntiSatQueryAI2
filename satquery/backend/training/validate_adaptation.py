# ============================================================
# FILE:
# satquery/backend/training/validate_adaptation.py
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def file_size(path: Path) -> int:

    try:
        return path.stat().st_size
    except OSError:
        return 0


def validate_adapter(
    adapter_dir: Path,
    load_test: bool = False,
) -> Dict[str, Any]:

    report: Dict[str, Any] = {
        "adapter_dir": str(
            adapter_dir
        ),
        "valid": False,
        "checks": {},
        "warnings": [],
    }

    config_path = (
        adapter_dir
        / "adapter_config.json"
    )

    safetensors_path = (
        adapter_dir
        / "adapter_model.safetensors"
    )

    pytorch_path = (
        adapter_dir
        / "adapter_model.bin"
    )

    report["checks"][
        "adapter_directory"
    ] = adapter_dir.exists()

    report["checks"][
        "adapter_config"
    ] = config_path.exists()

    weight_path = None

    if safetensors_path.exists():
        weight_path = safetensors_path

    elif pytorch_path.exists():
        weight_path = pytorch_path

    report["checks"][
        "adapter_weights"
    ] = weight_path is not None

    if weight_path:
        report["weight_file"] = str(
            weight_path
        )

        report["weight_size_bytes"] = (
            file_size(weight_path)
        )

        report["checks"][
            "non_empty_weights"
        ] = (
            file_size(weight_path) > 1024
        )

    if not config_path.exists():

        report["errors"] = [
            "adapter_config.json missing"
        ]

        return report

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    report["adapter_config"] = config

    report["checks"][
        "peft_type_is_lora"
    ] = (
        str(
            config.get(
                "peft_type",
                "",
            )
        ).upper()
        == "LORA"
    )

    target_modules = config.get(
        "target_modules"
    )

    if not target_modules:

        report["warnings"].append(
            "target_modules not present "
            "in adapter configuration."
        )

    required = [
        "adapter_directory",
        "adapter_config",
        "adapter_weights",
        "non_empty_weights",
        "peft_type_is_lora",
    ]

    structural_valid = all(
        report["checks"].get(
            key,
            False,
        )
        for key in required
    )

    report["checks"][
        "structurally_valid"
    ] = structural_valid

    if load_test and structural_valid:

        try:

            import torch
            from peft import PeftConfig, PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            peft_config = (
                PeftConfig.from_pretrained(
                    str(adapter_dir)
                )
            )

            base_model = (
                peft_config.base_model_name_or_path
            )

            report["load_test_base_model"] = (
                base_model
            )

            tokenizer = (
                AutoTokenizer.from_pretrained(
                    base_model,
                    trust_remote_code=True,
                )
            )

            model = (
                AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    torch_dtype=(
                        torch.float16
                        if torch.cuda.is_available()
                        else torch.float32
                    ),
                    trust_remote_code=True,
                )
            )

            model = PeftModel.from_pretrained(
                model,
                str(adapter_dir),
            )

            report["checks"][
                "load_test"
            ] = True

            del model
            del tokenizer

        except Exception as exc:

            report["checks"][
                "load_test"
            ] = False

            report["load_test_error"] = (
                repr(exc)
            )

    report["valid"] = (
        structural_valid
        and (
            not load_test
            or report["checks"].get(
                "load_test",
                False,
            )
        )
    )

    return report


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--load-test",
        action="store_true",
    )

    args = parser.parse_args()

    report = validate_adapter(
        args.adapter_dir,
        load_test=args.load_test,
    )

    output_path = (
        args.adapter_dir
        / "validation_report.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()