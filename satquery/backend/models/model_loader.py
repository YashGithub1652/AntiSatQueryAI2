"""
SatQuery AI — Centralized Scientific Model Loader
==================================================

Single source of truth for loading inference models.

Models:
    1. GeoChat-7B
    2. RemoteCLIP ViT-B/32
    3. ChangeFormer
    4. RSVG
    5. SAM

IMPORTANT SCIENTIFIC POLICY
---------------------------

This loader NEVER silently substitutes an unrelated model for a
scientific remote-sensing model.

Examples:

    GeoChat missing
        -> DO NOT silently replace with BLIP-1/BLIP-2

    ChangeFormer missing
        -> DO NOT use random weights

    RSVG missing
        -> DO NOT silently report GroundingDINO as RSVG

    RemoteCLIP checkpoint missing
        -> DO NOT silently claim RemoteCLIP while using OpenAI CLIP

A development fallback can be explicitly enabled through environment
variables, but fallback models are clearly marked as non-scientific
fallbacks.

Environment variables
---------------------

SATQUERY_ALLOW_NON_RS_VLM_FALLBACK=1
    Allows BLIP fallback for UI/development testing.

SATQUERY_ALLOW_BASE_CLIP_FALLBACK=1
    Allows OpenAI CLIP if RemoteCLIP checkpoint is unavailable.

SATQUERY_ALLOW_GROUNDING_DINO_FALLBACK=1
    Allows GroundingDINO if RSVG is unavailable.

By default ALL are disabled.

This means the production/scientific pipeline fails loudly instead of
returning scientifically misleading results.
"""

from __future__ import annotations

import logging
import importlib.util
import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.paths import (
    PROJECT_ROOT,
    MODELS_DIR,
    CHECKPOINT_DIR,
    model_path,
    checkpoint_path,
)

# ---------------------------------------------------------------------
# Torch
# ---------------------------------------------------------------------

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()

    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

    # 4-bit quantization is only useful when CUDA is available.
    USE_4BIT = CUDA_AVAILABLE

except ImportError:

    torch = None

    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

    DEVICE = "cpu"
    USE_4BIT = False


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

logger.info(
    "ModelLoader initialized: "
    f"device={DEVICE}, "
    f"cuda_available={CUDA_AVAILABLE}, "
    f"4bit={USE_4BIT}, "
    f"torch_available={TORCH_AVAILABLE}"
)


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

ALLOW_NON_RS_VLM_FALLBACK = (
    os.getenv(
        "SATQUERY_ALLOW_NON_RS_VLM_FALLBACK",
        "0",
    ).lower()
    in {"1", "true", "yes"}
)

ALLOW_BASE_CLIP_FALLBACK = (
    os.getenv(
        "SATQUERY_ALLOW_BASE_CLIP_FALLBACK",
        "0",
    ).lower()
    in {"1", "true", "yes"}
)

ALLOW_GROUNDING_DINO_FALLBACK = (
    os.getenv(
        "SATQUERY_ALLOW_GROUNDING_DINO_FALLBACK",
        "0",
    ).lower()
    in {"1", "true", "yes"}
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def _path_string(path: Path) -> str:
    """Return a normalized string representation of a Path."""
    return str(path.resolve())


def _file_exists(path: Path) -> bool:
    """Safe file existence check."""
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


def _directory_exists(path: Path) -> bool:
    """Safe directory existence check."""
    try:
        return path.exists() and path.is_dir()
    except Exception:
        return False


def _checkpoint_candidates(
    *names: str,
) -> list[Path]:
    """
    Build deterministic checkpoint search paths.

    Search order:
        1. models/<name>
        2. models/checkpoints/<name>
        3. ~/.cache/satquery/<name>
    """

    candidates: list[Path] = []

    cache_dir = (
        Path.home()
        / ".cache"
        / "satquery"
    )

    for name in names:

        candidates.append(
            model_path(name)
        )

        candidates.append(
            checkpoint_path(name)
        )

        candidates.append(
            cache_dir / name
        )

    # Remove duplicates while preserving order.
    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:

        key = _path_string(path)

        if key not in seen:

            seen.add(key)
            unique.append(path)

    return unique


def _find_checkpoint(
    *names: str,
) -> Optional[Path]:
    """
    Find the first existing checkpoint.
    """

    for path in _checkpoint_candidates(*names):

        if _file_exists(path):

            logger.info(
                "Checkpoint found: %s",
                _path_string(path),
            )

            return path

    return None


def _require_checkpoint(
    model_name: str,
    *names: str,
) -> Path:
    """
    Require a checkpoint.

    Raises FileNotFoundError instead of silently using random weights.
    """

    checkpoint = _find_checkpoint(*names)

    if checkpoint is not None:
        return checkpoint

    searched = "\n".join(
        f"  - {_path_string(path)}"
        for path in _checkpoint_candidates(*names)
    )

    raise FileNotFoundError(
        f"\n"
        f"{'=' * 72}\n"
        f"{model_name} CHECKPOINT NOT FOUND\n"
        f"{'=' * 72}\n"
        f"Searched:\n"
        f"{searched}\n\n"
        f"Scientific inference for {model_name} is BLOCKED.\n"
        f"Install a verified pretrained checkpoint before using this\n"
        f"capability for benchmark or SIH claims.\n"
        f"{'=' * 72}"
    )


def _validate_checkpoint_size(
    checkpoint: Path,
    minimum_mb: float = 1.0,
) -> None:
    """
    Basic sanity check.

    This does NOT prove that a checkpoint is valid.
    It only catches empty/corrupt-looking files.
    """

    size_bytes = checkpoint.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb < minimum_mb:

        raise RuntimeError(
            f"Checkpoint appears invalid or incomplete: "
            f"{checkpoint} "
            f"({size_mb:.3f} MB)"
        )


def _safe_state_dict(
    checkpoint_data: Any,
) -> Any:
    """
    Extract common checkpoint wrappers.

    Supports:
        state_dict
        model
        net
        params
    """

    if not isinstance(
        checkpoint_data,
        dict,
    ):
        return checkpoint_data

    for key in (
        "state_dict",
        "model_G_state_dict",
        "model",
        "net",
        "params",
    ):

        value = checkpoint_data.get(key)

        if isinstance(value, dict):

            return value

    return checkpoint_data


def _load_torch_checkpoint(
    checkpoint: Path,
) -> Any:
    """
    Load a PyTorch checkpoint safely.
    """

    if not TORCH_AVAILABLE:

        raise RuntimeError(
            "PyTorch is required to load model checkpoints."
        )

    _validate_checkpoint_size(
        checkpoint
    )

    logger.info(
        "Loading checkpoint: %s",
        _path_string(checkpoint),
    )

    return torch.load(
        checkpoint,
        weights_only=False,
        map_location=DEVICE,
    )


# ---------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------


class ModelLoader:
    """
    Singleton model registry.

    Models are loaded lazily and cached.

    Example:

        loader = get_model_loader()

        model, processor = loader.get_geochat()
    """

    def __init__(self):

        self._geochat_model = None
        self._geochat_processor = None

        self._remote_clip_model = None
        self._remote_clip_preprocess = None

        self._changeformer_model = None

        self._rsvg_model = None
        self._rsvg_tokenizer = None

        self._sam_predictor = None

        self._load_status: Dict[str, str] = {}

        self._checkpoint_paths: Dict[str, str] = {}

        self._scientific_flags: Dict[str, bool] = {}

        self._load_times: Dict[str, float] = {}

    # =================================================================
    # GEOCHAT
    # =================================================================

    def get_geochat(
        self,
    ) -> Tuple[Any, Any]:
        """
        Load GeoChat-7B.

        Primary:
            MBZUAI/geochat-7B

        Optional:
            BigEarthNet LoRA adapter.

        Returns:
            (model, processor)
        """

        if self._geochat_model is not None:

            return (
                self._geochat_model,
                self._geochat_processor,
            )

        if not TORCH_AVAILABLE:

            raise RuntimeError(
                "GeoChat requires PyTorch."
            )

        if DEVICE != "cuda":

            message = (
                "GeoChat-7B requires a GPU-backed environment "
                "for the intended SatQuery inference pipeline. "
                f"Current device: {DEVICE}"
            )

            self._load_status["geochat"] = (
                "BLOCKED: CUDA unavailable"
            )

            self._scientific_flags["geochat"] = False

            if not ALLOW_NON_RS_VLM_FALLBACK:

                raise RuntimeError(
                    message
                    + "\n"
                    "Install/use a CUDA environment or explicitly "
                    "enable SATQUERY_ALLOW_NON_RS_VLM_FALLBACK=1 "
                    "for development-only VLM fallback."
                )

        t0 = time.time()

        model_id = "MBZUAI/geochat-7B"

        logger.info(
            "Loading primary remote-sensing VLM: %s",
            model_id,
        )

        try:

            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
                BitsAndBytesConfig,
            )

            from peft import PeftModel

        except ImportError as exc:

            self._load_status["geochat"] = (
                f"FAILED: missing dependency: {exc}"
            )

            raise RuntimeError(
                "GeoChat requires transformers and peft.\n"
                "Install with:\n"
                "pip install transformers peft accelerate bitsandbytes"
            ) from exc

        try:

            quant_config = None

            if USE_4BIT:

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "device_map": "auto",
            }

            if quant_config is not None:

                model_kwargs[
                    "quantization_config"
                ] = quant_config

                model_kwargs[
                    "torch_dtype"
                ] = torch.float16

            elif DEVICE == "cuda":

                model_kwargs[
                    "torch_dtype"
                ] = torch.float16

            else:

                model_kwargs[
                    "torch_dtype"
                ] = torch.float32

                model_kwargs.pop(
                    "device_map",
                    None,
                )

            model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_id,
                    **model_kwargs,
                )
            )

            # ---------------------------------------------------------
            # Optional BigEarthNet LoRA
            # ---------------------------------------------------------

            lora_path = model_path(
                "geochat_lora_bigearthnet"
            )

            adapter_config = (
                lora_path
                / "adapter_config.json"
            )

            adapter_weights = [
                lora_path
                / "adapter_model.safetensors",
                lora_path
                / "adapter_model.bin",
            ]

            has_adapter_config = (
                adapter_config.exists()
            )

            has_adapter_weights = any(
                path.exists()
                for path in adapter_weights
            )

            if (
                has_adapter_config
                and has_adapter_weights
            ):

                logger.info(
                    "Loading BigEarthNet LoRA adapter: %s",
                    lora_path,
                )

                model = (
                    PeftModel
                    .from_pretrained(
                        model,
                        str(lora_path),
                    )
                )

                self._load_status[
                    "geochat"
                ] = (
                    "GeoChat-7B + "
                    "BigEarthNet-LoRA"
                )

                self._scientific_flags[
                    "geochat"
                ] = True

            elif has_adapter_config:

                logger.warning(
                    "GeoChat LoRA adapter_config.json exists "
                    "but adapter weights are missing."
                )

                self._load_status[
                    "geochat"
                ] = (
                    "GeoChat-7B base; "
                    "BigEarthNet-LoRA incomplete"
                )

                self._scientific_flags[
                    "geochat"
                ] = True

            else:

                logger.warning(
                    "BigEarthNet LoRA adapter not found. "
                    "Using base GeoChat-7B."
                )

                self._load_status[
                    "geochat"
                ] = (
                    "GeoChat-7B base; "
                    "no BigEarthNet-LoRA"
                )

                self._scientific_flags[
                    "geochat"
                ] = True

            processor = (
                AutoProcessor
                .from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
            )

            model.eval()

            self._geochat_model = model
            self._geochat_processor = processor

            elapsed = time.time() - t0

            self._load_times[
                "geochat"
            ] = elapsed

            logger.info(
                "GeoChat loaded in %.2fs",
                elapsed,
            )

            return (
                self._geochat_model,
                self._geochat_processor,
            )

        except Exception as exc:

            logger.exception(
                "GeoChat loading failed."
            )

            self._load_status[
                "geochat"
            ] = f"FAILED: {exc}"

            self._scientific_flags[
                "geochat"
            ] = False

            # ---------------------------------------------------------
            # Development-only fallback
            # ---------------------------------------------------------

            if ALLOW_NON_RS_VLM_FALLBACK:

                logger.warning(
                    "Development fallback explicitly enabled. "
                    "Attempting BLIP-2."
                )

                try:

                    from transformers import (
                        Blip2Processor,
                        Blip2ForConditionalGeneration,
                    )

                    fallback_id = (
                        "Salesforce/"
                        "blip2-opt-2.7b"
                    )

                    processor = (
                        Blip2Processor
                        .from_pretrained(
                            fallback_id
                        )
                    )

                    fallback_kwargs = {
                        "torch_dtype": (
                            torch.float16
                            if DEVICE == "cuda"
                            else torch.float32
                        )
                    }

                    if DEVICE == "cuda":

                        fallback_kwargs[
                            "device_map"
                        ] = "auto"

                    model = (
                        Blip2ForConditionalGeneration
                        .from_pretrained(
                            fallback_id,
                            **fallback_kwargs,
                        )
                    )

                    if DEVICE == "cpu":

                        model = model.to("cpu")

                    model.eval()

                    self._geochat_model = model
                    self._geochat_processor = processor

                    self._load_status[
                        "geochat"
                    ] = (
                        "DEVELOPMENT FALLBACK: "
                        "BLIP-2; NOT remote-sensing calibrated"
                    )

                    self._scientific_flags[
                        "geochat"
                    ] = False

                    return (
                        self._geochat_model,
                        self._geochat_processor,
                    )

                except Exception as fallback_exc:

                    logger.exception(
                        "BLIP-2 fallback failed."
                    )

                    raise RuntimeError(
                        "GeoChat failed and the explicitly enabled "
                        "development fallback also failed.\n"
                        f"GeoChat error: {exc}\n"
                        f"Fallback error: {fallback_exc}"
                    ) from fallback_exc

            raise RuntimeError(
                "GeoChat-7B could not be loaded.\n"
                f"Reason: {exc}\n\n"
                "Scientific VQA/captioning is BLOCKED."
            ) from exc

    # =================================================================
    # REMOTECLIP
    # =================================================================

    def get_remote_clip(
        self,
    ) -> Tuple[Any, Any]:
        """
        Load RemoteCLIP ViT-B/32.

        The OpenAI CLIP weights are NOT considered equivalent to
        RemoteCLIP.

        Therefore the RS checkpoint is required unless the explicit
        development fallback is enabled.
        """

        if self._remote_clip_model is not None:

            return (
                self._remote_clip_model,
                self._remote_clip_preprocess,
            )

        try:

            import open_clip

        except ImportError as exc:

            self._load_status[
                "remote_clip"
            ] = (
                f"FAILED: missing open_clip: {exc}"
            )

            raise RuntimeError(
                "RemoteCLIP requires open_clip_torch.\n"
                "Install:\n"
                "pip install open_clip_torch"
            ) from exc

        t0 = time.time()

        logger.info(
            "Loading RemoteCLIP ViT-B/32..."
        )

        try:

            model, _, preprocess = (
                open_clip.create_model_and_transforms(
                    "ViT-B-32",
                    pretrained="openai",
                )
            )

            checkpoint = _find_checkpoint(
                "RemoteCLIP-ViT-B-32.pt",
                "RemoteCLIP-ViT-B-32.pth",
            )

            # ---------------------------------------------------------
            # REAL RemoteCLIP
            # ---------------------------------------------------------

            if checkpoint is not None:

                state_dict = (
                    _load_torch_checkpoint(
                        checkpoint
                    )
                )

                state_dict = _safe_state_dict(
                    state_dict
                )

                missing, unexpected = (
                    model.load_state_dict(
                        state_dict,
                        strict=False,
                    )
                )

                logger.info(
                    "RemoteCLIP checkpoint loaded. "
                    "missing=%d unexpected=%d",
                    len(missing),
                    len(unexpected),
                )

                self._remote_clip_model = (
                    model.to(DEVICE)
                )

                self._remote_clip_preprocess = (
                    preprocess
                )

                self._checkpoint_paths[
                    "remote_clip"
                ] = _path_string(
                    checkpoint
                )

                self._load_status[
                    "remote_clip"
                ] = (
                    "RemoteCLIP-ViT-B/32 "
                    "RS-adapted checkpoint"
                )

                self._scientific_flags[
                    "remote_clip"
                ] = True

            # ---------------------------------------------------------
            # Explicit development fallback
            # ---------------------------------------------------------

            elif ALLOW_BASE_CLIP_FALLBACK:

                logger.warning(
                    "RemoteCLIP checkpoint missing. "
                    "Explicit base CLIP fallback enabled."
                )

                self._remote_clip_model = (
                    model.to(DEVICE)
                )

                self._remote_clip_preprocess = (
                    preprocess
                )

                self._load_status[
                    "remote_clip"
                ] = (
                    "DEVELOPMENT FALLBACK: "
                    "OpenAI CLIP; NOT RemoteCLIP"
                )

                self._scientific_flags[
                    "remote_clip"
                ] = False

            # ---------------------------------------------------------
            # Scientific mode: BLOCK
            # ---------------------------------------------------------

            else:

                self._load_status[
                    "remote_clip"
                ] = (
                    "BLOCKED: RemoteCLIP checkpoint missing"
                )

                self._scientific_flags[
                    "remote_clip"
                ] = False

                raise FileNotFoundError(
                    "RemoteCLIP checkpoint is missing.\n"
                    "Expected one of:\n"
                    "  models/RemoteCLIP-ViT-B-32.pt\n"
                    "  models/RemoteCLIP-ViT-B-32.pth\n"
                    "  models/checkpoints/RemoteCLIP-ViT-B-32.pt\n\n"
                    "Scientific RemoteCLIP inference is blocked."
                )

            model = self._remote_clip_model
            model.eval()

            tokenizer = (
                open_clip.get_tokenizer(
                    "ViT-B-32"
                )
            )

            self._remote_clip_model = (
                model,
                tokenizer,
            )

            elapsed = time.time() - t0

            self._load_times[
                "remote_clip"
            ] = elapsed

            logger.info(
                "RemoteCLIP loader completed in %.2fs",
                elapsed,
            )

            return (
                self._remote_clip_model,
                self._remote_clip_preprocess,
            )

        except Exception as exc:

            logger.exception(
                "RemoteCLIP loading failed."
            )

            self._load_status[
                "remote_clip"
            ] = f"FAILED: {exc}"

            self._scientific_flags[
                "remote_clip"
            ] = False

            raise RuntimeError(
                f"RemoteCLIP could not be loaded: {exc}"
            ) from exc

    # =================================================================
    # CHANGEFORMER
    # =================================================================

    def get_changeformer(
        self,
    ) -> Any:
        """
        Load ChangeFormer with a verified pretrained checkpoint.

        IMPORTANT:
        No random-weight fallback is allowed.
        """

        if self._changeformer_model is not None:

            return self._changeformer_model

        if not TORCH_AVAILABLE:

            raise RuntimeError(
                "ChangeFormer requires PyTorch."
            )

        t0 = time.time()

        logger.info(
            "Loading ChangeFormer..."
        )

        # -------------------------------------------------------------
        # Import official / locally integrated ChangeFormer
        # -------------------------------------------------------------

        ChangeFormerClass = None
        import_errors = []

        try:
            cf_root = PROJECT_ROOT / "external" / "ChangeFormer"
            cf_models = cf_root / "models"

            if not (cf_models / "ChangeFormer.py").is_file():
                raise FileNotFoundError(f"Official ChangeFormer package not found: {cf_models}")

            old_models = sys.modules.pop("models", None)
            old_models_changeformer = sys.modules.pop("models.ChangeFormer", None)
            old_models_base = sys.modules.pop("models.ChangeFormerBaseNetworks", None)
            old_models_networks = sys.modules.pop("models.networks", None)

            sys.path.insert(0, str(cf_root))
            try:
                import importlib
                cf_module = importlib.import_module("models.ChangeFormer")
                ChangeFormerClass = getattr(cf_module, "ChangeFormerV6")
            finally:
                if str(cf_root) in sys.path:
                    sys.path.remove(str(cf_root))

            logger.info("Official ChangeFormerV6 loaded from %s", cf_module.__file__)
        except Exception as exc:
            import_errors.append(f"official ChangeFormerV6: {exc}")
        # -------------------------------------------------------------
        # Checkpoint
        # -------------------------------------------------------------

        checkpoint = _find_checkpoint(
            "ChangeFormer_LEVIR.pth",
            "ChangeFormerV6_LEVIR.pth",
            "ChangeFormerV6.pth",
            "ChangeFormer_LEVIR_CD.pth",
        )

        if checkpoint is None:

            self._load_status[
                "changeformer"
            ] = (
                "BLOCKED: pretrained checkpoint missing"
            )

            self._scientific_flags[
                "changeformer"
            ] = False

            raise FileNotFoundError(
                "ChangeFormer pretrained checkpoint is missing.\n"
                "Scientific bi-temporal change detection is BLOCKED.\n\n"
                "Expected one of:\n"
                "  models/ChangeFormer_LEVIR.pth\n"
                "  models/ChangeFormerV6_LEVIR.pth\n"
                "  models/ChangeFormerV6.pth\n"
            )

        # -------------------------------------------------------------
        # Instantiate
        # -------------------------------------------------------------

        try:

            model = ChangeFormerClass()

        except TypeError as exc:

            raise RuntimeError(
                "ChangeFormer class was imported but its constructor "
                "requires additional configuration.\n"
                f"Constructor error: {exc}\n\n"
                "Adapt get_changeformer() to the exact upstream "
                "ChangeFormer configuration used by your checkpoint."
            ) from exc

        # -------------------------------------------------------------
        # Load checkpoint
        # -------------------------------------------------------------

        state_dict = (
            _load_torch_checkpoint(
                checkpoint
            )
        )

        state_dict = _safe_state_dict(
            state_dict
        )

        missing, unexpected = (
            model.load_state_dict(
                state_dict,
                strict=False,
            )
        )

        logger.info(
            "ChangeFormer checkpoint loaded: "
            "missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )

        # Do not hide severe incompatibility.
        if len(missing) > 50:

            raise RuntimeError(
                "ChangeFormer checkpoint appears incompatible "
                "with the loaded model architecture.\n"
                f"Missing keys: {len(missing)}\n"
                f"Unexpected keys: {len(unexpected)}"
            )

        model = model.to(DEVICE)
        model.eval()

        self._changeformer_model = model

        self._checkpoint_paths[
            "changeformer"
        ] = _path_string(
            checkpoint
        )

        self._load_status[
            "changeformer"
        ] = (
            "ChangeFormer pretrained checkpoint loaded"
        )

        self._scientific_flags[
            "changeformer"
        ] = True

        elapsed = time.time() - t0

        self._load_times[
            "changeformer"
        ] = elapsed

        logger.info(
            "ChangeFormer loaded in %.2fs",
            elapsed,
        )

        return self._changeformer_model

    # =================================================================
    # RSVG
    # =================================================================

    def get_rsvg(
        self,
    ) -> Tuple[Any, Any]:
        """
        Load RSVG visual grounding model.

        GroundingDINO fallback is available only when explicitly
        enabled and is always marked non-RSVG/non-scientific.
        """

        if self._rsvg_model is not None:

            return (
                self._rsvg_model,
                self._rsvg_tokenizer,
            )

        if not TORCH_AVAILABLE:

            raise RuntimeError(
                "RSVG requires PyTorch."
            )

        t0 = time.time()

        logger.info(
            "Loading remote-sensing grounding model..."
        )

        # -------------------------------------------------------------
        # Try RSVG
        # -------------------------------------------------------------

        try:

            from models.RSVG import (
                build_model as build_rsvg,
            )

            from transformers import (
                BertTokenizer,
            )

            model = build_rsvg()

            checkpoint = _find_checkpoint(
                "rsvg_best.pth",
                "RSVG_best.pth",
                "rsvg_vrsbench.pth",
            )

            if checkpoint is None:

                raise FileNotFoundError(
                    "RSVG checkpoint missing."
                )

            state_dict = (
                _load_torch_checkpoint(
                    checkpoint
                )
            )

            state_dict = _safe_state_dict(
                state_dict
            )

            missing, unexpected = (
                model.load_state_dict(
                    state_dict,
                    strict=False,
                )
            )

            logger.info(
                "RSVG checkpoint loaded: "
                "missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )

            tokenizer = (
                BertTokenizer
                .from_pretrained(
                    "bert-base-uncased"
                )
            )

            model = model.to(DEVICE)
            model.eval()

            self._rsvg_model = model
            self._rsvg_tokenizer = tokenizer

            self._checkpoint_paths[
                "rsvg"
            ] = _path_string(
                checkpoint
            )

            self._load_status[
                "rsvg"
            ] = (
                "RSVG remote-sensing grounding model"
            )

            self._scientific_flags[
                "rsvg"
            ] = True

            elapsed = time.time() - t0

            self._load_times[
                "rsvg"
            ] = elapsed

            logger.info(
                "RSVG loaded in %.2fs",
                elapsed,
            )

            return (
                self._rsvg_model,
                self._rsvg_tokenizer,
            )

        except Exception as rsvg_exc:

            logger.warning(
                "RSVG loading failed: %s",
                rsvg_exc,
            )

            self._scientific_flags[
                "rsvg"
            ] = False

            # ---------------------------------------------------------
            # Explicit development fallback
            # ---------------------------------------------------------

            if ALLOW_GROUNDING_DINO_FALLBACK:

                logger.warning(
                    "GroundingDINO development fallback explicitly enabled."
                )

                try:

                    from transformers import (
                        AutoProcessor,
                        AutoModelForZeroShotObjectDetection,
                    )

                    model_id = (
                        "IDEA-Research/"
                        "grounding-dino-base"
                    )

                    model = (
                        AutoModelForZeroShotObjectDetection
                        .from_pretrained(
                            model_id
                        )
                        .to(DEVICE)
                    )

                    processor = (
                        AutoProcessor
                        .from_pretrained(
                            model_id
                        )
                    )

                    model.eval()

                    self._rsvg_model = model
                    self._rsvg_tokenizer = processor

                    self._load_status[
                        "rsvg"
                    ] = (
                        "DEVELOPMENT FALLBACK: "
                        "GroundingDINO; NOT RSVG"
                    )

                    self._scientific_flags[
                        "rsvg"
                    ] = False

                    return (
                        self._rsvg_model,
                        self._rsvg_tokenizer,
                    )

                except Exception as fallback_exc:

                    raise RuntimeError(
                        "RSVG failed and the explicitly enabled "
                        "GroundingDINO fallback also failed.\n"
                        f"RSVG error: {rsvg_exc}\n"
                        f"Fallback error: {fallback_exc}"
                    ) from fallback_exc

            self._load_status[
                "rsvg"
            ] = (
                "BLOCKED: RSVG checkpoint or implementation unavailable"
            )

            raise RuntimeError(
                "RSVG grounding is unavailable.\n"
                f"Reason: {rsvg_exc}\n\n"
                "Scientific grounding is BLOCKED."
            ) from rsvg_exc

    # =================================================================
    # SAM
    # =================================================================

    def get_sam(
        self,
    ) -> Any:
        """
        Load SAM ViT-Base.

        SAM is an auxiliary segmentation model. It is not itself
        a remote-sensing grounding model.
        """

        if self._sam_predictor is not None:

            return self._sam_predictor

        try:

            from segment_anything import (
                sam_model_registry,
                SamPredictor,
            )

        except ImportError as exc:

            self._load_status[
                "sam"
            ] = (
                f"FAILED: missing segment-anything: {exc}"
            )

            raise RuntimeError(
                "SAM requires the segment-anything package."
            ) from exc

        t0 = time.time()

        checkpoint = _find_checkpoint(
            "sam_vit_b_01ec64.pth",
        )

        if checkpoint is None:

            self._load_status[
                "sam"
            ] = (
                "BLOCKED: SAM checkpoint missing"
            )

            self._scientific_flags[
                "sam"
            ] = False

            raise FileNotFoundError(
                "SAM ViT-B checkpoint not found.\n\n"
                "Expected:\n"
                "  models/sam_vit_b_01ec64.pth\n\n"
                "Download the official SAM ViT-B checkpoint "
                "and place it in the models directory."
            )

        try:

            _validate_checkpoint_size(
                checkpoint,
                minimum_mb=100.0,
            )

            logger.info(
                "Loading SAM checkpoint: %s",
                _path_string(checkpoint),
            )

            sam = sam_model_registry[
                "vit_b"
            ](
                checkpoint=_path_string(
                    checkpoint
                )
            )

            sam = sam.to(DEVICE)

            predictor = SamPredictor(
                sam
            )

            self._sam_predictor = predictor

            self._checkpoint_paths[
                "sam"
            ] = _path_string(
                checkpoint
            )

            self._load_status[
                "sam"
            ] = (
                "SAM-ViT-Base loaded"
            )

            self._scientific_flags[
                "sam"
            ] = True

            elapsed = time.time() - t0

            self._load_times[
                "sam"
            ] = elapsed

            logger.info(
                "SAM loaded in %.2fs",
                elapsed,
            )

            return self._sam_predictor

        except Exception as exc:

            logger.exception(
                "SAM loading failed."
            )

            self._load_status[
                "sam"
            ] = f"FAILED: {exc}"

            self._scientific_flags[
                "sam"
            ] = False

            self._sam_predictor = None

            raise RuntimeError(
                f"SAM could not be loaded: {exc}"
            ) from exc

    # =================================================================
    # MODEL STATUS
    # =================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return detailed runtime model status.

        Used by:
            /api/v1/models
            health endpoints
            preflight diagnostics
            frontend diagnostics
        """

        gpu_name = "N/A"
        vram_gb = 0.0

        if (
            TORCH_AVAILABLE
            and torch.cuda.is_available()
        ):

            try:

                gpu_name = (
                    torch.cuda
                    .get_device_name(0)
                )

                vram_gb = round(
                    torch.cuda
                    .get_device_properties(0)
                    .total_memory
                    / 1e9,
                    2,
                )

            except Exception:

                gpu_name = "CUDA available"
                vram_gb = 0.0

        models_loaded = {
            "geochat":
                self._geochat_model is not None,

            "remote_clip":
                self._remote_clip_model is not None,

            "changeformer":
                self._changeformer_model is not None,

            "rsvg":
                self._rsvg_model is not None,

            "sam":
                self._sam_predictor is not None,
        }

        scientific_ready = {
            name: bool(
                models_loaded.get(name)
                and self._scientific_flags.get(
                    name,
                    False,
                )
            )
            for name in models_loaded
        }

        return {
            "device": DEVICE,

            "cuda_available": (
                TORCH_AVAILABLE
                and torch.cuda.is_available()
            ),

            "gpu_name": gpu_name,

            "vram_gb": vram_gb,

            "project_root": str(
                PROJECT_ROOT
            ),

            "models_dir": str(
                MODELS_DIR
            ),

            "checkpoint_dir": str(
                CHECKPOINT_DIR
            ),

            "models_loaded": models_loaded,

            "scientific_ready": scientific_ready,

            "load_status": (
                self._load_status
            ),

            "checkpoint_paths": (
                self._checkpoint_paths
            ),

            "load_times_sec": {
                key: round(
                    value,
                    3,
                )
                for key, value
                in self._load_times.items()
            },

            "development_fallbacks": {
                "non_rs_vlm":
                    ALLOW_NON_RS_VLM_FALLBACK,

                "base_clip":
                    ALLOW_BASE_CLIP_FALLBACK,

                "grounding_dino":
                    ALLOW_GROUNDING_DINO_FALLBACK,
            },

            "scientific_policy": {
                "random_weights_allowed": False,
                "silent_model_substitution": False,
                "unverified_checkpoint_claims": False,
                "base_clip_claimed_as_remoteclip": False,
                "grounding_dino_claimed_as_rsvg": False,
                "fallbacks_are_marked_non_scientific": True,
            },
        }

    # =================================================================
    # READINESS
    # =================================================================

    def get_scientific_readiness(
        self,
    ) -> Dict[str, Any]:
        """
        Return a concise readiness report.

        This does not download or load models.
        It only reports whether runtime-loaded models are
        scientifically usable.
        """

        status = self.get_status()

        scientific_ready = status[
            "scientific_ready"
        ]

        return {
            "ready": all(
                scientific_ready.values()
            ),
            "models": scientific_ready,
            "failed_or_missing": [
                name
                for name, ready
                in scientific_ready.items()
                if not ready
            ],
        }

    # =================================================================
    # RESET
    # =================================================================

    def unload_all(
        self,
    ) -> None:
        """
        Release cached models.

        Useful for:
            testing
            development
            memory recovery
        """

        self._geochat_model = None
        self._geochat_processor = None

        self._remote_clip_model = None
        self._remote_clip_preprocess = None

        self._changeformer_model = None

        self._rsvg_model = None
        self._rsvg_tokenizer = None

        self._sam_predictor = None

        self._load_status.clear()
        self._checkpoint_paths.clear()
        self._scientific_flags.clear()
        self._load_times.clear()

        if (
            TORCH_AVAILABLE
            and torch.cuda.is_available()
        ):

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info(
            "All cached models unloaded."
        )


# =====================================================================
# MODULE-LEVEL SINGLETON
# =====================================================================

_model_loader: Optional[
    ModelLoader
] = None


def get_model_loader() -> ModelLoader:
    """
    Get the global ModelLoader instance.
    """

    global _model_loader

    if _model_loader is None:

        _model_loader = ModelLoader()

    return _model_loader
