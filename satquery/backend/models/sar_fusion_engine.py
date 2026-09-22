"""
SatQuery AI — SAR-Optical Fusion Engine
=======================================

Cross-modal analysis of SAR (Sentinel-1) and Optical (Sentinel-2)
imagery using:

    SAR:
        ResNet-50 adapted for 2-channel VV/VH input

    Optical:
        RemoteCLIP ViT optical encoder

    Fusion:
        Bi-directional Multi-Head Cross-Attention

The architecture may use a trained checkpoint when a verified
checkpoint is available.

IMPORTANT SCIENTIFIC POLICY
----------------------------
This module must not fabricate model performance, confidence,
feature statistics, alignment percentages, signal advantages,
or benchmark values.

All reported values must come from:

    1. Supplied imagery
    2. Actual model execution
    3. Loaded checkpoint state
    4. Explicitly computed statistics

If a trained checkpoint is unavailable, the system reports that
fact instead of pretending that random/untrained weights are trained.

No claims such as:

    "+14.2 dB signal advantage"
    "91.2% alignment"
    "0.91 confidence"
    "trained 8-head cross-attention"

are generated unless they are directly supported by actual
computation or verified model metadata.
"""

from __future__ import annotations

import os
import io
import base64
import time
import pickle
import logging
from typing import Optional, Dict, Any, Tuple

from ..core.paths import model_path, checkpoint_path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    nn_Module = nn.Module

except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False
    nn_Module = object

from PIL import Image

from .model_loader import get_model_loader


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# SAR Encoder
# ──────────────────────────────────────────────────────────────

class SAREncoder(nn_Module):
    """
    ResNet-50 modified to accept 2-channel SAR input (VV, VH).

    The ResNet backbone is initialized from ImageNet weights for
    the available layers. The first convolution is adapted to
    accept two SAR channels.

    NOTE:
        ImageNet initialization is not equivalent to a trained
        SAR-specific checkpoint.
    """

    def __init__(self, out_dim: int = 256):
        if not TORCH_AVAILABLE:
            return

        super().__init__()

        import torchvision.models as tv_models

        resnet = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
        )

        # Replace first convolution:
        # RGB 3 channels -> SAR VV/VH 2 channels
        self.conv1 = nn.Conv2d(
            2,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        with torch.no_grad():
            # Preserve the first two ImageNet channels as the
            # initialization for the two SAR channels.
            self.conv1.weight.copy_(
                resnet.conv1.weight[:, :2, :, :]
            )

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # layer3 output = 1024 channels
        self.proj = nn.Sequential(
            nn.Conv2d(
                1024,
                out_dim,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.proj(x)

        return x


# ──────────────────────────────────────────────────────────────
# Cross-Attention Fusion Module
# ──────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn_Module):
    """
    Bi-directional Multi-Head Cross-Attention between SAR and
    optical features.

    SAR queries Optical.
    Optical queries SAR.
    The resulting representations are projected into a common
    fusion space.

    IMPORTANT:
        The presence of this architecture does not itself prove
        that the architecture was trained. Training status is
        reported separately by SARFusionEngine.
    """

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
    ):
        if not TORCH_AVAILABLE:
            return

        super().__init__()

        self.sar_to_optical_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.optical_to_sar_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm_sar = nn.LayerNorm(dim)
        self.norm_opt = nn.LayerNorm(dim)

        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        sar_feat: torch.Tensor,
        opt_feat: torch.Tensor,
    ) -> torch.Tensor:

        B, C, H, W = sar_feat.shape

        sar_seq = (
            sar_feat
            .flatten(2)
            .permute(0, 2, 1)
        )

        opt_seq = (
            opt_feat
            .flatten(2)
            .permute(0, 2, 1)
        )

        sar_enriched, _ = self.sar_to_optical_attn(
            sar_seq,
            opt_seq,
            opt_seq,
        )

        opt_enriched, _ = self.optical_to_sar_attn(
            opt_seq,
            sar_seq,
            sar_seq,
        )

        sar_enriched = self.norm_sar(
            sar_seq + sar_enriched
        )

        opt_enriched = self.norm_opt(
            opt_seq + opt_enriched
        )

        fused_seq = self.fusion_proj(
            torch.cat(
                [
                    sar_enriched,
                    opt_enriched,
                ],
                dim=-1,
            )
        )

        fused = (
            fused_seq
            .permute(0, 2, 1)
            .reshape(B, C, H, W)
        )

        return fused


# ──────────────────────────────────────────────────────────────
# Full SAR-Optical Fusion Model
# ──────────────────────────────────────────────────────────────

class SAROpticalFusionModel(nn_Module):
    """
    Full SAR-Optical cross-modal fusion model.

    Encodes SAR and optical representations separately and
    combines them using cross-attention.
    """

    def __init__(self, dim: int = 256):
        if not TORCH_AVAILABLE:
            return

        super().__init__()

        self.sar_encoder = SAREncoder(
            out_dim=dim
        )

        self.optical_proj = nn.Sequential(
            nn.Linear(512, dim),
            nn.ReLU(inplace=True),
        )

        self.fusion = CrossAttentionFusion(
            dim=dim,
            n_heads=8,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        sar_tensor: torch.Tensor,
        opt_features: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        sar_feat = self.sar_encoder(
            sar_tensor
        )

        B, C_sar, H, W = sar_feat.shape

        opt_proj = self.optical_proj(
            opt_features
        )

        opt_feat = (
            opt_proj
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                B,
                -1,
                H,
                W,
            )
        )

        fused = self.fusion(
            sar_feat,
            opt_feat,
        )

        return (
            sar_feat,
            opt_feat,
            fused,
        )


# ──────────────────────────────────────────────────────────────
# SAR Fusion Engine
# ──────────────────────────────────────────────────────────────

class SARFusionEngine:
    """
    SAR-Optical cross-modal fusion engine.

    Handles:
        CROSS_MODAL_SAR_OPTICAL

    Scientific reporting rules:
        - No invented confidence
        - No invented feature statistics
        - No invented alignment
        - No invented signal advantage
        - No benchmark claims
        - Checkpoint status reported explicitly
    """

    def __init__(self):

        self.loader = get_model_loader()

        self._device = (
            "cuda"
            if (
                torch is not None
                and hasattr(torch, "cuda")
                and torch.cuda.is_available()
            )
            else "cpu"
        )

        self._fusion_model: Optional[
            SAROpticalFusionModel
        ] = None

        self._is_adapted_checkpoint = False
        self._checkpoint_path: Optional[str] = None
        self._checkpoint_load_error: Optional[str] = None

    # ──────────────────────────────────────────────────────────
    # MODEL INITIALIZATION
    # ──────────────────────────────────────────────────────────

    def _get_fusion_model(
            self,
        ) -> SAROpticalFusionModel:
    
            if self._fusion_model is None:
            
                self._fusion_model = (
                    SAROpticalFusionModel(
                        dim=256
                    ).to(self._device)
                )
    
                ckpt_paths = [
                    checkpoint_path(
                        "sar_optical_cross_attention_best.pth"
                    ),
                    model_path(
                        "sar_optical_fusion.pth"
                    ),
                ]
    
                loaded = False
    
                for p in ckpt_paths:
                
                    if not os.path.exists(p):
                        continue
                    
                    try:
                    
                        try:
                            ckpt = torch.load(
                                p,
                                map_location=self._device,
                                weights_only=False,
                            )
    
                        except Exception:
                        
                            with open(
                                p,
                                "rb",
                            ) as f:
                                ckpt = pickle.load(f)
    
                        if isinstance(
                            ckpt,
                            dict,
                        ):
                            raw_sd = ckpt.get(
                                "model_state_dict",
                                ckpt.get(
                                    "state_dict",
                                    ckpt,
                                ),
                            )
                        else:
                            raw_sd = ckpt
    
                        if not isinstance(
                            raw_sd,
                            dict,
                        ):
                            raise ValueError(
                                "Checkpoint does not contain "
                                "a valid state dictionary."
                            )
    
                        sd = {}
    
                        for k, v in raw_sd.items():
                        
                            if isinstance(
                                v,
                                np.ndarray,
                            ):
                                sd[k] = torch.from_numpy(v)
    
                            else:
                                sd[k] = v
    
                        missing_keys, unexpected_keys = (
                            self._fusion_model.load_state_dict(
                                sd,
                                strict=False,
                            )
                        )
    
                        # A partially loaded checkpoint must NOT be treated
                        # as a scientifically valid trained checkpoint.
                        if missing_keys or unexpected_keys:
                            details = (
                                f"missing_keys={list(missing_keys)}, "
                                f"unexpected_keys={list(unexpected_keys)}"
                            )
    
                            logger.error(
                                "Rejected SAR-optical checkpoint %s: "
                                "state-dict mismatch: %s",
                                p,
                                details,
                            )
    
                            raise ValueError(
                                "SAR-optical checkpoint rejected because "
                                "the checkpoint does not exactly match the "
                                f"current inference architecture: {details}"
                            )
    
                        self._is_adapted_checkpoint = True
                        self._checkpoint_path = p
                        self._checkpoint_load_error = None
                        loaded = True
    
                        logger.info(
                            "Loaded verified SAR-optical fusion checkpoint: %s",
                            p,
                        )
    
                        break
                    
                        if missing_keys:
                            logger.warning(
                                "SAR-optical checkpoint loaded with "
                                "missing keys: %s",
                                missing_keys,
                            )
    
                        if unexpected_keys:
                            logger.warning(
                                "SAR-optical checkpoint loaded with "
                                "unexpected keys: %s",
                                unexpected_keys,
                            )
    
                        self._is_adapted_checkpoint = True
                        self._checkpoint_path = p
                        self._checkpoint_load_error = None
                        loaded = True
            
                        logger.info(
                            "Loaded SAR-optical fusion checkpoint: %s",
                            p,
                        )
    
                        break
                    
                    except Exception as e:
                    
                        self._checkpoint_load_error = str(e)
    
                        logger.warning(
                            "Error loading SAR-optical checkpoint "
                            "from %s: %s",
                            p,
                            e,
                        )
    
                if not loaded:
                
                    self._is_adapted_checkpoint = False
                    self._checkpoint_path = None
    
                    logger.warning(
                        "SAR-optical fusion checkpoint was not loaded. "
                        "The configured fusion architecture remains "
                        "available, but its weights are not verified "
                        "as a trained SAR-optical checkpoint."
                    )
    
                self._fusion_model.eval()
    
            return self._fusion_model

    # ──────────────────────────────────────────────────────────
    # MAIN EXECUTION
    # ──────────────────────────────────────────────────────────

    def run(
        self,
        optical_array: np.ndarray,
        sar_array: np.ndarray,
        query: str = (
            "What does the combined SAR and optical "
            "analysis reveal?"
        ),
        optical_meta: Optional[Dict] = None,
        sar_meta: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run SAR-Optical fusion analysis.

        All reported statistics are derived from the supplied
        imagery or actual model execution.
        """

        t0 = time.time()

        optical_pil = self._array_to_pil(
            optical_array
        )

        optical_preview_b64 = (
            self._array_to_b64(
                optical_array
            )
        )

        sar_preview_b64 = (
            self._make_sar_preview_b64(
                sar_array
            )
        )

        # ──────────────────────────────────────────────
        # Physical SAR statistics
        # ──────────────────────────────────────────────

        vv = (
            sar_array[0]
            if sar_array.shape[0] > 0
            else sar_array.squeeze()
        )

        vh = (
            sar_array[1]
            if sar_array.shape[0] > 1
            else vv
        )

        # 22-bin histograms
        vv_hist_raw, _ = np.histogram(
            vv,
            bins=22,
            range=(0.0, 1.0),
        )

        vh_hist_raw, _ = np.histogram(
            vh,
            bins=22,
            range=(0.0, 1.0),
        )

        vv_hist = [
            round(
                float(
                    c
                    / (vv.size + 1e-8)
                    * 100
                    * 3.5
                ),
                1,
            )
            for c in vv_hist_raw
        ]

        vh_hist = [
            round(
                float(
                    c
                    / (vh.size + 1e-8)
                    * 100
                    * 3.5
                ),
                1,
            )
            for c in vh_hist_raw
        ]

        vv_mean_val = float(
            np.mean(vv)
        )

        vh_mean_val = float(
            np.mean(vh)
        )

        # These are derived from the current input normalization.
        # They should not be interpreted as externally calibrated
        # physical measurements unless the upstream preprocessing
        # is known to produce calibrated linear backscatter.
        vv_db = round(
            -28.0
            + vv_mean_val * 26.0,
            1,
        )

        vh_db = round(
            -32.0
            + vh_mean_val * 28.0,
            1,
        )

        pol_ratio = round(
            float(
                vv_mean_val
                / (vh_mean_val + 1e-6)
            ),
            2,
        )

        water_fraction = round(
            float(
                np.mean(
                    vv < 0.20
                )
            )
            * 100,
            1,
        )

        urban_double_bounce = round(
            float(
                np.mean(
                    vv > 0.62
                )
            )
            * 100,
            1,
        )

        veg_volume = round(
            float(
                np.mean(
                    (vv >= 0.20)
                    & (vv <= 0.62)
                )
            )
            * 100,
            1,
        )

        # ──────────────────────────────────────────────
        # Model execution
        # ──────────────────────────────────────────────

        confidence = 0.0

        fusion_overlay_b64 = (
            optical_preview_b64
        )

        fusion_stats: Dict[str, Any] = {
            "sar_feature_norm": 0.0,
            "optical_feature_norm": 0.0,
            "fusion_feature_norm": 0.0,
            "sar_optical_cosine_sim": 0.0,
            "cross_modal_agreement": 0.0,
            "feature_statistics_available": False,
        }

        if TORCH_AVAILABLE:

            optical_tensor = (
                self._prepare_optical_tensor(
                    optical_array
                )
            )

            sar_tensor = (
                self._prepare_sar_tensor(
                    sar_array
                )
            )

            try:

                (
                    opt_features,
                    confidence,
                ) = self._extract_optical_features(
                    optical_tensor,
                    query,
                )

                model = (
                    self._get_fusion_model()
                )

                with torch.no_grad():

                    (
                        sar_feat,
                        opt_feat,
                        fused_feat,
                    ) = model(
                        sar_tensor,
                        opt_features,
                    )

                fusion_overlay_b64 = (
                    self._visualize_attention(
                        fused_feat,
                        optical_array,
                    )
                )

                fusion_stats = (
                    self._compute_fusion_stats(
                        sar_feat,
                        opt_feat,
                        fused_feat,
                    )
                )

            except Exception as e:

                logger.exception(
                    "SAR-optical model execution failed: %s",
                    e,
                )

                confidence = 0.0

                fusion_stats = {
                    "sar_feature_norm": 0.0,
                    "optical_feature_norm": 0.0,
                    "fusion_feature_norm": 0.0,
                    "sar_optical_cosine_sim": 0.0,
                    "cross_modal_agreement": 0.0,
                    "feature_statistics_available": False,
                    "execution_error": str(e),
                }

        else:

            logger.warning(
                "PyTorch is unavailable. "
                "Deep SAR-optical feature fusion was not executed."
            )

            confidence = 0.0

        # ──────────────────────────────────────────────
        # Inject actual SAR statistics
        # ──────────────────────────────────────────────

        fusion_stats[
            "sar_vv_hist"
        ] = vv_hist

        fusion_stats[
            "sar_vh_hist"
        ] = vh_hist

        fusion_stats[
            "vv_mean_db"
        ] = vv_db

        fusion_stats[
            "vh_mean_db"
        ] = vh_db

        fusion_stats[
            "sar_vv_mean_db"
        ] = vv_db

        fusion_stats[
            "sar_vh_mean_db"
        ] = vh_db

        fusion_stats[
            "pol_ratio"
        ] = pol_ratio

        fusion_stats[
            "water_coverage_pct"
        ] = water_fraction

        fusion_stats[
            "builtup_coverage_pct"
        ] = urban_double_bounce

        fusion_stats[
            "vegetation_coverage_pct"
        ] = veg_volume

        # ──────────────────────────────────────────────
        # Dominant scattering interpretation
        # ──────────────────────────────────────────────

        if water_fraction > 20.0:

            dominant = (
                "Specular Reflection "
                "(Open Water / Inundation)"
            )

        elif urban_double_bounce > 20.0:

            dominant = (
                "Double-Bounce "
                "(Urban Built-up Structures)"
            )

        else:

            dominant = (
                "Volume Scattering "
                "(Forest / Agricultural Canopy)"
            )

        fusion_stats[
            "dominant_scattering"
        ] = dominant

        # ──────────────────────────────────────────────
        # Checkpoint metadata
        # ──────────────────────────────────────────────

        is_adapted = getattr(
            self,
            "_is_adapted_checkpoint",
            False,
        )

        fusion_stats[
            "trained_checkpoint"
        ] = bool(is_adapted)

        fusion_stats[
            "checkpoint_path"
        ] = self._checkpoint_path

        fusion_stats[
            "checkpoint_load_error"
        ] = self._checkpoint_load_error

        fusion_stats[
            "architecture"
        ] = (
            "ResNet-50 SAR encoder + "
            "RemoteCLIP optical encoder + "
            "bi-directional 8-head cross-attention"
        )

        if is_adapted:

            fusion_stats[
                "interpretation_mode"
            ] = (
                "Cross-modal feature fusion using "
                "the loaded SAR-optical checkpoint."
            )

            model_used = (
                "CrossModal-ResNet50 + "
                "RemoteCLIP-ViT + "
                "8-Head Cross-Attention "
                "(Loaded Checkpoint)"
            )

        else:

            fusion_stats[
                "interpretation_mode"
            ] = (
                "Configured cross-modal architecture "
                "without a verified trained SAR-optical "
                "checkpoint."
            )

            model_used = (
                "CrossModal-ResNet50 + "
                "RemoteCLIP-ViT + "
                "8-Head Cross-Attention "
                "(Checkpoint Not Verified)"
            )

        # ──────────────────────────────────────────────
        # Independent modality findings
        # ──────────────────────────────────────────────

        optical_findings = (
            self._analyze_optical(
                optical_pil,
                optical_meta,
            )
        )

        sar_findings = (
            self._analyze_sar(
                vv_db,
                vh_db,
                pol_ratio,
                water_fraction,
                urban_double_bounce,
                veg_volume,
                sar_meta,
            )
        )

        cross_modal_agreement = float(
            fusion_stats.get(
                "sar_optical_cosine_sim",
                0.0,
            )
        )

        fused_findings = (
            self._analyze_fused(
                optical_pil,
                query,
                optical_findings,
                sar_findings,
                confidence,
                water_fraction,
                urban_double_bounce,
                veg_volume,
                vv_db,
                vh_db,
                optical_meta,
                sar_meta,
                cross_modal_agreement,
            )
        )

        # ──────────────────────────────────────────────
        # Final response
        # ──────────────────────────────────────────────

        return {
            "optical_preview_b64": (
                optical_preview_b64
            ),

            "sar_preview_b64": (
                sar_preview_b64
            ),

            "fusion_overlay_b64": (
                fusion_overlay_b64
            ),

            "optical_findings": (
                optical_findings
            ),

            "sar_findings": (
                sar_findings
            ),

            "fused_findings": (
                fused_findings
            ),

            # This is model-derived similarity only.
            # It is NOT a calibrated probability unless
            # a calibration procedure has been performed.
            "confidence": float(
                confidence
            ),

            # Explicit cross-modal agreement.
            "cross_modal_agreement": (
                cross_modal_agreement
            ),

            "fusion_stats": (
                fusion_stats
            ),

            "latency_sec": round(
                time.time() - t0,
                2,
            ),

            "model_used": (
                model_used
            ),
        }

    # ──────────────────────────────────────────────────────────
    # OPTICAL FEATURE EXTRACTION
    # ──────────────────────────────────────────────────────────

    def _extract_optical_features(
        self,
        optical_tensor: torch.Tensor,
        query: str,
    ) -> Tuple[
        torch.Tensor,
        float,
    ]:
        """
        Extract optical features using RemoteCLIP.

        The returned confidence is derived from actual image/text
        similarity when available.

        IMPORTANT:
            The value is a similarity-derived score, not a
            calibrated probability.
        """

        try:

            (
                clip_model,
                tokenizer,
            ), preprocess = (
                self.loader.get_remote_clip()
            )

            pil_img = Image.fromarray(
                (
                    optical_tensor
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    * 255
                ).astype(
                    np.uint8
                )
            )

            img_t = (
                preprocess(
                    pil_img
                )
                .unsqueeze(0)
                .to(self._device)
            )

            with torch.no_grad():

                img_feat = (
                    clip_model.encode_image(
                        img_t
                    )
                )

                img_feat_norm = (
                    img_feat
                    / (
                        img_feat.norm(
                            dim=-1,
                            keepdim=True,
                        )
                        + 1e-8
                    )
                )

                q_tok = (
                    tokenizer(
                        [query]
                    ).to(
                        self._device
                    )
                )

                q_feat = (
                    clip_model.encode_text(
                        q_tok
                    )
                )

                q_feat_norm = (
                    q_feat
                    / (
                        q_feat.norm(
                            dim=-1,
                            keepdim=True,
                        )
                        + 1e-8
                    )
                )

                sim = (
                    img_feat_norm
                    @ q_feat_norm.T
                ).item()

                # Convert similarity to a bounded score for UI.
                # This is explicitly NOT a calibrated probability.
                similarity_score = (
                    (sim + 1.0) / 2.0
                )

                confidence = round(
                    max(
                        0.0,
                        min(
                            1.0,
                            similarity_score,
                        ),
                    ),
                    3,
                )

            return (
                img_feat,
                confidence,
            )

        except Exception as e:

            logger.warning(
                "RemoteCLIP feature extraction failed: %s",
                e,
            )

            # Do not fabricate a random feature vector or confidence.
            raise

    # ──────────────────────────────────────────────────────────
    # OPTICAL ANALYSIS
    # ──────────────────────────────────────────────────────────

    def _analyze_optical(
        self,
        pil: Image.Image,
        meta: Optional[Dict],
    ) -> str:
        """
        Analyze optical imagery using simple spectral/color
        characteristics derived from the supplied image.
        """

        sensor = (
            (meta or {}).get(
                "sensor",
                "Sentinel-2 MSI",
            )
        )

        arr = np.array(
            pil,
            dtype=np.float32,
        )

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        exg = (
            2 * g
            - r
            - b
        )

        veg_pct = round(
            float(
                np.mean(
                    exg > 15
                )
            )
            * 100,
            1,
        )

        bright_pct = round(
            float(
                np.mean(
                    arr.mean(
                        axis=-1
                    )
                    > 175
                )
            )
            * 100,
            1,
        )

        dark_pct = round(
            float(
                np.mean(
                    arr.mean(
                        axis=-1
                    )
                    < 45
                )
            )
            * 100,
            1,
        )

        return (
            f"Optical analysis "
            f"({sensor.replace('_', ' ').title()}): "
            f"scene exhibits {veg_pct}% "
            f"pixels above the configured ExG vegetation "
            f"threshold, {bright_pct}% high-intensity pixels, "
            f"and {dark_pct}% low-intensity pixels. "
            f"These are image-derived spectral/color statistics "
            f"and are not independently validated land-cover labels."
        )

    # ──────────────────────────────────────────────────────────
    # SAR ANALYSIS
    # ──────────────────────────────────────────────────────────

    def _analyze_sar(
        self,
        vv_db: float,
        vh_db: float,
        pol_ratio: float,
        water_pct: float,
        urban_pct: float,
        veg_pct: float,
        meta: Optional[Dict],
    ) -> str:
        """
        Physical/statistical interpretation of the supplied SAR
        array.

        Threshold-based percentages should be interpreted as
        heuristic image-derived indicators unless the upstream
        calibration and thresholds have been validated for the
        sensor/product.
        """

        sensor = (
            (meta or {}).get(
                "sensor",
                "Sentinel-1 C-Band SAR",
            )
        )

        return (
            f"SAR Polarimetric Assessment "
            f"({sensor.replace('_', ' ').title()}): "
            f"mean derived backscatter values are "
            f"VV = {vv_db} dB and VH = {vh_db} dB "
            f"(VV/VH ratio: {pol_ratio}). "
            f"The supplied imagery contains approximately "
            f"{water_pct}% pixels below the configured low-VV "
            f"threshold, {urban_pct}% above the configured "
            f"high-VV threshold, and {veg_pct}% within the "
            f"configured intermediate-VV range. "
            f"These percentages are image-derived indicators, "
            f"not independently verified semantic classifications."
        )

    # ──────────────────────────────────────────────────────────
    # FUSED INTERPRETATION
    # ──────────────────────────────────────────────────────────

    def _analyze_fused(
        self,
        optical_pil: Image.Image,
        query: str,
        optical_findings: str,
        sar_findings: str,
        confidence: float,
        water_pct: float,
        urban_pct: float,
        veg_pct: float,
        vv_db: float,
        vh_db: float,
        optical_meta: Optional[Dict],
        sar_meta: Optional[Dict],
        cross_modal_agreement: float = 0.0,
    ) -> str:
        """
        Produce a conservative SAR-optical interpretation.

        No fabricated signal advantage, alignment percentage,
        validation result, or ground-truth classification is
        reported here.
        """

        return (
            "SAR-optical cross-modal analysis completed using "
            "the configured fusion architecture. "
            f"SAR backscatter statistics: VV={vv_db:.2f} dB, "
            f"VH={vh_db:.2f} dB. "
            f"Estimated water-like coverage={water_pct:.2f}%, "
            f"urban-like coverage={urban_pct:.2f}%, "
            f"vegetation-like coverage={veg_pct:.2f}%. "
            "These statistics are derived from the supplied imagery. "
            f"Model confidence score={confidence:.3f}. "
            f"Cross-modal feature agreement="
            f"{cross_modal_agreement:.3f}. "
            "These model-derived scores are reported separately "
            "and should not be interpreted as calibrated probabilities "
            "or validated semantic accuracy without a held-out "
            "evaluation/calibration dataset."
        )

    # ──────────────────────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────────────────────

    def _visualize_attention(
        self,
        fused_feat: torch.Tensor,
        optical_array: np.ndarray,
    ) -> str:
        """
        Create a visualization from the fused feature activation.

        NOTE:
            This is a feature-activation visualization. It is not
            necessarily an attention probability map.
        """

        attn = (
            fused_feat
            .squeeze(0)
            .mean(0)
            .detach()
            .cpu()
            .numpy()
        )

        attn = (
            attn - attn.min()
        ) / (
            attn.max()
            - attn.min()
            + 1e-8
        )

        target_h = optical_array.shape[1]
        target_w = optical_array.shape[2]

        attn_pil = Image.fromarray(
            (
                attn * 255
            ).astype(
                np.uint8
            )
        )

        attn_pil = attn_pil.resize(
            (
                target_w,
                target_h,
            ),
            Image.BILINEAR,
        )

        attn_np = (
            np.array(
                attn_pil
            )
            / 255.0
        )

        # Visualization heatmap.
        heatmap = np.zeros(
            (
                target_h,
                target_w,
                3,
            ),
            dtype=np.float32,
        )

        heatmap[:, :, 0] = np.clip(
            attn_np * 2,
            0,
            1,
        )

        heatmap[:, :, 1] = np.clip(
            (
                attn_np
                - 0.5
            )
            * 2,
            0,
            1,
        )

        heatmap[:, :, 2] = 0.0

        # Blend with optical image.
        if optical_array.shape[0] >= 3:

            opt_rgb = (
                optical_array[:3]
                .transpose(
                    1,
                    2,
                    0,
                )
            )

        else:

            opt_rgb = np.repeat(
                optical_array[:1]
                .transpose(
                    1,
                    2,
                    0,
                ),
                3,
                axis=-1,
            )

        blended = (
            0.6 * opt_rgb
            + 0.4 * heatmap
        )

        blended = (
            np.clip(
                blended,
                0,
                1,
            )
            * 255
        ).astype(
            np.uint8
        )

        return self._pil_to_b64(
            Image.fromarray(
                blended
            )
        )

    # ──────────────────────────────────────────────────────────
    # SAR PREVIEW
    # ──────────────────────────────────────────────────────────

    def _make_sar_preview_b64(
        self,
        sar_array: np.ndarray,
    ) -> str:
        """
        Generate a SAR false-color preview from VV/VH channels.
        """

        vv = (
            sar_array[0]
            if sar_array.shape[0] > 0
            else sar_array.squeeze()
        )

        vh = (
            sar_array[1]
            if sar_array.shape[0] > 1
            else vv
        )

        ratio = np.clip(
            vv - vh,
            0,
            1,
        )

        rgb = np.stack(
            [
                vv,
                vh,
                ratio,
            ],
            axis=-1,
        )

        rgb = (
            np.clip(
                rgb,
                0,
                1,
            )
            * 255
        ).astype(
            np.uint8
        )

        return self._pil_to_b64(
            Image.fromarray(
                rgb
            )
        )

    # ──────────────────────────────────────────────────────────
    # FEATURE STATISTICS
    # ──────────────────────────────────────────────────────────

    def _compute_fusion_stats(
        self,
        sar_feat: torch.Tensor,
        opt_feat: torch.Tensor,
        fused: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute feature statistics from actual model outputs.

        sar_optical_cosine_sim is a direct feature-space cosine
        similarity.

        It is NOT a calibrated probability and must not be
        described as percentage alignment or accuracy.
        """

        if (
            sar_feat is None
            or opt_feat is None
            or fused is None
        ):
            return {
                "sar_feature_norm": 0.0,
                "optical_feature_norm": 0.0,
                "fusion_feature_norm": 0.0,
                "sar_optical_cosine_sim": 0.0,
                "cross_modal_agreement": 0.0,
                "feature_statistics_available": False,
            }

        try:

            sar_flat = (
                sar_feat.flatten(1)
            )

            opt_flat = (
                opt_feat.flatten(1)
            )

            # Ensure feature dimensionality matches.
            if sar_flat.shape != opt_flat.shape:

                raise ValueError(
                    "SAR and optical feature tensors "
                    f"have incompatible shapes: "
                    f"{tuple(sar_flat.shape)} vs "
                    f"{tuple(opt_flat.shape)}"
                )

            cosine_sim = (
                F.cosine_similarity(
                    sar_flat,
                    opt_flat,
                    dim=1,
                )
                .mean()
                .item()
            )

            cosine_sim = float(
                np.clip(
                    cosine_sim,
                    -1.0,
                    1.0,
                )
            )

            return {
                "sar_feature_norm": round(
                    float(
                        sar_feat.norm().item()
                    ),
                    3,
                ),

                "optical_feature_norm": round(
                    float(
                        opt_feat.norm().item()
                    ),
                    3,
                ),

                "fusion_feature_norm": round(
                    float(
                        fused.norm().item()
                    ),
                    3,
                ),

                "sar_optical_cosine_sim": round(
                    cosine_sim,
                    3,
                ),

                "cross_modal_agreement": round(
                    cosine_sim,
                    3,
                ),

                "feature_statistics_available": True,
            }

        except Exception as e:

            logger.warning(
                "Unable to compute cross-modal feature "
                "agreement: %s",
                e,
            )

            return {
                "sar_feature_norm": 0.0,
                "optical_feature_norm": 0.0,
                "fusion_feature_norm": 0.0,
                "sar_optical_cosine_sim": 0.0,
                "cross_modal_agreement": 0.0,
                "feature_statistics_available": False,
                "agreement_error": str(e),
            }

    # ──────────────────────────────────────────────────────────
    # TENSOR PREPARATION
    # ──────────────────────────────────────────────────────────

    def _prepare_optical_tensor(
        self,
        array: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert [C, H, W] optical array to [1, 3, H, W].
        """

        array = np.asarray(
            array,
            dtype=np.float32,
        )

        if array.ndim != 3:

            raise ValueError(
                "Optical array must have shape [C, H, W]."
            )

        if array.shape[0] > 3:

            array = array[:3]

        elif array.shape[0] < 3:

            array = np.repeat(
                array[:1],
                3,
                axis=0,
            )

        return (
            torch.from_numpy(
                np.ascontiguousarray(
                    array
                )
            )
            .float()
            .unsqueeze(0)
            .to(self._device)
        )

    def _prepare_sar_tensor(
        self,
        array: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert [C, H, W] SAR array to [1, 2, H, W].
        """

        array = np.asarray(
            array,
            dtype=np.float32,
        )

        if array.ndim != 3:

            raise ValueError(
                "SAR array must have shape [C, H, W]."
            )

        if array.shape[0] >= 2:

            arr = array[:2]

        else:

            arr = np.repeat(
                array[:1],
                2,
                axis=0,
            )

        return (
            torch.from_numpy(
                np.ascontiguousarray(
                    arr
                )
            )
            .float()
            .unsqueeze(0)
            .to(self._device)
        )

    # ──────────────────────────────────────────────────────────
    # IMAGE CONVERSION
    # ──────────────────────────────────────────────────────────

    def _array_to_pil(
        self,
        array: np.ndarray,
    ) -> Image.Image:
        """
        Convert [C, H, W] normalized array to RGB PIL image.
        """

        array = np.asarray(
            array,
            dtype=np.float32,
        )

        if array.ndim != 3:

            raise ValueError(
                "Image array must have shape [C, H, W]."
            )

        if array.shape[0] >= 3:

            rgb = (
                array[:3]
                .transpose(
                    1,
                    2,
                    0,
                )
            )

        else:

            rgb = np.repeat(
                array[:1]
                .transpose(
                    1,
                    2,
                    0,
                ),
                3,
                axis=-1,
            )

        rgb = (
            np.clip(
                rgb,
                0,
                1,
            )
            * 255
        ).astype(
            np.uint8
        )

        return Image.fromarray(
            rgb
        )

    def _array_to_b64(
        self,
        array: np.ndarray,
    ) -> str:
        return self._pil_to_b64(
            self._array_to_pil(
                array
            )
        )

    def _pil_to_b64(
        self,
        pil: Image.Image,
    ) -> str:

        buf = io.BytesIO()

        pil.convert(
            "RGB"
        ).save(
            buf,
            format="JPEG",
            quality=85,
        )

        return (
            "data:image/jpeg;base64,"
            + base64.b64encode(
                buf.getvalue()
            ).decode()
        )


# ──────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────

_sar_engine: Optional[
    SARFusionEngine
] = None


def get_sar_engine() -> SARFusionEngine:

    global _sar_engine

    if _sar_engine is None:

        _sar_engine = SARFusionEngine()

    return _sar_engine


# Backward-compatible alias
get_sar_fusion_engine = get_sar_engine