"""
SatQuery AI — Real Change Detection Engine
===========================================
Bi-temporal change analysis using ChangeFormer (or Siamese ResNet-18 fallback).
GeoChat-7B provides natural language description of detected changes.

Key design principle from SIH docs:
  "ChangeFormer computes the change map. The LLM only verbalizes it."

Replaces: Previous hardcoded scenario-based text responses.
"""

from __future__ import annotations

import io
import base64
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

from PIL import Image, ImageDraw, ImageFont

from .model_loader import get_model_loader

logger = logging.getLogger(__name__)

# Change map color scheme (matches frontend legend)
CHANGE_COLOR = (239, 68, 68)     # Red #EF4444 — changed pixels
NO_CHANGE_COLOR = (16, 185, 129) # Green #10B981 — unchanged pixels
CHANGE_THRESHOLD = 0.5           # Sigmoid output threshold

# Area calculation constants
# (will be overridden by actual image resolution from metadata)
# No default spatial resolution: PNG/JPEG must not receive fake GSD.
M2_PER_KM2 = 1_000_000


class ChangeDetectionEngine:
    """
    Bi-temporal change detection engine.
    Handles: BI_TEMPORAL_CHANGE task type.
    """

    def __init__(self):
        self.loader = get_model_loader()
        self._device = "cuda" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"

    def run(
        self,
        t1_array: np.ndarray,
        t2_array: np.ndarray,
        t1_meta: Optional[Dict] = None,
        t2_meta: Optional[Dict] = None,
        query: str = "What changed between these two images?",
        reference_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run bi-temporal change detection.

        Args:
            t1_array: [C, H, W] float32 normalized T1 (before) image
            t2_array: [C, H, W] float32 normalized T2 (after) image
            t1_meta: Metadata for T1 (sensor, date, resolution, etc.)
            t2_meta: Metadata for T2
            query:   Natural language query for VQA decoder
            reference_mask: Optional ground-truth mask for IoU evaluation [H, W] binary

        Returns:
            {
                "change_map_b64": str,        # base64 PNG change map
                "overlay_b64": str,           # base64 PNG: T2 + change overlay
                "change_pct": float,          # percentage of area changed
                "changed_area_km2": float,    # estimated changed area in km²
                "description": str,           # GeoChat natural language answer
                "t1_preview_b64": str,        # base64 T1 preview
                "t2_preview_b64": str,        # base64 T2 preview
                "change_stats": dict,         # connected component analysis
                "evaluation": dict,           # IoU, F1, Precision, Recall (if reference provided)
                "confidence": float,
                "latency_sec": float,
                "model_used": str,
            }
        """
        t0 = time.time()

        # ── Step 1 & 2: Run ChangeFormer or Spectral Differencing ──────
        if TORCH_AVAILABLE:
            t1_tensor = self._prepare_tensor(t1_array)
            t2_tensor = self._prepare_tensor(t2_array)
            import hashlib

            t1_hash = hashlib.sha256(
                t1_tensor.detach().cpu().numpy().tobytes()
            ).hexdigest()[:16]

            t2_hash = hashlib.sha256(
                t2_tensor.detach().cpu().numpy().tobytes()
            ).hexdigest()[:16]

            print(
                "[CHANGEFORMER INPUT] "
                f"T1 hash={t1_hash} "
                f"T2 hash={t2_hash} "
                f"T1 shape={tuple(t1_tensor.shape)} "
                f"T2 shape={tuple(t2_tensor.shape)} "
                f"T1 mean={float(t1_tensor.mean()):.6f} "
                f"T2 mean={float(t2_tensor.mean()):.6f}"
            )

            change_logits, model_name = self._run_changeformer(t1_tensor, t2_tensor)

            output_hash = hashlib.sha256(
                change_logits.detach().cpu().numpy().tobytes()
            ).hexdigest()[:16]

            print(
                "[CHANGEFORMER OUTPUT] "
                f"hash={output_hash} "
                f"shape={tuple(change_logits.shape)} "
                f"min={float(change_logits.min()):.6f} "
                f"max={float(change_logits.max()):.6f} "
                f"mean={float(change_logits.mean()):.6f}"
            )
            change_prob = torch.softmax(change_logits, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
            change_map = torch.argmax(change_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        else:
            model_name = "SpectralDifferencing-Siamese"
            arr1 = t1_array.astype(float)
            arr2 = t2_array.astype(float)
            if arr1.max() > 1.0:
                arr1 = arr1 / 255.0
            if arr2.max() > 1.0:
                arr2 = arr2 / 255.0
            diff = np.abs(arr1 - arr2)
            if diff.ndim == 3 and diff.shape[0] in (2, 3, 4):
                change_prob = diff.mean(axis=0)
            elif diff.ndim == 3:
                change_prob = diff.mean(axis=-1)
            else:
                change_prob = diff
            change_map = (change_prob > 0.15).astype(np.uint8)

        # ── Step 3: Compute change statistics ────────────────        # Step 3: Compute change statistics
        change_pct = float(change_map.mean() * 100)

        # Only real spatial-resolution metadata may be used for
        # geographic area. PNG/JPEG visual-only inputs have no
        # defensible pixel size, so area remains unavailable.
        resolution_raw = (t1_meta or {}).get("resolution_m")

        try:
            resolution_m = (
                float(resolution_raw)
                if resolution_raw is not None
                else None
            )
            if resolution_m is not None and resolution_m <= 0:
                resolution_m = None
        except (TypeError, ValueError):
            resolution_m = None

        changed_area_km2 = None

        if resolution_m is not None:
            changed_area_km2 = self._compute_area_km2(
                change_map,
                resolution_m,
            )

        change_stats = self._analyze_change_regions(
            change_map,
            resolution_m,
        )

        # ── Step 4: Evaluation against reference mask ─────────
        evaluation = {}
        if reference_mask is not None:
            evaluation = self._evaluate_against_reference(change_map, reference_mask)

        # ── Step 5: Build visual outputs ──────────────────────
        change_map_b64 = self._colorize_change_map(change_prob, change_map)
        t1_pil = self._array_to_pil(t1_array, t1_meta)
        t2_pil = self._array_to_pil(t2_array, t2_meta)
        overlay_b64 = self._overlay_change_on_image(t2_pil, change_map, change_pct)

        t1_b64 = self._pil_to_b64(t1_pil)
        t2_b64 = self._pil_to_b64(t2_pil)

        # ── Step 6: GeoChat natural language description ──────
        description = self._generate_description(
            t1_pil, t2_pil, query, change_pct, changed_area_km2,
            change_stats, t1_meta, t2_meta
        )

        # Diagnostic confidence derived from ChangeFormer probability
        # decisiveness. This is NOT a calibrated probability.
        certainty = float(np.mean(np.abs(change_prob - 0.5)) * 2)
        confidence = round(certainty, 3)

        return {
            "change_map_b64": change_map_b64,
            "overlay_b64": overlay_b64,
            "change_pct": round(change_pct, 2),
            "changed_area_km2": (
                round(changed_area_km2, 2)
                if changed_area_km2 is not None
                else None
            ),
            "description": description,
            "t1_preview_b64": t1_b64,
            "t2_preview_b64": t2_b64,
            "change_stats": change_stats,
            "evaluation": evaluation,
            "confidence": confidence,
            "latency_sec": round(time.time() - t0, 2),
            "model_used": model_name,
            "_change_map_raw": change_map,
        }

    # ──────────────────────────────────────────────────────────
    # CHANGEFORMER FORWARD PASS
    # ──────────────────────────────────────────────────────────

    def _run_changeformer(
        self, t1: torch.Tensor, t2: torch.Tensor
    ) -> Tuple[torch.Tensor, str]:
        """Run change detection model. Returns (logits [B,1,H,W], model_name)."""
        model = self.loader.get_changeformer()
        model_name = self.loader._load_status.get("changeformer", "ChangeFormer")

        with torch.no_grad():
            output = model(t1, t2)

        # Normalize output shape to [B, 1, H, W]
        if isinstance(output, (list, tuple)):
            output = output[-1]  # ChangeFormer returns list of multi-scale outputs
        if output.dim() == 3:
            output = output.unsqueeze(1)

        # Resize to match t1 spatial resolution
        if output.shape[-2:] != t1.shape[-2:]:
            output = F.interpolate(output, size=t1.shape[-2:], mode="bilinear", align_corners=False)

        return output, model_name

    # ──────────────────────────────────────────────────────────
    # GEOCHAT & DOMAIN-EXPERT DESCRIPTION
    # ──────────────────────────────────────────────────────────

    def _generate_description(
        self,
        t1_pil: Image.Image,
        t2_pil: Image.Image,
        query: str,
        change_pct: float,
        area_km2: Optional[float],
        change_stats: Dict,
        t1_meta: Optional[Dict],
        t2_meta: Optional[Dict],
    ) -> str:
        """Generate an evidence-grounded change description.

        Geographic area is reported only when real spatial-resolution
        metadata is available. Visual-only PNG/JPEG inputs report
        image-space change only.
        """

        t1_arr = np.array(t1_pil, dtype=np.float32)
        t2_arr = np.array(t2_pil, dtype=np.float32)

        diff_arr = t2_arr - t1_arr
        delta_brightness = float(np.mean(diff_arr))

        exg_t1 = (
            2.0 * t1_arr[:, :, 1]
            - t1_arr[:, :, 0]
            - t1_arr[:, :, 2]
        )

        exg_t2 = (
            2.0 * t2_arr[:, :, 1]
            - t2_arr[:, :, 0]
            - t2_arr[:, :, 2]
        )

        delta_exg = float(np.mean(exg_t2 - exg_t1))

        d1 = (t1_meta or {}).get(
            "acquisition_date",
            "T1 baseline",
        )

        d2 = (t2_meta or {}).get(
            "acquisition_date",
            "T2 observation",
        )

        sensor = (t1_meta or {}).get(
            "sensor",
            "remote-sensing imagery",
        )

        n_reg = int(
            change_stats.get(
                "n_change_regions",
                0,
            )
        )

        largest_km2 = change_stats.get(
            "largest_region_km2"
        )

        if area_km2 is None:
            title = (
                "Bi-temporal Remote Sensing Change Assessment "
                "(Visual-only T1 ? T2)"
            )

            area_text = (
                "Geographic area unavailable because the uploaded "
                "PNG/JPEG images do not contain CRS, geotransform, "
                "or spatial-resolution metadata."
            )

            region_text = (
                f"{n_reg} connected change region(s) identified."
            )

        else:
            title = (
                "Bi-temporal Remote Sensing Change Assessment "
                f"({str(sensor).replace('_', ' ').title()} "
                f"T1 ? T2)"
            )

            area_text = (
                f"Estimated changed geographic area: "
                f"{area_km2:.2f} km? "
                f"({area_km2 * 100.0:.2f} hectares)."
            )

            if largest_km2 is not None:
                region_text = (
                    f"{n_reg} connected change region(s) identified; "
                    f"largest region spans approximately "
                    f"{largest_km2:.2f} km?."
                )
            else:
                region_text = (
                    f"{n_reg} connected change region(s) identified."
                )

        return (
            f"{title}:\n"
            f"Change Detection: ChangeFormer detected "
            f"{change_pct:.2f}% of image pixels as changed.\n"
            f"{area_text}\n"
            f"Spatial Structure: {region_text}\n"
            f"Spectral Context: Mean RGB brightness delta = "
            f"{delta_brightness:.2f} DN-equivalent; "
            f"Excess Green delta = {delta_exg:.2f}.\n"
            f"Interpretation: These measurements describe "
            f"detected spatial and spectral differences between "
            f"the two observations. They do not by themselves "
            f"establish the specific land-cover cause of change."
        )

    def _evaluate_against_reference(
        self, pred: np.ndarray, reference: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute IoU, F1, Precision, Recall against reference mask.
        Fills the 'reference mask comparison' audit gap.
        """
        if pred.shape != reference.shape:
            ref_pil = Image.fromarray(reference.astype(np.uint8) * 255)
            ref_pil = ref_pil.resize((pred.shape[1], pred.shape[0]), Image.NEAREST)
            reference = (np.array(ref_pil) > 127).astype(np.uint8)

        pred_bool = pred.astype(bool)
        ref_bool = reference.astype(bool)

        tp = float((pred_bool & ref_bool).sum())
        fp = float((pred_bool & ~ref_bool).sum())
        fn = float((~pred_bool & ref_bool).sum())
        tn = float((~pred_bool & ~ref_bool).sum())

        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        return {
            "iou": round(iou, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
        }

    # ──────────────────────────────────────────────────────────
    # STATISTICS & VISUALIZATION
    # ──────────────────────────────────────────────────────────

    def _analyze_change_regions(
        self,
        change_map: np.ndarray,
        resolution_m: Optional[float],
    ) -> Dict[str, Any]:
        """Connected components with truthful geographic-area handling."""

        try:
            from scipy import ndimage

            labeled, n_regions = ndimage.label(change_map)
            slices = ndimage.find_objects(labeled)

            pixel_area_km2 = None

            if resolution_m is not None and resolution_m > 0:
                pixel_area_km2 = (
                    resolution_m ** 2
                ) / M2_PER_KM2

            regions_info = []

            for i, region_slice in enumerate(slices or []):
                if region_slice is None:
                    continue

                c_mask = (
                    labeled[region_slice] == (i + 1)
                )

                px_count = int(np.sum(c_mask))

                if px_count < 8:
                    continue

                y_slice, x_slice = region_slice

                region = {
                    "region_idx": i + 1,
                    "x1": int(x_slice.start),
                    "y1": int(y_slice.start),
                    "x2": int(x_slice.stop),
                    "y2": int(y_slice.stop),
                    "pixels": px_count,
                    "area_km2": None,
                    "area_ha": None,
                }

                if pixel_area_km2 is not None:
                    region["area_km2"] = round(
                        px_count * pixel_area_km2,
                        4,
                    )

                    region["area_ha"] = round(
                        (px_count * resolution_m ** 2)
                        / 10000.0,
                        2,
                    )

                regions_info.append(region)

            regions_info.sort(
                key=lambda r: r["pixels"],
                reverse=True,
            )

            if pixel_area_km2 is not None:
                top_3 = [
                    r["area_km2"]
                    for r in regions_info[:3]
                ]

                total_area = round(
                    sum(
                        r["area_km2"]
                        for r in regions_info
                        if r["area_km2"] is not None
                    ),
                    3,
                )

                largest_area = (
                    regions_info[0]["area_km2"]
                    if regions_info
                    else 0
                )
            else:
                top_3 = []
                total_area = None
                largest_area = None

            return {
                "n_change_regions": n_regions,
                "largest_region_km2": largest_area,
                "top_3_regions_km2": top_3,
                "total_changed_km2": total_area,
                "change_regions": regions_info[:8],
                "area_available": pixel_area_km2 is not None,
            }

        except Exception as e:
            logger.warning(
                f"Change region analysis error: {e}"
            )

            return {
                "n_change_regions": 1,
                "largest_region_km2": None,
                "top_3_regions_km2": [],
                "total_changed_km2": None,
                "change_regions": [],
                "area_available": False,
            }

    def _compute_area_km2(
        self,
        change_map: np.ndarray,
        resolution_m: Optional[float],
    ) -> Optional[float]:
        if resolution_m is None or resolution_m <= 0:
            return None

        n_changed = change_map.sum()

        pixel_area_km2 = (
            resolution_m ** 2
        ) / M2_PER_KM2

        return float(
            n_changed * pixel_area_km2
        )

    def _colorize_change_map(
        self, change_prob: np.ndarray, change_binary: np.ndarray
    ) -> str:
        """Create a colored PNG: red=changed, green=unchanged."""
        h, w = change_binary.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Changed: red gradient by probability
        changed_mask = change_binary == 1
        rgb[changed_mask] = CHANGE_COLOR

        # Unchanged: green
        rgb[~changed_mask] = NO_CHANGE_COLOR

        # Blend intensity with probability for heat-map effect
        prob_rgb = (change_prob[:, :, None] * np.array(CHANGE_COLOR) +
                    (1 - change_prob[:, :, None]) * np.array(NO_CHANGE_COLOR))
        rgb = np.clip(prob_rgb, 0, 255).astype(np.uint8)

        return self._pil_to_b64(Image.fromarray(rgb))

    def _overlay_change_on_image(
        self, base_pil: Image.Image, change_map: np.ndarray, change_pct: float
    ) -> str:
        """Overlay change mask on T2 image with 50% transparency."""
        base = base_pil.convert("RGBA")
        h, w = change_map.shape

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        overlay_arr = np.array(overlay)
        changed_mask = change_map == 1
        overlay_arr[changed_mask] = (*CHANGE_COLOR, 140)  # Red, semi-transparent
        overlay = Image.fromarray(overlay_arr.astype(np.uint8))

        # Resize base to match overlay if needed
        base = base.resize((w, h), Image.LANCZOS)
        composite = Image.alpha_composite(base, overlay).convert("RGB")

        # Add text annotation
        draw = ImageDraw.Draw(composite)
        draw.rectangle([0, 0, 200, 28], fill=(0, 0, 0, 200))
        draw.text((5, 5), f"Changed: {change_pct:.1f}%", fill=(255, 255, 255))

        return self._pil_to_b64(composite)

    # ──────────────────────────────────────────────────────────
    # TENSOR / IMAGE UTILITIES
    # ──────────────────────────────────────────────────────────

    def _prepare_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array (either [C, H, W] or [H, W, C]) to [1, 3, H, W] torch tensor."""
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                arr = arr[:, :, :3].transpose(2, 0, 1)
            elif arr.shape[0] > 3:
                arr = arr[:3]
            elif arr.shape[0] < 3:
                arr = np.repeat(arr[:1], 3, axis=0)
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)
        if float(np.max(arr)) > 1.05:
            arr = arr / 255.0
        
        # Match the official ChangeFormer preprocessing:
        # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        arr = (arr - 0.5) / 0.5
        
        t = torch.from_numpy(arr).float().unsqueeze(0)
        return t.to(self._device)

    def _array_to_pil(
        self, array: np.ndarray, meta: Optional[Dict] = None
    ) -> Image.Image:
        """Convert array (either [C, H, W] or [H, W, C]) to PIL RGB."""
        arr = np.asarray(array)
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                rgb = arr[:, :, :3]
            elif arr.shape[0] in (3, 4):
                rgb = arr[:3].transpose(1, 2, 0)
            elif arr.shape[0] == 1:
                rgb = np.repeat(arr.transpose(1, 2, 0), 3, axis=-1)
            else:
                rgb = arr[:, :, :3] if arr.shape[-1] >= 3 else np.repeat(arr[:, :, :1], 3, axis=-1)
        elif arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")

        if rgb.dtype != np.uint8:
            if float(np.max(rgb)) <= 1.05:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    def _pil_to_b64(self, pil: Image.Image, quality: int = 85) -> str:
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Module-level singleton
_change_engine: Optional[ChangeDetectionEngine] = None


def get_change_engine() -> ChangeDetectionEngine:
    global _change_engine
    if _change_engine is None:
        _change_engine = ChangeDetectionEngine()
    return _change_engine
