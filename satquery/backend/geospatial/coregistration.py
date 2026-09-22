"""
SatQuery AI — Spatial Compatibility + Registration
====================================================

Two levels are performed:

LEVEL 1
-------
Geospatial compatibility:
    - CRS
    - footprint overlap
    - resolution
    - acquisition date

LEVEL 2
-------
Image-space registration refinement:
    - phase cross-correlation
    - sub-pixel translation estimate
    - registration error
    - aligned moving image

Important:
Phase correlation estimates translation. It is NOT a substitute for
full orthorectification/reprojection when the source products are
geometrically incompatible.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from scipy.ndimage import shift as ndi_shift

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage.registration import phase_cross_correlation

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


MIN_OVERLAP_FRACTION = 0.80
MAX_RESOLUTION_RATIO = 10.0


class CoregistrationChecker:

    def check(
        self,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any],
        task: str = "bitemporal",
    ) -> Dict[str, Any]:

        warnings: List[str] = []
        errors: List[str] = []
        checks_passed: List[str] = []

        is_geo1 = meta1.get("bbox_wgs84") is not None
        is_geo2 = meta2.get("bbox_wgs84") is not None

        if not is_geo1 or not is_geo2:

            message = (
                "Both images contain sufficient geospatial metadata."
                if task != "sar_optical"
                else
                "SAR-optical fusion requires GeoTIFF inputs with CRS "
                "and footprint metadata."
            )

            if task == "sar_optical":
                errors.append(message)
            else:
                warnings.append(
                    "One or both images lack geospatial metadata. "
                    "Only image-space registration can be attempted."
                )

            return self._result(
                False,
                0.0,
                False,
                1.0,
                warnings,
                errors,
                "Upload georeferenced GeoTIFF products for geospatially "
                "validated analysis.",
            )

        # ---------------------------------------------------------
        # CRS
        # ---------------------------------------------------------

        crs1 = meta1.get("crs", "")
        crs2 = meta2.get("crs", "")

        crs_match = self._crs_equal(crs1, crs2)

        if crs_match:
            checks_passed.append("CRS_MATCH")
        else:
            warnings.append(
                f"CRS mismatch: {crs1} vs {crs2}. "
                "Images should be reprojected to a common CRS before "
                "scientific pixel-level comparison."
            )

        # ---------------------------------------------------------
        # Spatial overlap
        # ---------------------------------------------------------

        bbox1 = meta1.get("bbox_wgs84")
        bbox2 = meta2.get("bbox_wgs84")

        overlap_pct = self._compute_overlap(bbox1, bbox2)

        if overlap_pct < MIN_OVERLAP_FRACTION * 100:

            errors.append(
                f"Spatial overlap is only {overlap_pct:.2f}%. "
                f"Required minimum is {MIN_OVERLAP_FRACTION * 100:.0f}%."
            )

        elif overlap_pct < 95:

            warnings.append(
                f"Partial footprint overlap: {overlap_pct:.2f}%. "
                "Analysis should be restricted to the common footprint."
            )

        else:

            checks_passed.append(
                f"OVERLAP_{overlap_pct:.1f}%"
            )

        # ---------------------------------------------------------
        # Resolution
        # ---------------------------------------------------------

        res1 = float(meta1.get("resolution_m") or 10.0)
        res2 = float(meta2.get("resolution_m") or 10.0)

        ratio = max(res1, res2) / max(min(res1, res2), 1e-6)

        if ratio > MAX_RESOLUTION_RATIO:

            warnings.append(
                f"Resolution ratio is {ratio:.2f}x "
                f"({res1}m vs {res2}m)."
            )

        else:

            checks_passed.append(
                f"RESOLUTION_RATIO_{ratio:.2f}x"
            )

        # ---------------------------------------------------------
        # Temporal
        # ---------------------------------------------------------

        if task == "bitemporal":

            date1 = meta1.get("acquisition_date")
            date2 = meta2.get("acquisition_date")

            if date1 and date2:

                if date1 == date2:

                    warnings.append(
                        f"Both images have acquisition date {date1}. "
                        "A temporal change analysis normally requires "
                        "different acquisition dates."
                    )

                else:

                    checks_passed.append(
                        f"TEMPORAL_{date1}_VS_{date2}"
                    )

            else:

                warnings.append(
                    "Acquisition date missing from one or both images."
                )

        passed = (
            len(errors) == 0
            and overlap_pct >= MIN_OVERLAP_FRACTION * 100
        )

        if passed:

            recommendation = (
                f"Geospatial compatibility passed for "
                f"{task}. {len(checks_passed)} checks passed."
            )

        else:

            recommendation = (
                "Geospatial compatibility failed. "
                "Scientific pixel-level comparison should not proceed."
            )

        return self._result(
            passed,
            overlap_pct,
            crs_match,
            ratio,
            warnings,
            errors,
            recommendation,
        )

    # ============================================================
    # IMAGE-SPACE REGISTRATION
    # ============================================================

    def refine_registration(
        self,
        reference: np.ndarray,
        moving: np.ndarray,
        *,
        upsample_factor: int = 20,
        apply_alignment: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        """
        Estimate translation between reference and moving image.

        Inputs:
            [C,H,W] or [H,W]

        Returns:
            aligned_moving
            registration_metrics
        """

        if reference is None or moving is None:

            return moving, {
                "performed": False,
                "reason": "Missing image array",
                "quality": 0.0,
            }

        if not SKIMAGE_AVAILABLE:

            return moving, {
                "performed": False,
                "reason": "scikit-image not installed",
                "quality": 0.0,
            }

        ref_gray = self._registration_image(reference)
        mov_gray = self._registration_image(moving)

        if ref_gray.shape != mov_gray.shape:

            return moving, {
                "performed": False,
                "reason": (
                    f"Shape mismatch: "
                    f"{ref_gray.shape} vs {mov_gray.shape}"
                ),
                "quality": 0.0,
            }

        try:

            shift, error, phase_difference = phase_cross_correlation(
                ref_gray,
                mov_gray,
                upsample_factor=upsample_factor,
            )

            shift_y = float(shift[0])
            shift_x = float(shift[1])

            # Lower error = better registration.
            quality = float(
                np.exp(-min(max(float(error), 0.0), 5.0))
            )

            aligned = moving

            if apply_alignment and SCIPY_AVAILABLE:

                if moving.ndim == 3:

                    aligned_channels = []

                    for channel in moving:

                        aligned_channel = ndi_shift(
                            channel,
                            shift=(shift_y, shift_x),
                            order=1,
                            mode="nearest",
                            prefilter=False,
                        )

                        aligned_channels.append(aligned_channel)

                    aligned = np.stack(
                        aligned_channels,
                        axis=0,
                    ).astype(np.float32)

                else:

                    aligned = ndi_shift(
                        moving,
                        shift=(shift_y, shift_x),
                        order=1,
                        mode="nearest",
                        prefilter=False,
                    ).astype(np.float32)

            return aligned, {
                "performed": True,
                "shift_y_px": round(shift_y, 4),
                "shift_x_px": round(shift_x, 4),
                "registration_error": round(float(error), 6),
                "phase_difference": round(
                    float(phase_difference), 6
                ),
                "quality": round(quality, 4),
                "method": "phase_cross_correlation",
                "upsample_factor": upsample_factor,
                "aligned": bool(apply_alignment and SCIPY_AVAILABLE),
            }

        except Exception as exc:

            logger.exception(
                "Image registration failed"
            )

            return moving, {
                "performed": False,
                "reason": str(exc),
                "quality": 0.0,
            }

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _registration_image(array: np.ndarray) -> np.ndarray:

        if array.ndim == 2:

            image = array

        elif array.ndim == 3:

            channels = array.shape[0]

            if channels == 1:

                image = array[0]

            elif channels >= 3:

                image = (
                    0.299 * array[0]
                    + 0.587 * array[1]
                    + 0.114 * array[2]
                )

            else:

                image = np.mean(array, axis=0)

        else:

            raise ValueError(
                f"Unsupported image shape: {array.shape}"
            )

        image = image.astype(np.float32)

        image -= np.nanmean(image)

        std = np.nanstd(image)

        if std > 1e-8:
            image /= std

        image = np.nan_to_num(image)

        return image

    @staticmethod
    def _compute_overlap(
        bbox1: Optional[List[float]],
        bbox2: Optional[List[float]],
    ) -> float:

        if not bbox1 or not bbox2:
            return 0.0

        w1, s1, e1, n1 = bbox1
        w2, s2, e2, n2 = bbox2

        iw = max(w1, w2)
        isouth = max(s1, s2)
        ie = min(e1, e2)
        inn = min(n1, n2)

        if ie <= iw or inn <= isouth:
            return 0.0

        intersection = (
            (ie - iw) * (inn - isouth)
        )

        area1 = (e1 - w1) * (n1 - s1)
        area2 = (e2 - w2) * (n2 - s2)

        smaller_area = min(area1, area2)

        if smaller_area <= 0:
            return 0.0

        return min(
            100.0,
            (intersection / smaller_area) * 100.0,
        )

    @staticmethod
    def _crs_equal(
        crs1_str: str,
        crs2_str: str,
    ) -> bool:

        if not crs1_str or not crs2_str:
            return False

        c1 = (
            str(crs1_str)
            .strip()
            .upper()
            .replace("EPSG:", "")
            .replace(" ", "")
        )

        c2 = (
            str(crs2_str)
            .strip()
            .upper()
            .replace("EPSG:", "")
            .replace(" ", "")
        )

        if c1 == c2:
            return True

        if RASTERIO_AVAILABLE:

            try:

                return (
                    CRS.from_string(str(crs1_str))
                    == CRS.from_string(str(crs2_str))
                )

            except Exception:
                pass

        return False

    @staticmethod
    def _result(
        is_coregistered: bool,
        overlap_pct: float,
        crs_match: bool,
        resolution_ratio: float,
        warnings: List[str],
        errors: List[str],
        recommendation: str,
    ) -> Dict[str, Any]:

        return {
            "passed": bool(
                is_coregistered
                and not errors
            ),
            "is_coregistered": bool(
                is_coregistered
            ),
            "message": recommendation,
            "overlap_pct": round(
                overlap_pct, 2
            ),
            "crs_match": bool(crs_match),
            "resolution_ratio": round(
                resolution_ratio, 3
            ),
            "warnings": warnings,
            "errors": errors,
            "recommendation": recommendation,
            "checks_summary": {
                "spatial_overlap": round(
                    overlap_pct, 2
                ),
                "crs_compatible": crs_match,
                "resolution_compatible":
                    resolution_ratio
                    <= MAX_RESOLUTION_RATIO,
            },
        }


_checker: Optional[CoregistrationChecker] = None


def get_checker() -> CoregistrationChecker:

    global _checker

    if _checker is None:
        _checker = CoregistrationChecker()

    return _checker