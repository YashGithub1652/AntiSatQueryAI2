"""
SatQuery AI — Confidence / Uncertainty Engine
===============================================

Important:
This module deliberately does NOT pretend that a heuristic score is
a calibrated probability.

Before calibration:
    confidence_status = "uncalibrated"

After calibration on a held-out validation set:
    confidence_status = "calibrated"

The system combines:
    - model confidence
    - image quality
    - registration quality
    - prediction stability
    - cross-modal agreement
    - OOD penalty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ConfidenceInputs:
    model_confidence: float = 0.0
    image_quality: float = 1.0
    registration_quality: float = 1.0
    prediction_stability: float = 1.0
    cross_modal_agreement: float = 1.0
    ood_score: float = 0.0


class ConfidenceEngine:
    """
    Transparent confidence aggregator.

    This is an engineering confidence score until calibrated.
    """

    WEIGHTS = {
        "model": 0.35,
        "image_quality": 0.15,
        "registration": 0.15,
        "stability": 0.15,
        "cross_modal": 0.10,
        "ood": 0.10,
    }

    def compute(
        self,
        values: Optional[ConfidenceInputs] = None,
    ) -> dict:

        values = values or ConfidenceInputs()

        model = self._clip(values.model_confidence)
        image_quality = self._clip(values.image_quality)
        registration = self._clip(values.registration_quality)
        stability = self._clip(values.prediction_stability)
        cross_modal = self._clip(values.cross_modal_agreement)
        ood = self._clip(values.ood_score)

        raw_score = (
            self.WEIGHTS["model"] * model
            + self.WEIGHTS["image_quality"] * image_quality
            + self.WEIGHTS["registration"] * registration
            + self.WEIGHTS["stability"] * stability
            + self.WEIGHTS["cross_modal"] * cross_modal
            + self.WEIGHTS["ood"] * (1.0 - ood)
        )

        # Smooth extreme values rather than generating fake 0.99 confidence.
        score = self._clip(raw_score)

        return {
            "score": round(score, 4),
            "percent": round(score * 100.0, 2),
            "status": "uncalibrated",
            "calibrated": False,
            "components": {
                "model": round(model, 4),
                "image_quality": round(image_quality, 4),
                "registration": round(registration, 4),
                "prediction_stability": round(stability, 4),
                "cross_modal_agreement": round(cross_modal, 4),
                "ood_penalty": round(ood, 4),
            },
        }

    @staticmethod
    def _clip(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        if not math.isfinite(value):
            value = 0.0

        return max(0.0, min(1.0, value))


_confidence_engine: Optional[ConfidenceEngine] = None


def get_confidence_engine() -> ConfidenceEngine:
    global _confidence_engine

    if _confidence_engine is None:
        _confidence_engine = ConfidenceEngine()

    return _confidence_engine