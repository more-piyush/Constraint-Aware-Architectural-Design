"""Probabilistic confidence scoring for bylaw compliance.

The confidence score C is a weighted composite:
    C = w_d * D + w_p * P + w_m * M

Where:
- D = Deterministic compliance ratio (hard constraint pass rate)
- P = Probabilistic margin (how far from constraint boundaries)
- M = Model confidence (posterior probability of the MAP assignment)
"""

from __future__ import annotations

from car.config import (
    WEIGHT_DETERMINISTIC,
    WEIGHT_MODEL_CONFIDENCE,
    WEIGHT_PROBABILISTIC_MARGIN,
)
from car.models.constraints import SiteConstraints
from car.models.design import BuildingDesign
from car.models.results import ComplianceResult


class ConfidenceScorer:
    """Computes a probabilistic confidence score for bylaw compliance."""

    def __init__(
        self,
        weight_deterministic: float = WEIGHT_DETERMINISTIC,
        weight_probabilistic_margin: float = WEIGHT_PROBABILISTIC_MARGIN,
        weight_model_confidence: float = WEIGHT_MODEL_CONFIDENCE,
    ) -> None:
        self._w_d = weight_deterministic
        self._w_p = weight_probabilistic_margin
        self._w_m = weight_model_confidence

    def score(
        self,
        design: BuildingDesign,
        constraints: SiteConstraints,
        compliance_result: ComplianceResult,
        marginal_probs: dict[str, dict[str, float]],
    ) -> float:
        """Compute the composite confidence score in [0, 1]."""
        D = (
            compliance_result.passed_constraints_count
            / compliance_result.checked_constraints_count
            if compliance_result.checked_constraints_count > 0
            else 1.0
        )

        P = self._compute_probabilistic_margin(design, constraints)
        M = self._compute_model_confidence(marginal_probs, design)

        raw_score = self._w_d * D + self._w_p * P + self._w_m * M
        return max(0.0, min(1.0, raw_score))

    def _compute_probabilistic_margin(
        self, design: BuildingDesign, constraints: SiteConstraints
    ) -> float:
        """Compute how far the design is from constraint boundaries.

        A design exactly at a boundary scores 0; well within scores 1.
        """
        margins = []

        # FAR margin
        actual_far = (design.floor_area_sqm * design.num_floors) / constraints.site_area_sqm
        far_margin = 1.0 - (actual_far / constraints.regulatory.far_limit)
        margins.append(max(0.0, min(1.0, far_margin)))

        # Height margin
        height_margin = 1.0 - (design.building_height_m / constraints.regulatory.height_limit_m)
        margins.append(max(0.0, min(1.0, height_margin)))

        # Wall thickness margin (distance from bounds, normalized)
        min_t = constraints.technical.wall_thickness_min_mm
        max_t = constraints.technical.wall_thickness_max_mm
        range_t = max_t - min_t
        if range_t > 0:
            center = (min_t + max_t) / 2
            dist_from_center = abs(design.wall_thickness_mm - center)
            thickness_margin = 1.0 - (dist_from_center / (range_t / 2))
            margins.append(max(0.0, min(1.0, thickness_margin)))

        return sum(margins) / len(margins) if margins else 1.0

    def _compute_model_confidence(
        self, marginal_probs: dict[str, dict[str, float]], design: BuildingDesign
    ) -> float:
        """Compute how confident the model is in the chosen assignment.

        Uses the average marginal probability of the MAP states.
        """
        design_to_bn = {
            "structural_system": design.structural_system.value,
            "window_size": design.window_size.value,
            "wall_type": design.wall_type.value,
            "roof_type": design.roof_type.value,
        }

        probs = []
        for var_name, chosen_state in design_to_bn.items():
            if var_name in marginal_probs and chosen_state in marginal_probs[var_name]:
                probs.append(marginal_probs[var_name][chosen_state])

        return sum(probs) / len(probs) if probs else 0.5
