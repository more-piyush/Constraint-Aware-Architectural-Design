"""Tests for compliance checking and confidence scoring."""

import pytest

from car.models.design import (
    BuildingDesign,
    RoofType,
    StructuralSystem,
    WallType,
    WindowSize,
)
from car.models.results import ComplianceResult
from car.scoring.compliance import ComplianceChecker
from car.scoring.confidence import ConfidenceScorer


def _make_design(**kwargs) -> BuildingDesign:
    """Helper to create a BuildingDesign with defaults."""
    defaults = dict(
        num_floors=2,
        floor_area_sqm=120.0,
        building_height_m=7.0,
        structural_system=StructuralSystem.STEEL_FRAME,
        wall_type=WallType.CURTAIN_WALL,
        wall_thickness_mm=200.0,
        window_size=WindowSize.LARGE,
        roof_type=RoofType.FLAT,
        primary_material="steel",
        footprint_width_m=10.0,
        footprint_depth_m=12.0,
    )
    defaults.update(kwargs)
    return BuildingDesign(**defaults)


class TestComplianceChecker:
    def test_compliant_design(self, residential_constraints):
        design = _make_design(
            num_floors=1,
            floor_area_sqm=150.0,
            building_height_m=3.0,
        )
        checker = ComplianceChecker()
        result = checker.check(design, residential_constraints)
        assert result.is_compliant

    def test_far_violation(self, residential_constraints):
        # FAR limit is 0.6, site is 500 sqm -> max total area = 300 sqm
        # 2 floors * 200 sqm = 400 sqm -> FAR = 0.8 > 0.6
        design = _make_design(
            num_floors=2,
            floor_area_sqm=200.0,
            building_height_m=7.0,
        )
        checker = ComplianceChecker()
        result = checker.check(design, residential_constraints)
        far_violations = [v for v in result.violations if v.constraint_name == "Floor Area Ratio"]
        assert len(far_violations) == 1
        assert far_violations[0].severity == "hard"

    def test_height_violation(self, residential_constraints):
        design = _make_design(
            building_height_m=15.0,  # limit is 10m
        )
        checker = ComplianceChecker()
        result = checker.check(design, residential_constraints)
        height_violations = [v for v in result.violations if v.constraint_name == "Building Height"]
        assert len(height_violations) == 1

    def test_seismic_violation(self, commercial_constraints):
        design = _make_design(
            structural_system=StructuralSystem.TIMBER_FRAME,
        )
        checker = ComplianceChecker()
        result = checker.check(design, commercial_constraints)
        seismic_violations = [
            v for v in result.violations
            if v.constraint_name == "Seismic Zone Material Restriction"
        ]
        assert len(seismic_violations) == 1
        assert not result.is_compliant


class TestConfidenceScorer:
    def test_perfect_compliance_high_confidence(self, residential_constraints):
        design = _make_design(
            num_floors=1,
            floor_area_sqm=100.0,
            building_height_m=3.0,
            wall_thickness_mm=175.0,
        )
        compliance = ComplianceResult(
            is_compliant=True,
            confidence_score=1.0,
            violations=[],
            checked_constraints_count=6,
            passed_constraints_count=6,
        )
        marginals = {
            "structural_system": {"steel_frame": 0.8, "reinforced_concrete": 0.1,
                                   "timber_frame": 0.05, "masonry": 0.03,
                                   "hybrid_steel_concrete": 0.02},
            "window_size": {"large": 0.6, "full_glass": 0.2, "medium": 0.15, "small": 0.05},
            "wall_type": {"curtain_wall": 0.7, "partition": 0.2, "load_bearing": 0.1},
            "roof_type": {"flat": 0.6, "pitched": 0.25, "green_roof": 0.15},
        }

        scorer = ConfidenceScorer()
        score = scorer.score(design, residential_constraints, compliance, marginals)
        assert score > 0.7, f"Perfect compliance should give high confidence, got {score}"

    def test_score_in_valid_range(self, residential_constraints):
        design = _make_design()
        compliance = ComplianceResult(
            is_compliant=True,
            confidence_score=0.8,
            violations=[],
            checked_constraints_count=6,
            passed_constraints_count=5,
        )
        scorer = ConfidenceScorer()
        score = scorer.score(design, residential_constraints, compliance, {})
        assert 0.0 <= score <= 1.0
