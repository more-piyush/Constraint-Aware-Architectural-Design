"""Tests for data models."""

import pytest
from pydantic import ValidationError

from car.models.constraints import (
    RegulatoryConstraints,
    SeismicZone,
    SiteConstraints,
)
from car.models.design import (
    AestheticFeel,
    BuildingDesign,
    DesignIntent,
    RoofType,
    StructuralSystem,
    ViewPriority,
    WallType,
    WindowSize,
)
from car.models.results import ComplianceResult, ConstraintViolation, DesignIteration


class TestConstraintModels:
    def test_seismic_zone_range(self):
        assert SeismicZone.ZONE_0.value == 0
        assert SeismicZone.ZONE_5.value == 5

    def test_regulatory_constraints_validation(self):
        with pytest.raises(ValidationError):
            RegulatoryConstraints(
                far_limit=-1.0,  # must be > 0
                height_limit_m=10.0,
                setback_front_m=0,
                setback_side_m=0,
                setback_rear_m=0,
            )

    def test_site_constraints_creation(self, residential_constraints):
        assert residential_constraints.site_area_sqm == 500.0
        assert residential_constraints.regulatory.far_limit == 0.6
        assert residential_constraints.geophysical.seismic_zone == SeismicZone.ZONE_2


class TestDesignModels:
    def test_design_intent_defaults(self):
        intent = DesignIntent(
            aesthetic_feel=AestheticFeel.MINIMALIST,
            view_priority=ViewPriority.HIGH,
        )
        assert intent.sustainability_priority == ViewPriority.MEDIUM
        assert intent.budget_level == "medium"

    def test_building_design_creation(self):
        design = BuildingDesign(
            num_floors=3,
            floor_area_sqm=150.0,
            building_height_m=10.5,
            structural_system=StructuralSystem.STEEL_FRAME,
            wall_type=WallType.CURTAIN_WALL,
            wall_thickness_mm=200.0,
            window_size=WindowSize.LARGE,
            roof_type=RoofType.FLAT,
            primary_material="steel",
            footprint_width_m=12.0,
            footprint_depth_m=12.5,
        )
        assert design.num_floors == 3
        assert design.structural_system == StructuralSystem.STEEL_FRAME


class TestResultModels:
    def test_compliance_result(self):
        result = ComplianceResult(
            is_compliant=True,
            confidence_score=0.85,
            violations=[],
            checked_constraints_count=6,
            passed_constraints_count=6,
        )
        assert result.is_compliant
        assert result.confidence_score == 0.85

    def test_compliance_with_violation(self):
        violation = ConstraintViolation(
            constraint_name="Height",
            constraint_type="regulatory",
            required_value="<= 10m",
            actual_value="12m",
            severity="hard",
        )
        result = ComplianceResult(
            is_compliant=False,
            confidence_score=0.5,
            violations=[violation],
            checked_constraints_count=6,
            passed_constraints_count=5,
        )
        assert not result.is_compliant
        assert len(result.violations) == 1
