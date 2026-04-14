"""Tests for visualization modules."""

import pytest
import tempfile
from pathlib import Path

from car.models.design import (
    BuildingDesign,
    RoofType,
    StructuralSystem,
    WallType,
    WindowSize,
)
from car.models.results import ComplianceResult, DesignIteration
from car.network.builder import NetworkBuilder
from car.visualization.network_plot import NetworkPlotter
from car.visualization.design_plot import DesignPlotter


def _make_design(**kwargs) -> BuildingDesign:
    defaults = dict(
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
    defaults.update(kwargs)
    return BuildingDesign(**defaults)


class TestNetworkPlotter:
    def test_plot_creates_file(self):
        builder = NetworkBuilder()
        model = builder.build()
        topology = builder.build_topology()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_network.png"
            NetworkPlotter().plot(model, topology, output)
            assert output.exists()
            assert output.stat().st_size > 0


class TestDesignPlotter:
    def test_floor_plan_creates_file(self):
        design = _make_design()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_floor_plan.png"
            DesignPlotter().plot_floor_plan(design, output)
            assert output.exists()
            assert output.stat().st_size > 0

    def test_building_section_creates_file(self):
        design = _make_design()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_section.png"
            DesignPlotter().plot_building_section(design, output)
            assert output.exists()

    def test_design_comparison(self):
        compliance = ComplianceResult(
            is_compliant=True, confidence_score=0.9,
            violations=[], checked_constraints_count=6, passed_constraints_count=6,
        )
        designs = [
            DesignIteration(
                iteration_id=i,
                design=_make_design(num_floors=i + 1, building_height_m=(i + 1) * 3.5),
                compliance=compliance,
                overall_score=0.8 - i * 0.1,
            )
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_comparison.png"
            DesignPlotter().plot_design_comparison(designs, output)
            assert output.exists()
