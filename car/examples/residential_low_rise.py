"""Example: Single-family residential home in Los Angeles.

- Suburban lot, 500 sqm
- Seismic zone 2 (moderate)
- FAR limit 0.6
- Minimalist aesthetic, high view priority
- Timber and steel available
"""

from __future__ import annotations

import click

from car.models.constraints import (
    EnvironmentalConstraints,
    GeophysicalConstraints,
    MaterialProperties,
    RegulatoryConstraints,
    SeismicZone,
    SiteConstraints,
    TechnicalConstraints,
)
from car.models.design import AestheticFeel, DesignIntent, ViewPriority


def get_constraints() -> SiteConstraints:
    return SiteConstraints(
        site_area_sqm=500.0,
        regulatory=RegulatoryConstraints(
            far_limit=0.6,
            height_limit_m=10.0,
            is_airport_zone=False,
            setback_front_m=6.0,
            setback_side_m=3.0,
            setback_rear_m=5.0,
            min_parking_spaces=2,
            fire_escape_required=False,
        ),
        environmental=EnvironmentalConstraints(
            latitude=34.05,
            longitude=-118.24,
            solar_azimuth_peak_deg=180.0,
            solar_elevation_peak_deg=73.0,
            prevailing_wind_direction_deg=270.0,
            prevailing_wind_speed_kmh=12.0,
            annual_rainfall_mm=380.0,
        ),
        geophysical=GeophysicalConstraints(
            seismic_zone=SeismicZone.ZONE_2,
            soil_bearing_capacity_kpa=200.0,
            water_table_depth_m=8.0,
        ),
        technical=TechnicalConstraints(
            available_materials=[
                MaterialProperties(
                    name="timber",
                    youngs_modulus_gpa=12.0,
                    thermal_mass_kj_per_m3k=750.0,
                    density_kg_per_m3=500.0,
                    cost_per_m3_usd=600.0,
                ),
                MaterialProperties(
                    name="steel",
                    youngs_modulus_gpa=200.0,
                    thermal_mass_kj_per_m3k=3500.0,
                    density_kg_per_m3=7850.0,
                    cost_per_m3_usd=5000.0,
                ),
            ],
            wall_thickness_min_mm=100.0,
            wall_thickness_max_mm=250.0,
            floor_to_floor_height_min_m=2.7,
            floor_to_floor_height_max_m=3.2,
        ),
    )


def get_intent() -> DesignIntent:
    return DesignIntent(
        aesthetic_feel=AestheticFeel.MINIMALIST,
        view_priority=ViewPriority.HIGH,
        sustainability_priority=ViewPriority.MEDIUM,
        budget_level="medium",
    )


def run(output_dir: str = "./output", method: str = "map") -> None:
    """Run the residential low-rise example."""
    click.echo("=" * 60)
    click.echo("EXAMPLE: Residential Low-Rise (Los Angeles)")
    click.echo("  Minimalist aesthetic | High view priority")
    click.echo("  500 sqm lot | FAR 0.6 | Seismic Zone 2")
    click.echo("=" * 60)

    constraints = get_constraints()
    intent = get_intent()

    from car.inference.pipeline import InferencePipeline, PipelineConfig

    config = PipelineConfig(inference_method=method)
    pipeline = InferencePipeline(config)
    result = pipeline.run(constraints, intent)

    _print_result(result, constraints, output_dir)


def _print_result(result, constraints, output_dir):
    from pathlib import Path
    from car.network.builder import NetworkBuilder
    from car.visualization.network_plot import NetworkPlotter
    from car.visualization.design_plot import DesignPlotter
    from car.visualization.compliance_report import ComplianceReportGenerator

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    builder = NetworkBuilder()
    model = builder.build()
    topology = builder.build_topology()

    ng = output_path / "network_graph.png"
    fp = output_path / "floor_plan.png"
    bs = output_path / "building_section.png"
    rp = output_path / "compliance_report.html"

    NetworkPlotter().plot(model, topology, ng)
    DesignPlotter().plot_floor_plan(result.map_design.design, fp)
    DesignPlotter().plot_building_section(result.map_design.design, bs)
    ComplianceReportGenerator().generate(result, constraints, rp, ng, fp)

    if result.sampled_designs:
        DesignPlotter().plot_design_comparison(
            result.sampled_designs, output_path / "design_comparison.png"
        )

    md = result.map_design
    click.echo(f"\nConfidence: {md.compliance.confidence_score:.1%}")
    click.echo(f"Compliant:  {'YES' if md.compliance.is_compliant else 'NO'}")
    click.echo(f"Structure:  {md.design.structural_system.value}")
    click.echo(f"Floors:     {md.design.num_floors} ({md.design.building_height_m:.1f}m)")
    click.echo(f"Windows:    {md.design.window_size.value}")
    click.echo(f"Roof:       {md.design.roof_type.value}")
    click.echo(f"Output:     {output_path.resolve()}")
