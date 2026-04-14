"""Example: Mixed-use urban building in Copenhagen.

- Medium site, 800 sqm
- Seismic zone 1 (low risk)
- FAR limit 3.5, height limit 25m
- Organic aesthetic, high sustainability priority
- All material types available
- Generous setbacks with courtyard preference
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
        site_area_sqm=800.0,
        regulatory=RegulatoryConstraints(
            far_limit=3.5,
            height_limit_m=25.0,
            is_airport_zone=False,
            setback_front_m=5.0,
            setback_side_m=4.0,
            setback_rear_m=5.0,
            min_parking_spaces=15,
            fire_escape_required=True,
        ),
        environmental=EnvironmentalConstraints(
            latitude=55.68,
            longitude=12.57,
            solar_azimuth_peak_deg=180.0,
            solar_elevation_peak_deg=58.0,
            prevailing_wind_direction_deg=250.0,
            prevailing_wind_speed_kmh=22.0,
            annual_rainfall_mm=600.0,
        ),
        geophysical=GeophysicalConstraints(
            seismic_zone=SeismicZone.ZONE_1,
            soil_bearing_capacity_kpa=250.0,
            water_table_depth_m=4.0,
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
                    name="concrete",
                    youngs_modulus_gpa=30.0,
                    thermal_mass_kj_per_m3k=2000.0,
                    density_kg_per_m3=2400.0,
                    cost_per_m3_usd=150.0,
                ),
                MaterialProperties(
                    name="steel",
                    youngs_modulus_gpa=200.0,
                    thermal_mass_kj_per_m3k=3500.0,
                    density_kg_per_m3=7850.0,
                    cost_per_m3_usd=5000.0,
                ),
                MaterialProperties(
                    name="masonry",
                    youngs_modulus_gpa=20.0,
                    thermal_mass_kj_per_m3k=1800.0,
                    density_kg_per_m3=2000.0,
                    cost_per_m3_usd=200.0,
                ),
            ],
            wall_thickness_min_mm=120.0,
            wall_thickness_max_mm=400.0,
            floor_to_floor_height_min_m=2.8,
            floor_to_floor_height_max_m=4.0,
        ),
    )


def get_intent() -> DesignIntent:
    return DesignIntent(
        aesthetic_feel=AestheticFeel.ORGANIC,
        view_priority=ViewPriority.MEDIUM,
        sustainability_priority=ViewPriority.HIGH,
        budget_level="medium",
    )


def run(output_dir: str = "./output", method: str = "map") -> None:
    """Run the mixed-use urban example."""
    click.echo("=" * 60)
    click.echo("EXAMPLE: Mixed-Use Urban (Copenhagen)")
    click.echo("  Organic aesthetic | High sustainability | Medium views")
    click.echo("  800 sqm lot | FAR 3.5 | Seismic Zone 1")
    click.echo("=" * 60)

    constraints = get_constraints()
    intent = get_intent()

    from car.inference.pipeline import InferencePipeline, PipelineConfig

    config = PipelineConfig(inference_method=method)
    pipeline = InferencePipeline(config)
    result = pipeline.run(constraints, intent)

    from car.examples.residential_low_rise import _print_result
    _print_result(result, constraints, output_dir)
