"""Example: Commercial office tower in Tokyo airport zone.

- Large site, 2000 sqm
- Seismic zone 4 (high risk)
- FAR limit 8.0, height limit 45m, airport zone
- Industrial aesthetic, medium view priority
- Steel and concrete available
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
        site_area_sqm=2000.0,
        regulatory=RegulatoryConstraints(
            far_limit=8.0,
            height_limit_m=45.0,
            is_airport_zone=True,
            setback_front_m=8.0,
            setback_side_m=5.0,
            setback_rear_m=6.0,
            min_parking_spaces=50,
            fire_escape_required=True,
        ),
        environmental=EnvironmentalConstraints(
            latitude=35.68,
            longitude=139.69,
            solar_azimuth_peak_deg=180.0,
            solar_elevation_peak_deg=78.0,
            prevailing_wind_direction_deg=315.0,
            prevailing_wind_speed_kmh=18.0,
            annual_rainfall_mm=1530.0,
        ),
        geophysical=GeophysicalConstraints(
            seismic_zone=SeismicZone.ZONE_4,
            soil_bearing_capacity_kpa=180.0,
            water_table_depth_m=3.0,
        ),
        technical=TechnicalConstraints(
            available_materials=[
                MaterialProperties(
                    name="steel",
                    youngs_modulus_gpa=200.0,
                    thermal_mass_kj_per_m3k=3500.0,
                    density_kg_per_m3=7850.0,
                    cost_per_m3_usd=5000.0,
                ),
                MaterialProperties(
                    name="concrete",
                    youngs_modulus_gpa=30.0,
                    thermal_mass_kj_per_m3k=2000.0,
                    density_kg_per_m3=2400.0,
                    cost_per_m3_usd=150.0,
                ),
            ],
            wall_thickness_min_mm=150.0,
            wall_thickness_max_mm=450.0,
            floor_to_floor_height_min_m=3.0,
            floor_to_floor_height_max_m=4.5,
        ),
    )


def get_intent() -> DesignIntent:
    return DesignIntent(
        aesthetic_feel=AestheticFeel.INDUSTRIAL,
        view_priority=ViewPriority.MEDIUM,
        sustainability_priority=ViewPriority.LOW,
        budget_level="high",
    )


def run(output_dir: str = "./output", method: str = "map") -> None:
    """Run the commercial high-rise example."""
    click.echo("=" * 60)
    click.echo("EXAMPLE: Commercial High-Rise (Tokyo Airport Zone)")
    click.echo("  Industrial aesthetic | Medium view priority")
    click.echo("  2000 sqm lot | FAR 8.0 | Seismic Zone 4 | Airport")
    click.echo("=" * 60)

    constraints = get_constraints()
    intent = get_intent()

    from car.inference.pipeline import InferencePipeline, PipelineConfig

    config = PipelineConfig(inference_method=method)
    pipeline = InferencePipeline(config)
    result = pipeline.run(constraints, intent)

    from car.examples.residential_low_rise import _print_result
    _print_result(result, constraints, output_dir)
