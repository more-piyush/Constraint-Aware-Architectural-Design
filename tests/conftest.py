"""Shared test fixtures for CAR."""

import pytest

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


@pytest.fixture
def residential_constraints() -> SiteConstraints:
    """Standard residential site constraints."""
    return SiteConstraints(
        site_area_sqm=500.0,
        regulatory=RegulatoryConstraints(
            far_limit=0.6,
            height_limit_m=10.0,
            is_airport_zone=False,
            setback_front_m=6.0,
            setback_side_m=3.0,
            setback_rear_m=5.0,
        ),
        environmental=EnvironmentalConstraints(
            latitude=34.05,
            longitude=-118.24,
            solar_azimuth_peak_deg=180.0,
            solar_elevation_peak_deg=73.0,
            prevailing_wind_direction_deg=270.0,
            prevailing_wind_speed_kmh=12.0,
        ),
        geophysical=GeophysicalConstraints(
            seismic_zone=SeismicZone.ZONE_2,
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
            ],
            wall_thickness_min_mm=100.0,
            wall_thickness_max_mm=250.0,
        ),
    )


@pytest.fixture
def minimalist_intent() -> DesignIntent:
    """Minimalist design intent with high view priority."""
    return DesignIntent(
        aesthetic_feel=AestheticFeel.MINIMALIST,
        view_priority=ViewPriority.HIGH,
    )


@pytest.fixture
def commercial_constraints() -> SiteConstraints:
    """Commercial high-rise site constraints."""
    return SiteConstraints(
        site_area_sqm=2000.0,
        regulatory=RegulatoryConstraints(
            far_limit=8.0,
            height_limit_m=45.0,
            is_airport_zone=True,
            setback_front_m=8.0,
            setback_side_m=5.0,
            setback_rear_m=6.0,
        ),
        environmental=EnvironmentalConstraints(
            latitude=35.68,
            longitude=139.69,
            solar_azimuth_peak_deg=180.0,
            solar_elevation_peak_deg=78.0,
            prevailing_wind_direction_deg=315.0,
            prevailing_wind_speed_kmh=18.0,
        ),
        geophysical=GeophysicalConstraints(
            seismic_zone=SeismicZone.ZONE_4,
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
        ),
    )
