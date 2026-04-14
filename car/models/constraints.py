"""Constraint data models for site, regulatory, environmental, geophysical, and technical inputs."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class SeismicZone(IntEnum):
    ZONE_0 = 0
    ZONE_1 = 1
    ZONE_2 = 2
    ZONE_3 = 3
    ZONE_4 = 4
    ZONE_5 = 5


class RegulatoryConstraints(BaseModel):
    """Hard constraints from building codes and local bylaws."""

    far_limit: float = Field(..., gt=0, description="Maximum Floor Area Ratio")
    height_limit_m: float = Field(..., gt=0, description="Maximum building height in meters")
    is_airport_zone: bool = Field(default=False)
    setback_front_m: float = Field(..., ge=0)
    setback_side_m: float = Field(..., ge=0)
    setback_rear_m: float = Field(..., ge=0)
    min_parking_spaces: int = Field(default=0, ge=0)
    fire_escape_required: bool = Field(default=True)


class EnvironmentalConstraints(BaseModel):
    """Environmental conditions affecting design."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    solar_azimuth_peak_deg: float = Field(..., ge=0, lt=360)
    solar_elevation_peak_deg: float = Field(..., ge=0, le=90)
    prevailing_wind_direction_deg: float = Field(..., ge=0, lt=360)
    prevailing_wind_speed_kmh: float = Field(default=15.0, ge=0)
    annual_rainfall_mm: float = Field(default=800.0, ge=0)


class GeophysicalConstraints(BaseModel):
    """Geophysical site conditions."""

    seismic_zone: SeismicZone
    soil_bearing_capacity_kpa: float = Field(default=150.0, gt=0)
    water_table_depth_m: float = Field(default=5.0, gt=0)


class MaterialProperties(BaseModel):
    """Technical properties of a structural material."""

    name: str
    youngs_modulus_gpa: float = Field(..., gt=0)
    thermal_mass_kj_per_m3k: float = Field(..., gt=0)
    density_kg_per_m3: float = Field(..., gt=0)
    cost_per_m3_usd: float = Field(..., gt=0)


class TechnicalConstraints(BaseModel):
    """Technical building constraints."""

    available_materials: list[MaterialProperties]
    wall_thickness_min_mm: float = Field(default=100.0, gt=0)
    wall_thickness_max_mm: float = Field(default=500.0, gt=0)
    floor_to_floor_height_min_m: float = Field(default=2.7, gt=0)
    floor_to_floor_height_max_m: float = Field(default=4.5, gt=0)


class SiteConstraints(BaseModel):
    """Aggregate of all constraints for a building site."""

    site_area_sqm: float = Field(..., gt=0)
    regulatory: RegulatoryConstraints
    environmental: EnvironmentalConstraints
    geophysical: GeophysicalConstraints
    technical: TechnicalConstraints
