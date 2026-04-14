"""Building component and design intent data models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AestheticFeel(str, Enum):
    MINIMALIST = "minimalist"
    INDUSTRIAL = "industrial"
    ORGANIC = "organic"
    CLASSICAL = "classical"


class ViewPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StructuralSystem(str, Enum):
    STEEL_FRAME = "steel_frame"
    REINFORCED_CONCRETE = "reinforced_concrete"
    TIMBER_FRAME = "timber_frame"
    MASONRY = "masonry"
    HYBRID_STEEL_CONCRETE = "hybrid_steel_concrete"


class WallType(str, Enum):
    LOAD_BEARING = "load_bearing"
    CURTAIN_WALL = "curtain_wall"
    PARTITION = "partition"


class WindowSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    FULL_GLASS = "full_glass"


class RoofType(str, Enum):
    FLAT = "flat"
    PITCHED = "pitched"
    GREEN_ROOF = "green_roof"


class DesignIntent(BaseModel):
    """Latent variables representing the architect's design intent (primary driver)."""

    aesthetic_feel: AestheticFeel
    view_priority: ViewPriority
    sustainability_priority: ViewPriority = ViewPriority.MEDIUM
    budget_level: str = Field(default="medium", pattern="^(low|medium|high)$")


class Room(BaseModel):
    """A room within the building."""

    name: str
    area_sqm: float = Field(..., gt=0)
    floor_level: int = Field(default=0, ge=0)
    requires_natural_light: bool = Field(default=True)
    is_wet_room: bool = Field(default=False)


class BuildingDesign(BaseModel):
    """A complete building design configuration -- the output of inference."""

    num_floors: int = Field(..., ge=1)
    floor_area_sqm: float = Field(..., gt=0)
    building_height_m: float = Field(..., gt=0)
    structural_system: StructuralSystem
    wall_type: WallType
    wall_thickness_mm: float
    window_size: WindowSize
    roof_type: RoofType
    primary_material: str
    window_orientation_deg: float = Field(default=180.0)
    rooms: list[Room] = Field(default_factory=list)
    footprint_width_m: float = Field(..., gt=0)
    footprint_depth_m: float = Field(..., gt=0)
