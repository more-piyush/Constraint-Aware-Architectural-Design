"""Hard-constraint compliance checker for building designs."""

from __future__ import annotations

from car.models.constraints import SiteConstraints
from car.models.design import BuildingDesign, StructuralSystem
from car.models.results import ComplianceResult, ConstraintViolation


class ComplianceChecker:
    """Checks a BuildingDesign against all hard constraints."""

    def check(
        self, design: BuildingDesign, constraints: SiteConstraints
    ) -> ComplianceResult:
        """Run all constraint checks and aggregate results."""
        checks = [
            self._check_far,
            self._check_height,
            self._check_setback_footprint,
            self._check_seismic_material,
            self._check_wall_thickness,
            self._check_floor_height,
        ]

        violations = []
        passed = 0
        for check_fn in checks:
            violation = check_fn(design, constraints)
            if violation is None:
                passed += 1
            else:
                violations.append(violation)

        total = len(checks)
        is_compliant = all(v.severity != "hard" for v in violations)
        confidence = passed / total if total > 0 else 1.0

        return ComplianceResult(
            is_compliant=is_compliant,
            confidence_score=confidence,
            violations=violations,
            checked_constraints_count=total,
            passed_constraints_count=passed,
        )

    def _check_far(
        self, design: BuildingDesign, constraints: SiteConstraints
    ) -> ConstraintViolation | None:
        actual_far = (design.floor_area_sqm * design.num_floors) / constraints.site_area_sqm
        if actual_far > constraints.regulatory.far_limit:
            return ConstraintViolation(
                constraint_name="Floor Area Ratio",
                constraint_type="regulatory",
                required_value=f"<= {constraints.regulatory.far_limit:.2f}",
                actual_value=f"{actual_far:.2f}",
                severity="hard",
            )
        return None

    def _check_height(
        self, design: BuildingDesign, constraints: SiteConstraints
    ) -> ConstraintViolation | None:
        if design.building_height_m > constraints.regulatory.height_limit_m:
            return ConstraintViolation(
                constraint_name="Building Height",
                constraint_type="regulatory",
                required_value=f"<= {constraints.regulatory.height_limit_m:.1f}m",
                actual_value=f"{design.building_height_m:.1f}m",
                severity="hard",
            )
        return None

    def _check_setback_footprint(
        self, design: BuildingDesign, constraints: SiteConstraints
    ) -> ConstraintViolation | None:
        """Check that the building footprint fits within setback boundaries."""
        import math

        site_side = math.sqrt(constraints.site_area_sqm)
        available_width = site_side - constraints.regulatory.setback_side_m * 2
        available_depth = (
            site_side
            - constraints.regulatory.setback_front_m
            - constraints.regulatory.setback_rear_m
        )

        if available_width <= 0 or available_depth <= 0:
            return ConstraintViolation(
                constraint_name="Setback Feasibility",
                constraint_type="regulatory",
                required_value="Positive buildable area after setbacks",
                actual_value="No buildable area",
                severity="hard",
            )

        if design.footprint_width_m > available_width or design.footprint_depth_m > available_depth:
            return ConstraintViolation(
                constraint_name="Setback Compliance",
                constraint_type="regulatory",
                required_value=f"Footprint within {available_width:.1f}m x {available_depth:.1f}m",
                actual_value=f"{design.footprint_width_m:.1f}m x {design.footprint_depth_m:.1f}m",
                severity="soft",
            )
        return None

    def _check_seismic_material(
        self, design: BuildingDesign, constraints: SiteConstraints
    ) -> ConstraintViolation | None:
        zone = constraints.geophysical.seismic_zone.value
        if zone >= 4 and design.structural_system in (
            StructuralSystem.TIMBER_FRAME,
            StructuralSystem.MASONRY,
        ):
            return ConstraintViolation(
                constraint_name="Seismic Zone Material Restriction",
                constraint_type="geophysical",
                required_value="steel_frame, reinforced_concrete, or hybrid in zone >= 4",
                actual_value=design.structural_system.value,
                severity="hard",
            )
        return None

    def _check_wall_thickness(
        self, design: BuildingDesign, constraints: SiteConstraints
    ) -> ConstraintViolation | None:
        min_t = constraints.technical.wall_thickness_min_mm
        max_t = constraints.technical.wall_thickness_max_mm
        if design.wall_thickness_mm < min_t or design.wall_thickness_mm > max_t:
            return ConstraintViolation(
                constraint_name="Wall Thickness Range",
                constraint_type="technical",
                required_value=f"{min_t:.0f}mm - {max_t:.0f}mm",
                actual_value=f"{design.wall_thickness_mm:.0f}mm",
                severity="soft",
            )
        return None

    def _check_floor_height(
        self, design: BuildingDesign, constraints: SiteConstraints
    ) -> ConstraintViolation | None:
        if design.num_floors > 0:
            floor_height = design.building_height_m / design.num_floors
            min_h = constraints.technical.floor_to_floor_height_min_m
            max_h = constraints.technical.floor_to_floor_height_max_m
            if floor_height < min_h or floor_height > max_h:
                return ConstraintViolation(
                    constraint_name="Floor-to-Floor Height",
                    constraint_type="technical",
                    required_value=f"{min_h:.1f}m - {max_h:.1f}m",
                    actual_value=f"{floor_height:.1f}m",
                    severity="soft",
                )
        return None
