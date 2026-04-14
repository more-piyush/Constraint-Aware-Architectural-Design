"""Inference pipeline orchestrator.

Design-first approach:
1. Architect's design intent forms strong priors (primary driver)
2. Constraints are discretized and layered on as evidence (filters)
3. MAP inference finds the best design, MCMC samples alternatives
4. Compliance checking and confidence scoring evaluate each design
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from car.config import (
    DEFAULT_FLOOR_HEIGHT_M,
    DEFAULT_MCMC_CHAINS,
    DEFAULT_MCMC_DRAWS,
    DEFAULT_MCMC_TUNE,
    DEFAULT_RANDOM_SEED,
    FLOOR_COUNT_MIDPOINTS,
    FLOOR_HEIGHT_MIDPOINTS,
    WALL_THICKNESS_MIDPOINTS,
    WINDOW_WALL_RATIO,
)
from car.models.constraints import SiteConstraints
from car.models.design import (
    BuildingDesign,
    DesignIntent,
    RoofType,
    StructuralSystem,
    WallType,
    WindowSize,
)
from car.models.results import ComplianceResult, DesignIteration, InferenceResult
from car.inference.map_inference import DESIGN_VARIABLES, MAPInferenceEngine
from car.inference.mcmc_sampler import MCMCSampler
from car.inference.variational import VariationalEngine
from car.network.builder import NetworkBuilder
from car.scoring.compliance import ComplianceChecker
from car.scoring.confidence import ConfidenceScorer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    inference_method: Literal["map", "mcmc", "variational"] = "map"
    num_samples: int = 500
    mcmc_chains: int = DEFAULT_MCMC_CHAINS
    mcmc_tune: int = DEFAULT_MCMC_TUNE
    mcmc_draws: int = DEFAULT_MCMC_DRAWS
    vi_iterations: int = 10000
    random_seed: int = DEFAULT_RANDOM_SEED


class InferencePipeline:
    """Orchestrates the end-to-end inference process."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._network_builder = NetworkBuilder()
        self._map_engine = MAPInferenceEngine()
        self._mcmc_sampler = MCMCSampler()
        self._vi_engine = VariationalEngine()
        self._compliance_checker = ComplianceChecker()
        self._confidence_scorer = ConfidenceScorer()

    def run(
        self, site_constraints: SiteConstraints, design_intent: DesignIntent
    ) -> InferenceResult:
        """Full pipeline execution.

        Steps:
        1. Discretize continuous constraints into categorical states
        2. Build the Bayesian network with CPDs
        3. Set evidence: design intent FIRST (strong priors), then constraints
        4. Run MAP inference for best single design
        5. Optionally run MCMC/VI for design alternatives
        6. Score all designs for compliance and confidence
        7. Package results
        """
        start = time.time()

        # 1. Discretize constraints into BN evidence
        constraint_evidence = self._discretize_constraints(site_constraints)
        logger.info(f"Discretized constraints: {constraint_evidence}")

        # 2. Build the Bayesian network
        model = self._network_builder.build()
        logger.info("Bayesian network built and validated")

        # 3. Design intent as evidence (primary driver)
        intent_evidence = {
            "aesthetic_feel": design_intent.aesthetic_feel.value,
            "view_priority": design_intent.view_priority.value,
        }

        full_evidence = {**intent_evidence, **constraint_evidence}

        # 4. MAP inference
        map_assignment = self._map_engine.find_map_design(
            model, constraint_evidence, intent_evidence
        )
        logger.info(f"MAP assignment: {map_assignment}")

        # Query marginals for confidence scoring
        marginal_probs = self._map_engine.query_marginals(
            model, full_evidence, DESIGN_VARIABLES
        )

        # 5. Decode MAP to BuildingDesign
        map_design = self._decode_design(map_assignment, site_constraints)

        # 6. Score compliance and confidence
        map_compliance = self._compliance_checker.check(map_design, site_constraints)
        map_confidence = self._confidence_scorer.score(
            map_design, site_constraints, map_compliance, marginal_probs
        )
        map_compliance = ComplianceResult(
            is_compliant=map_compliance.is_compliant,
            confidence_score=map_confidence,
            violations=map_compliance.violations,
            checked_constraints_count=map_compliance.checked_constraints_count,
            passed_constraints_count=map_compliance.passed_constraints_count,
        )

        # Compute aesthetic and view scores
        aesthetic_score = self._compute_aesthetic_score(map_assignment, design_intent)
        view_score = self._compute_view_score(map_assignment, design_intent)
        overall_score = 0.4 * aesthetic_score + 0.3 * view_score + 0.3 * map_confidence

        map_iteration = DesignIteration(
            iteration_id=0,
            design=map_design,
            compliance=map_compliance,
            aesthetic_score=aesthetic_score,
            view_score=view_score,
            overall_score=overall_score,
        )

        # 7. Sample alternatives if requested
        sampled_designs: list[DesignIteration] = []
        convergence_diagnostics: dict[str, Any] = {}

        if self._config.inference_method in ("mcmc", "variational"):
            idata, decoded_samples = self._run_sampling(
                marginal_probs, site_constraints
            )

            if idata is not None:
                convergence_diagnostics = self._mcmc_sampler.get_diagnostics(idata)

            # Deduplicate and score samples
            seen = set()
            for i, sample in enumerate(decoded_samples):
                key = tuple(sorted(sample.items()))
                if key in seen:
                    continue
                seen.add(key)

                design = self._decode_design(sample, site_constraints)
                compliance = self._compliance_checker.check(design, site_constraints)
                confidence = self._confidence_scorer.score(
                    design, site_constraints, compliance, marginal_probs
                )
                compliance = ComplianceResult(
                    is_compliant=compliance.is_compliant,
                    confidence_score=confidence,
                    violations=compliance.violations,
                    checked_constraints_count=compliance.checked_constraints_count,
                    passed_constraints_count=compliance.passed_constraints_count,
                )

                a_score = self._compute_aesthetic_score(sample, design_intent)
                v_score = self._compute_view_score(sample, design_intent)
                o_score = 0.4 * a_score + 0.3 * v_score + 0.3 * confidence

                sampled_designs.append(
                    DesignIteration(
                        iteration_id=i + 1,
                        design=design,
                        compliance=compliance,
                        aesthetic_score=a_score,
                        view_score=v_score,
                        overall_score=o_score,
                    )
                )

            # Sort by overall score descending
            sampled_designs.sort(key=lambda d: d.overall_score, reverse=True)

        elapsed = time.time() - start

        return InferenceResult(
            map_design=map_iteration,
            sampled_designs=sampled_designs[:50],  # top 50
            convergence_diagnostics=convergence_diagnostics,
            inference_method=self._config.inference_method,
            elapsed_seconds=elapsed,
        )

    def _run_sampling(
        self,
        marginal_probs: dict[str, dict[str, float]],
        site_constraints: SiteConstraints,
    ) -> tuple[Any, list[dict[str, str]]]:
        """Run MCMC or VI sampling."""
        if self._config.inference_method == "mcmc":
            return self._mcmc_sampler.sample_designs(
                marginal_probs=marginal_probs,
                site_constraints=site_constraints,
                num_chains=self._config.mcmc_chains,
                num_draws=self._config.mcmc_draws,
                num_tune=self._config.mcmc_tune,
                random_seed=self._config.random_seed,
            )
        else:
            return self._vi_engine.fit_and_sample(
                marginal_probs=marginal_probs,
                n_iterations=self._config.vi_iterations,
                n_samples=self._config.num_samples,
                random_seed=self._config.random_seed,
            )

    def _discretize_constraints(self, constraints: SiteConstraints) -> dict[str, str]:
        """Convert continuous constraint values to discrete BN states."""
        evidence: dict[str, str] = {}

        # FAR classification
        far = constraints.regulatory.far_limit
        if far <= 1.5:
            evidence["far_class"] = "low_far"
        elif far <= 3.0:
            evidence["far_class"] = "medium_far"
        else:
            evidence["far_class"] = "high_far"

        # Height restriction
        h = constraints.regulatory.height_limit_m
        airport = constraints.regulatory.is_airport_zone
        if airport or h < 12:
            evidence["height_restriction"] = "strict"
        elif h < 30:
            evidence["height_restriction"] = "moderate"
        else:
            evidence["height_restriction"] = "unrestricted"

        # Setback classification
        total_setback = (
            constraints.regulatory.setback_front_m
            + constraints.regulatory.setback_side_m * 2
            + constraints.regulatory.setback_rear_m
        )
        evidence["setback_class"] = "generous" if total_setback > 15 else "standard"

        # Solar orientation (quantize azimuth into cardinal direction)
        azimuth = constraints.environmental.solar_azimuth_peak_deg
        if 45 <= azimuth < 135:
            evidence["solar_orientation"] = "east"
        elif 135 <= azimuth < 225:
            evidence["solar_orientation"] = "south"
        elif 225 <= azimuth < 315:
            evidence["solar_orientation"] = "west"
        else:
            evidence["solar_orientation"] = "north"

        # Wind exposure
        wind = constraints.environmental.prevailing_wind_speed_kmh
        if wind < 10:
            evidence["wind_exposure"] = "sheltered"
        elif wind < 25:
            evidence["wind_exposure"] = "moderate"
        else:
            evidence["wind_exposure"] = "exposed"

        # Seismic zone classification
        zone = constraints.geophysical.seismic_zone.value
        if zone <= 1:
            evidence["seismic_zone_class"] = "low_risk"
        elif zone <= 3:
            evidence["seismic_zone_class"] = "moderate_risk"
        else:
            evidence["seismic_zone_class"] = "high_risk"

        # Material class (dominant available material)
        evidence["material_class"] = self._classify_material(
            constraints.technical.available_materials
        )

        # Wall thickness classification
        max_thick = constraints.technical.wall_thickness_max_mm
        if max_thick <= 200:
            evidence["wall_thickness_class"] = "thin"
        elif max_thick <= 350:
            evidence["wall_thickness_class"] = "standard"
        else:
            evidence["wall_thickness_class"] = "thick"

        return evidence

    def _classify_material(self, materials: list) -> str:
        """Classify the dominant available material."""
        material_map = {
            "steel": "steel",
            "concrete": "concrete",
            "timber": "timber",
            "masonry": "masonry",
            "wood": "timber",
            "brick": "masonry",
        }
        for mat in materials:
            name_lower = mat.name.lower()
            for keyword, category in material_map.items():
                if keyword in name_lower:
                    return category
        return "concrete"  # default

    def _decode_design(
        self, assignment: dict[str, str], constraints: SiteConstraints
    ) -> BuildingDesign:
        """Convert categorical BN state assignment to a concrete BuildingDesign."""
        num_floors_cat = assignment.get("num_floors", "2_3_floors")
        num_floors = FLOOR_COUNT_MIDPOINTS.get(num_floors_cat, 2)

        floor_height = DEFAULT_FLOOR_HEIGHT_M
        # Clamp to technical constraints
        floor_height = max(
            constraints.technical.floor_to_floor_height_min_m,
            min(floor_height, constraints.technical.floor_to_floor_height_max_m),
        )
        building_height = num_floors * floor_height

        # Clamp height to limit
        if building_height > constraints.regulatory.height_limit_m:
            building_height = constraints.regulatory.height_limit_m
            num_floors = max(1, int(building_height / floor_height))

        structural_system = StructuralSystem(
            assignment.get("structural_system", "reinforced_concrete")
        )
        wall_type = WallType(assignment.get("wall_type", "load_bearing"))
        window_size = WindowSize(assignment.get("window_size", "medium"))
        roof_type = RoofType(assignment.get("roof_type", "flat"))

        # Wall thickness from category
        wall_thickness_cat = assignment.get("wall_thickness_class", "standard")
        wall_thickness = WALL_THICKNESS_MIDPOINTS.get(wall_thickness_cat, 250.0)
        wall_thickness = max(
            constraints.technical.wall_thickness_min_mm,
            min(wall_thickness, constraints.technical.wall_thickness_max_mm),
        )

        # Compute floor area from FAR and site area
        max_total_area = constraints.regulatory.far_limit * constraints.site_area_sqm
        floor_area = max_total_area / max(num_floors, 1)

        # Compute footprint dimensions (aspect ratio from topology intent)
        site_side = math.sqrt(constraints.site_area_sqm)
        available_width = max(
            1.0,
            site_side - constraints.regulatory.setback_side_m * 2,
        )
        available_depth = max(
            1.0,
            site_side
            - constraints.regulatory.setback_front_m
            - constraints.regulatory.setback_rear_m,
        )

        footprint_area = min(floor_area, available_width * available_depth)
        aspect_ratio = min(available_width, available_depth) / max(
            available_width, available_depth
        )
        footprint_width = min(math.sqrt(footprint_area / max(aspect_ratio, 0.3)), available_width)
        footprint_depth = min(footprint_area / max(footprint_width, 1.0), available_depth)

        # Primary material
        primary_material = "concrete"
        if constraints.technical.available_materials:
            primary_material = constraints.technical.available_materials[0].name

        # Window orientation from solar data
        window_orientation = constraints.environmental.solar_azimuth_peak_deg

        return BuildingDesign(
            num_floors=num_floors,
            floor_area_sqm=round(floor_area, 1),
            building_height_m=round(building_height, 1),
            structural_system=structural_system,
            wall_type=wall_type,
            wall_thickness_mm=round(wall_thickness, 0),
            window_size=window_size,
            roof_type=roof_type,
            primary_material=primary_material,
            window_orientation_deg=window_orientation,
            footprint_width_m=round(footprint_width, 1),
            footprint_depth_m=round(footprint_depth, 1),
        )

    def _compute_aesthetic_score(
        self, assignment: dict[str, str], intent: DesignIntent
    ) -> float:
        """Score how well the design matches the aesthetic intent."""
        aesthetic = intent.aesthetic_feel.value
        score = 0.5  # base

        # Structural system alignment
        system = assignment.get("structural_system", "")
        aesthetic_system_map = {
            "minimalist": ["steel_frame", "hybrid_steel_concrete"],
            "industrial": ["reinforced_concrete", "steel_frame"],
            "organic": ["timber_frame", "masonry"],
            "classical": ["masonry", "reinforced_concrete"],
        }
        preferred = aesthetic_system_map.get(aesthetic, [])
        if system in preferred:
            score += 0.2

        # Window size alignment
        window = assignment.get("window_size", "")
        aesthetic_window_map = {
            "minimalist": ["large", "full_glass"],
            "industrial": ["large", "full_glass"],
            "organic": ["medium", "large"],
            "classical": ["medium", "small"],
        }
        preferred_w = aesthetic_window_map.get(aesthetic, [])
        if window in preferred_w:
            score += 0.15

        # Roof type alignment
        roof = assignment.get("roof_type", "")
        aesthetic_roof_map = {
            "minimalist": ["flat"],
            "industrial": ["flat"],
            "organic": ["green_roof", "pitched"],
            "classical": ["pitched"],
        }
        preferred_r = aesthetic_roof_map.get(aesthetic, [])
        if roof in preferred_r:
            score += 0.15

        return min(1.0, score)

    def _compute_view_score(
        self, assignment: dict[str, str], intent: DesignIntent
    ) -> float:
        """Score how well the design optimizes for views."""
        view_priority = intent.view_priority.value
        window = assignment.get("window_size", "medium")

        window_scores = {"small": 0.2, "medium": 0.5, "large": 0.8, "full_glass": 1.0}
        window_score = window_scores.get(window, 0.5)

        if view_priority == "high":
            return window_score
        elif view_priority == "medium":
            return 0.3 + 0.4 * window_score
        else:
            return 0.5 + 0.2 * window_score
