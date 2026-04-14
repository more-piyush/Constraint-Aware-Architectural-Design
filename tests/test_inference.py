"""Tests for inference engines and pipeline."""

import pytest

from car.inference.map_inference import MAPInferenceEngine, DESIGN_VARIABLES
from car.inference.pipeline import InferencePipeline, PipelineConfig
from car.network.builder import NetworkBuilder


class TestMAPInference:
    def test_map_returns_all_design_variables(self, residential_constraints, minimalist_intent):
        builder = NetworkBuilder()
        model = builder.build()

        engine = MAPInferenceEngine()
        evidence = {
            "far_class": "low_far",
            "height_restriction": "strict",
            "setback_class": "standard",
            "solar_orientation": "south",
            "wind_exposure": "moderate",
            "seismic_zone_class": "moderate_risk",
            "material_class": "timber",
            "wall_thickness_class": "standard",
        }
        intent = {
            "aesthetic_feel": "minimalist",
            "view_priority": "high",
        }
        result = engine.find_map_design(model, evidence, intent)

        # All non-evidence design variables should be assigned
        for var in DESIGN_VARIABLES:
            if var not in evidence and var not in intent:
                assert var in result, f"Missing design variable: {var}"

    def test_marginal_probabilities(self):
        builder = NetworkBuilder()
        model = builder.build()

        engine = MAPInferenceEngine()
        evidence = {
            "aesthetic_feel": "minimalist",
            "view_priority": "high",
            "seismic_zone_class": "low_risk",
            "material_class": "steel",
        }
        marginals = engine.query_marginals(model, evidence)

        for var, probs in marginals.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01, f"Marginals for {var} don't sum to 1: {total}"


class TestInferencePipeline:
    def test_map_pipeline(self, residential_constraints, minimalist_intent):
        config = PipelineConfig(inference_method="map")
        pipeline = InferencePipeline(config)
        result = pipeline.run(residential_constraints, minimalist_intent)

        assert result.map_design is not None
        assert result.map_design.design.num_floors >= 1
        assert result.map_design.compliance.checked_constraints_count > 0
        assert 0 <= result.map_design.compliance.confidence_score <= 1
        assert 0 <= result.map_design.aesthetic_score <= 1
        assert 0 <= result.map_design.view_score <= 1

    def test_design_first_priority(self, residential_constraints, minimalist_intent):
        """Verify that design intent drives the output (minimalist -> steel/glass)."""
        config = PipelineConfig(inference_method="map")
        pipeline = InferencePipeline(config)
        result = pipeline.run(residential_constraints, minimalist_intent)

        design = result.map_design.design
        # Minimalist + high view priority should favor large windows
        assert design.window_size.value in ("large", "full_glass"), (
            f"Minimalist + high view should produce large windows, got {design.window_size.value}"
        )

    def test_seismic_constraint_enforcement(self, commercial_constraints):
        """Verify high seismic zones don't produce timber/masonry structures."""
        from car.models.design import AestheticFeel, DesignIntent, ViewPriority

        intent = DesignIntent(
            aesthetic_feel=AestheticFeel.ORGANIC,  # would normally favor timber
            view_priority=ViewPriority.MEDIUM,
        )
        config = PipelineConfig(inference_method="map")
        pipeline = InferencePipeline(config)
        result = pipeline.run(commercial_constraints, intent)

        design = result.map_design.design
        # Seismic zone 4 should override organic preference for timber
        assert design.structural_system.value not in ("timber_frame", "masonry"), (
            f"Seismic zone 4 should not allow {design.structural_system.value}"
        )

    def test_height_constraint_respected(self, residential_constraints, minimalist_intent):
        """Verify building height doesn't exceed limit."""
        config = PipelineConfig(inference_method="map")
        pipeline = InferencePipeline(config)
        result = pipeline.run(residential_constraints, minimalist_intent)

        design = result.map_design.design
        assert design.building_height_m <= residential_constraints.regulatory.height_limit_m + 0.1

    def test_variational_pipeline(self, residential_constraints, minimalist_intent):
        config = PipelineConfig(inference_method="variational", num_samples=20)
        pipeline = InferencePipeline(config)
        result = pipeline.run(residential_constraints, minimalist_intent)

        assert result.map_design is not None
        assert len(result.sampled_designs) > 0

    def test_pipeline_discretization(self, residential_constraints):
        pipeline = InferencePipeline()
        evidence = pipeline._discretize_constraints(residential_constraints)

        assert evidence["far_class"] == "low_far"
        assert evidence["height_restriction"] == "strict"
        assert evidence["seismic_zone_class"] == "moderate_risk"
        assert evidence["solar_orientation"] == "south"
        assert evidence["wind_exposure"] == "moderate"
        assert evidence["material_class"] == "timber"
