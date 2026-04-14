"""Tests for Bayesian network construction."""

import pytest

from car.network.builder import NetworkBuilder
from car.network.nodes import NodeRegistry
from car.network.edges import EdgeRegistry
from car.network.cpd_factory import CPDFactory


class TestNodeRegistry:
    def test_total_node_count(self):
        registry = NodeRegistry()
        nodes = registry.get_all_nodes()
        assert len(nodes) == 16

    def test_latent_nodes(self):
        registry = NodeRegistry()
        latent = [n for n in registry.get_all_nodes() if n.variable_type == "latent"]
        assert len(latent) == 3
        names = {n.name for n in latent}
        assert "aesthetic_feel" in names
        assert "view_priority" in names
        assert "sketch_topology" in names

    def test_observed_nodes(self):
        registry = NodeRegistry()
        observed = [n for n in registry.get_all_nodes() if n.variable_type == "observed"]
        assert len(observed) == 8

    def test_decision_nodes(self):
        registry = NodeRegistry()
        decision = [n for n in registry.get_all_nodes() if n.variable_type == "decision"]
        assert len(decision) == 5

    def test_get_node_by_name(self):
        registry = NodeRegistry()
        node = registry.get_node("seismic_zone_class")
        assert node.cardinality == 3
        assert "high_risk" in node.state_names

    def test_get_nonexistent_node(self):
        registry = NodeRegistry()
        with pytest.raises(KeyError):
            registry.get_node("nonexistent")


class TestEdgeRegistry:
    def test_total_edge_count(self):
        registry = EdgeRegistry()
        edges = registry.get_all_edges()
        assert len(edges) == 23

    def test_design_intent_edges_have_higher_weights(self):
        registry = EdgeRegistry()
        intent_edges = registry._design_intent_edges()
        regulatory_edges = registry._regulatory_edges()
        avg_intent = sum(e.weight for e in intent_edges) / len(intent_edges)
        avg_reg = sum(e.weight for e in regulatory_edges) / len(regulatory_edges)
        assert avg_intent > avg_reg, "Design intent edges should have higher average weight"


class TestCPDFactory:
    def test_structural_system_weights_minimalist_seismic(self):
        factory = CPDFactory()
        weights = factory._weights_structural_system({
            "aesthetic_feel": "minimalist",
            "seismic_zone_class": "high_risk",
            "material_class": "steel",
            "num_floors": "4_7_floors",
        })
        assert len(weights) == 5
        # Steel frame should be dominant for minimalist + steel + high seismic
        steel_idx = 0  # steel_frame is first
        assert weights[steel_idx] == max(weights)

    def test_structural_system_weights_organic_low_seismic(self):
        factory = CPDFactory()
        weights = factory._weights_structural_system({
            "aesthetic_feel": "organic",
            "seismic_zone_class": "low_risk",
            "material_class": "timber",
            "num_floors": "1_floor",
        })
        timber_idx = 2  # timber_frame is third
        assert weights[timber_idx] == max(weights), "Organic + timber + low risk should favor timber"

    def test_all_weights_positive(self):
        factory = CPDFactory()
        for method_name in dir(factory):
            if method_name.startswith("_weights_"):
                method = getattr(factory, method_name)
                # Test with empty parents (should return defaults)
                weights = method({})
                assert all(w >= 0 for w in weights), f"{method_name} produced negative weights"


class TestNetworkBuilder:
    def test_build_succeeds(self):
        builder = NetworkBuilder()
        model = builder.build()
        assert model is not None

    def test_model_has_correct_nodes(self):
        builder = NetworkBuilder()
        model = builder.build()
        nodes = set(model.nodes())
        assert "aesthetic_feel" in nodes
        assert "structural_system" in nodes
        assert "seismic_zone_class" in nodes

    def test_model_has_correct_edges(self):
        builder = NetworkBuilder()
        model = builder.build()
        edges = set(model.edges())
        assert ("aesthetic_feel", "structural_system") in edges
        assert ("seismic_zone_class", "structural_system") in edges

    def test_model_cpds_valid(self):
        builder = NetworkBuilder()
        model = builder.build()
        assert model.check_model()

    def test_topology_structure(self):
        builder = NetworkBuilder()
        topology = builder.build_topology()
        assert len(topology.nodes) == 16
        assert len(topology.edges) == 23
