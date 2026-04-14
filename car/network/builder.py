"""Assembles the complete Bayesian network from node and edge registries."""

from __future__ import annotations

from pgmpy.models import DiscreteBayesianNetwork

from car.models.network_spec import NetworkTopology
from car.network.cpd_factory import CPDFactory
from car.network.edges import EdgeRegistry
from car.network.nodes import NodeRegistry


class NetworkBuilder:
    """Assembles the complete Bayesian network from node and edge registries."""

    def __init__(self) -> None:
        self._node_registry = NodeRegistry()
        self._edge_registry = EdgeRegistry()
        self._cpd_factory = CPDFactory()

    def build(self) -> DiscreteBayesianNetwork:
        """Construct and validate the complete Bayesian network."""
        topology = self.build_topology()
        edges = [(e.parent, e.child) for e in topology.edges]
        model = DiscreteBayesianNetwork(edges)

        cpds = self._cpd_factory.build_all_cpds(topology)
        model.add_cpds(*cpds)

        if not model.check_model():
            raise ValueError("Model validation failed: CPDs do not match graph structure")

        return model

    def build_topology(self) -> NetworkTopology:
        """Combine node and edge registries into a topology spec."""
        return NetworkTopology(
            nodes=self._node_registry.get_all_nodes(),
            edges=self._edge_registry.get_all_edges(),
        )
