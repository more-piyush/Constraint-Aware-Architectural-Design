"""Conditional Probability Distribution factory using parametric rule-based weight computation.

Instead of hand-coding thousands of CPD table entries, we use domain rules as
multiplicative weight adjustments that are then normalized to valid probability
distributions. This is auditable, testable, and modifiable by domain experts.

Design-first approach: design intent CPDs use stronger priors so the architect's
vision drives the generative process before constraints filter/shape it.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from pgmpy.factors.discrete import TabularCPD

from car.models.network_spec import NetworkTopology, NodeSpec
from car.network.nodes import NodeRegistry


class CPDFactory:
    """Constructs TabularCPD objects for every node in the network."""

    def __init__(self) -> None:
        self._node_registry = NodeRegistry()
        self._nodes_by_name: dict[str, NodeSpec] = {
            n.name: n for n in self._node_registry.get_all_nodes()
        }

    def build_all_cpds(self, topology: NetworkTopology) -> list[TabularCPD]:
        """Build CPDs for all nodes in the network."""
        parent_map: dict[str, list[str]] = {n.name: [] for n in topology.nodes}
        for edge in topology.edges:
            parent_map[edge.child].append(edge.parent)

        cpds = []
        for node in topology.nodes:
            parents = parent_map[node.name]
            if not parents:
                cpd = self._build_root_cpd(node)
            else:
                cpd = self._build_conditional_cpd(node, parents)
            cpds.append(cpd)
        return cpds

    def _build_root_cpd(self, node: NodeSpec) -> TabularCPD:
        """Build CPD for a root node (no parents) -- uniform prior."""
        n = node.cardinality
        values = [[1.0 / n]] * n
        return TabularCPD(
            variable=node.name,
            variable_card=node.cardinality,
            values=values,
            state_names={node.name: node.state_names},
        )

    def _build_conditional_cpd(self, node: NodeSpec, parents: list[str]) -> TabularCPD:
        """Build CPD for a node with parents using parametric weight computation."""
        parent_nodes = [self._nodes_by_name[p] for p in parents]
        parent_cards = [p.cardinality for p in parent_nodes]
        parent_state_names_list = [p.state_names for p in parent_nodes]

        num_cols = 1
        for c in parent_cards:
            num_cols *= c

        values = np.zeros((node.cardinality, num_cols))

        for col_idx, parent_combo in enumerate(
            itertools.product(*parent_state_names_list)
        ):
            parent_assignment = dict(zip(parents, parent_combo))
            weights = self._compute_weights(node.name, node.state_names, parent_assignment)
            total = sum(weights)
            if total > 0:
                values[:, col_idx] = [w / total for w in weights]
            else:
                values[:, col_idx] = 1.0 / node.cardinality

        state_names = {node.name: node.state_names}
        for p_node in parent_nodes:
            state_names[p_node.name] = p_node.state_names

        return TabularCPD(
            variable=node.name,
            variable_card=node.cardinality,
            values=values.tolist(),
            evidence=parents,
            evidence_card=parent_cards,
            state_names=state_names,
        )

    def _compute_weights(
        self, node_name: str, state_names: list[str], parents: dict[str, str]
    ) -> list[float]:
        """Dispatch to the appropriate weight computation method."""
        method_name = f"_weights_{node_name}"
        method = getattr(self, method_name, None)
        if method is None:
            return [1.0] * len(state_names)
        return method(parents)

    # -------------------------------------------------------------------------
    # Design intent -> decision node weight methods (STRONG influence)
    # -------------------------------------------------------------------------

    def _weights_structural_system(self, parents: dict[str, str]) -> list[float]:
        """P(structural_system | aesthetic_feel, seismic_zone_class, material_class, num_floors)
        Returns weights for: steel_frame, reinforced_concrete, timber_frame, masonry, hybrid_steel_concrete
        """
        w = {"steel_frame": 1.0, "reinforced_concrete": 1.0, "timber_frame": 1.0,
             "masonry": 0.8, "hybrid_steel_concrete": 0.8}

        # Design intent drives first (strong multipliers)
        aesthetic = parents.get("aesthetic_feel")
        if aesthetic == "minimalist":
            w["steel_frame"] *= 3.5
            w["hybrid_steel_concrete"] *= 1.5
        elif aesthetic == "industrial":
            w["reinforced_concrete"] *= 3.0
            w["steel_frame"] *= 2.0
        elif aesthetic == "organic":
            w["timber_frame"] *= 3.5
            w["masonry"] *= 1.5
        elif aesthetic == "classical":
            w["masonry"] *= 3.0
            w["reinforced_concrete"] *= 1.5

        # Constraints filter second (moderate multipliers)
        seismic = parents.get("seismic_zone_class")
        if seismic == "high_risk":
            w["reinforced_concrete"] *= 2.5
            w["hybrid_steel_concrete"] *= 2.5
            w["steel_frame"] *= 1.5
            w["timber_frame"] *= 0.1
            w["masonry"] *= 0.1
        elif seismic == "moderate_risk":
            w["reinforced_concrete"] *= 1.5
            w["hybrid_steel_concrete"] *= 1.5
            w["timber_frame"] *= 0.5
            w["masonry"] *= 0.5

        material = parents.get("material_class")
        if material == "timber":
            w["timber_frame"] *= 4.0
            w["steel_frame"] *= 0.3
            w["masonry"] *= 0.2
        elif material == "steel":
            w["steel_frame"] *= 3.0
            w["hybrid_steel_concrete"] *= 1.5
        elif material == "concrete":
            w["reinforced_concrete"] *= 3.0
            w["hybrid_steel_concrete"] *= 1.5
        elif material == "masonry":
            w["masonry"] *= 3.0
            w["reinforced_concrete"] *= 1.2

        floors = parents.get("num_floors")
        if floors == "8_plus_floors":
            w["timber_frame"] *= 0.01
            w["masonry"] *= 0.01
            w["steel_frame"] *= 2.0
            w["hybrid_steel_concrete"] *= 2.5
        elif floors == "4_7_floors":
            w["timber_frame"] *= 0.3
            w["masonry"] *= 0.3
            w["steel_frame"] *= 1.5
        elif floors == "1_floor":
            w["timber_frame"] *= 1.5
            w["masonry"] *= 1.5

        return [w[s] for s in [
            "steel_frame", "reinforced_concrete", "timber_frame",
            "masonry", "hybrid_steel_concrete"
        ]]

    def _weights_num_floors(self, parents: dict[str, str]) -> list[float]:
        """P(num_floors | sketch_topology, far_class, height_restriction)
        Returns weights for: 1_floor, 2_3_floors, 4_7_floors, 8_plus_floors
        """
        w = {"1_floor": 1.0, "2_3_floors": 1.0, "4_7_floors": 1.0, "8_plus_floors": 1.0}

        # Design intent first
        topology = parents.get("sketch_topology")
        if topology == "compact":
            w["4_7_floors"] *= 2.5
            w["8_plus_floors"] *= 2.0
            w["1_floor"] *= 0.5
        elif topology == "linear":
            w["1_floor"] *= 2.0
            w["2_3_floors"] *= 2.5
            w["8_plus_floors"] *= 0.3
        elif topology == "courtyard":
            w["2_3_floors"] *= 2.5
            w["4_7_floors"] *= 1.5
            w["8_plus_floors"] *= 0.2

        # Constraints filter
        far = parents.get("far_class")
        if far == "low_far":
            w["1_floor"] *= 2.5
            w["2_3_floors"] *= 1.5
            w["8_plus_floors"] *= 0.1
        elif far == "high_far":
            w["4_7_floors"] *= 2.0
            w["8_plus_floors"] *= 3.0
            w["1_floor"] *= 0.3

        height = parents.get("height_restriction")
        if height == "strict":
            w["8_plus_floors"] *= 0.01
            w["4_7_floors"] *= 0.3
            w["1_floor"] *= 2.0
        elif height == "moderate":
            w["8_plus_floors"] *= 0.2
            w["4_7_floors"] *= 1.0

        return [w[s] for s in ["1_floor", "2_3_floors", "4_7_floors", "8_plus_floors"]]

    def _weights_window_size(self, parents: dict[str, str]) -> list[float]:
        """P(window_size | aesthetic_feel, view_priority, solar_orientation, wind_exposure, far_class)
        Returns weights for: small, medium, large, full_glass
        """
        w = {"small": 1.0, "medium": 1.0, "large": 1.0, "full_glass": 1.0}

        # Design intent drives strongly
        aesthetic = parents.get("aesthetic_feel")
        if aesthetic == "minimalist":
            w["large"] *= 3.0
            w["full_glass"] *= 2.5
            w["small"] *= 0.3
        elif aesthetic == "industrial":
            w["large"] *= 2.0
            w["full_glass"] *= 2.0
        elif aesthetic == "classical":
            w["medium"] *= 3.0
            w["small"] *= 1.5
            w["full_glass"] *= 0.3
        elif aesthetic == "organic":
            w["medium"] *= 2.0
            w["large"] *= 1.5

        view = parents.get("view_priority")
        if view == "high":
            w["large"] *= 3.0
            w["full_glass"] *= 4.0
            w["small"] *= 0.2
        elif view == "medium":
            w["medium"] *= 1.5
            w["large"] *= 1.5
        elif view == "low":
            w["small"] *= 2.0
            w["medium"] *= 1.5

        # Constraints filter
        solar = parents.get("solar_orientation")
        if solar == "south":
            w["large"] *= 1.5
            w["full_glass"] *= 1.3
        elif solar == "north":
            w["small"] *= 1.3
            w["full_glass"] *= 0.8

        wind = parents.get("wind_exposure")
        if wind == "exposed":
            w["full_glass"] *= 0.4
            w["large"] *= 0.6
            w["small"] *= 1.5
        elif wind == "sheltered":
            w["large"] *= 1.3
            w["full_glass"] *= 1.2

        far = parents.get("far_class")
        if far == "high_far":
            w["full_glass"] *= 1.3
            w["large"] *= 1.2

        return [w[s] for s in ["small", "medium", "large", "full_glass"]]

    def _weights_wall_type(self, parents: dict[str, str]) -> list[float]:
        """P(wall_type | aesthetic_feel, structural_system, wall_thickness_class)
        Returns weights for: load_bearing, curtain_wall, partition
        """
        w = {"load_bearing": 1.0, "curtain_wall": 1.0, "partition": 1.0}

        # Design intent first
        aesthetic = parents.get("aesthetic_feel")
        if aesthetic == "minimalist":
            w["curtain_wall"] *= 2.5
            w["partition"] *= 1.5
        elif aesthetic == "industrial":
            w["curtain_wall"] *= 3.0
        elif aesthetic == "classical":
            w["load_bearing"] *= 3.0
            w["curtain_wall"] *= 0.5
        elif aesthetic == "organic":
            w["load_bearing"] *= 2.0
            w["partition"] *= 1.5

        # Structural constraints
        system = parents.get("structural_system")
        if system == "steel_frame":
            w["curtain_wall"] *= 3.0
            w["partition"] *= 2.0
            w["load_bearing"] *= 0.3
        elif system == "reinforced_concrete":
            w["curtain_wall"] *= 2.0
            w["partition"] *= 1.5
        elif system == "masonry":
            w["load_bearing"] *= 4.0
            w["curtain_wall"] *= 0.2
        elif system == "timber_frame":
            w["partition"] *= 2.5
            w["load_bearing"] *= 1.5
            w["curtain_wall"] *= 0.5

        thickness = parents.get("wall_thickness_class")
        if thickness == "thick":
            w["load_bearing"] *= 2.0
            w["curtain_wall"] *= 0.5
        elif thickness == "thin":
            w["curtain_wall"] *= 2.0
            w["partition"] *= 1.5
            w["load_bearing"] *= 0.4

        return [w[s] for s in ["load_bearing", "curtain_wall", "partition"]]

    def _weights_roof_type(self, parents: dict[str, str]) -> list[float]:
        """P(roof_type | aesthetic_feel, height_restriction, wind_exposure)
        Returns weights for: flat, pitched, green_roof
        """
        w = {"flat": 1.0, "pitched": 1.0, "green_roof": 1.0}

        # Design intent first
        aesthetic = parents.get("aesthetic_feel")
        if aesthetic == "minimalist":
            w["flat"] *= 3.0
            w["pitched"] *= 0.5
        elif aesthetic == "industrial":
            w["flat"] *= 2.5
        elif aesthetic == "organic":
            w["green_roof"] *= 4.0
            w["pitched"] *= 1.5
        elif aesthetic == "classical":
            w["pitched"] *= 3.5
            w["flat"] *= 0.5

        # Constraints
        height = parents.get("height_restriction")
        if height == "strict":
            w["flat"] *= 2.0
            w["pitched"] *= 0.5

        wind = parents.get("wind_exposure")
        if wind == "exposed":
            w["flat"] *= 1.5
            w["green_roof"] *= 0.6
        elif wind == "sheltered":
            w["green_roof"] *= 1.3

        return [w[s] for s in ["flat", "pitched", "green_roof"]]

    def _weights_sketch_topology(self, parents: dict[str, str]) -> list[float]:
        """P(sketch_topology | setback_class, solar_orientation)
        Returns weights for: compact, linear, courtyard
        """
        w = {"compact": 1.0, "linear": 1.0, "courtyard": 1.0}

        setback = parents.get("setback_class")
        if setback == "generous":
            w["courtyard"] *= 2.5
            w["linear"] *= 1.5
            w["compact"] *= 0.7
        elif setback == "standard":
            w["compact"] *= 2.0
            w["courtyard"] *= 0.5

        solar = parents.get("solar_orientation")
        if solar == "south":
            w["linear"] *= 1.5  # elongate east-west for max south exposure
        elif solar == "east" or solar == "west":
            w["compact"] *= 1.3

        return [w[s] for s in ["compact", "linear", "courtyard"]]

    def _weights_wall_thickness_class(self, parents: dict[str, str]) -> list[float]:
        """P(wall_thickness_class | seismic_zone_class, structural_system, num_floors)
        Returns weights for: thin, standard, thick
        """
        w = {"thin": 1.0, "standard": 1.0, "thick": 1.0}

        seismic = parents.get("seismic_zone_class")
        if seismic == "high_risk":
            w["thick"] *= 3.0
            w["thin"] *= 0.2
        elif seismic == "moderate_risk":
            w["standard"] *= 1.5
            w["thick"] *= 1.5
            w["thin"] *= 0.5

        system = parents.get("structural_system")
        if system == "timber_frame":
            w["thin"] *= 2.0
            w["thick"] *= 0.5
        elif system == "reinforced_concrete":
            w["standard"] *= 1.5
            w["thick"] *= 1.5
        elif system == "masonry":
            w["thick"] *= 2.0
            w["standard"] *= 1.5

        floors = parents.get("num_floors")
        if floors == "8_plus_floors":
            w["thick"] *= 2.5
            w["thin"] *= 0.2
        elif floors == "4_7_floors":
            w["standard"] *= 1.5
            w["thick"] *= 1.3
        elif floors == "1_floor":
            w["thin"] *= 1.5

        return [w[s] for s in ["thin", "standard", "thick"]]
