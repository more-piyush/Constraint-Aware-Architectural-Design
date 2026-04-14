"""Bayesian network DAG visualization with category-colored nodes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork

from car.models.network_spec import NetworkTopology


CATEGORY_COLORS = {
    "intent": "#E8D5B7",
    "regulatory": "#FFB3B3",
    "environmental": "#B3FFB3",
    "geophysical": "#FFE0B3",
    "technical": "#B3D9FF",
    "design": "#D9B3FF",
}


class NetworkPlotter:
    """Visualizes the Bayesian network DAG with categorical coloring."""

    def plot(
        self,
        model: DiscreteBayesianNetwork,
        topology: NetworkTopology,
        output_path: str | Path = "network_graph.png",
        figsize: tuple[int, int] = (18, 12),
    ) -> None:
        """Render the Bayesian network as a layered DAG."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        G = nx.DiGraph(model.edges())

        node_category = {n.name: n.category for n in topology.nodes}

        # Layer assignment for hierarchical layout
        layer_map = {}
        for node in topology.nodes:
            if node.variable_type == "latent":
                layer_map[node.name] = 0
            elif node.variable_type == "observed":
                layer_map[node.name] = 1
            else:
                layer_map[node.name] = 2

        # Use multipartite layout for layered hierarchy
        for node_name in G.nodes():
            G.nodes[node_name]["subset"] = layer_map.get(node_name, 2)

        try:
            pos = nx.multipartite_layout(G, subset_key="subset", scale=3)
        except Exception:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        node_colors = [
            CATEGORY_COLORS.get(node_category.get(n, "design"), "#CCCCCC")
            for n in G.nodes()
        ]

        # Draw
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=2500,
            edgecolors="#333333",
            linewidths=1.5,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=7,
            font_weight="bold",
            font_family="monospace",
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            arrows=True,
            arrowsize=15,
            edge_color="#666666",
            width=1.5,
            connectionstyle="arc3,rad=0.1",
        )

        # Legend
        patches = [
            mpatches.Patch(color=color, label=category.title())
            for category, color in CATEGORY_COLORS.items()
        ]
        ax.legend(
            handles=patches, loc="upper left", fontsize=9,
            framealpha=0.9, edgecolor="#333333",
        )

        # Layer labels
        ax.set_title(
            "Architectural Design Bayesian Network\n"
            "(Layer 0: Design Intent | Layer 1: Constraints | Layer 2: Design Decisions)",
            fontsize=13, fontweight="bold", pad=20,
        )

        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
