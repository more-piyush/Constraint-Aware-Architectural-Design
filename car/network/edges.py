"""Edge registry for the architectural design Bayesian network.

23 directed edges encoding structural, regulatory, and aesthetic dependencies.
Design intent edges have higher weights to prioritize the architect's vision.
"""

from __future__ import annotations

from car.models.network_spec import EdgeSpec


class EdgeRegistry:
    """Registry of all directed edges in the Bayesian network."""

    def get_all_edges(self) -> list[EdgeSpec]:
        return (
            self._design_intent_edges()
            + self._regulatory_edges()
            + self._environmental_edges()
            + self._structural_edges()
            + self._cross_domain_edges()
        )

    def _design_intent_edges(self) -> list[EdgeSpec]:
        """Design intent drives the generative process (higher weights)."""
        return [
            EdgeSpec(
                parent="aesthetic_feel",
                child="structural_system",
                weight=0.9,
                rationale="Minimalist favors steel; industrial favors exposed concrete",
            ),
            EdgeSpec(
                parent="aesthetic_feel",
                child="window_size",
                weight=0.85,
                rationale="Minimalist favors large glass; classical favors medium",
            ),
            EdgeSpec(
                parent="aesthetic_feel",
                child="wall_type",
                weight=0.8,
                rationale="Industrial favors curtain wall; classical favors load-bearing",
            ),
            EdgeSpec(
                parent="aesthetic_feel",
                child="roof_type",
                weight=0.8,
                rationale="Organic favors green roofs; minimalist favors flat",
            ),
            EdgeSpec(
                parent="view_priority",
                child="window_size",
                weight=0.9,
                rationale="High view priority drives large/full-glass windows",
            ),
            EdgeSpec(
                parent="sketch_topology",
                child="num_floors",
                weight=0.7,
                rationale="Compact topology allows taller buildings; linear stays low",
            ),
        ]

    def _regulatory_edges(self) -> list[EdgeSpec]:
        """Regulatory constraints filter/shape the design."""
        return [
            EdgeSpec(
                parent="far_class",
                child="num_floors",
                weight=0.7,
                rationale="Higher FAR allows more floors",
            ),
            EdgeSpec(
                parent="height_restriction",
                child="num_floors",
                weight=0.8,
                rationale="Strict height limits cap floor count",
            ),
            EdgeSpec(
                parent="height_restriction",
                child="roof_type",
                weight=0.6,
                rationale="Strict limits favor flat roofs to maximize usable height",
            ),
            EdgeSpec(
                parent="setback_class",
                child="sketch_topology",
                weight=0.5,
                rationale="Generous setbacks favor courtyard layouts",
            ),
            EdgeSpec(
                parent="seismic_zone_class",
                child="structural_system",
                weight=0.8,
                rationale="High seismic risk strongly favors reinforced concrete or hybrid",
            ),
            EdgeSpec(
                parent="seismic_zone_class",
                child="wall_thickness_class",
                weight=0.7,
                rationale="High seismic risk requires thicker walls",
            ),
            EdgeSpec(
                parent="far_class",
                child="window_size",
                weight=0.4,
                rationale="High FAR incentivizes curtain walls for density",
            ),
        ]

    def _environmental_edges(self) -> list[EdgeSpec]:
        """Environmental conditions influence design choices."""
        return [
            EdgeSpec(
                parent="solar_orientation",
                child="window_size",
                weight=0.5,
                rationale="South-facing (northern hemisphere) encourages larger windows",
            ),
            EdgeSpec(
                parent="solar_orientation",
                child="sketch_topology",
                weight=0.4,
                rationale="Orientation influences optimal building shape",
            ),
            EdgeSpec(
                parent="wind_exposure",
                child="window_size",
                weight=0.5,
                rationale="Exposed sites reduce viable window area",
            ),
            EdgeSpec(
                parent="wind_exposure",
                child="roof_type",
                weight=0.5,
                rationale="Exposed sites favor flat or aerodynamic roofs",
            ),
        ]

    def _structural_edges(self) -> list[EdgeSpec]:
        """Structural dependencies between design components."""
        return [
            EdgeSpec(
                parent="structural_system",
                child="wall_type",
                weight=0.9,
                rationale="Steel frame enables curtain walls; masonry requires load-bearing",
            ),
            EdgeSpec(
                parent="structural_system",
                child="wall_thickness_class",
                weight=0.7,
                rationale="Timber needs thinner walls; concrete thicker",
            ),
            EdgeSpec(
                parent="material_class",
                child="structural_system",
                weight=0.7,
                rationale="Available materials constrain the structural system",
            ),
            EdgeSpec(
                parent="num_floors",
                child="structural_system",
                weight=0.6,
                rationale="Many floors require steel or concrete",
            ),
        ]

    def _cross_domain_edges(self) -> list[EdgeSpec]:
        """Cross-domain dependencies."""
        return [
            EdgeSpec(
                parent="num_floors",
                child="wall_thickness_class",
                weight=0.5,
                rationale="More floors require thicker structural walls",
            ),
            EdgeSpec(
                parent="wall_thickness_class",
                child="wall_type",
                weight=0.6,
                rationale="Thick walls enable load-bearing; thin require framing",
            ),
        ]
