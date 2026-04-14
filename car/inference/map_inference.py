"""MAP inference engine using pgmpy VariableElimination."""

from __future__ import annotations

from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


DESIGN_VARIABLES = [
    "structural_system",
    "num_floors",
    "window_size",
    "wall_type",
    "roof_type",
]


class MAPInferenceEngine:
    """Performs MAP inference to find the single best design configuration."""

    def find_map_design(
        self,
        model: DiscreteBayesianNetwork,
        evidence: dict[str, str],
        design_intent: dict[str, str],
    ) -> dict[str, str]:
        """Find the MAP assignment of design variables given evidence.

        Design intent is set as evidence with high priority (it was already
        encoded with strong priors in the CPDs). Constraints are layered on.
        """
        full_evidence = {**evidence, **design_intent}

        # Remove any evidence for variables that are also decision variables
        query_vars = [v for v in DESIGN_VARIABLES if v not in full_evidence]

        if not query_vars:
            return {v: full_evidence[v] for v in DESIGN_VARIABLES if v in full_evidence}

        inference = VariableElimination(model)
        map_result = inference.map_query(
            variables=query_vars,
            evidence=full_evidence,
            show_progress=False,
        )
        return dict(map_result)

    def query_marginals(
        self,
        model: DiscreteBayesianNetwork,
        evidence: dict[str, str],
        variables: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Query marginal probabilities for design variables.

        Used for confidence scoring -- shows how confident the model is
        in each design choice given the evidence.
        """
        if variables is None:
            variables = DESIGN_VARIABLES

        # Filter out variables that are in evidence
        query_vars = [v for v in variables if v not in evidence]

        inference = VariableElimination(model)
        results = {}
        for var in query_vars:
            factor = inference.query(
                variables=[var],
                evidence=evidence,
                show_progress=False,
            )
            state_names = factor.state_names[var]
            values = factor.values
            results[var] = {
                state: float(values[i]) for i, state in enumerate(state_names)
            }
        return results
