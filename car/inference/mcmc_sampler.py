"""MCMC sampler using PyMC for generating multiple design iterations.

Translates the discrete Bayesian network into a PyMC model for posterior
sampling. Hard constraints are encoded as pm.Potential log-probability penalties.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from car.config import (
    DEFAULT_MCMC_CHAINS,
    DEFAULT_MCMC_DRAWS,
    DEFAULT_MCMC_TUNE,
    DEFAULT_RANDOM_SEED,
    FLOOR_HEIGHT_MIDPOINTS,
    HARD_CONSTRAINT_PENALTY,
)
from car.models.constraints import SiteConstraints

logger = logging.getLogger(__name__)

DESIGN_VARIABLES = [
    "structural_system",
    "num_floors",
    "window_size",
    "wall_type",
    "roof_type",
]

# State name mappings for decoding
STATE_NAMES = {
    "structural_system": [
        "steel_frame", "reinforced_concrete", "timber_frame",
        "masonry", "hybrid_steel_concrete",
    ],
    "num_floors": ["1_floor", "2_3_floors", "4_7_floors", "8_plus_floors"],
    "window_size": ["small", "medium", "large", "full_glass"],
    "wall_type": ["load_bearing", "curtain_wall", "partition"],
    "roof_type": ["flat", "pitched", "green_roof"],
}


class MCMCSampler:
    """Generates multiple design iterations via MCMC sampling using PyMC."""

    def sample_designs(
        self,
        marginal_probs: dict[str, dict[str, float]],
        site_constraints: SiteConstraints,
        num_chains: int = DEFAULT_MCMC_CHAINS,
        num_draws: int = DEFAULT_MCMC_DRAWS,
        num_tune: int = DEFAULT_MCMC_TUNE,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> tuple[Any, list[dict[str, str]]]:
        """Sample from the posterior distribution of design variables.

        Uses marginal probabilities from the BN as priors for PyMC categoricals,
        then adds hard constraint potentials to filter infeasible designs.

        Returns (inference_data, list_of_decoded_design_dicts).
        """
        try:
            import pymc as pm
        except ImportError:
            logger.warning("PyMC not available, falling back to direct sampling from marginals")
            return self._fallback_sample(marginal_probs, num_draws * num_chains)

        with pm.Model():
            design_vars = {}

            for var_name in DESIGN_VARIABLES:
                if var_name in marginal_probs:
                    probs = marginal_probs[var_name]
                    states = STATE_NAMES[var_name]
                    p = np.array([probs.get(s, 1.0 / len(states)) for s in states])
                    p = p / p.sum()
                else:
                    states = STATE_NAMES[var_name]
                    p = np.ones(len(states)) / len(states)

                design_vars[var_name] = pm.Categorical(var_name, p=p)

            # Hard constraint potentials
            self._add_height_constraint(pm, design_vars, site_constraints)
            self._add_seismic_constraint(pm, design_vars, site_constraints)

            try:
                idata = pm.sample(
                    draws=num_draws,
                    tune=num_tune,
                    chains=num_chains,
                    random_seed=random_seed,
                    return_inferencedata=True,
                    progressbar=False,
                )
            except Exception as e:
                logger.warning(f"PyMC sampling failed ({e}), falling back to direct sampling")
                return self._fallback_sample(marginal_probs, num_draws * num_chains)

        decoded = self._decode_samples(idata)
        return idata, decoded

    def _add_height_constraint(self, pm: Any, design_vars: dict, constraints: SiteConstraints) -> None:
        """Penalize floor counts that exceed the height limit."""
        max_height = constraints.regulatory.height_limit_m
        floor_heights = list(FLOOR_HEIGHT_MIDPOINTS.values())

        num_floors_var = design_vars["num_floors"]

        for idx, h in enumerate(floor_heights):
            if h > max_height:
                penalty = pm.math.switch(
                    pm.math.eq(num_floors_var, idx),
                    HARD_CONSTRAINT_PENALTY,
                    0.0,
                )
                pm.Potential(f"height_constraint_{idx}", penalty)

    def _add_seismic_constraint(self, pm: Any, design_vars: dict, constraints: SiteConstraints) -> None:
        """Penalize weak structural systems in high seismic zones."""
        zone = constraints.geophysical.seismic_zone.value
        if zone < 4:
            return

        structural_var = design_vars["structural_system"]
        # timber_frame=2, masonry=3 should be penalized in high seismic zones
        for idx, name in enumerate(STATE_NAMES["structural_system"]):
            if name in ("timber_frame", "masonry"):
                penalty = pm.math.switch(
                    pm.math.eq(structural_var, idx),
                    HARD_CONSTRAINT_PENALTY,
                    0.0,
                )
                pm.Potential(f"seismic_constraint_{name}", penalty)

    def _decode_samples(self, idata: Any) -> list[dict[str, str]]:
        """Convert integer indices in MCMC trace back to state names."""
        decoded = []
        posterior = idata.posterior
        for chain in range(posterior.sizes["chain"]):
            for draw in range(posterior.sizes["draw"]):
                sample = {}
                for var_name in DESIGN_VARIABLES:
                    idx = int(posterior[var_name].values[chain, draw])
                    states = STATE_NAMES[var_name]
                    sample[var_name] = states[min(idx, len(states) - 1)]
                decoded.append(sample)
        return decoded

    def _fallback_sample(
        self,
        marginal_probs: dict[str, dict[str, float]],
        n_samples: int,
    ) -> tuple[None, list[dict[str, str]]]:
        """Direct sampling from marginals when PyMC is unavailable."""
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
        decoded = []
        for _ in range(n_samples):
            sample = {}
            for var_name in DESIGN_VARIABLES:
                if var_name in marginal_probs:
                    probs = marginal_probs[var_name]
                    states = list(probs.keys())
                    p = np.array(list(probs.values()))
                    p = p / p.sum()
                    sample[var_name] = rng.choice(states, p=p)
                else:
                    states = STATE_NAMES[var_name]
                    sample[var_name] = rng.choice(states)
            decoded.append(sample)
        return None, decoded

    def get_diagnostics(self, idata: Any) -> dict[str, Any]:
        """Extract convergence diagnostics from inference data."""
        if idata is None:
            return {"method": "fallback_sampling", "note": "PyMC unavailable"}

        try:
            import arviz as az
            summary = az.summary(idata, var_names=DESIGN_VARIABLES)
            return {
                "r_hat": summary["r_hat"].to_dict() if "r_hat" in summary.columns else {},
                "ess_bulk": summary["ess_bulk"].to_dict() if "ess_bulk" in summary.columns else {},
                "ess_tail": summary["ess_tail"].to_dict() if "ess_tail" in summary.columns else {},
            }
        except Exception:
            return {"note": "Diagnostics unavailable"}
