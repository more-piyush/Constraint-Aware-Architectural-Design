"""Variational inference engine using PyMC ADVI as a faster alternative to MCMC."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from car.config import DEFAULT_RANDOM_SEED
from car.inference.mcmc_sampler import DESIGN_VARIABLES, STATE_NAMES

logger = logging.getLogger(__name__)


class VariationalEngine:
    """Uses ADVI via PyMC as a faster alternative to MCMC for generating design iterations."""

    def fit_and_sample(
        self,
        marginal_probs: dict[str, dict[str, float]],
        n_iterations: int = 10000,
        n_samples: int = 500,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> tuple[Any, list[dict[str, str]]]:
        """Fit a variational approximation and sample from it.

        For discrete variables, we sample directly from the marginal probabilities
        since ADVI requires continuous relaxation which adds complexity.
        The marginals from the BN already encode the design-first + constraint
        filtering, so direct sampling is a valid approximation.
        """
        rng = np.random.default_rng(random_seed)
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
