"""ArviZ-based MCMC diagnostic plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DESIGN_VARIABLES = [
    "structural_system",
    "num_floors",
    "window_size",
    "wall_type",
    "roof_type",
]


class TracePlotter:
    """ArviZ-based MCMC diagnostic plots."""

    def plot_trace(self, idata: Any, output_path: str | Path = "trace_plot.png") -> None:
        """Standard trace plot for all design variables."""
        if idata is None:
            logger.warning("No inference data available for trace plot")
            return

        try:
            import arviz as az
            az.plot_trace(idata, var_names=DESIGN_VARIABLES)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate trace plot: {e}")

    def plot_posterior(self, idata: Any, output_path: str | Path = "posterior_plot.png") -> None:
        """Posterior distribution plots for design variables."""
        if idata is None:
            logger.warning("No inference data available for posterior plot")
            return

        try:
            import arviz as az
            az.plot_posterior(idata, var_names=DESIGN_VARIABLES)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate posterior plot: {e}")

    def get_diagnostics(self, idata: Any) -> dict[str, Any]:
        """Extract R-hat, ESS, and divergence counts."""
        if idata is None:
            return {"note": "No inference data available"}

        try:
            import arviz as az
            summary = az.summary(idata, var_names=DESIGN_VARIABLES)
            return {
                "r_hat": summary["r_hat"].to_dict() if "r_hat" in summary.columns else {},
                "ess_bulk": summary["ess_bulk"].to_dict() if "ess_bulk" in summary.columns else {},
                "ess_tail": summary["ess_tail"].to_dict() if "ess_tail" in summary.columns else {},
            }
        except Exception as e:
            logger.warning(f"Could not compute diagnostics: {e}")
            return {"error": str(e)}
