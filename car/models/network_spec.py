"""Bayesian network structure specification models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    """Specification for a single node in the Bayesian network."""

    name: str
    variable_type: str  # "latent", "observed", "decision"
    category: str  # "intent", "regulatory", "environmental", "geophysical", "technical", "design"
    cardinality: int = Field(..., ge=2)
    state_names: list[str]
    description: str


class EdgeSpec(BaseModel):
    """Specification for a directed edge in the Bayesian network."""

    parent: str
    child: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str


class NetworkTopology(BaseModel):
    """Complete specification of the Bayesian network structure."""

    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
