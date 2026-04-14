"""Inference result data models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from car.models.design import BuildingDesign


class ConstraintViolation(BaseModel):
    """A single constraint that was violated."""

    constraint_name: str
    constraint_type: str  # "regulatory", "environmental", "geophysical", "technical"
    required_value: str
    actual_value: str
    severity: str  # "hard" or "soft"


class ComplianceResult(BaseModel):
    """Result of checking a design against all constraints."""

    is_compliant: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    violations: list[ConstraintViolation] = Field(default_factory=list)
    checked_constraints_count: int
    passed_constraints_count: int


class DesignIteration(BaseModel):
    """A single design produced by sampling."""

    iteration_id: int
    design: BuildingDesign
    compliance: ComplianceResult
    log_probability: float = 0.0
    aesthetic_score: float = Field(default=0.5, ge=0.0, le=1.0)
    view_score: float = Field(default=0.5, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.5, ge=0.0, le=1.0)


class InferenceResult(BaseModel):
    """Complete result from the inference pipeline."""

    map_design: DesignIteration
    sampled_designs: list[DesignIteration] = Field(default_factory=list)
    convergence_diagnostics: dict[str, Any] = Field(default_factory=dict)
    inference_method: str = "map"
    elapsed_seconds: float = 0.0
