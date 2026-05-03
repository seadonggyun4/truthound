"""Versioned coverage contract for the AI validation intent DSL.

The matrix is deliberately data-backed so supported intent/param policy changes
are reviewed as contract diffs instead of being hidden inside compiler branches.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

from pydantic import Field, model_validator

from truthound.ai.context import SUPPORTED_INTENT_NAMES
from truthound.ai.models import BaseStrictModel

ColumnDTypePolicy = Literal["any", "numeric", "string"]
ApproximationPolicy = Literal["forbidden"]

_DEFAULT_COVERAGE_RESOURCE = "data/validation_dsl_coverage.v1.json"


class IntentCoverage(BaseStrictModel):
    required_columns: bool
    allowed_params: tuple[str, ...] = ()
    required_any_params: tuple[str, ...] = ()
    column_dtype_policy: ColumnDTypePolicy = "any"
    validator_mapping: str
    deterministic_extractor_supported: bool = False
    automatic_approximation: ApproximationPolicy = "forbidden"

    @model_validator(mode="after")
    def _validate_param_contract(self) -> IntentCoverage:
        unknown_required = sorted(set(self.required_any_params) - set(self.allowed_params))
        if unknown_required:
            raise ValueError(f"required_any_params must be allowed params: {unknown_required}")
        if not self.validator_mapping.strip():
            raise ValueError("validator_mapping must be set")
        return self


class ValidationDSLCoverageMatrix(BaseStrictModel):
    schema_version: str
    dsl_version: str
    intents: dict[str, IntentCoverage] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> ValidationDSLCoverageMatrix:
        if self.schema_version != "1":
            raise ValueError("validation DSL coverage schema_version must be '1'")
        if self.dsl_version != "validation-dsl-v1":
            raise ValueError("validation DSL coverage dsl_version must be 'validation-dsl-v1'")
        expected = set(SUPPORTED_INTENT_NAMES)
        actual = set(self.intents)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"validation DSL coverage intents mismatch: missing={missing}, extra={extra}")
        return self

    @property
    def deterministic_intents(self) -> tuple[str, ...]:
        return tuple(
            intent
            for intent, coverage in self.intents.items()
            if coverage.deterministic_extractor_supported
        )


def load_validation_dsl_coverage_from_text(text: str) -> ValidationDSLCoverageMatrix:
    payload: Any = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("validation DSL coverage matrix must be a JSON object")
    return ValidationDSLCoverageMatrix.model_validate(payload)


@lru_cache(maxsize=1)
def get_default_validation_dsl_coverage() -> ValidationDSLCoverageMatrix:
    text = (
        resources.files("truthound.ai")
        .joinpath(_DEFAULT_COVERAGE_RESOURCE)
        .read_text(encoding="utf-8")
    )
    return load_validation_dsl_coverage_from_text(text)


__all__ = [
    "IntentCoverage",
    "ValidationDSLCoverageMatrix",
    "get_default_validation_dsl_coverage",
    "load_validation_dsl_coverage_from_text",
]
