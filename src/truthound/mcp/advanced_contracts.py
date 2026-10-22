"""Typed advanced-domain inputs; no generic namespace/method invocation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.contracts import Name, Names, WireModel
from truthound.mcp.session_contracts import Handle  # noqa: TC001 - Pydantic runtime resolution


class DriftCompare(WireModel):
    baseline_id: Handle
    current_id: Handle
    columns: Names | None = None
    method: Literal[
        "auto",
        "ks",
        "psi",
        "chi2",
        "js",
        "kl",
        "wasserstein",
        "cvm",
        "anderson",
        "hellinger",
        "bhattacharyya",
        "tv",
        "total_variation",
        "energy",
        "mmd",
    ] = "auto"
    threshold: float | None = None
    sample_size: Annotated[int, Field(ge=1, le=10_000_000)] | None = None


class ModelSpec(WireModel):
    algorithm: Name
    config: dict[str, JsonValue] = Field(default_factory=dict)
    detectors: Annotated[list[ModelSpec], Field(max_length=8)] | None = None

    @model_validator(mode="after")
    def constructor(self):
        from truthound.mcp.model_wire import validate_spec

        validate_spec(self.model_dump())
        return self


class DriftDetect(WireModel):
    detector: Name
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    baseline_id: Handle
    current_id: Handle
    baseline_column: Name
    current_column: Name | None = None

    @model_validator(mode="after")
    def constructor(self):
        from truthound.mcp.policy import MCPError
        from truthound.mcp.validator_wire import contracts, fields

        row = contracts()["drift_detectors"].get(self.detector)
        if row is None:
            raise MCPError("unknown_detector")
        fields(row["fields"], self.settings)
        return self


class ModelCatalog(WireModel):
    offset: Annotated[int, Field(ge=0, le=10000)] = 0
    limit: Annotated[int, Field(ge=1, le=5)] = 3


class ModelInput(WireModel):
    model_id: Handle


class ModelData(ModelInput):
    asset_id: Handle
    expected_revision: Annotated[int, Field(ge=0)]


class ModelDetect(ModelData):
    reference_id: Handle
    columns: Names | None = None


class ModelRuleQuery(ModelData):
    action: Literal["suite", "by_type", "by_column", "validator_specs"]
    rule_type: Name | None = None
    column: Name | None = None

    @model_validator(mode="after")
    def selectors(self):
        if (self.action == "by_type") != (self.rule_type is not None):
            raise ValueError("rule_type is required only for by_type")
        if (self.action == "by_column") != (self.column is not None):
            raise ValueError("column is required only for by_column")
        return self


class ModelInspect(ModelInput):
    expected_revision: Annotated[int, Field(ge=0)]


class ModelRestore(WireModel):
    resource_id: Handle


class ModelState(WireModel):
    model_id: Handle
    algorithm: str
    revision: int
    state: str
    trained: bool
