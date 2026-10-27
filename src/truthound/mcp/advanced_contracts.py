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
    threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
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


class ModelResultQuery(ModelData):
    action: Literal["all_scores", "anomalies", "drifted_columns"]
    threshold: Annotated[float, Field(allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def selector(self):
        if self.threshold is not None and self.action != "drifted_columns":
            raise ValueError("threshold is supported only for drifted_columns")
        return self


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


class EnsembleAdd(ModelInspect):
    detector: ModelSpec | None = None
    source_model_id: Handle | None = None
    source_revision: Annotated[int, Field(ge=0)] | None = None
    weight: Annotated[float, Field(allow_inf_nan=False)] = 1.0

    @model_validator(mode="after")
    def source(self):
        if (self.detector is None) == (self.source_model_id is None):
            raise ValueError("Select a constructor or an existing model, not both")
        if (self.source_model_id is None) != (self.source_revision is None):
            raise ValueError("Existing models require their exact revision")
        if self.source_model_id == self.model_id:
            raise ValueError("An ensemble cannot contain itself")
        return self


class EnsembleRemove(ModelInspect):
    index: Annotated[int, Field(ge=-8, le=7)]


class ModelState(WireModel):
    model_id: Handle
    algorithm: str
    revision: int
    state: str
    trained: bool
