"""Strict, transport-independent wire contracts for the opt-in Core profile.

The optional MCP extra supplies Pydantic. Never import this module from Core.
No callable, provider URI, SQL or Python import path is a wire value.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Name = Annotated[str, Field(min_length=1, max_length=512)]
Names = Annotated[list[Name], Field(max_length=100)]
Scalar = str | int | float | bool | None


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Page(WireModel):
    offset: Annotated[int, Field(ge=0, le=10_000_000)] = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 50


class Describe(WireModel):
    operation_id: Annotated[str, Field(min_length=1, max_length=100)]


class FileInput(WireModel):
    path: Annotated[str, Field(min_length=1, max_length=4096)]


class EvidenceInput(FileInput):
    evidence: Literal["summary", "full"] = "summary"


class ValidatorOptions(WireModel):
    columns: Names | None = None
    exclude_columns: Names | None = None
    severity_override: Literal["low", "medium", "high", "critical"] | None = None
    mostly: Annotated[float, Field(ge=0, le=1)] | None = None
    sample_size: Annotated[int, Field(ge=0, le=100)] = 5
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] | None = 300.0
    graceful_degradation: bool = True
    log_errors: bool = True
    partial_failure_mode: Literal["collect", "raise"] = "collect"


class BasicCheck(ValidatorOptions):
    validator: Literal["null", "not_null", "unique"]


class DuplicateCheck(ValidatorOptions):
    validator: Literal["duplicate"]
    subset: Names | None = None


class BetweenCheck(ValidatorOptions):
    validator: Literal["between"]
    min_value: float | None = None
    max_value: float | None = None
    inclusive: bool = True

    @model_validator(mode="after")
    def bounds(self):
        if self.min_value is None and self.max_value is None:
            raise ValueError("A bound is required")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("Inverted bounds")
        return self


class LengthCheck(ValidatorOptions):
    validator: Literal["length"]
    min_length: Annotated[int, Field(ge=0)] | None = None
    max_length: Annotated[int, Field(ge=0)] | None = None
    exact_length: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def bounds(self):
        if all(v is None for v in (self.min_length, self.max_length, self.exact_length)):
            raise ValueError("A length constraint is required")
        if self.exact_length is not None and (
            self.min_length is not None or self.max_length is not None
        ):
            raise ValueError("Use exact_length or range, not both")
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("Inverted bounds")
        return self


Check = Annotated[
    BasicCheck | DuplicateCheck | BetweenCheck | LengthCheck,
    Field(discriminator="validator"),
]


class ColumnSpec(WireModel):
    name: Name
    dtype: Name
    nullable: bool = True
    unique: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: Annotated[list[Scalar], Field(max_length=100)] | None = None
    min_length: Annotated[int, Field(ge=0)] | None = None
    max_length: Annotated[int, Field(ge=0)] | None = None
    null_ratio: Annotated[float, Field(ge=0, le=1)] | None = None
    unique_ratio: Annotated[float, Field(ge=0, le=1)] | None = None
    mean: float | None = None
    std: float | None = None
    quantiles: dict[Name, float] | None = None


class SchemaSpec(WireModel):
    columns: Annotated[dict[Name, ColumnSpec], Field(max_length=100)]
    row_count: Annotated[int, Field(ge=0)] | None = None
    version: Name = "1.0"

    @model_validator(mode="after")
    def matching_names(self):
        if not self.columns or any(key != col.name for key, col in self.columns.items()):
            raise ValueError("Schema column identities must match")
        return self


class ValidationInput(EvidenceInput):
    checks: Annotated[list[Check], Field(min_length=1, max_length=100)] | None = None
    schema_spec: SchemaSpec | None = None
    auto_schema: bool = False
    min_severity: Literal["low", "medium", "high", "critical"] | None = None
    parallel: bool = False
    max_workers: Annotated[int, Field(ge=1, le=8)] | None = None
    pushdown: bool | None = None
    result_format: Literal["boolean_only", "basic", "summary", "complete"] = "summary"
    catch_exceptions: bool = True
    max_retries: Annotated[int, Field(ge=0, le=3)] = 0
    exclude_columns: Names | None = None


class LearnInput(FileInput):
    infer_constraints: bool = True
    categorical_threshold: Annotated[int, Field(ge=0, le=100)] = 20


class ReadInput(FileInput, Page):
    sample_size: Annotated[int, Field(ge=1, le=100_000)] | None = None

    @model_validator(mode="after")
    def no_resampling_cursor(self):
        if self.sample_size is not None and self.offset != 0:
            raise ValueError("Core random samples cannot serve stable offset pages")
        return self


class MaskInput(FileInput, Page):
    columns: Names | None = None
    strategy: Literal["redact", "hash", "fake"] = "redact"
    strict: bool = False

    @model_validator(mode="after")
    def no_random_mask_cursor(self):
        if self.strategy == "fake" and self.offset != 0:
            raise ValueError("Nondeterministic masking requires a retained artifact for paging")
        return self


class FullResult(WireModel):
    encoding: Literal["tagged-v1"]
    result: JsonValue


class CheckSummary(WireModel):
    name: str
    success: bool
    issue_count: int


class ValidationSummary(WireModel):
    row_count: int
    column_count: int
    success: bool
    issue_count: int
    execution_issue_count: int
    checks: list[CheckSummary]


class ColumnSummary(WireModel):
    name: str
    dtype: str
    null_pct: str | None = None
    unique_pct: str | None = None


class ProfileSummary(WireModel):
    row_count: int
    column_count: int
    columns: list[ColumnSummary]


class Finding(WireModel):
    column: str
    pii_type: str
    count: int
    confidence: float


class ScanSummary(WireModel):
    row_count: int
    finding_count: int
    count_semantics: Literal["estimated_from_first_1000_non_null_values"]
    findings: list[Finding]


class DataPage(WireModel):
    encoding: Literal["tagged-v1"]
    rows: list[JsonValue]
    data_schema: dict[str, str] = Field(alias="schema")
    total_rows: int
    next_offset: int | None


class CapabilitySummary(WireModel):
    operation_id: str
    description: str
    availability: Literal[
        "available",
        "approval_required",
        "operator_configuration_required",
        "missing_dependency",
        "unimplemented",
    ]


class CapabilityPage(WireModel):
    operations: list[CapabilitySummary]
    total: int
    next_offset: int | None
    full_core_parity: Literal[False]
    limitations: list[str]


class CapabilityDetail(WireModel):
    operation_id: str
    contract_version: Literal[1]
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    public_target: str
    effects: list[str]
    required_grants: list[str]
    conditional_grants: dict[str, list[str]]
    execution_mode: Literal[
        "disposable_worker", "inline_discovery", "inline_session", "resident_worker"
    ]
    context_mode: Literal["transient", "retained", "persistent"]
    required_extras: list[str]


class Coverage(WireModel):
    returned_checks: int | None = None
    total_checks: int | None = None
    execution_issue_count: int | None = None
    truncated: bool | None = None
    accounting_complete: bool | None = None
    rows: int | None = None
    columns: int | None = None
    sampling: str | None = None
    returned_rows: int | None = None
    total_rows: int | None = None


class Provenance(WireModel):
    core_version: str | None = None
    input_identity: str | None = None
    input_sha256: str | None = None
    context_mode: Literal["transient", "retained", "persistent"] | None = None
    authorization_plan_id: str | None = None
    configuration_identity: str | None = None


class ExecutionProblem(WireModel):
    code: str


class ResultEnvelope(WireModel):
    operation_id: str
    contract_version: Literal[1]
    execution_status: Literal["succeeded", "failed", "outcome_unknown"]
    quality_status: Literal["pass", "fail", "unknown", "not_applicable"]
    issues: list[ExecutionProblem]
    coverage: Coverage
    provenance: Provenance

    @model_validator(mode="after")
    def failure_is_not_quality_pass(self):
        if self.execution_status != "succeeded" and self.quality_status != "unknown":
            raise ValueError("Failed execution cannot claim a quality verdict")
        return self
