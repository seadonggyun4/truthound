"""Typed retained-resource and persistent-workspace contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.contracts import (
    FileInput,
    Name,
    Names,
    Page,
    SchemaSpec,  # noqa: TC001 - runtime schema
    ValidationInput,
    WireModel,
)

Handle = Annotated[
    str,
    Field(
        pattern=r"^(asset|artifact|schema|context|source|graph|model|suite|history|monitor)_[A-Za-z0-9_-]{32}$"
    ),
]


class FileReadOptions(WireModel):
    name: Name | None = None
    max_rows: Annotated[int, Field(ge=1)] = 10_000_000
    max_memory_mb: Annotated[int, Field(ge=1)] = 4096
    sample_size: Annotated[int, Field(ge=1)] = 100_000
    sample_seed: int | None = 42
    cache_schema: bool = True
    strict_types: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    infer_schema_length: Annotated[int, Field(ge=0, le=1_000_000)] = 10000
    ignore_errors: bool = False
    encoding: Literal["utf8", "utf8-lossy"] = "utf8"
    separator: Annotated[str, Field(min_length=1, max_length=1)] = ","
    rechunk: bool = False
    streaming: bool = False


class AssetCapture(FileInput):
    options: FileReadOptions = Field(default_factory=FileReadOptions)
    sample_size: Annotated[int, Field(ge=1, le=10_000_000)] | None = None


class SQLiteCapture(FileInput):
    table: Name
    materialization_row_limit: Annotated[int, Field(ge=1, le=10_000_000)] = 100_000
    fetch_size: Annotated[int, Field(ge=1, le=100_000)] = 10000
    sample_size: Annotated[int, Field(ge=1, le=100_000)] | None = None
    sample_seed: int = 42


class DType(WireModel):
    kind: Literal[
        "Null",
        "Boolean",
        "String",
        "Binary",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "Date",
        "Time",
        "Datetime",
        "Duration",
        "Decimal",
        "List",
        "Array",
        "Struct",
        "Categorical",
        "Enum",
    ]
    unit: Literal["ns", "us", "ms"] | None = None
    time_zone: Name | None = None
    inner: DType | None = None
    width: Annotated[int, Field(ge=1, le=1000)] | None = None
    fields: Annotated[dict[Name, DType], Field(max_length=100)] | None = None
    precision: Annotated[int, Field(ge=1, le=38)] | None = None
    scale: Annotated[int, Field(ge=0, le=38)] | None = None
    categories: Annotated[list[str], Field(max_length=1000)] | None = None

    @model_validator(mode="after")
    def parameters(self):
        allowed = {
            "Datetime": {"unit", "time_zone"},
            "Duration": {"unit"},
            "Decimal": {"precision", "scale"},
            "List": {"inner"},
            "Array": {"inner", "width"},
            "Struct": {"fields"},
            "Enum": {"categories"},
        }.get(self.kind, set())
        if any(
            getattr(self, name) is not None and name not in allowed
            for name in type(self).model_fields
            if name != "kind"
        ):
            raise ValueError("Datatype parameter does not apply")
        for name in {
            "List": ("inner",),
            "Array": ("inner", "width"),
            "Struct": ("fields",),
            "Enum": ("categories",),
        }.get(self.kind, ()):
            if getattr(self, name) is None:
                raise ValueError("Missing datatype parameter")
        if self.categories is not None and len(set(self.categories)) != len(self.categories):
            raise ValueError("Enum categories must be unique")
        return self


class InlineColumn(WireModel):
    name: Name
    dtype: DType
    values: Annotated[list[JsonValue], Field(max_length=10000)]


class AssetInline(WireModel):
    columns: Annotated[list[InlineColumn], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def table(self):
        if (
            len({column.name for column in self.columns}) != len(self.columns)
            or len({len(column.values) for column in self.columns}) != 1
        ):
            raise ValueError("Column names/lengths do not form a table")
        return self


class AssetInput(WireModel):
    asset_id: Handle


class AssetPage(AssetInput, Page):
    pass


class AssetMask(AssetInput):
    columns: Names | None = None
    strategy: Literal["redact", "hash", "fake"] = "redact"
    strict: bool = False


class AssetSample(AssetInput):
    n: Annotated[int, Field(ge=1, le=10_000_000)] = 1000
    seed: int | None = None


class AssetLearn(AssetInput):
    infer_constraints: bool = True
    categorical_threshold: Annotated[int, Field(ge=0, le=10000)] = 20


class ResourceInput(WireModel):
    resource_id: Handle


class SchemaCreate(WireModel):
    schema_spec: SchemaSpec


class SchemaExport(ResourceInput):
    format: Literal["json", "yaml"] = "yaml"


class ArtifactRender(ResourceInput):
    format: Literal["json", "markdown", "console", "html"] = "json"
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def closed_renderer(self):
        from truthound.mcp.policy import MCPError
        from truthound.mcp.validator_wire import contracts, fields, record_fields

        if any(self.settings.get(key) is not None for key in ("output_path", "template_path")):
            raise MCPError("operator_configuration_required")
        fields(record_fields(contracts()["reporters"][self.format]), self.settings)
        return self


class ArtifactDocs(ResourceInput):
    title: Annotated[str, Field(max_length=512)] = "Truthound Validation Data Docs"
    subtitle: Annotated[str, Field(max_length=1024)] = ""
    theme: Literal["light", "dark", "minimal"] = "light"
    locale: Literal["en", "ko"] = "en"


class ResourceExport(ResourceInput):
    format: Literal["ipc", "parquet", "csv", "json", "ndjson"] = "ipc"


class ArtifactRead(ResourceInput):
    offset: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=32768)] = 16384


class ResourceInfo(WireModel):
    resource_id: str
    kind: str
    media_type: str
    bytes: int
    ttl_seconds: int
    sha256: str | None = None
    row_count: int | None = None
    column_count: int | None = None


class Released(WireModel):
    resource_id: str
    released: Literal[True]


class ArtifactChunk(WireModel):
    resource_id: str
    media_type: str
    offset: int
    total_bytes: int
    base64: str
    sha256: str
    next_offset: int | None


class ContextConfig(WireModel):
    persist_runs: bool = True
    persist_docs: bool = True
    auto_create_baseline: bool = True
    auto_create_workspace: bool = True
    default_result_format: Literal["boolean_only", "basic", "summary", "complete"] = "summary"
    docs_theme: Name = "light"
    max_metric_history_entries: Annotated[int, Field(ge=1, le=10000)] = 100


class ContextOpen(WireModel):
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    config: ContextConfig | None = None


class ContextInput(WireModel):
    context_id: Handle


class ContextInitialize(ContextInput):
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]


class PlanInput(WireModel):
    plan_id: Annotated[str, Field(pattern=r"^plan_[A-Za-z0-9_-]{32}$")]


class ContextInfo(WireModel):
    context_id: str
    config: ContextConfig
    workspace_exists: bool


class ContextMutationResult(ContextInfo):
    plan_id: str
    replayed: bool


class RegistryCheck(WireModel):
    validator: Name
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    binding_id: Name | None = None

    @model_validator(mode="after")
    def closed_constructor(self):
        from truthound.mcp.validator_wire import settings

        settings(self.validator, self.settings, partial=self.binding_id is not None)
        return self


class ValidatorDescribe(WireModel):
    validator: Name


class ValidatorPage(WireModel):
    offset: Annotated[int, Field(ge=0, le=10000)] = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 50


class AssetValidation(ValidationInput):
    # The source is a retained DataFrame. The legacy file operation is unchanged.
    path: Literal[""] = ""
    asset_id: Handle
    checks: Annotated[list[RegistryCheck], Field(min_length=1, max_length=1000)] | None = None
    schema_id: Handle | None = None
    context_id: Handle | None = None
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)] | None = None

    @model_validator(mode="after")
    def persistent_intent(self):
        needs_intent = bool(self.context_id) or any(check.binding_id for check in self.checks or ())
        if needs_intent != bool(self.idempotency_key):
            raise ValueError("Persistent validation requires a separate idempotent intent")
        return self


class ValidationArtifact(WireModel):
    summary: dict[str, JsonValue]
    artifact: ResourceInfo | None
    artifact_available: bool
    replayed: bool


class OperationPlan(WireModel):
    plan_id: str
    operation_id: str
    state: Literal[
        "awaiting_approval",
        "approved",
        "executing",
        "succeeded",
        "failed",
        "outcome_unknown",
        "stale",
        "expired",
        "revoked",
    ]
    expires_at: float
    request_digest: str
    effects: list[str]
