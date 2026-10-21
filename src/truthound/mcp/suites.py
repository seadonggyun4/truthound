"""Versioned declarative suites compiled into native immutable kernel intent."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import AssetValidation, Handle, RegistryCheck
from truthound.mcp.validator_wire import _known_type, contracts, fields, record_fields


class SuiteCheck(RegistryCheck):
    check_id: Name
    check_name: Name | None = None
    category: Name | None = None
    tags: Annotated[list[Name], Field(max_length=100)] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SuiteSpec(WireModel):
    contract_version: Annotated[int, Field(ge=1, le=1)] = 1
    name: Name = "mcp-suite"
    checks: Annotated[list[SuiteCheck], Field(max_length=1000)] = Field(default_factory=list)
    schema_id: Handle | None = None
    schema_auto_learned: bool = False
    schema_source: Name | None = None
    evidence_policy: dict[str, JsonValue] = Field(default_factory=dict)
    severity_policy: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def policies(self):
        if not self.checks and self.schema_id is None:
            raise ValueError("A suite requires checks or a schema")
        if self.schema_id is None and (self.schema_auto_learned or self.schema_source is not None):
            raise ValueError("Schema metadata requires a schema")
        if self.schema_id is not None and any(row.check_id == "schema" for row in self.checks):
            raise ValueError("Schema check ID is reserved")
        if len({row.check_id for row in self.checks}) != len(self.checks):
            raise ValueError("Duplicate suite check ID")
        for key in ("evidence_policy", "severity_policy"):
            fields(record_fields(contracts()["suite"][key]), getattr(self, key))
        return self


class SuiteCreate(WireModel):
    spec: SuiteSpec


class SuiteValidation(WireModel):
    asset_id: Handle
    suite_id: Handle
    context_id: Handle | None = None
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)] | None = None
    evidence: Literal["summary", "full"] = "summary"
    parallel: bool = False
    max_workers: Annotated[int, Field(ge=1, le=8)] | None = None
    pushdown: bool | None = None


def effective_arguments(arguments, suite):
    values = arguments.model_dump(exclude={"suite_id"})
    needs_intent = bool(arguments.context_id) or any(check.binding_id for check in suite.checks)
    if needs_intent != bool(arguments.idempotency_key):
        raise MCPError("invalid_arguments")
    # Values and SuiteCheck objects were validated by closed public contracts.
    # Keep this internal adaptation separate from public override acceptance.
    return AssetValidation.model_construct(**values, checks=suite.checks, schema_id=suite.schema_id)


def compile_suite(specification, validators, schema=None):
    """Only reviewed factories; the wire never selects import paths or code."""
    if len(validators) != len(specification.checks):
        raise MCPError("invalid_suite")
    row = contracts()["suite"]
    evidence = _known_type(row["evidence_policy"]["target"])(
        **fields(
            record_fields(row["evidence_policy"]),
            specification.evidence_policy,
            build=True,
        )
    )
    severity = _known_type(row["severity_policy"]["target"])(
        **fields(
            record_fields(row["severity_policy"]),
            specification.severity_policy,
            build=True,
        )
    )
    CheckSpec = _known_type("truthound.core.suite.CheckSpec")
    Suite = _known_type("truthound.core.suite.ValidationSuite")
    checks = tuple(
        CheckSpec(
            id=check.check_id,
            name=check.check_name or validator.name,
            category=check.category or validator.category,
            factory=lambda v=validator: v,
            tags=tuple(check.tags),
            metadata=check.metadata,
        )
        for check, validator in zip(specification.checks, validators, strict=True)
    )
    schema_spec = None
    if schema is not None:
        SchemaSpec = _known_type("truthound.core.suite.SchemaSpec")
        schema_spec = SchemaSpec(
            schema=schema,
            auto_learned=specification.schema_auto_learned,
            source=specification.schema_source,
        )
        checks = (schema_spec.to_check_spec(evidence_policy=evidence), *checks)
    return Suite(
        name=specification.name,
        checks=checks,
        evidence_policy=evidence,
        severity_policy=severity,
        schema_spec=schema_spec,
        metadata=specification.metadata,
    )


def run_suite(specification, validators, frame, arguments, context, schema=None):
    from truthound.core import ScanPlanner, ValidationRuntime, build_validation_asset

    suite = compile_suite(specification, validators, schema)
    asset = build_validation_asset(frame, pushdown=arguments.pushdown)
    plan = ScanPlanner().plan(
        suite=suite,
        asset=asset,
        parallel=arguments.parallel,
        max_workers=arguments.max_workers,
        pushdown=arguments.pushdown,
    )
    result = ValidationRuntime().execute(asset=asset, plan=plan)
    if suite.severity_policy.min_severity is not None:
        result = result.filter_by_severity(suite.severity_policy.min_severity)
    # A kernel suite does not synthesize a facade auto-suite/baseline. Explicit
    # context persistence still delegates to its public Core lifecycle methods.
    if arguments.context_id:
        context.track_asset(data=frame)
        result = result.with_metadata(
            context_root=str(context.root_dir),
            context_source_key=context.resolve_source_key(data=frame),
            context_history_key=context.resolve_source_key(data=frame),
            context_source_fingerprint=context.resolve_fingerprint(data=frame),
        )
        if context.config.persist_runs:
            path = context.persist_run(result)
            result = result.with_metadata(
                context_run_artifact=str(path), _truthound_run_artifact=str(path)
            )
        if context.config.persist_docs:
            path = context.persist_docs(result)
            result = result.with_metadata(context_docs_artifact=str(path))
    return result
