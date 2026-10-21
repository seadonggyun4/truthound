"""Explicit Core bindings. No model-selected imports, evaluation or shell calls."""

from __future__ import annotations

from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.projection import project


def envelope(
    operation: str,
    *,
    data=None,
    quality="not_applicable",
    error=None,
    coverage=None,
    provenance=None,
) -> dict:
    return {
        "operation_id": operation,
        "contract_version": 1,
        "execution_status": "outcome_unknown"
        if error == "outcome_unknown"
        else ("failed" if error else "succeeded"),
        "quality_status": "unknown" if error else quality,
        "data": data,
        "issues": [{"code": error}] if error else [],
        "coverage": coverage or {},
        "provenance": provenance or {},
    }


def _context(path: str):
    from truthound.context import TruthoundContext, TruthoundContextConfig

    return TruthoundContext(
        Path(path).parent,
        config=TruthoundContextConfig(
            persist_runs=False,
            persist_docs=False,
            # Auto-suite semantics require a baseline. It is owned by the
            # disposable worker directory, never the admitted user project.
            auto_create_baseline=True,
            auto_create_workspace=False,
        ),
    )


def _require_lossless_rows(dtype):
    import polars as pl

    # Polars to_dicts() converts temporal scalars to Python microseconds.
    # Reject unsupported precision rather than silently dropping nanoseconds.
    if dtype.base_type() in {pl.Datetime, pl.Duration} and dtype.time_unit == "ns":
        raise MCPError("unsupported_value")
    if dtype.base_type() == pl.Time:
        raise MCPError("unsupported_value")
    if dtype.base_type() in {pl.List, pl.Array}:
        _require_lossless_rows(dtype.inner)
    if dtype.base_type() == pl.Struct:
        for field in dtype.fields:
            _require_lossless_rows(field.dtype)


def _complete_targets(path, arguments, validators) -> bool:
    """Use Core's selector, not an issue count, to detect unapplied wire rules.

    Core results do not yet record auto-suite target coverage. Preserve that as
    unknown rather than inventing completeness from declared CheckResult rows.
    """
    if validators is None:
        return False
    from truthound.adapters import to_lazyframe
    from truthound.validators.base import NUMERIC_TYPES, STRING_TYPES, SchemaValidator

    lf = to_lazyframe(path)
    for validator in validators:
        requested = list(validator.config.columns) if validator.config.columns else None
        if validator.name == "duplicate" and validator.subset:
            requested = validator.subset
        dtype_filter = {"between": NUMERIC_TYPES, "length": STRING_TYPES}.get(validator.name)
        available = SchemaValidator.get_safe_columns(lf, requested, dtype_filter)
        excluded = set(arguments.exclude_columns or ()) | set(
            validator.config.exclude_columns or ()
        )
        targets = set(available) - excluded
        if not targets or (requested and set(requested) - excluded - targets):
            return False
    return True


def execute(request: dict) -> dict:
    import truthound as th
    from truthound.mcp.catalog import CATALOG

    name = request["operation"]
    if CATALOG[name].route == "session":
        from truthound.mcp.session_worker import execute as session_execute

        return session_execute(request)
    arguments = CATALOG[name].input_type.model_validate(request["arguments"])
    path = request["path"]
    maximum = request["max_items"]
    coverage = {}
    quality = "not_applicable"
    if name == "validation_check":
        from truthound.schema import Schema
        from truthound.types import Severity
        from truthound.validators import get_validator

        validators = None
        if arguments.checks is not None:
            validators = []
            for check in arguments.checks:
                config = check.model_dump(exclude={"validator"})
                if config["severity_override"] is not None:
                    config["severity_override"] = Severity(config["severity_override"])
                validators.append(get_validator(check.validator)(**config))
        options = arguments.model_dump(exclude={"path", "evidence", "checks", "schema_spec"})
        schema = (
            Schema.from_dict(arguments.schema_spec.model_dump(exclude_none=True))
            if arguments.schema_spec
            else None
        )
        result = th.check(
            path, context=_context(path), validators=validators, schema=schema, **options
        )
        data, truncated = project("check", result, maximum)
        # Core's parallel runtime currently does not retain complete per-step
        # failure/skip accounting. Never infer a pass from its aggregate alone.
        target_accounting = _complete_targets(path, arguments, validators)
        uncertain = (
            result.execution_mode == "parallel"
            or not result.checks
            or bool(result.execution_issues)
            or not target_accounting
        )
        quality = "unknown" if uncertain else ("pass" if result.success else "fail")
        coverage = {
            "returned_checks": min(len(result.checks), maximum),
            "total_checks": len(result.checks),
            "execution_issue_count": len(result.execution_issues),
            "truncated": truncated,
            "accounting_complete": not uncertain,
            "rows": result.row_count,
            "columns": result.column_count,
        }
        if result.execution_issues:
            return envelope(
                name,
                data=data,
                error="core_execution_failed",
                coverage=coverage,
                provenance=request["provenance"],
            )
        if arguments.evidence == "full":
            data = {"encoding": "tagged-v1", "result": encode(result.to_dict())}
            coverage.update(returned_checks=len(result.checks), truncated=False)
    elif name in {"data_profile", "privacy_scan"}:
        operation = "profile" if name == "data_profile" else "scan"
        result = th.profile(path) if operation == "profile" else th.scan(path)
        data, truncated = project(operation, result, maximum)
        coverage = {"rows": result.row_count, "truncated": truncated}
        if operation == "scan":
            coverage["sampling"] = "first_1000_non_null_values_per_column"
        if arguments.evidence == "full":
            data = {"encoding": "tagged-v1", "result": encode(result.to_dict())}
            coverage["truncated"] = False
    elif name == "schema_learn":
        result = th.learn(
            path,
            infer_constraints=arguments.infer_constraints,
            categorical_threshold=arguments.categorical_threshold,
        )
        data = {"encoding": "tagged-v1", "result": encode(result.to_dict())}
    elif name in {"data_read", "data_mask"}:
        if arguments.limit > maximum:
            raise MCPError("resource_exhausted")
        if name == "data_read":
            result = th.read(path, sample_size=arguments.sample_size)
        else:
            result = th.mask(
                path,
                columns=arguments.columns,
                strategy=arguments.strategy,
                strict=arguments.strict,
            )
        for dtype in result.schema.values():
            _require_lossless_rows(dtype)
        page = result.slice(arguments.offset, arguments.limit)
        end = arguments.offset + page.height
        unstable = (name == "data_read" and arguments.sample_size is not None) or (
            name == "data_mask" and arguments.strategy == "fake"
        )
        data = {
            "encoding": "tagged-v1",
            "rows": encode(page.to_dicts()),
            "schema": {key: str(dtype) for key, dtype in result.schema.items()},
            "total_rows": result.height,
            "next_offset": end if end < result.height and not unstable else None,
        }
        coverage = {
            "returned_rows": page.height,
            "total_rows": result.height,
            "truncated": end < result.height or arguments.offset > 0,
        }
        if unstable:
            coverage["sampling"] = "nondeterministic_core_result_no_cursor"
    else:
        raise MCPError("unknown_operation")
    return envelope(
        name, data=data, quality=quality, coverage=coverage, provenance=request["provenance"]
    )
