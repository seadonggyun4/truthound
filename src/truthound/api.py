"""Main API functions for Truthound.

This module uses lazy imports for heavy submodules (profiler, validators)
to improve import performance. The imports are deferred until the functions
that need them are actually called.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from truthound.adapters import to_lazyframe
from truthound.maskers import mask_data
from truthound.scanners import scan_pii
from truthound.types import ResultFormat, ResultFormatConfig, Severity


# =============================================================================
# Cached helpers - avoid repeated parsing overhead
# =============================================================================

@functools.lru_cache(maxsize=8)
def _parse_severity(s: str) -> Severity:
    """Parse severity string with caching to avoid repeated conversions."""
    return Severity(s.lower())

if TYPE_CHECKING:
    from truthound.context import TruthoundContext
    from truthound.schema import Schema
    from truthound.datasources.base import BaseDataSource
    from truthound.execution.base import BaseExecutionEngine
    from truthound.core.results import ValidationRunResult
    from truthound.report import PIIReport, ProfileReport
    from truthound.validators.base import Validator


# =============================================================================
# Lazy import helpers - these avoid loading heavy modules at import time
# =============================================================================

def _get_report_classes() -> tuple[type, type, type]:
    """Lazily import Report classes."""
    from truthound.report import PIIReport, ProfileReport, Report
    return PIIReport, ProfileReport, Report


def _get_validator_utils() -> tuple[dict, type, Any]:
    """Lazily import validator utilities."""
    from truthound.validators import BUILTIN_VALIDATORS, Validator, get_validator
    return BUILTIN_VALIDATORS, Validator, get_validator


def _get_profile_data() -> Any:
    """Lazily import profile_data function."""
    from truthound.profiler import profile_data
    return profile_data



def check(
    data: Any = None,
    source: "BaseDataSource | None" = None,
    context: "TruthoundContext | None" = None,
    validators: list[str | Validator] | None = None,
    validator_config: dict[str, dict[str, Any]] | None = None,
    min_severity: str | Severity | None = None,
    schema: str | Path | Schema | None = None,
    auto_schema: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    pushdown: bool | None = None,
    result_format: str | ResultFormat | ResultFormatConfig = ResultFormat.SUMMARY,
    catch_exceptions: bool = True,
    max_retries: int = 0,
    exclude_columns: list[str] | None = None,
) -> ValidationRunResult:
    """Perform data quality validation through the Truthound 3.0 kernel.

    ``th.check()`` now returns ``ValidationRunResult`` directly. Truthound
    automatically discovers or creates the local ``.truthound`` workspace,
    resolves a baseline schema when needed, synthesizes an auto-suite when
    validators are omitted, and persists run metadata/docs through the active
    project context.
    """
    from truthound.core import (
        ScanPlanner,
        ValidationRuntime,
        ValidationSuite,
        build_validation_asset,
    )
    from truthound.context import get_context

    active_context = context or get_context()
    asset = build_validation_asset(data=data, source=source, pushdown=pushdown)
    source_key = active_context.resolve_source_key(data=data, source=source)
    source_fingerprint = active_context.resolve_fingerprint(data=data, source=source)
    active_context.track_asset(data=data, source=source)
    suite = ValidationSuite.from_legacy(
        context=active_context,
        validators=validators,
        validator_config=validator_config,
        schema=schema,
        auto_schema=auto_schema,
        data=data,
        source=source,
        catch_exceptions=catch_exceptions,
        max_retries=max_retries,
        exclude_columns=exclude_columns,
        result_format=result_format,
        min_severity=min_severity,
    )
    plan = ScanPlanner().plan(
        suite=suite,
        asset=asset,
        parallel=parallel,
        max_workers=max_workers,
        pushdown=pushdown,
    )
    run_result = ValidationRuntime().execute(asset=asset, plan=plan)
    run_result = run_result.with_metadata(
        context_root=str(active_context.root_dir),
        context_source_key=source_key,
        context_history_key=source_key,
        context_source_fingerprint=source_fingerprint,
    )

    if min_severity is not None:
        if isinstance(min_severity, str):
            min_severity = _parse_severity(min_severity)
        run_result = run_result.filter_by_severity(min_severity)

    if active_context.config.persist_runs:
        run_path = active_context.persist_run(run_result)
        run_result = run_result.with_metadata(
            context_run_artifact=str(run_path),
            _truthound_run_artifact=str(run_path),
        )
    if active_context.config.persist_docs:
        docs_path = active_context.persist_docs(run_result)
        run_result = run_result.with_metadata(
            context_docs_artifact=str(docs_path),
            _truthound_docs_artifact=str(docs_path),
        )

    return run_result


def scan(
    data: Any = None,
    source: "BaseDataSource | None" = None,
) -> PIIReport:
    """Scan data for personally identifiable information (PII).

    Args:
        data: Input data (file path, DataFrame, dict, etc.)
        source: Optional DataSource instance. If provided, data is ignored.
                This enables scanning on SQL databases, Spark, etc.

    Returns:
        PIIReport containing all PII findings.

    Example:
        >>> import truthound as th
        >>> pii_report = th.scan("data.csv")
        >>> print(pii_report)

        >>> # Using DataSource for SQL database
        >>> from truthound.datasources import get_sql_datasource
        >>> source = get_sql_datasource("mydb.db", table="users")
        >>> pii_report = th.scan(source=source)
    """
    # Handle DataSource if provided
    if source is not None:
        from truthound.datasources.base import BaseDataSource

        if not isinstance(source, BaseDataSource):
            raise ValueError(
                f"source must be a DataSource instance, got {type(source).__name__}"
            )
        lf = source.to_polars_lazyframe()
        source_name = source.name
    else:
        if data is None:
            raise ValueError("Either 'data' or 'source' must be provided")
        lf = to_lazyframe(data)
        source_name = str(data) if isinstance(data, str) else type(data).__name__

    df = lf.collect()
    row_count = len(df)

    findings = scan_pii(df.lazy())

    # Lazy import PIIReport
    PIIReport, _, _ = _get_report_classes()
    return PIIReport(
        findings=findings,
        source=source_name,
        row_count=row_count,
    )


def mask(
    data: Any = None,
    source: "BaseDataSource | None" = None,
    columns: list[str] | None = None,
    strategy: str = "redact",
    *,
    strict: bool = False,
) -> pl.DataFrame:
    """Mask sensitive data in the input.

    Args:
        data: Input data (file path, DataFrame, dict, etc.)
        source: Optional DataSource instance. If provided, data is ignored.
                This enables masking on SQL databases, Spark, etc.
        columns: Optional list of columns to mask.
                If None, auto-detects PII columns.
        strategy: Masking strategy - "redact", "hash", or "fake".
        strict: If True, raise ValueError for non-existent columns.
                If False (default), emit a warning and skip missing columns.

    Returns:
        Polars DataFrame with masked values.

    Raises:
        ValueError: If strict=True and a specified column doesn't exist.

    Warnings:
        MaskingWarning: When a specified column does not exist in the data
                        (only if strict=False).

    Example:
        >>> import truthound as th
        >>> masked_df = th.mask("data.csv")

        >>> # Mask specific columns
        >>> masked_df = th.mask(df, columns=["email", "phone"])

        >>> # Use hash strategy
        >>> masked_df = th.mask(df, strategy="hash")

        >>> # Strict mode - fail if columns don't exist
        >>> masked_df = th.mask(df, columns=["email"], strict=True)

        >>> # Using DataSource for SQL database
        >>> from truthound.datasources import get_sql_datasource
        >>> source = get_sql_datasource("mydb.db", table="users")
        >>> masked_df = th.mask(source=source)
    """
    # Handle DataSource if provided
    if source is not None:
        from truthound.datasources.base import BaseDataSource

        if not isinstance(source, BaseDataSource):
            raise ValueError(
                f"source must be a DataSource instance, got {type(source).__name__}"
            )
        lf = source.to_polars_lazyframe()
    else:
        if data is None:
            raise ValueError("Either 'data' or 'source' must be provided")
        lf = to_lazyframe(data)

    return mask_data(lf, columns=columns, strategy=strategy, strict=strict)


def profile(
    data: Any = None,
    source: "BaseDataSource | None" = None,
) -> ProfileReport:
    """Generate a statistical profile of the dataset.

    Args:
        data: Input data (file path, DataFrame, dict, etc.)
        source: Optional DataSource instance. If provided, data is ignored.
                This enables profiling on SQL databases, Spark, etc.

    Returns:
        ProfileReport containing statistical summary.

    Example:
        >>> import truthound as th
        >>> profile = th.profile("data.csv")
        >>> print(profile)

        >>> # Using DataSource for SQL database
        >>> from truthound.datasources import get_sql_datasource
        >>> source = get_sql_datasource("mydb.db", table="users")
        >>> profile = th.profile(source=source)
    """
    # Handle DataSource if provided
    if source is not None:
        from truthound.datasources.base import BaseDataSource

        if not isinstance(source, BaseDataSource):
            raise ValueError(
                f"source must be a DataSource instance, got {type(source).__name__}"
            )
        lf = source.to_polars_lazyframe()
        source_name = source.name
    else:
        if data is None:
            raise ValueError("Either 'data' or 'source' must be provided")
        lf = to_lazyframe(data)
        source_name = str(data) if isinstance(data, str) else type(data).__name__

    # Lazy import profile_data and ProfileReport
    profile_data = _get_profile_data()
    _, ProfileReport, _ = _get_report_classes()

    profile_dict = profile_data(lf, source=source_name)

    return ProfileReport(
        source=profile_dict["source"],
        row_count=profile_dict["row_count"],
        column_count=profile_dict["column_count"],
        size_bytes=profile_dict["size_bytes"],
        columns=profile_dict["columns"],
    )


def read(
    data: Any,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Read data from various sources and return as Polars DataFrame.

    This is a convenience function that wraps datasources.get_datasource()
    for simpler data loading. It supports file paths, dict configurations,
    DataFrames, and DataSource objects.

    Supported Input Types:
        - str: File path (CSV, JSON, Parquet, etc.)
        - pl.DataFrame: Polars DataFrame (returned as-is with optional sampling)
        - pl.LazyFrame: Polars LazyFrame (collected to DataFrame)
        - dict: Either raw data dict ({"col": [values]}) or
                configuration dict with "path" key ({"path": "file.csv", ...})
        - DataSource: Existing DataSource instance

    Args:
        data: File path, DataFrame, dict data, dict config, or DataSource object.
        sample_size: Optional sample size for large datasets. If the dataset
                    exceeds this size, a random sample is returned.
        **kwargs: Additional arguments passed to get_datasource().
                 Common options include:
                 - delimiter: CSV delimiter character
                 - has_header: Whether CSV has headers
                 - schema: Explicit column schema

    Returns:
        Polars DataFrame containing the loaded data.

    Example:
        >>> import truthound as th
        >>> # Simple file reading
        >>> df = th.read("data.csv")
        >>> df = th.read("data.parquet")
        >>> df = th.read("data.json")

        >>> # Raw data dict (like a DataFrame)
        >>> df = th.read({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        >>> # Config dict with "path" key
        >>> df = th.read({"path": "data.csv", "delimiter": ";"})

        >>> # With sampling for large datasets
        >>> df = th.read("large_data.csv", sample_size=10000)

        >>> # With additional options
        >>> df = th.read("data.csv", has_header=False)
    """
    from truthound.datasources import get_datasource
    from truthound.datasources.base import BaseDataSource

    # Handle DataSource instance
    if isinstance(data, BaseDataSource):
        source = data
        df = source.to_polars_lazyframe().collect()
    # Handle dict - distinguish between data dict and config dict
    elif isinstance(data, dict):
        # Check if this looks like a config dict (has "path" key)
        if "path" in data:
            path = data["path"]
            # Merge dict config with kwargs
            config = {k: v for k, v in data.items() if k != "path"}
            config.update(kwargs)
            source = get_datasource(path, **config)
            df = source.to_polars_lazyframe().collect()
        else:
            # Treat as raw data dict (column-oriented data)
            # Check if values are list-like (data dict) vs scalar (likely config)
            first_value = next(iter(data.values()), None) if data else None
            if first_value is not None and isinstance(first_value, (list, tuple)):
                # This is a data dict like {"col": [1, 2, 3]}
                df = pl.DataFrame(data)
            else:
                # Likely intended as config but missing "path"
                raise ValueError(
                    "Dict configuration must include 'path' key.\n\n"
                    "Example:\n"
                    "  th.read({'path': 'data.csv', 'delimiter': ';'})\n\n"
                    "For raw data, use column-oriented format:\n"
                    "  th.read({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})"
                )
    # Handle file path or other data types (DataFrame, LazyFrame, etc.)
    else:
        source = get_datasource(data, **kwargs)
        df = source.to_polars_lazyframe().collect()

    # Apply sampling if requested
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, seed=42)

    return df
