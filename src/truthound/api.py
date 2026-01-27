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
from truthound.types import Severity


# =============================================================================
# Cached helpers - avoid repeated parsing overhead
# =============================================================================

@functools.lru_cache(maxsize=8)
def _parse_severity(s: str) -> Severity:
    """Parse severity string with caching to avoid repeated conversions."""
    return Severity(s.lower())

if TYPE_CHECKING:
    from truthound.schema import Schema
    from truthound.datasources.base import BaseDataSource
    from truthound.execution.base import BaseExecutionEngine
    from truthound.report import PIIReport, ProfileReport, Report
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
    validators: list[str | Validator] | None = None,
    validator_config: dict[str, dict[str, Any]] | None = None,
    min_severity: str | Severity | None = None,
    schema: str | Path | Schema | None = None,
    auto_schema: bool = False,
    use_engine: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    pushdown: bool | None = None,
) -> Report:
    """Perform data quality validation on the input data.

    This is the main entry point for data quality validation. It accepts various
    input types and automatically converts them to Polars LazyFrame internally.

    Supported Input Types:
        - str: File path (CSV, JSON, Parquet)
        - pl.DataFrame: Polars DataFrame (converted to LazyFrame)
        - pl.LazyFrame: Polars LazyFrame (used directly)
        - pd.DataFrame: pandas DataFrame (converted via Polars)
        - dict: Python dictionary (converted to DataFrame then LazyFrame)
        - BaseDataSource: DataSource instance for SQL databases, Spark, etc.

    Note:
        Individual Validator classes only accept pl.LazyFrame directly.
        This API handles the conversion for convenience. If using validators
        directly, use ``truthound.adapters.to_lazyframe()`` to convert your data.

    Args:
        data: Input data (file path, DataFrame, dict, etc.)
        source: Optional DataSource instance. If provided, data is ignored.
                This enables validation on SQL databases, Spark, etc.
        validators: Optional list of validator names or Validator instances.
                   If None, all built-in validators are used.
        validator_config: Optional configuration dict for validators.
                         Maps validator name to configuration dict.
                         Example: {"regex": {"patterns": {"email": r"^[\\w.+-]+@..."}}}
        min_severity: Minimum severity level to include in results.
                     Can be "low", "medium", "high", or "critical".
        schema: Optional schema for validation. Can be:
               - Path to a schema YAML file
               - Schema object from th.learn()
               When provided, schema validation runs in addition to other validators.
        auto_schema: If True, automatically learns and caches a schema from the data.
                    On subsequent runs with the same data source, validates against
                    the cached schema. This enables true "zero-config" validation.
        use_engine: If True, uses execution engine for validation (experimental).
                   Currently validators still use Polars LazyFrame fallback.
        parallel: If True, uses DAG-based parallel execution for validators.
                 Validators are grouped by dependency and executed in parallel
                 when possible. This can significantly improve performance for
                 large datasets with many validators.
        max_workers: Maximum number of worker threads for parallel execution.
                    Only used when parallel=True. Defaults to min(32, cpu_count + 4).
        pushdown: If True, enables query pushdown for SQL data sources.
                 Validation logic is executed server-side when possible,
                 reducing data transfer and improving performance.
                 If None (default), auto-detects based on data source type.

    Returns:
        Report containing all validation issues found.

    Example:
        >>> import truthound as th
        >>> report = th.check("data.csv")
        >>> print(report)

        >>> # With specific validators
        >>> report = th.check(df, validators=["null", "duplicate"])

        >>> # Filter by severity
        >>> report = th.check(df, min_severity="medium")

        >>> # With schema validation
        >>> schema = th.learn("baseline.csv")
        >>> report = th.check("new_data.csv", schema=schema)

        >>> # Zero-config with auto schema caching
        >>> report = th.check("data.csv", auto_schema=True)

        >>> # Using DataSource for SQL database
        >>> from truthound.datasources.sql import PostgreSQLDataSource
        >>> source = PostgreSQLDataSource(
        ...     table="users",
        ...     host="localhost",
        ...     database="mydb",
        ...     user="postgres",
        ... )
        >>> report = th.check(source=source, validators=["null", "duplicate"])

        >>> # Using auto-detection with DataSource
        >>> from truthound.datasources import get_datasource
        >>> source = get_datasource(spark_df)  # PySpark DataFrame
        >>> if source.needs_sampling():
        ...     source = source.sample(n=100_000)
        >>> report = th.check(source=source)

        >>> # With query pushdown for SQL data sources
        >>> from truthound.datasources.sql import PostgreSQLDataSource
        >>> source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
        >>> report = th.check(source=source, pushdown=True)  # Execute validations server-side
    """
    # Handle DataSource if provided
    use_pushdown = False
    sql_source = None

    if source is not None:
        from truthound.datasources.base import BaseDataSource
        from truthound.datasources._protocols import DataSourceCapability

        if not isinstance(source, BaseDataSource):
            raise ValueError(
                f"source must be a DataSource instance, got {type(source).__name__}"
            )

        # Determine if pushdown should be used
        if pushdown is True:
            use_pushdown = True
        elif pushdown is None:
            # Auto-detect: use pushdown for SQL sources with SQL_PUSHDOWN capability
            use_pushdown = DataSourceCapability.SQL_PUSHDOWN in source.capabilities

        if use_pushdown:
            # Verify it's actually a SQL data source
            try:
                from truthound.datasources.sql.base import BaseSQLDataSource
                if isinstance(source, BaseSQLDataSource):
                    sql_source = source
                else:
                    use_pushdown = False
            except ImportError:
                use_pushdown = False

        # Check size limits and warn if needed (only if not using pushdown)
        if not use_pushdown and source.needs_sampling():
            import warnings
            warnings.warn(
                f"Data source '{source.name}' has {source.row_count:,} rows, "
                f"which exceeds the limit of {source.config.max_rows:,}. "
                "Consider using source.sample() for better performance.",
                UserWarning,
            )

        source_name = source.name
    else:
        if data is None:
            raise ValueError("Either 'data' or 'source' must be provided")

        # Convert input to LazyFrame (legacy path)
        source_name = str(data) if isinstance(data, str) else type(data).__name__

    # For pushdown path, get metadata without loading all data
    if use_pushdown and sql_source is not None:
        row_count = sql_source.row_count or 0
        column_count = len(sql_source.columns)
        lf = None  # Will be loaded lazily if needed for non-pushdown validators
    else:
        # Standard path: load data into Polars
        if source is not None:
            lf = source.to_polars_lazyframe()
        else:
            lf = to_lazyframe(data)

        # Collect metadata without materializing the entire DataFrame
        polars_schema = lf.collect_schema()  # Lazy operation - only reads schema
        row_count = lf.select(pl.len()).collect().item()  # Efficient row count
        column_count = len(polars_schema)
        # lf remains lazy - validators will collect only when needed

    # Lazy load validator utilities
    BUILTIN_VALIDATORS, Validator, get_validator = _get_validator_utils()

    # Determine which validators to use
    validator_instances: list[Validator] = []

    # Add schema validator if schema is provided or auto_schema is enabled
    if schema is not None or auto_schema:
        from truthound.schema import Schema as SchemaClass
        from truthound.validators.schema_validator import SchemaValidator

        if schema is not None:
            if isinstance(schema, (str, Path)):
                schema_obj = SchemaClass.load(schema)
            else:
                schema_obj = schema
        else:
            # Auto schema mode: get from cache or learn new
            from truthound.cache import get_or_learn_schema
            schema_obj, was_cached = get_or_learn_schema(data)

        validator_instances.append(SchemaValidator(schema_obj))

    # Normalize validator_config to empty dict if None
    validator_config = validator_config or {}

    if validators is None:
        # Use all built-in validators with their configs
        for name, cls in BUILTIN_VALIDATORS.items():
            config = validator_config.get(name, {})
            validator_instances.append(cls(**config) if config else cls())
    else:
        for v in validators:
            if isinstance(v, str):
                validator_cls = get_validator(v)
                config = validator_config.get(v, {})
                validator_instances.append(validator_cls(**config) if config else validator_cls())
            elif isinstance(v, Validator):
                validator_instances.append(v)
            else:
                raise ValueError(f"Invalid validator: {v}. Expected str or Validator instance.")

    # Run all validators and collect issues
    all_issues = []

    if use_pushdown and sql_source is not None:
        # Use pushdown validation engine for SQL data sources
        from truthound.validators.pushdown_support import PushdownValidationEngine

        engine = PushdownValidationEngine(sql_source)
        all_issues = engine.validate(validator_instances)

    elif parallel and len(validator_instances) > 1:
        # Use DAG-based parallel execution
        from truthound.validators.optimization.orchestrator import (
            ValidatorDAG,
            ParallelExecutionStrategy,
            AdaptiveExecutionStrategy,
        )

        dag = ValidatorDAG()
        dag.add_validators(validator_instances)
        plan = dag.build_execution_plan()

        # Choose strategy based on max_workers
        if max_workers is not None:
            strategy = ParallelExecutionStrategy(max_workers=max_workers)
        else:
            strategy = AdaptiveExecutionStrategy()

        result = plan.execute(lf, strategy)
        all_issues = result.all_issues
    else:
        # Sequential or lightweight parallel execution
        # For small validator sets, avoid ThreadPool creation overhead
        if len(validator_instances) < 5:
            # Sequential execution for small sets
            for validator in validator_instances:
                issues = validator.validate(lf)
                all_issues.extend(issues)
        else:
            # Lightweight parallel execution for larger sets
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda v: v.validate(lf), validator_instances))
            for issues in results:
                all_issues.extend(issues)

    # Create report (lazy import Report class)
    _, _, Report = _get_report_classes()
    report = Report(
        issues=all_issues,
        source=source_name,
        row_count=row_count,
        column_count=column_count,
    )

    # Filter by severity if specified
    if min_severity is not None:
        if isinstance(min_severity, str):
            min_severity = _parse_severity(min_severity)
        report = report.filter_by_severity(min_severity)

    return report


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
