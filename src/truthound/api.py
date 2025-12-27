"""Main API functions for Truthound."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from truthound.adapters import to_lazyframe
from truthound.maskers import mask_data
from truthound.profiler import profile_data
from truthound.report import PIIReport, ProfileReport, Report
from truthound.scanners import scan_pii
from truthound.types import Severity
from truthound.validators import BUILTIN_VALIDATORS, Validator, get_validator

if TYPE_CHECKING:
    from truthound.schema import Schema
    from truthound.datasources.base import BaseDataSource
    from truthound.execution.base import BaseExecutionEngine


def check(
    data: Any = None,
    source: "BaseDataSource | None" = None,
    validators: list[str | Validator] | None = None,
    min_severity: str | Severity | None = None,
    schema: str | Path | Schema | None = None,
    auto_schema: bool = False,
    use_engine: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    pushdown: bool | None = None,
) -> Report:
    """Perform data quality validation on the input data.

    Args:
        data: Input data (file path, DataFrame, dict, etc.)
        source: Optional DataSource instance. If provided, data is ignored.
                This enables validation on SQL databases, Spark, etc.
        validators: Optional list of validator names or Validator instances.
                   If None, all built-in validators are used.
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

        # Collect metadata
        polars_schema = lf.collect_schema()
        df_collected = lf.collect()
        row_count = len(df_collected)
        column_count = len(polars_schema)

        # Re-create lazy frame after collecting metadata
        lf = df_collected.lazy()

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

    if validators is None:
        # Use all built-in validators
        validator_instances.extend([cls() for cls in BUILTIN_VALIDATORS.values()])
    else:
        for v in validators:
            if isinstance(v, str):
                validator_cls = get_validator(v)
                validator_instances.append(validator_cls())
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
        # Sequential execution (original behavior)
        for validator in validator_instances:
            issues = validator.validate(lf)
            all_issues.extend(issues)

    # Create report
    report = Report(
        issues=all_issues,
        source=source_name,
        row_count=row_count,
        column_count=column_count,
    )

    # Filter by severity if specified
    if min_severity is not None:
        if isinstance(min_severity, str):
            min_severity = Severity(min_severity.lower())
        report = report.filter_by_severity(min_severity)

    return report


def scan(data: Any) -> PIIReport:
    """Scan data for personally identifiable information (PII).

    Args:
        data: Input data (file path, DataFrame, dict, etc.)

    Returns:
        PIIReport containing all PII findings.

    Example:
        >>> import truthound as th
        >>> pii_report = th.scan("data.csv")
        >>> print(pii_report)
    """
    lf = to_lazyframe(data)
    source = str(data) if isinstance(data, str) else type(data).__name__

    df = lf.collect()
    row_count = len(df)

    findings = scan_pii(df.lazy())

    return PIIReport(
        findings=findings,
        source=source,
        row_count=row_count,
    )


def mask(
    data: Any,
    columns: list[str] | None = None,
    strategy: str = "redact",
) -> pl.DataFrame:
    """Mask sensitive data in the input.

    Args:
        data: Input data (file path, DataFrame, dict, etc.)
        columns: Optional list of columns to mask.
                If None, auto-detects PII columns.
        strategy: Masking strategy - "redact", "hash", or "fake".

    Returns:
        Polars DataFrame with masked values.

    Example:
        >>> import truthound as th
        >>> masked_df = th.mask("data.csv")

        >>> # Mask specific columns
        >>> masked_df = th.mask(df, columns=["email", "phone"])

        >>> # Use hash strategy
        >>> masked_df = th.mask(df, strategy="hash")
    """
    lf = to_lazyframe(data)
    return mask_data(lf, columns=columns, strategy=strategy)


def profile(data: Any) -> ProfileReport:
    """Generate a statistical profile of the dataset.

    Args:
        data: Input data (file path, DataFrame, dict, etc.)

    Returns:
        ProfileReport containing statistical summary.

    Example:
        >>> import truthound as th
        >>> profile = th.profile("data.csv")
        >>> print(profile)
    """
    lf = to_lazyframe(data)
    source = str(data) if isinstance(data, str) else type(data).__name__

    profile_dict = profile_data(lf, source=source)

    return ProfileReport(
        source=profile_dict["source"],
        row_count=profile_dict["row_count"],
        column_count=profile_dict["column_count"],
        size_bytes=profile_dict["size_bytes"],
        columns=profile_dict["columns"],
    )
