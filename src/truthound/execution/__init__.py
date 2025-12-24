"""Execution engine implementations for Truthound.

This package provides execution engines that run validation operations
on different data backends.

Execution engines are responsible for:
- Running validation operations (count, aggregate, filter, etc.)
- Optimizing operations for the specific backend
- Converting to Polars for validators that require it
- Managing caching and performance

Available engines:
- PolarsExecutionEngine: Primary engine using Polars (default)
- PandasExecutionEngine: Native Pandas operations
- SQLExecutionEngine: SQL pushdown for database sources

Example:
    >>> from truthound.datasources import get_datasource
    >>> from truthound.execution import PolarsExecutionEngine
    >>>
    >>> source = get_datasource("data.csv")
    >>> engine = source.get_execution_engine()
    >>>
    >>> # Use engine for validation operations
    >>> null_count = engine.count_nulls("column_name")
    >>> stats = engine.get_stats("numeric_column")
"""

from truthound.execution._protocols import (
    AggregationType,
    ValidatorCapability,
    ExecutionEngineProtocol,
    SQLExecutionEngineProtocol,
)

from truthound.execution.base import (
    BaseExecutionEngine,
    ExecutionConfig,
    ExecutionError,
    UnsupportedOperationError,
    ExecutionSizeError,
)

from truthound.execution.polars_engine import PolarsExecutionEngine
from truthound.execution.pandas_engine import PandasExecutionEngine
from truthound.execution.sql_engine import SQLExecutionEngine

__all__ = [
    # Protocols
    "AggregationType",
    "ValidatorCapability",
    "ExecutionEngineProtocol",
    "SQLExecutionEngineProtocol",
    # Base classes
    "BaseExecutionEngine",
    "ExecutionConfig",
    # Exceptions
    "ExecutionError",
    "UnsupportedOperationError",
    "ExecutionSizeError",
    # Engines
    "PolarsExecutionEngine",
    "PandasExecutionEngine",
    "SQLExecutionEngine",
]
