"""Stores module for persisting validation results and expectations.

This module provides a unified interface for storing and retrieving validation
results across different backends (filesystem, S3, GCS, databases).

Example:
    >>> from truthound.stores import get_store, ValidationResult
    >>>
    >>> # Use filesystem store (default)
    >>> store = get_store("filesystem", base_path=".truthound/results")
    >>>
    >>> # Save validation result
    >>> result = ValidationResult.from_report(report, data_asset="customers.csv")
    >>> run_id = store.save(result)
    >>>
    >>> # Retrieve result
    >>> result = store.get(run_id)
    >>>
    >>> # List all results for a data asset
    >>> run_ids = store.list_runs("customers.csv")
"""

from truthound.stores.base import (
    BaseStore,
    StoreConfig,
    StoreQuery,
    StoreError,
    StoreNotFoundError,
    StoreConnectionError,
)
from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatistics,
    ResultStatus,
)
from truthound.stores.expectations import (
    Expectation,
    ExpectationSuite,
)
from truthound.stores.factory import get_store, register_store

__all__ = [
    # Base classes
    "BaseStore",
    "StoreConfig",
    "StoreQuery",
    "StoreError",
    "StoreNotFoundError",
    "StoreConnectionError",
    # Result types
    "ValidationResult",
    "ValidatorResult",
    "ResultStatistics",
    "ResultStatus",
    # Expectation types
    "Expectation",
    "ExpectationSuite",
    # Factory functions
    "get_store",
    "register_store",
]
