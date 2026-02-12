"""Shared metric infrastructure for deduplication.

This module implements PHASE 3 of the validation enhancement plan:
metric deduplication across validators. When multiple validators need
the same base metric (e.g., null_count, row_count), it is computed
once and shared via SharedMetricStore.

Architecture:
    MetricKey       — Immutable key identifying a unique metric
    SharedMetricStore — Session-scoped cache for computed metrics
    CommonMetrics   — Standard metric expressions (row_count, null_count, etc.)

GX Reference:
    Inspired by GX's MetricConfigurationID + resolved_metrics pattern,
    adapted for Polars expression-based computation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

logger = logging.getLogger("truthound.metrics")


# ============================================================================
# MetricKey — unique identifier for a metric
# ============================================================================


@dataclass(frozen=True)
class MetricKey:
    """Immutable key that uniquely identifies a metric.

    Corresponds to GX's MetricConfigurationID. The combination of
    (metric_name, column, kwargs_hash) guarantees uniqueness.

    Attributes:
        metric_name: Metric identifier (e.g. "null_count", "row_count")
        column: Column name, or None for table-level metrics
        kwargs_hash: Hash of additional parameters (e.g. min_value, q)
    """

    metric_name: str
    column: str | None = None
    kwargs_hash: str = ""

    @staticmethod
    def create(
        metric_name: str,
        column: str | None = None,
        **kwargs: Any,
    ) -> MetricKey:
        """Factory: build a MetricKey, hashing extra kwargs for identity."""
        if kwargs:
            # Deterministic hash of sorted kwargs
            raw = json.dumps(kwargs, sort_keys=True, default=str)
            kwargs_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
        else:
            kwargs_hash = ""
        return MetricKey(
            metric_name=metric_name,
            column=column,
            kwargs_hash=kwargs_hash,
        )


# ============================================================================
# MetricStoreStats — performance counters
# ============================================================================


@dataclass
class MetricStoreStats:
    """Performance counters for SharedMetricStore."""

    total_lookups: int = 0
    cache_hits: int = 0
    metrics_computed: int = 0
    deduplication_saves: int = 0

    @property
    def hit_ratio(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_lookups

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "hit_ratio": round(self.hit_ratio, 4),
            "metrics_computed": self.metrics_computed,
            "deduplication_saves": self.deduplication_saves,
        }


# ============================================================================
# SharedMetricStore — session-scoped metric cache
# ============================================================================


class SharedMetricStore:
    """Session-scoped cache for computed metrics.

    Corresponds to GX's resolved_metrics dict. Multiple validators
    register their metric needs; the store deduplicates and serves
    results so each metric is computed at most once per session.

    Thread-safe via an RLock. Designed to live for exactly one
    ``check()`` call and then be discarded.

    Usage::

        store = SharedMetricStore()

        # Direct put/get
        key = MetricKey("row_count")
        store.put(key, 1000)
        assert store.get(key) == 1000

        # Compute-if-absent (thread-safe)
        val = store.get_or_compute(key, lambda: lf.select(pl.len()).collect().item())
    """

    __slots__ = ("_store", "_lock", "_pending", "_stats")

    def __init__(self) -> None:
        self._store: dict[MetricKey, Any] = {}
        self._lock = threading.RLock()
        self._pending: dict[MetricKey, threading.Event] = {}
        self._stats = MetricStoreStats()

    # -- read / write ---------------------------------------------------------

    def get(self, key: MetricKey) -> Any | None:
        """Look up a cached metric value. Returns None on miss."""
        with self._lock:
            self._stats.total_lookups += 1
            value = self._store.get(key)
            if value is not None:
                self._stats.cache_hits += 1
            return value

    def put(self, key: MetricKey, value: Any) -> None:
        """Store a computed metric value."""
        with self._lock:
            is_new = key not in self._store
            self._store[key] = value
            if is_new:
                self._stats.metrics_computed += 1
            # Wake up any threads waiting for this key
            if key in self._pending:
                self._pending[key].set()

    def get_or_compute(
        self,
        key: MetricKey,
        compute_fn: Callable[[], Any],
    ) -> Any:
        """Return cached value or compute, store, and return.

        Thread-safe: concurrent calls for the same key will compute once;
        other callers wait on an Event.
        """
        # Fast path
        value = self.get(key)
        if value is not None:
            return value

        with self._lock:
            # Double-check after acquiring lock
            value = self._store.get(key)
            if value is not None:
                self._stats.total_lookups += 1
                self._stats.cache_hits += 1
                return value

            if key in self._pending:
                # Another thread is already computing this metric
                event = self._pending[key]
            else:
                # This thread will compute
                event = threading.Event()
                self._pending[key] = event
                try:
                    value = compute_fn()
                    self.put(key, value)
                    return value
                finally:
                    self._pending.pop(key, None)

        # Wait for the other thread to finish
        event.wait(timeout=300)
        return self._store.get(key)

    # -- bulk operations ------------------------------------------------------

    def put_many(self, pairs: dict[MetricKey, Any]) -> None:
        """Store multiple metric values at once."""
        with self._lock:
            for key, value in pairs.items():
                is_new = key not in self._store
                self._store[key] = value
                if is_new:
                    self._stats.metrics_computed += 1
                if key in self._pending:
                    self._pending[key].set()

    def get_many(self, keys: list[MetricKey]) -> dict[MetricKey, Any]:
        """Look up multiple keys. Missing keys are omitted."""
        result: dict[MetricKey, Any] = {}
        with self._lock:
            for key in keys:
                self._stats.total_lookups += 1
                value = self._store.get(key)
                if value is not None:
                    self._stats.cache_hits += 1
                    result[key] = value
        return result

    def missing_keys(self, keys: list[MetricKey]) -> list[MetricKey]:
        """Return subset of *keys* not yet in the store."""
        with self._lock:
            return [k for k in keys if k not in self._store]

    # -- lifecycle ------------------------------------------------------------

    def clear(self) -> None:
        """Release all cached values. Call at session end."""
        with self._lock:
            self._store.clear()
            self._pending.clear()

    @property
    def stats(self) -> MetricStoreStats:
        return self._stats

    def get_stats_dict(self) -> dict[str, Any]:
        return self._stats.to_dict()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: MetricKey) -> bool:
        return key in self._store

    def __repr__(self) -> str:
        return (
            f"SharedMetricStore(size={len(self._store)}, "
            f"hits={self._stats.cache_hits}/{self._stats.total_lookups})"
        )


# ============================================================================
# CommonMetrics — standard metric expressions
# ============================================================================


class CommonMetrics:
    """Library of standard metric expressions used by multiple validators.

    Each method returns a ``(MetricKey, pl.Expr)`` pair so that the
    caller can register the expression with a SharedMetricStore and
    deduplicate identical requests across validators.

    The alias format ``_metric_{name}_{column}`` is used to avoid
    collisions in a multi-expression collect().

    Adding new metrics:
        1. Add a ``@staticmethod`` method returning ``tuple[MetricKey, pl.Expr]``
        2. Use ``MetricKey.create(...)`` for parameterised metrics
        3. Register in ``METRIC_REGISTRY`` if it should be discoverable
    """

    # -- Table-level metrics --------------------------------------------------

    @staticmethod
    def row_count() -> tuple[MetricKey, pl.Expr]:
        """Total row count (table-level)."""
        key = MetricKey(metric_name="row_count", column=None)
        expr = pl.len().alias("_metric_row_count")
        return key, expr

    # -- Column-level metrics -------------------------------------------------

    @staticmethod
    def null_count(column: str) -> tuple[MetricKey, pl.Expr]:
        """Count of null values in a column."""
        key = MetricKey(metric_name="null_count", column=column)
        expr = pl.col(column).null_count().alias(f"_metric_null_count_{column}")
        return key, expr

    @staticmethod
    def non_null_count(column: str) -> tuple[MetricKey, pl.Expr]:
        """Count of non-null values in a column."""
        key = MetricKey(metric_name="non_null_count", column=column)
        expr = pl.col(column).count().alias(f"_metric_non_null_count_{column}")
        return key, expr

    @staticmethod
    def n_unique(column: str) -> tuple[MetricKey, pl.Expr]:
        """Number of unique values in a column."""
        key = MetricKey(metric_name="n_unique", column=column)
        expr = pl.col(column).n_unique().alias(f"_metric_n_unique_{column}")
        return key, expr

    @staticmethod
    def mean(column: str) -> tuple[MetricKey, pl.Expr]:
        """Mean of a numeric column."""
        key = MetricKey(metric_name="mean", column=column)
        expr = pl.col(column).mean().alias(f"_metric_mean_{column}")
        return key, expr

    @staticmethod
    def std(column: str) -> tuple[MetricKey, pl.Expr]:
        """Standard deviation of a numeric column."""
        key = MetricKey(metric_name="std", column=column)
        expr = pl.col(column).std().alias(f"_metric_std_{column}")
        return key, expr

    @staticmethod
    def min(column: str) -> tuple[MetricKey, pl.Expr]:
        """Minimum value of a column."""
        key = MetricKey(metric_name="min", column=column)
        expr = pl.col(column).min().alias(f"_metric_min_{column}")
        return key, expr

    @staticmethod
    def max(column: str) -> tuple[MetricKey, pl.Expr]:
        """Maximum value of a column."""
        key = MetricKey(metric_name="max", column=column)
        expr = pl.col(column).max().alias(f"_metric_max_{column}")
        return key, expr

    @staticmethod
    def sum(column: str) -> tuple[MetricKey, pl.Expr]:
        """Sum of a numeric column."""
        key = MetricKey(metric_name="sum", column=column)
        expr = pl.col(column).sum().alias(f"_metric_sum_{column}")
        return key, expr

    @staticmethod
    def quantile(column: str, q: float) -> tuple[MetricKey, pl.Expr]:
        """Quantile of a numeric column."""
        key = MetricKey.create("quantile", column=column, q=q)
        expr = pl.col(column).quantile(q).alias(f"_metric_q{q}_{column}")
        return key, expr

    @staticmethod
    def median(column: str) -> tuple[MetricKey, pl.Expr]:
        """Median of a numeric column."""
        key = MetricKey(metric_name="median", column=column)
        expr = pl.col(column).median().alias(f"_metric_median_{column}")
        return key, expr


# ============================================================================
# METRIC_REGISTRY — maps metric names to their factory functions
# ============================================================================

_METRIC_FACTORIES: dict[str, Callable[..., tuple[MetricKey, pl.Expr]]] = {
    "row_count": lambda **_: CommonMetrics.row_count(),
    "null_count": lambda column, **_: CommonMetrics.null_count(column),
    "non_null_count": lambda column, **_: CommonMetrics.non_null_count(column),
    "n_unique": lambda column, **_: CommonMetrics.n_unique(column),
    "mean": lambda column, **_: CommonMetrics.mean(column),
    "std": lambda column, **_: CommonMetrics.std(column),
    "min": lambda column, **_: CommonMetrics.min(column),
    "max": lambda column, **_: CommonMetrics.max(column),
    "sum": lambda column, **_: CommonMetrics.sum(column),
    "quantile": lambda column, q, **_: CommonMetrics.quantile(column, q),
    "median": lambda column, **_: CommonMetrics.median(column),
}


def metric_key_to_expr(key: MetricKey) -> pl.Expr | None:
    """Resolve a MetricKey to its Polars expression.

    Returns None if the metric_name is not in the registry (i.e. a
    custom metric that was registered externally).
    """
    factory = _METRIC_FACTORIES.get(key.metric_name)
    if factory is None:
        return None
    try:
        if key.column is not None:
            _, expr = factory(column=key.column)
        else:
            _, expr = factory()
        return expr
    except TypeError:
        return None
