"""Enterprise features for validators.

This module provides production-ready integrations:
- #14: Audit logging integration (who/when/what)
- #15: Metrics collection (Prometheus/StatsD)
- #16: Reference data caching
- #17: Parallel processing support
- #18: Configuration validation
- #19: Polars version compatibility
- #20: Internationalization support

These features integrate with the existing audit, observability, and cache modules.
"""

from __future__ import annotations

import functools
import hashlib
import locale
import os
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    overload,
)

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    Validator,
    ValidatorConfig,
    ValidationIssue,
    ValidatorExecutionResult,
    ValidationResult,
    ValidatorLogger,
    ValidationErrorContext,
    GracefulValidator,
)


# =============================================================================
# #14: Audit Logging Integration
# =============================================================================


@dataclass
class ValidationAuditRecord:
    """Audit record for validation operations.

    Tracks who, when, what, and results of validation.
    """

    # Who
    user_id: str | None = None
    user_name: str | None = None
    service_name: str | None = None
    client_ip: str | None = None

    # When
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0

    # What
    validator_name: str = ""
    validator_category: str = ""
    data_source: str = ""
    row_count: int = 0
    column_count: int = 0
    columns_validated: tuple[str, ...] = ()

    # Results
    issues_found: int = 0
    severity_counts: dict[str, int] = field(default_factory=dict)
    status: str = "unknown"
    error_message: str | None = None

    # Context
    session_id: str | None = None
    request_id: str | None = None
    environment: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "service_name": self.service_name,
            "client_ip": self.client_ip,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "validator_name": self.validator_name,
            "validator_category": self.validator_category,
            "data_source": self.data_source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns_validated": list(self.columns_validated),
            "issues_found": self.issues_found,
            "severity_counts": self.severity_counts,
            "status": self.status,
            "error_message": self.error_message,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "environment": self.environment,
            "metadata": self.metadata,
        }


class ValidationAuditLogger:
    """Audit logger specifically for validation operations.

    Integrates with the main audit system when available.
    """

    _instance: "ValidationAuditLogger | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._records: list[ValidationAuditRecord] = []
        self._audit_logger: Any = None
        self._enabled = True
        self._max_records = 10000  # In-memory limit
        self.logger = ValidatorLogger("ValidationAuditLogger")

        # Try to integrate with main audit system
        self._init_audit_integration()

    def _init_audit_integration(self) -> None:
        """Initialize integration with truthound.audit if available."""
        try:
            from truthound.audit import get_audit_logger, AuditEventType

            self._audit_logger = get_audit_logger()
            self._audit_event_type = AuditEventType
            self.logger.debug("Integrated with truthound.audit system")
        except (ImportError, Exception):
            self._audit_logger = None
            self.logger.debug("truthound.audit not available, using standalone mode")

    @classmethod
    def get_instance(cls) -> "ValidationAuditLogger":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def log_validation(
        self,
        validator: Validator,
        lf: pl.LazyFrame,
        result: ValidatorExecutionResult,
        user_id: str | None = None,
        session_id: str | None = None,
        data_source: str | None = None,
        **metadata: Any,
    ) -> ValidationAuditRecord:
        """Log a validation operation.

        Args:
            validator: The validator that was executed
            lf: The LazyFrame that was validated
            result: The execution result
            user_id: Optional user identifier
            session_id: Optional session identifier
            data_source: Description of data source
            **metadata: Additional metadata

        Returns:
            The created audit record
        """
        if not self._enabled:
            return ValidationAuditRecord()

        # Collect schema info
        try:
            schema = lf.collect_schema()
            columns = schema.names()
            row_count = lf.select(pl.len()).collect().item()
        except Exception:
            columns = []
            row_count = 0

        # Build severity counts
        severity_counts: dict[str, int] = {}
        for issue in result.issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Create audit record
        record = ValidationAuditRecord(
            user_id=user_id or os.environ.get("USER"),
            service_name=os.environ.get("SERVICE_NAME"),
            timestamp=datetime.utcnow(),
            duration_ms=result.execution_time_ms,
            validator_name=result.validator_name,
            validator_category=getattr(validator, "category", "unknown"),
            data_source=data_source or "unknown",
            row_count=row_count,
            column_count=len(columns),
            columns_validated=tuple(columns),
            issues_found=len(result.issues),
            severity_counts=severity_counts,
            status=result.status.value,
            error_message=(
                result.error_context.message if result.error_context else None
            ),
            session_id=session_id,
            environment=os.environ.get("ENVIRONMENT", "development"),
            metadata=metadata,
        )

        # Store locally
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

        # Forward to main audit system if available
        if self._audit_logger:
            try:
                from truthound.audit import AuditResource, AuditActor

                self._audit_logger.log(
                    event_type=self._audit_event_type.READ,
                    action=f"validate_{validator.name}",
                    actor=AuditActor(id=user_id or "system"),
                    resource=AuditResource(
                        id=data_source or "unknown",
                        type="dataset",
                    ),
                )
            except Exception as e:
                self.logger.debug(f"Audit system forwarding skipped: {e}")

        return record

    def get_records(
        self,
        validator_name: str | None = None,
        user_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[ValidationAuditRecord]:
        """Query audit records.

        Args:
            validator_name: Filter by validator name
            user_id: Filter by user ID
            since: Filter by timestamp
            limit: Maximum records to return

        Returns:
            List of matching audit records
        """
        with self._lock:
            records = self._records.copy()

        if validator_name:
            records = [r for r in records if r.validator_name == validator_name]
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if since:
            records = [r for r in records if r.timestamp >= since]

        return records[-limit:]

    def clear(self) -> None:
        """Clear all audit records."""
        with self._lock:
            self._records.clear()

    def enable(self) -> None:
        """Enable audit logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable audit logging."""
        self._enabled = False


def get_validation_audit_logger() -> ValidationAuditLogger:
    """Get the global validation audit logger."""
    return ValidationAuditLogger.get_instance()


# =============================================================================
# #15: Metrics Collection Integration
# =============================================================================


class MetricsCollector:
    """Collects metrics for validation operations.

    Integrates with Prometheus/StatsD through truthound.observability.
    """

    _instance: "MetricsCollector | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._metrics_system: Any = None
        self._enabled = True
        self._local_stats: dict[str, float] = {}
        self._local_counts: dict[str, int] = {}
        self.logger = ValidatorLogger("MetricsCollector")

        self._init_metrics_integration()

    def _init_metrics_integration(self) -> None:
        """Initialize metrics backend integration."""
        try:
            from truthound.observability.metrics import Counter, Histogram, Gauge

            self._validation_counter = Counter(
                "truthound_validations_total",
                "Total number of validations",
                labels=("validator", "status", "category"),
            )
            self._validation_duration = Histogram(
                "truthound_validation_duration_ms",
                "Validation duration in milliseconds",
                labels=("validator", "category"),
            )
            self._issues_counter = Counter(
                "truthound_issues_total",
                "Total issues found",
                labels=("validator", "severity", "category"),
            )
            self._active_validations = Gauge(
                "truthound_active_validations",
                "Currently running validations",
                labels=("category",),
            )
            self._metrics_system = True
            self.logger.debug("Integrated with truthound.observability.metrics")
        except (ImportError, Exception) as e:
            self._metrics_system = None
            self._validation_counter = None
            self._validation_duration = None
            self._issues_counter = None
            self._active_validations = None
            self.logger.debug(f"Metrics integration not available: {e}")

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_validation(
        self,
        validator_name: str,
        category: str,
        status: str,
        duration_ms: float,
        issues: list[ValidationIssue],
    ) -> None:
        """Record metrics for a validation.

        Args:
            validator_name: Name of the validator
            category: Validator category
            status: Execution status
            duration_ms: Duration in milliseconds
            issues: List of validation issues
        """
        if not self._enabled:
            return

        # Local stats (always available)
        key = f"{validator_name}_{status}"
        with self._lock:
            self._local_counts[key] = self._local_counts.get(key, 0) + 1
            self._local_stats[f"{validator_name}_duration_sum"] = (
                self._local_stats.get(f"{validator_name}_duration_sum", 0) + duration_ms
            )

        # Forward to metrics system if available
        if self._metrics_system and self._validation_counter:
            try:
                self._validation_counter.inc(
                    validator=validator_name,
                    status=status,
                    category=category,
                )

                if self._validation_duration:
                    self._validation_duration.observe(
                        duration_ms,
                        validator=validator_name,
                        category=category,
                    )

                if self._issues_counter:
                    for issue in issues:
                        self._issues_counter.inc(
                            validator=validator_name,
                            severity=issue.severity.value,
                            category=category,
                        )
            except Exception as e:
                self.logger.debug(f"Metrics recording skipped: {e}")

    @contextmanager
    def track_validation(
        self,
        validator_name: str,
        category: str,
    ) -> Iterator[dict[str, Any]]:
        """Context manager to track validation execution.

        Args:
            validator_name: Name of the validator
            category: Validator category

        Yields:
            Dict to store results
        """
        start_time = time.time()
        result: dict[str, Any] = {"status": "unknown", "issues": []}

        # Track active validations
        if self._metrics_system and self._active_validations:
            try:
                self._active_validations.inc(category=category)
            except Exception:
                pass

        try:
            yield result
        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Decrement active
            if self._metrics_system and self._active_validations:
                try:
                    self._active_validations.dec(category=category)
                except Exception:
                    pass

            # Record final metrics
            self.record_validation(
                validator_name=validator_name,
                category=category,
                status=result.get("status", "unknown"),
                duration_ms=duration_ms,
                issues=result.get("issues", []),
            )

    def get_stats(self) -> dict[str, Any]:
        """Get local statistics summary."""
        with self._lock:
            return {
                "counts": self._local_counts.copy(),
                "stats": self._local_stats.copy(),
            }


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return MetricsCollector.get_instance()


# =============================================================================
# #16: Reference Data Caching
# =============================================================================


@dataclass
class CacheEntry:
    """Entry in the reference data cache."""

    data: pl.LazyFrame | pl.DataFrame
    created_at: datetime
    expires_at: datetime | None
    hits: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class ReferentialDataCache:
    """Cache for reference data used in referential integrity checks.

    Features:
    - LRU eviction policy
    - TTL-based expiration
    - Memory-aware sizing
    - Thread-safe operations
    """

    _instance: "ReferentialDataCache | None" = None
    _lock = threading.Lock()

    def __init__(
        self,
        max_entries: int = 100,
        max_size_mb: float = 500,
        default_ttl_seconds: float = 3600,
    ) -> None:
        """Initialize the cache.

        Args:
            max_entries: Maximum number of entries
            max_size_mb: Maximum total size in MB
            default_ttl_seconds: Default TTL in seconds
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._default_ttl = timedelta(seconds=default_ttl_seconds)
        self._total_size = 0
        self._hits = 0
        self._misses = 0
        self.logger = ValidatorLogger("ReferentialDataCache")

    @classmethod
    def get_instance(cls) -> "ReferentialDataCache":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _generate_key(
        self,
        source: str,
        column: str,
        query_hash: str | None = None,
    ) -> str:
        """Generate cache key for reference data."""
        parts = [source, column]
        if query_hash:
            parts.append(query_hash)
        key_str = ":".join(parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _estimate_size(self, data: pl.LazyFrame | pl.DataFrame) -> int:
        """Estimate memory size of data in bytes."""
        try:
            if isinstance(data, pl.LazyFrame):
                # Collect schema only
                schema = data.collect_schema()
                return len(schema.names()) * 1000  # Rough estimate
            else:
                return data.estimated_size()
        except Exception:
            return 0

    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries if needed to make room."""
        # Evict expired entries first
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            self._remove_entry(key)

        # Check entry count
        while len(self._cache) >= self._max_entries:
            self._evict_lru()

        # Check size
        while self._total_size + new_size > self._max_size_bytes and self._cache:
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find entry with lowest hit count
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].hits)
        self._remove_entry(lru_key)

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_size -= entry.size_bytes

    def get(
        self,
        source: str,
        column: str,
        query_hash: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame | None:
        """Get cached reference data.

        Args:
            source: Data source identifier
            column: Reference column name
            query_hash: Optional query hash for filtered data

        Returns:
            Cached data or None
        """
        key = self._generate_key(source, column, query_hash)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                return None

            entry.hits += 1
            self._hits += 1
            return entry.data

    def set(
        self,
        source: str,
        column: str,
        data: pl.LazyFrame | pl.DataFrame,
        ttl_seconds: float | None = None,
        query_hash: str | None = None,
    ) -> None:
        """Store reference data in cache.

        Args:
            source: Data source identifier
            column: Reference column name
            data: Data to cache
            ttl_seconds: Optional TTL override
            query_hash: Optional query hash for filtered data
        """
        key = self._generate_key(source, column, query_hash)
        size = self._estimate_size(data)

        # Determine expiration
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self._default_ttl
        expires_at = datetime.utcnow() + ttl

        entry = CacheEntry(
            data=data,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            size_bytes=size,
        )

        with self._lock:
            # Remove existing entry if any
            if key in self._cache:
                self._remove_entry(key)

            # Evict if needed
            self._evict_if_needed(size)

            # Add new entry
            self._cache[key] = entry
            self._total_size += size

    def invalidate(
        self,
        source: str | None = None,
        column: str | None = None,
    ) -> int:
        """Invalidate cached entries.

        Args:
            source: Optional source to match
            column: Optional column to match

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if source is None and column is None:
                count = len(self._cache)
                self._cache.clear()
                self._total_size = 0
                return count

            keys_to_remove = []
            for key in self._cache:
                # This is a simplistic match - in production you'd want
                # to store source/column in the entry for proper matching
                if source and source in key:
                    keys_to_remove.append(key)
                elif column and column in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self._remove_entry(key)

            return len(keys_to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "total_size_mb": self._total_size / (1024 * 1024),
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(total_requests, 1),
            }


def get_reference_cache() -> ReferentialDataCache:
    """Get the global reference data cache."""
    return ReferentialDataCache.get_instance()


# =============================================================================
# #17: Parallel Processing Support
# =============================================================================


class ParallelExecutionMode(Enum):
    """Execution mode for parallel validation."""

    SEQUENTIAL = "sequential"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel validation execution."""

    mode: ParallelExecutionMode = ParallelExecutionMode.THREADING
    max_workers: int | None = None  # None = auto (CPU count)
    chunk_size: int = 10000
    timeout_seconds: float = 300.0
    fail_fast: bool = False  # Stop on first error

    def get_workers(self) -> int:
        """Get effective worker count."""
        if self.max_workers:
            return self.max_workers
        import os
        return min(os.cpu_count() or 4, 8)


class ParallelValidator:
    """Executes multiple validators in parallel.

    Features:
    - Thread and process-based parallelism
    - Chunked processing for large datasets
    - Graceful error handling
    - Progress tracking
    """

    def __init__(
        self,
        validators: list[Validator],
        config: ParallelExecutionConfig | None = None,
    ) -> None:
        """Initialize parallel validator.

        Args:
            validators: List of validators to execute
            config: Execution configuration
        """
        self.validators = validators
        self.config = config or ParallelExecutionConfig()
        self.logger = ValidatorLogger("ParallelValidator")

    def validate(
        self,
        lf: pl.LazyFrame,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ValidatorExecutionResult]:
        """Execute all validators.

        Args:
            lf: LazyFrame to validate
            progress_callback: Optional callback(completed, total)

        Returns:
            List of execution results
        """
        if self.config.mode == ParallelExecutionMode.SEQUENTIAL:
            return self._validate_sequential(lf, progress_callback)
        elif self.config.mode == ParallelExecutionMode.THREADING:
            return self._validate_threaded(lf, progress_callback)
        else:
            return self._validate_multiprocess(lf, progress_callback)

    def _validate_sequential(
        self,
        lf: pl.LazyFrame,
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[ValidatorExecutionResult]:
        """Execute validators sequentially."""
        results: list[ValidatorExecutionResult] = []
        total = len(self.validators)

        for i, validator in enumerate(self.validators):
            try:
                result = validator.validate_safe(lf)
                results.append(result)

                if (
                    self.config.fail_fast
                    and result.status == ValidationResult.FAILED
                ):
                    break

            except Exception as e:
                self.logger.error(f"Validator {validator.name} failed: {e}")
                results.append(
                    ValidatorExecutionResult(
                        validator_name=validator.name,
                        status=ValidationResult.FAILED,
                        issues=[],
                        error_context=ValidationErrorContext(
                            validator_name=validator.name,
                            error_type="execution_error",
                            message=str(e),
                            exception=e,
                        ),
                    )
                )

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def _validate_threaded(
        self,
        lf: pl.LazyFrame,
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[ValidatorExecutionResult]:
        """Execute validators using threading."""
        results: list[ValidatorExecutionResult] = []
        total = len(self.validators)
        completed = 0

        def validate_one(validator: Validator) -> ValidatorExecutionResult:
            try:
                return validator.validate_safe(lf)
            except Exception as e:
                return ValidatorExecutionResult(
                    validator_name=validator.name,
                    status=ValidationResult.FAILED,
                    issues=[],
                    error_context=ValidationErrorContext(
                        validator_name=validator.name,
                        error_type="execution_error",
                        message=str(e),
                        exception=e,
                    ),
                )

        with ThreadPoolExecutor(max_workers=self.config.get_workers()) as executor:
            futures = {
                executor.submit(validate_one, v): v for v in self.validators
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, total)

                if (
                    self.config.fail_fast
                    and result.status == ValidationResult.FAILED
                ):
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        return results

    def _validate_multiprocess(
        self,
        lf: pl.LazyFrame,
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[ValidatorExecutionResult]:
        """Execute validators using multiprocessing.

        Note: This requires validators to be picklable.
        Falls back to threading if multiprocessing fails.
        """
        try:
            # Collect DataFrame for multiprocessing
            df = lf.collect()
            results: list[ValidatorExecutionResult] = []
            total = len(self.validators)
            completed = 0

            def validate_one(args: tuple) -> ValidatorExecutionResult:
                validator, data = args
                try:
                    return validator.validate_safe(data.lazy())
                except Exception as e:
                    return ValidatorExecutionResult(
                        validator_name=validator.name,
                        status=ValidationResult.FAILED,
                        issues=[],
                    )

            with ProcessPoolExecutor(max_workers=self.config.get_workers()) as executor:
                futures = {
                    executor.submit(validate_one, (v, df)): v
                    for v in self.validators
                }

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        results.append(result)
                    except Exception as e:
                        validator = futures[future]
                        results.append(
                            ValidatorExecutionResult(
                                validator_name=validator.name,
                                status=ValidationResult.FAILED,
                                issues=[],
                            )
                        )

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

            return results

        except Exception as e:
            self.logger.warning(
                f"Multiprocessing failed, falling back to threading: {e}"
            )
            return self._validate_threaded(lf, progress_callback)


# =============================================================================
# #18: Configuration Validation
# =============================================================================


@dataclass
class ConfigValidationError:
    """Error in configuration validation."""

    field: str
    message: str
    value: Any
    suggestion: str | None = None


class ConfigValidator:
    """Validates ValidatorConfig settings.

    Catches configuration errors early with helpful messages.
    """

    @classmethod
    def validate(
        cls,
        config: ValidatorConfig,
        validator_name: str = "",
    ) -> list[ConfigValidationError]:
        """Validate configuration.

        Args:
            config: Configuration to validate
            validator_name: Name of validator for context

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[ConfigValidationError] = []

        # Validate sample_size
        if config.sample_size < 0:
            errors.append(
                ConfigValidationError(
                    field="sample_size",
                    message="sample_size must be >= 0",
                    value=config.sample_size,
                    suggestion="Use sample_size=0 to disable sampling",
                )
            )
        elif config.sample_size > 10000:
            errors.append(
                ConfigValidationError(
                    field="sample_size",
                    message="sample_size > 10000 may cause memory issues",
                    value=config.sample_size,
                    suggestion="Consider using sample_size=100 for typical use cases",
                )
            )

        # Validate mostly
        if config.mostly is not None:
            if not (0.0 <= config.mostly <= 1.0):
                errors.append(
                    ConfigValidationError(
                        field="mostly",
                        message="mostly must be in [0.0, 1.0]",
                        value=config.mostly,
                        suggestion="Use mostly=0.95 for 95% pass rate",
                    )
                )

        # Validate timeout
        if config.timeout_seconds is not None:
            if config.timeout_seconds <= 0:
                errors.append(
                    ConfigValidationError(
                        field="timeout_seconds",
                        message="timeout_seconds must be > 0",
                        value=config.timeout_seconds,
                        suggestion="Use timeout_seconds=None to disable timeout",
                    )
                )
            elif config.timeout_seconds < 1:
                errors.append(
                    ConfigValidationError(
                        field="timeout_seconds",
                        message="timeout_seconds < 1 may cause false timeouts",
                        value=config.timeout_seconds,
                        suggestion="Use at least timeout_seconds=1",
                    )
                )

        # Validate memory limit
        if config.memory_limit_mb is not None:
            if config.memory_limit_mb <= 0:
                errors.append(
                    ConfigValidationError(
                        field="memory_limit_mb",
                        message="memory_limit_mb must be > 0",
                        value=config.memory_limit_mb,
                        suggestion="Use memory_limit_mb=None to disable limit",
                    )
                )
            elif config.memory_limit_mb < 10:
                errors.append(
                    ConfigValidationError(
                        field="memory_limit_mb",
                        message="memory_limit_mb < 10 may be too restrictive",
                        value=config.memory_limit_mb,
                        suggestion="Use at least memory_limit_mb=100",
                    )
                )

        # Validate columns
        if config.columns:
            for col in config.columns:
                if not col or not col.strip():
                    errors.append(
                        ConfigValidationError(
                            field="columns",
                            message="Column name cannot be empty",
                            value=col,
                            suggestion="Remove empty column names",
                        )
                    )

        return errors

    @classmethod
    def validate_or_raise(
        cls,
        config: ValidatorConfig,
        validator_name: str = "",
    ) -> None:
        """Validate configuration and raise on error.

        Raises:
            ValueError: If configuration is invalid
        """
        errors = cls.validate(config, validator_name)
        if errors:
            error_msgs = [f"  - {e.field}: {e.message}" for e in errors]
            raise ValueError(
                f"Invalid configuration for {validator_name or 'validator'}:\n"
                + "\n".join(error_msgs)
            )


# =============================================================================
# #19: Polars Version Compatibility
# =============================================================================


@dataclass
class PolarsVersionInfo:
    """Polars version information."""

    major: int
    minor: int
    patch: int
    raw: str

    @classmethod
    def current(cls) -> "PolarsVersionInfo":
        """Get current Polars version."""
        version_str = pl.__version__
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 0,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2].split("-")[0]) if len(parts) > 2 else 0,
            raw=version_str,
        )

    def __ge__(self, other: tuple[int, int, int]) -> bool:
        return (self.major, self.minor, self.patch) >= other

    def __lt__(self, other: tuple[int, int, int]) -> bool:
        return (self.major, self.minor, self.patch) < other


class PolarsCompat:
    """Polars version compatibility layer.

    Provides compatible implementations for API changes between versions.
    """

    _version: PolarsVersionInfo | None = None

    @classmethod
    def version(cls) -> PolarsVersionInfo:
        """Get cached Polars version."""
        if cls._version is None:
            cls._version = PolarsVersionInfo.current()
        return cls._version

    @classmethod
    def collect_schema(cls, lf: pl.LazyFrame) -> pl.Schema:
        """Get schema from LazyFrame (compatible across versions)."""
        v = cls.version()
        if v >= (0, 20, 0):
            return lf.collect_schema()
        else:
            # Older versions
            return lf.schema  # type: ignore

    @classmethod
    def estimated_size(cls, df: pl.DataFrame) -> int:
        """Get estimated size in bytes (compatible across versions)."""
        v = cls.version()
        try:
            if v >= (0, 19, 0):
                return df.estimated_size()
            else:
                return df.estimated_size("b")  # type: ignore
        except Exception:
            # Fallback estimation
            return len(df) * len(df.columns) * 8

    @classmethod
    def str_contains(
        cls,
        expr: pl.Expr,
        pattern: str,
        literal: bool = False,
    ) -> pl.Expr:
        """String contains (compatible across versions)."""
        v = cls.version()
        if v >= (0, 19, 0):
            return expr.str.contains(pattern, literal=literal)
        else:
            # Older API
            if literal:
                return expr.str.contains(pattern, literal=True)  # type: ignore
            return expr.str.contains(pattern)

    @classmethod
    def null_count(cls, lf: pl.LazyFrame, col: str) -> pl.Expr:
        """Count nulls in column (compatible across versions)."""
        v = cls.version()
        if v >= (0, 18, 0):
            return pl.col(col).null_count()
        else:
            return pl.col(col).is_null().sum()

    @classmethod
    def check_min_version(
        cls,
        min_version: tuple[int, int, int],
        feature: str = "",
    ) -> bool:
        """Check if current Polars meets minimum version.

        Args:
            min_version: Minimum required version (major, minor, patch)
            feature: Feature name for warning message

        Returns:
            True if version requirement is met
        """
        v = cls.version()
        if v < min_version:
            ver_str = ".".join(map(str, min_version))
            msg = f"Polars {ver_str}+ required"
            if feature:
                msg += f" for {feature}"
            msg += f", current: {v.raw}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        return True


# =============================================================================
# #20: Internationalization Support
# =============================================================================


class Language(Enum):
    """Supported languages for error messages."""

    EN = "en"
    KO = "ko"
    JA = "ja"
    ZH = "zh"
    ES = "es"
    FR = "fr"
    DE = "de"


# Translation dictionaries
_TRANSLATIONS: dict[str, dict[str, str]] = {
    # Issue types
    "null_values": {
        "en": "Null values found",
        "ko": "null 값 발견",
        "ja": "null値が見つかりました",
        "zh": "发现空值",
        "es": "Valores nulos encontrados",
        "fr": "Valeurs nulles trouvées",
        "de": "Null-Werte gefunden",
    },
    "out_of_range": {
        "en": "Values out of range",
        "ko": "범위를 벗어난 값",
        "ja": "範囲外の値",
        "zh": "超出范围的值",
        "es": "Valores fuera de rango",
        "fr": "Valeurs hors limites",
        "de": "Werte außerhalb des Bereichs",
    },
    "duplicate_values": {
        "en": "Duplicate values found",
        "ko": "중복 값 발견",
        "ja": "重複値が見つかりました",
        "zh": "发现重复值",
        "es": "Valores duplicados encontrados",
        "fr": "Valeurs en double trouvées",
        "de": "Doppelte Werte gefunden",
    },
    "invalid_format": {
        "en": "Invalid format",
        "ko": "잘못된 형식",
        "ja": "無効な形式",
        "zh": "格式无效",
        "es": "Formato inválido",
        "fr": "Format invalide",
        "de": "Ungültiges Format",
    },
    "referential_integrity_violation": {
        "en": "Referential integrity violation",
        "ko": "참조 무결성 위반",
        "ja": "参照整合性違反",
        "zh": "引用完整性违规",
        "es": "Violación de integridad referencial",
        "fr": "Violation d'intégrité référentielle",
        "de": "Referenzielle Integritätsverletzung",
    },
    "schema_mismatch": {
        "en": "Schema mismatch",
        "ko": "스키마 불일치",
        "ja": "スキーマ不一致",
        "zh": "模式不匹配",
        "es": "Desajuste de esquema",
        "fr": "Non-concordance de schéma",
        "de": "Schema-Abweichung",
    },
    # Severity levels
    "critical": {
        "en": "Critical",
        "ko": "심각",
        "ja": "重大",
        "zh": "严重",
        "es": "Crítico",
        "fr": "Critique",
        "de": "Kritisch",
    },
    "high": {
        "en": "High",
        "ko": "높음",
        "ja": "高",
        "zh": "高",
        "es": "Alto",
        "fr": "Élevé",
        "de": "Hoch",
    },
    "medium": {
        "en": "Medium",
        "ko": "중간",
        "ja": "中",
        "zh": "中",
        "es": "Medio",
        "fr": "Moyen",
        "de": "Mittel",
    },
    "low": {
        "en": "Low",
        "ko": "낮음",
        "ja": "低",
        "zh": "低",
        "es": "Bajo",
        "fr": "Faible",
        "de": "Niedrig",
    },
    # Common messages
    "values_found": {
        "en": "{count} values found",
        "ko": "{count}개의 값 발견",
        "ja": "{count}個の値が見つかりました",
        "zh": "发现{count}个值",
        "es": "{count} valores encontrados",
        "fr": "{count} valeurs trouvées",
        "de": "{count} Werte gefunden",
    },
    "column_not_found": {
        "en": "Column '{column}' not found",
        "ko": "'{column}' 컬럼을 찾을 수 없습니다",
        "ja": "'{column}' 列が見つかりません",
        "zh": "未找到'{column}'列",
        "es": "Columna '{column}' no encontrada",
        "fr": "Colonne '{column}' non trouvée",
        "de": "Spalte '{column}' nicht gefunden",
    },
    "validation_passed": {
        "en": "Validation passed",
        "ko": "검증 통과",
        "ja": "検証通過",
        "zh": "验证通过",
        "es": "Validación aprobada",
        "fr": "Validation réussie",
        "de": "Validierung bestanden",
    },
    "validation_failed": {
        "en": "Validation failed",
        "ko": "검증 실패",
        "ja": "検証失敗",
        "zh": "验证失败",
        "es": "Validación fallida",
        "fr": "Validation échouée",
        "de": "Validierung fehlgeschlagen",
    },
}


class I18n:
    """Internationalization support for validation messages.

    Features:
    - Multiple language support
    - Fallback to English
    - Auto-detection from locale
    - Template interpolation
    """

    _current_language: Language = Language.EN
    _custom_translations: dict[str, dict[str, str]] = {}

    @classmethod
    def set_language(cls, lang: Language | str) -> None:
        """Set the current language.

        Args:
            lang: Language enum or code string
        """
        if isinstance(lang, str):
            lang = Language(lang.lower())
        cls._current_language = lang

    @classmethod
    def get_language(cls) -> Language:
        """Get the current language."""
        return cls._current_language

    @classmethod
    def detect_language(cls) -> Language:
        """Detect language from system locale."""
        try:
            loc = locale.getlocale()[0]
            if loc:
                code = loc.split("_")[0].lower()
                try:
                    return Language(code)
                except ValueError:
                    pass
        except Exception:
            pass
        return Language.EN

    @classmethod
    def auto_configure(cls) -> None:
        """Auto-configure language from environment."""
        # Check environment variable first
        env_lang = os.environ.get("TRUTHOUND_LANGUAGE")
        if env_lang:
            try:
                cls.set_language(env_lang)
                return
            except ValueError:
                pass

        # Fall back to locale detection
        cls.set_language(cls.detect_language())

    @classmethod
    def add_translations(cls, key: str, translations: dict[str, str]) -> None:
        """Add custom translations.

        Args:
            key: Translation key
            translations: Dict of language code -> text
        """
        cls._custom_translations[key] = translations

    @classmethod
    def t(
        cls,
        key: str,
        lang: Language | None = None,
        **kwargs: Any,
    ) -> str:
        """Translate a key to the specified language.

        Args:
            key: Translation key
            lang: Language (default: current)
            **kwargs: Template interpolation values

        Returns:
            Translated string
        """
        lang = lang or cls._current_language
        lang_code = lang.value

        # Check custom translations first
        if key in cls._custom_translations:
            translations = cls._custom_translations[key]
        elif key in _TRANSLATIONS:
            translations = _TRANSLATIONS[key]
        else:
            return key  # Return key if not found

        # Get translation for language, fallback to English
        text = translations.get(lang_code, translations.get("en", key))

        # Apply template interpolation
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass  # Keep original if interpolation fails

        return text

    @classmethod
    def translate_issue(
        cls,
        issue: ValidationIssue,
        lang: Language | None = None,
    ) -> ValidationIssue:
        """Translate validation issue to specified language.

        Args:
            issue: Original validation issue
            lang: Target language

        Returns:
            New issue with translated messages
        """
        lang = lang or cls._current_language

        # Translate issue type
        translated_type = cls.t(issue.issue_type, lang)

        # Translate details if it matches a template
        translated_details = issue.details
        if issue.details:
            # Try to match common patterns
            for key in _TRANSLATIONS:
                if key in issue.issue_type.lower():
                    translated_details = cls.t(
                        "values_found",
                        lang,
                        count=issue.count,
                    )
                    break

        # Create new issue with translated content
        return ValidationIssue(
            column=issue.column,
            issue_type=translated_type,
            count=issue.count,
            severity=issue.severity,
            details=translated_details,
            expected=issue.expected,
            actual=issue.actual,
            sample_values=issue.sample_values,
            error_context=issue.error_context,
            validator_name=issue.validator_name,
            execution_time_ms=issue.execution_time_ms,
        )

    @classmethod
    def translate_severity(cls, severity: Severity, lang: Language | None = None) -> str:
        """Translate severity level to specified language."""
        return cls.t(severity.value.lower(), lang)


# Convenience function
def translate(key: str, **kwargs: Any) -> str:
    """Translate a key using current language."""
    return I18n.t(key, **kwargs)


# =============================================================================
# Integration: Enhanced Validator with Enterprise Features
# =============================================================================


class EnterpriseValidator(Validator):
    """Validator with enterprise features enabled.

    Automatically integrates:
    - Audit logging
    - Metrics collection
    - Configuration validation
    - Polars compatibility checks
    - Internationalized messages

    Usage:
        class MyValidator(EnterpriseValidator):
            name = "my_validator"
            category = "custom"

            def validate(self, lf):
                # Your validation logic
                pass
    """

    # Enterprise features
    enable_audit: bool = True
    enable_metrics: bool = True
    validate_config: bool = True
    translate_messages: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Validate config if enabled
        if self.validate_config:
            ConfigValidator.validate_or_raise(self.config, self.name)

    def validate_safe(self, lf: pl.LazyFrame) -> ValidatorExecutionResult:
        """Execute validation with enterprise features."""
        audit_logger = get_validation_audit_logger() if self.enable_audit else None
        metrics = get_metrics_collector() if self.enable_metrics else None

        # Execute with metrics tracking
        if metrics:
            with metrics.track_validation(self.name, self.category) as ctx:
                result = super().validate_safe(lf)
                ctx["status"] = result.status.value
                ctx["issues"] = result.issues
        else:
            result = super().validate_safe(lf)

        # Log to audit
        if audit_logger:
            audit_logger.log_validation(
                validator=self,
                lf=lf,
                result=result,
            )

        # Translate messages if enabled
        if self.translate_messages:
            result.issues = [
                I18n.translate_issue(issue) for issue in result.issues
            ]

        return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Audit (#14)
    "ValidationAuditRecord",
    "ValidationAuditLogger",
    "get_validation_audit_logger",
    # Metrics (#15)
    "MetricsCollector",
    "get_metrics_collector",
    # Caching (#16)
    "CacheEntry",
    "ReferentialDataCache",
    "get_reference_cache",
    # Parallel (#17)
    "ParallelExecutionMode",
    "ParallelExecutionConfig",
    "ParallelValidator",
    # Config Validation (#18)
    "ConfigValidationError",
    "ConfigValidator",
    # Polars Compat (#19)
    "PolarsVersionInfo",
    "PolarsCompat",
    # I18n (#20)
    "Language",
    "I18n",
    "translate",
    # Enterprise Validator
    "EnterpriseValidator",
]
