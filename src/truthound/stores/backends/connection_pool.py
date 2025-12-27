"""Enterprise-grade connection pool management for database stores.

This module provides a highly extensible and maintainable connection pool
abstraction layer with:
- Multiple pooling strategies (QueuePool, NullPool, StaticPool, AsyncAdaptedQueuePool)
- Connection health monitoring and automatic recovery
- Circuit breaker pattern for fault tolerance
- Comprehensive metrics and observability
- Database-specific optimizations
- Retry logic with exponential backoff

Install with: pip install truthound[database]
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

# Lazy import SQLAlchemy
try:
    from sqlalchemy import create_engine, event, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import (
        DBAPIError,
        DisconnectionError,
        InterfaceError,
        OperationalError,
        SQLAlchemyError,
    )
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.pool import (
        NullPool,
        QueuePool,
        SingletonThreadPool,
        StaticPool,
    )

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    SQLAlchemyError = Exception  # type: ignore
    DBAPIError = Exception  # type: ignore
    OperationalError = Exception  # type: ignore
    DisconnectionError = Exception  # type: ignore
    InterfaceError = Exception  # type: ignore

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import Pool


# =============================================================================
# Enums and Constants
# =============================================================================


class PoolStrategy(Enum):
    """Connection pool strategy types."""

    QUEUE_POOL = auto()  # Standard pool with overflow
    NULL_POOL = auto()  # No pooling, new connection each time
    STATIC_POOL = auto()  # Single connection for all requests
    SINGLETON_THREAD = auto()  # One connection per thread
    ASYNC_QUEUE = auto()  # Async-compatible queue pool


class DatabaseDialect(Enum):
    """Supported database dialects with specific optimizations."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MSSQL = "mssql"
    ORACLE = "oracle"
    UNKNOWN = "unknown"

    @classmethod
    def from_url(cls, url: str) -> "DatabaseDialect":
        """Detect dialect from connection URL."""
        url_lower = url.lower()
        if url_lower.startswith("postgresql") or url_lower.startswith("postgres"):
            return cls.POSTGRESQL
        elif url_lower.startswith("mysql"):
            return cls.MYSQL
        elif url_lower.startswith("sqlite"):
            return cls.SQLITE
        elif url_lower.startswith("mssql") or "pyodbc" in url_lower:
            return cls.MSSQL
        elif url_lower.startswith("oracle"):
            return cls.ORACLE
        return cls.UNKNOWN


class ConnectionState(Enum):
    """State of a pooled connection."""

    IDLE = auto()
    IN_USE = auto()
    STALE = auto()
    BROKEN = auto()
    RECYCLING = auto()


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, reject requests
    HALF_OPEN = auto()  # Testing if recovered


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class PoolConfig:
    """Connection pool configuration.

    Attributes:
        strategy: Pooling strategy to use.
        pool_size: Number of connections to maintain in pool.
        max_overflow: Maximum overflow connections beyond pool_size.
        pool_timeout: Seconds to wait for available connection.
        pool_recycle: Seconds before a connection is recycled (-1 = never).
        pool_pre_ping: Whether to test connections before use.
        echo_pool: Log pool checkouts/checkins for debugging.
        reset_on_return: How to reset connections when returned.
    """

    strategy: PoolStrategy = PoolStrategy.QUEUE_POOL
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    echo_pool: bool = False
    reset_on_return: str = "rollback"  # "rollback", "commit", or None


def _get_default_retryable_errors() -> tuple[type[Exception], ...]:
    """Get default retryable error types based on available imports."""
    if HAS_SQLALCHEMY:
        return (OperationalError, DisconnectionError, InterfaceError)
    # When SQLAlchemy is not available, use a placeholder that won't match anything
    # Users should provide their own retryable_errors in this case
    return ()


@dataclass
class RetryConfig:
    """Retry behavior configuration.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter to delays.
        retryable_errors: Error types that should trigger retry.
    """

    max_retries: int = 3
    base_delay: float = 0.1
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: tuple[type[Exception], ...] = field(
        default_factory=_get_default_retryable_errors
    )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration.

    Attributes:
        failure_threshold: Failures before opening circuit.
        success_threshold: Successes before closing circuit.
        timeout: Seconds before attempting recovery.
        half_open_max_calls: Max calls in half-open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class HealthCheckConfig:
    """Health check configuration.

    Attributes:
        enabled: Whether health checks are enabled.
        interval: Seconds between health checks.
        timeout: Seconds before health check times out.
        query: SQL query to use for health check.
    """

    enabled: bool = True
    interval: float = 30.0
    timeout: float = 5.0
    query: str = "SELECT 1"


@dataclass
class ConnectionPoolConfig:
    """Complete connection pool manager configuration.

    Combines all sub-configurations into a single config object.
    """

    connection_url: str = "sqlite:///:memory:"
    pool: PoolConfig = field(default_factory=PoolConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    echo: bool = False
    connect_args: dict[str, Any] = field(default_factory=dict)
    engine_options: dict[str, Any] = field(default_factory=dict)

    @property
    def dialect(self) -> DatabaseDialect:
        """Get detected database dialect."""
        return DatabaseDialect.from_url(self.connection_url)

    def with_dialect_defaults(self) -> "ConnectionPoolConfig":
        """Apply dialect-specific default settings."""
        dialect = self.dialect
        config = ConnectionPoolConfig(
            connection_url=self.connection_url,
            pool=PoolConfig(
                strategy=self.pool.strategy,
                pool_size=self.pool.pool_size,
                max_overflow=self.pool.max_overflow,
                pool_timeout=self.pool.pool_timeout,
                pool_recycle=self.pool.pool_recycle,
                pool_pre_ping=self.pool.pool_pre_ping,
                echo_pool=self.pool.echo_pool,
                reset_on_return=self.pool.reset_on_return,
            ),
            retry=self.retry,
            circuit_breaker=self.circuit_breaker,
            health_check=self.health_check,
            echo=self.echo,
            connect_args=dict(self.connect_args),
            engine_options=dict(self.engine_options),
        )

        # Apply dialect-specific defaults
        if dialect == DatabaseDialect.SQLITE:
            config.connect_args.setdefault("check_same_thread", False)
            # SQLite doesn't support connection pooling well
            if config.pool.strategy == PoolStrategy.QUEUE_POOL:
                config.pool.pool_size = 1
                config.pool.max_overflow = 0

        elif dialect == DatabaseDialect.POSTGRESQL:
            # PostgreSQL-specific optimizations
            config.engine_options.setdefault("pool_use_lifo", True)
            config.pool.pool_recycle = min(config.pool.pool_recycle, 1800)

        elif dialect == DatabaseDialect.MYSQL:
            # MySQL typically needs more aggressive recycling
            config.pool.pool_recycle = min(config.pool.pool_recycle, 3600)
            config.connect_args.setdefault("connect_timeout", 10)

        return config


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class PoolMetricsProtocol(Protocol):
    """Protocol for pool metrics providers."""

    @property
    def total_connections(self) -> int:
        """Total connections in pool."""
        ...

    @property
    def active_connections(self) -> int:
        """Currently checked out connections."""
        ...

    @property
    def idle_connections(self) -> int:
        """Available connections in pool."""
        ...

    @property
    def overflow_connections(self) -> int:
        """Connections beyond pool_size."""
        ...


@runtime_checkable
class ConnectionPoolProtocol(Protocol):
    """Protocol for connection pool implementations."""

    def get_connection(self) -> Any:
        """Get a connection from the pool."""
        ...

    def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool."""
        ...

    def dispose(self) -> None:
        """Dispose of all connections."""
        ...

    @property
    def metrics(self) -> PoolMetricsProtocol:
        """Get pool metrics."""
        ...


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class PoolMetrics:
    """Connection pool metrics for monitoring and observability.

    Thread-safe metrics collection with atomic operations.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Connection counts
    total_created: int = 0
    total_closed: int = 0
    current_size: int = 0
    current_checked_out: int = 0
    current_overflow: int = 0
    peak_connections: int = 0

    # Operation counts
    checkouts: int = 0
    checkins: int = 0
    checkout_failures: int = 0
    recycles: int = 0
    invalidations: int = 0

    # Timing metrics
    total_checkout_time_ms: float = 0.0
    max_checkout_time_ms: float = 0.0
    total_connection_time_ms: float = 0.0

    # Health check metrics
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    last_health_check: datetime | None = None

    # Circuit breaker metrics
    circuit_opens: int = 0
    circuit_closes: int = 0
    requests_rejected: int = 0

    @property
    def total_connections(self) -> int:
        """Total connections currently managed."""
        return self.current_size + self.current_overflow

    @property
    def active_connections(self) -> int:
        """Currently checked out connections."""
        return self.current_checked_out

    @property
    def idle_connections(self) -> int:
        """Available connections in pool."""
        return self.current_size - self.current_checked_out

    @property
    def overflow_connections(self) -> int:
        """Connections beyond pool_size."""
        return self.current_overflow

    @property
    def avg_checkout_time_ms(self) -> float:
        """Average checkout time in milliseconds."""
        if self.checkouts == 0:
            return 0.0
        return self.total_checkout_time_ms / self.checkouts

    def record_checkout(self, duration_ms: float) -> None:
        """Record a successful checkout."""
        with self._lock:
            self.checkouts += 1
            self.current_checked_out += 1
            self.total_checkout_time_ms += duration_ms
            self.max_checkout_time_ms = max(self.max_checkout_time_ms, duration_ms)
            if self.current_checked_out > self.peak_connections:
                self.peak_connections = self.current_checked_out

    def record_checkin(self) -> None:
        """Record a connection checkin."""
        with self._lock:
            self.checkins += 1
            self.current_checked_out = max(0, self.current_checked_out - 1)

    def record_creation(self) -> None:
        """Record a new connection creation."""
        with self._lock:
            self.total_created += 1
            self.current_size += 1

    def record_close(self) -> None:
        """Record a connection close."""
        with self._lock:
            self.total_closed += 1
            self.current_size = max(0, self.current_size - 1)

    def record_health_check(self, passed: bool) -> None:
        """Record health check result."""
        with self._lock:
            self.last_health_check = datetime.now(timezone.utc)
            if passed:
                self.health_checks_passed += 1
            else:
                self.health_checks_failed += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "connections": {
                "total_created": self.total_created,
                "total_closed": self.total_closed,
                "current_size": self.current_size,
                "current_checked_out": self.current_checked_out,
                "current_overflow": self.current_overflow,
                "peak": self.peak_connections,
                "idle": self.idle_connections,
            },
            "operations": {
                "checkouts": self.checkouts,
                "checkins": self.checkins,
                "checkout_failures": self.checkout_failures,
                "recycles": self.recycles,
                "invalidations": self.invalidations,
            },
            "timing": {
                "avg_checkout_ms": round(self.avg_checkout_time_ms, 2),
                "max_checkout_ms": round(self.max_checkout_time_ms, 2),
                "total_connection_time_ms": round(self.total_connection_time_ms, 2),
            },
            "health": {
                "checks_passed": self.health_checks_passed,
                "checks_failed": self.health_checks_failed,
                "last_check": (
                    self.last_health_check.isoformat() if self.last_health_check else None
                ),
            },
            "circuit_breaker": {
                "opens": self.circuit_opens,
                "closes": self.circuit_closes,
                "requests_rejected": self.requests_rejected,
            },
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = [
            "# HELP truthound_pool_connections_total Total connections created",
            f"truthound_pool_connections_total {self.total_created}",
            "# HELP truthound_pool_connections_current Current pool size",
            f"truthound_pool_connections_current {self.current_size}",
            "# HELP truthound_pool_connections_active Active connections",
            f"truthound_pool_connections_active {self.current_checked_out}",
            "# HELP truthound_pool_connections_idle Idle connections",
            f"truthound_pool_connections_idle {self.idle_connections}",
            "# HELP truthound_pool_checkouts_total Total checkouts",
            f"truthound_pool_checkouts_total {self.checkouts}",
            "# HELP truthound_pool_checkout_time_avg_ms Average checkout time",
            f"truthound_pool_checkout_time_avg_ms {self.avg_checkout_time_ms:.2f}",
            "# HELP truthound_pool_health_checks_passed Health checks passed",
            f"truthound_pool_health_checks_passed {self.health_checks_passed}",
            "# HELP truthound_pool_health_checks_failed Health checks failed",
            f"truthound_pool_health_checks_failed {self.health_checks_failed}",
        ]
        return "\n".join(lines)


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """Circuit breaker implementation for connection pool fault tolerance.

    Prevents cascading failures by temporarily rejecting requests when
    the database is experiencing issues.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration.
        """
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time:
                    elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                    if elapsed >= self._config.timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
                        # First call in half-open, increment counter
                        self._half_open_calls += 1
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            logger.info(f"Circuit breaker closed (was {old_state.name})")

        elif new_state == CircuitState.OPEN:
            self._success_count = 0
            self._half_open_calls = 0
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )

        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
            logger.info("Circuit breaker half-open, testing recovery")

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)


# =============================================================================
# Retry Handler
# =============================================================================


class RetryHandler:
    """Retry handler with exponential backoff and jitter.

    Provides configurable retry behavior for transient database errors.
    """

    def __init__(self, config: RetryConfig) -> None:
        """Initialize retry handler.

        Args:
            config: Retry configuration.
        """
        self._config = config
        import random

        self._random = random

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if operation should be retried.

        Args:
            error: The exception that occurred.
            attempt: Current attempt number (1-based).

        Returns:
            True if should retry, False otherwise.
        """
        if attempt >= self._config.max_retries:
            return False
        return isinstance(error, self._config.retryable_errors)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry.

        Args:
            attempt: Current attempt number (1-based).

        Returns:
            Delay in seconds.
        """
        delay = self._config.base_delay * (self._config.exponential_base ** (attempt - 1))
        delay = min(delay, self._config.max_delay)

        if self._config.jitter:
            # Add ±25% jitter
            jitter = delay * 0.25 * (2 * self._random.random() - 1)
            delay += jitter

        return max(0, delay)

    def execute_with_retry(
        self,
        operation: Callable[[], Any],
        on_retry: Callable[[Exception, int], None] | None = None,
    ) -> Any:
        """Execute operation with retry logic.

        Args:
            operation: Callable to execute.
            on_retry: Optional callback called before each retry.

        Returns:
            Result of the operation.

        Raises:
            Exception: The last exception if all retries exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 2):
            try:
                return operation()
            except Exception as e:
                last_error = e

                if not self.should_retry(e, attempt):
                    raise

                delay = self.get_delay(attempt)

                if on_retry:
                    on_retry(e, attempt)

                logger.warning(
                    f"Retry attempt {attempt}/{self._config.max_retries} "
                    f"after {delay:.2f}s: {e}"
                )

                time.sleep(delay)

        # Should not reach here, but for type safety
        if last_error:
            raise last_error
        raise RuntimeError("Retry logic error")


# =============================================================================
# Connection Pool Manager
# =============================================================================


class ConnectionPoolManager:
    """Enterprise-grade connection pool manager.

    Provides:
    - Multiple pooling strategies
    - Health monitoring
    - Circuit breaker protection
    - Retry logic
    - Comprehensive metrics
    - Database-specific optimizations

    Example:
        >>> config = ConnectionPoolConfig(
        ...     connection_url="postgresql://user:pass@localhost/db",
        ...     pool=PoolConfig(pool_size=10, max_overflow=20),
        ... )
        >>> manager = ConnectionPoolManager(config)
        >>> with manager.session() as session:
        ...     session.execute(text("SELECT 1"))
    """

    def __init__(
        self,
        config: ConnectionPoolConfig | None = None,
        *,
        connection_url: str | None = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize connection pool manager.

        Args:
            config: Complete configuration object.
            connection_url: Database connection URL (if not using config).
            pool_size: Pool size (if not using config).
            max_overflow: Max overflow (if not using config).
            **kwargs: Additional options passed to config.
        """
        if not HAS_SQLALCHEMY:
            raise ImportError(
                "SQLAlchemy is required for ConnectionPoolManager. "
                "Install with: pip install truthound[database]"
            )

        # Build config from parameters if not provided
        if config is None:
            url = connection_url or kwargs.pop("url", "sqlite:///:memory:")
            pool_config = PoolConfig(
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=kwargs.pop("pool_timeout", 30.0),
                pool_recycle=kwargs.pop("pool_recycle", 3600),
                pool_pre_ping=kwargs.pop("pool_pre_ping", True),
            )
            config = ConnectionPoolConfig(
                connection_url=url,
                pool=pool_config,
                echo=kwargs.pop("echo", False),
                connect_args=kwargs.pop("connect_args", {}),
                engine_options=kwargs,
            )

        # Apply dialect-specific defaults
        self._config = config.with_dialect_defaults()

        # Initialize components
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._metrics = PoolMetrics()
        self._circuit_breaker = CircuitBreaker(self._config.circuit_breaker)
        self._retry_handler = RetryHandler(self._config.retry)

        # Health check management
        self._health_check_thread: threading.Thread | None = None
        self._health_check_stop = threading.Event()
        self._is_healthy = True

        # Lifecycle management
        self._lock = threading.RLock()
        self._initialized = False
        self._disposed = False

        # Tracked sessions for cleanup
        self._active_sessions: weakref.WeakSet[Session] = weakref.WeakSet()

    @property
    def config(self) -> ConnectionPoolConfig:
        """Get pool configuration."""
        return self._config

    @property
    def metrics(self) -> PoolMetrics:
        """Get pool metrics."""
        return self._metrics

    @property
    def is_healthy(self) -> bool:
        """Check if pool is healthy."""
        return self._is_healthy and self._circuit_breaker.is_closed

    @property
    def circuit_state(self) -> CircuitState:
        """Get circuit breaker state."""
        return self._circuit_breaker.state

    def initialize(self) -> None:
        """Initialize the connection pool.

        Creates the engine, session factory, and starts health checks.
        """
        with self._lock:
            if self._initialized:
                return

            if self._disposed:
                raise RuntimeError("Cannot initialize disposed pool manager")

            self._create_engine()
            self._test_connection()
            self._setup_event_listeners()
            self._start_health_checks()
            self._initialized = True

            logger.info(
                f"ConnectionPoolManager initialized: "
                f"dialect={self._config.dialect.value}, "
                f"pool_size={self._config.pool.pool_size}, "
                f"max_overflow={self._config.pool.max_overflow}"
            )

    def _create_engine(self) -> None:
        """Create SQLAlchemy engine with configured pool."""
        pool_config = self._config.pool
        strategy = pool_config.strategy

        # Build pool class based on strategy
        pool_class: type | None = None
        pool_kwargs: dict[str, Any] = {}

        if strategy == PoolStrategy.QUEUE_POOL:
            pool_class = QueuePool
            pool_kwargs = {
                "pool_size": pool_config.pool_size,
                "max_overflow": pool_config.max_overflow,
                "timeout": pool_config.pool_timeout,
                "pool_recycle": pool_config.pool_recycle,
                "pool_pre_ping": pool_config.pool_pre_ping,
            }
        elif strategy == PoolStrategy.NULL_POOL:
            pool_class = NullPool
        elif strategy == PoolStrategy.STATIC_POOL:
            pool_class = StaticPool
        elif strategy == PoolStrategy.SINGLETON_THREAD:
            pool_class = SingletonThreadPool
            pool_kwargs = {"pool_size": pool_config.pool_size}

        # Build engine options
        engine_options = {
            "echo": self._config.echo,
            "echo_pool": pool_config.echo_pool,
            "connect_args": self._config.connect_args,
            **self._config.engine_options,
        }

        if pool_class:
            engine_options["poolclass"] = pool_class
            engine_options.update(pool_kwargs)

        self._engine = create_engine(self._config.connection_url, **engine_options)
        self._session_factory = sessionmaker(bind=self._engine)

    def _test_connection(self) -> None:
        """Test database connection."""
        if self._engine is None:
            raise RuntimeError("Engine not created")

        try:
            with self._engine.connect() as conn:
                conn.execute(text(self._config.health_check.query))
                conn.commit()
        except SQLAlchemyError as e:
            logger.error(f"Connection test failed: {e}")
            raise

    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for metrics."""
        if self._engine is None:
            return

        @event.listens_for(self._engine, "checkout")
        def on_checkout(
            dbapi_conn: Any, connection_record: Any, connection_proxy: Any
        ) -> None:
            self._metrics.record_checkout(0)  # Duration tracked elsewhere

        @event.listens_for(self._engine, "checkin")
        def on_checkin(dbapi_conn: Any, connection_record: Any) -> None:
            self._metrics.record_checkin()

        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_conn: Any, connection_record: Any) -> None:
            self._metrics.record_creation()

        @event.listens_for(self._engine, "close")
        def on_close(dbapi_conn: Any, connection_record: Any) -> None:
            self._metrics.record_close()

        @event.listens_for(self._engine, "invalidate")
        def on_invalidate(
            dbapi_conn: Any, connection_record: Any, exception: Any
        ) -> None:
            self._metrics._lock.acquire()
            try:
                self._metrics.invalidations += 1
            finally:
                self._metrics._lock.release()

    def _start_health_checks(self) -> None:
        """Start background health check thread."""
        if not self._config.health_check.enabled:
            return

        self._health_check_stop.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="truthound-pool-health-check",
            daemon=True,
        )
        self._health_check_thread.start()

    def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._health_check_stop.is_set():
            try:
                self._perform_health_check()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            self._health_check_stop.wait(self._config.health_check.interval)

    def _perform_health_check(self) -> None:
        """Perform a single health check."""
        if self._engine is None:
            self._is_healthy = False
            self._metrics.record_health_check(False)
            return

        try:
            with self._engine.connect() as conn:
                conn.execute(text(self._config.health_check.query))
                conn.commit()
            self._is_healthy = True
            self._metrics.record_health_check(True)
        except SQLAlchemyError:
            self._is_healthy = False
            self._metrics.record_health_check(False)

    def get_engine(self) -> "Engine":
        """Get the SQLAlchemy engine.

        Returns:
            The configured engine.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            self.initialize()
        if self._engine is None:
            raise RuntimeError("Engine not available")
        return self._engine

    def get_session(self) -> "Session":
        """Get a new database session.

        Returns:
            A new session instance.

        Raises:
            RuntimeError: If circuit is open.
        """
        if not self._initialized:
            self.initialize()

        if not self._circuit_breaker.can_execute():
            self._metrics._lock.acquire()
            try:
                self._metrics.requests_rejected += 1
            finally:
                self._metrics._lock.release()
            raise RuntimeError(
                "Circuit breaker is open, database connections temporarily unavailable"
            )

        if self._session_factory is None:
            raise RuntimeError("Session factory not available")

        session = self._session_factory()
        self._active_sessions.add(session)
        return session

    @contextmanager
    def session(self) -> Iterator["Session"]:
        """Context manager for database sessions.

        Provides automatic commit/rollback and connection return.

        Yields:
            Database session.

        Example:
            >>> with pool_manager.session() as session:
            ...     result = session.execute(text("SELECT 1"))
        """
        session = None
        start_time = time.time()
        success = False

        try:
            session = self.get_session()
            yield session
            session.commit()
            success = True
            self._circuit_breaker.record_success()

        except SQLAlchemyError as e:
            if session:
                session.rollback()
            self._circuit_breaker.record_failure()
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            if session:
                try:
                    session.close()
                except Exception:
                    pass

            # Record checkout timing
            if success:
                with self._metrics._lock:
                    self._metrics.total_checkout_time_ms += duration_ms
                    self._metrics.max_checkout_time_ms = max(
                        self._metrics.max_checkout_time_ms, duration_ms
                    )

    def execute_with_retry(
        self,
        operation: Callable[["Session"], Any],
    ) -> Any:
        """Execute operation with automatic retry.

        Args:
            operation: Callable that takes a session and returns a result.

        Returns:
            Result of the operation.
        """

        def wrapped_operation() -> Any:
            with self.session() as session:
                return operation(session)

        return self._retry_handler.execute_with_retry(wrapped_operation)

    def recycle_connections(self) -> int:
        """Manually recycle all pool connections.

        Returns:
            Number of connections recycled.
        """
        if self._engine is None:
            return 0

        pool = self._engine.pool
        if pool is None:
            return 0

        # Dispose and recreate
        old_size = self._metrics.current_size
        self._engine.dispose()
        self._metrics._lock.acquire()
        try:
            self._metrics.recycles += old_size
            self._metrics.current_size = 0
        finally:
            self._metrics._lock.release()

        logger.info(f"Recycled {old_size} connections")
        return old_size

    def get_pool_status(self) -> dict[str, Any]:
        """Get comprehensive pool status.

        Returns:
            Dictionary with pool status information.
        """
        return {
            "initialized": self._initialized,
            "disposed": self._disposed,
            "healthy": self.is_healthy,
            "circuit_state": self._circuit_breaker.state.name,
            "dialect": self._config.dialect.value,
            "config": {
                "connection_url": self._mask_password(self._config.connection_url),
                "pool_size": self._config.pool.pool_size,
                "max_overflow": self._config.pool.max_overflow,
                "pool_timeout": self._config.pool.pool_timeout,
                "pool_recycle": self._config.pool.pool_recycle,
            },
            "metrics": self._metrics.to_dict(),
        }

    def _mask_password(self, url: str) -> str:
        """Mask password in connection URL."""
        import re

        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)

    def dispose(self) -> None:
        """Dispose of the connection pool.

        Closes all connections and stops health checks.
        """
        with self._lock:
            if self._disposed:
                return

            # Stop health checks
            if self._health_check_thread:
                self._health_check_stop.set()
                self._health_check_thread.join(timeout=5.0)
                self._health_check_thread = None

            # Close active sessions
            for session in list(self._active_sessions):
                try:
                    session.close()
                except Exception:
                    pass

            # Dispose engine
            if self._engine:
                self._engine.dispose()
                self._engine = None
                self._session_factory = None

            self._disposed = True
            self._initialized = False

            logger.info("ConnectionPoolManager disposed")

    def __enter__(self) -> "ConnectionPoolManager":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.dispose()

    def __del__(self) -> None:
        """Destructor - ensure cleanup."""
        try:
            self.dispose()
        except Exception:
            pass


# =============================================================================
# Factory Functions
# =============================================================================


def create_pool_manager(
    connection_url: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    strategy: PoolStrategy = PoolStrategy.QUEUE_POOL,
    enable_circuit_breaker: bool = True,
    enable_health_checks: bool = True,
    **kwargs: Any,
) -> ConnectionPoolManager:
    """Create a connection pool manager with sensible defaults.

    Args:
        connection_url: Database connection URL.
        pool_size: Number of connections to maintain.
        max_overflow: Maximum overflow connections.
        strategy: Pooling strategy to use.
        enable_circuit_breaker: Whether to enable circuit breaker.
        enable_health_checks: Whether to enable health checks.
        **kwargs: Additional configuration options.

    Returns:
        Configured ConnectionPoolManager instance.

    Example:
        >>> manager = create_pool_manager(
        ...     "postgresql://user:pass@localhost/db",
        ...     pool_size=10,
        ...     max_overflow=20,
        ... )
    """
    pool_config = PoolConfig(
        strategy=strategy,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=kwargs.pop("pool_timeout", 30.0),
        pool_recycle=kwargs.pop("pool_recycle", 3600),
        pool_pre_ping=kwargs.pop("pool_pre_ping", True),
    )

    circuit_config = CircuitBreakerConfig()
    if not enable_circuit_breaker:
        circuit_config.failure_threshold = 999999  # Effectively disabled

    health_config = HealthCheckConfig(enabled=enable_health_checks)

    config = ConnectionPoolConfig(
        connection_url=connection_url,
        pool=pool_config,
        circuit_breaker=circuit_config,
        health_check=health_config,
        echo=kwargs.pop("echo", False),
        connect_args=kwargs.pop("connect_args", {}),
        engine_options=kwargs,
    )

    return ConnectionPoolManager(config)


def create_pool_for_dialect(
    dialect: DatabaseDialect | str,
    host: str = "localhost",
    port: int | None = None,
    database: str = "truthound",
    username: str = "",
    password: str = "",
    **kwargs: Any,
) -> ConnectionPoolManager:
    """Create a pool manager for a specific database dialect.

    Args:
        dialect: Database dialect (or string name).
        host: Database host.
        port: Database port (uses dialect default if None).
        database: Database name.
        username: Database username.
        password: Database password.
        **kwargs: Additional pool configuration.

    Returns:
        Configured ConnectionPoolManager instance.

    Example:
        >>> manager = create_pool_for_dialect(
        ...     DatabaseDialect.POSTGRESQL,
        ...     host="localhost",
        ...     database="mydb",
        ...     username="user",
        ...     password="pass",
        ... )
    """
    if isinstance(dialect, str):
        dialect = DatabaseDialect(dialect)

    # Build connection URL based on dialect
    default_ports = {
        DatabaseDialect.POSTGRESQL: 5432,
        DatabaseDialect.MYSQL: 3306,
        DatabaseDialect.MSSQL: 1433,
        DatabaseDialect.ORACLE: 1521,
    }

    port = port or default_ports.get(dialect)

    if dialect == DatabaseDialect.SQLITE:
        url = f"sqlite:///{database}"
    elif dialect == DatabaseDialect.POSTGRESQL:
        url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    elif dialect == DatabaseDialect.MYSQL:
        url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif dialect == DatabaseDialect.MSSQL:
        url = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    elif dialect == DatabaseDialect.ORACLE:
        url = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError(f"Unsupported dialect: {dialect}")

    return create_pool_manager(url, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "PoolStrategy",
    "DatabaseDialect",
    "ConnectionState",
    "CircuitState",
    # Configuration
    "PoolConfig",
    "RetryConfig",
    "CircuitBreakerConfig",
    "HealthCheckConfig",
    "ConnectionPoolConfig",
    # Core classes
    "PoolMetrics",
    "CircuitBreaker",
    "RetryHandler",
    "ConnectionPoolManager",
    # Factory functions
    "create_pool_manager",
    "create_pool_for_dialect",
    # Protocols
    "PoolMetricsProtocol",
    "ConnectionPoolProtocol",
]
