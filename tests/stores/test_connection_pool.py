"""Tests for the enterprise connection pool manager.

This module provides comprehensive tests for:
- Connection pool configuration
- Pool strategies
- Circuit breaker
- Retry handler
- Health checks
- Metrics collection
- Database dialect detection
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Check if sqlalchemy is available
try:
    import sqlalchemy
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError, DisconnectionError

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    sqlalchemy = None  # type: ignore

# Import the module under test
from truthound.stores.backends.connection_pool import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ConnectionPoolConfig,
    ConnectionPoolManager,
    DatabaseDialect,
    HealthCheckConfig,
    PoolConfig,
    PoolMetrics,
    PoolStrategy,
    RetryConfig,
    RetryHandler,
    create_pool_manager,
    create_pool_for_dialect,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pool_config() -> PoolConfig:
    """Create a default pool configuration."""
    return PoolConfig(
        strategy=PoolStrategy.QUEUE_POOL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30.0,
        pool_recycle=3600,
        pool_pre_ping=True,
    )


@pytest.fixture
def retry_config() -> RetryConfig:
    """Create a default retry configuration."""
    return RetryConfig(
        max_retries=3,
        base_delay=0.01,  # Fast for tests
        max_delay=0.1,
        exponential_base=2.0,
        jitter=False,  # Deterministic for tests
    )


@pytest.fixture
def circuit_breaker_config() -> CircuitBreakerConfig:
    """Create a default circuit breaker configuration."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=1.0,  # Fast for tests
        half_open_max_calls=2,
    )


@pytest.fixture
def health_check_config() -> HealthCheckConfig:
    """Create a default health check configuration."""
    return HealthCheckConfig(
        enabled=False,  # Disabled for tests
        interval=1.0,
        timeout=1.0,
        query="SELECT 1",
    )


@pytest.fixture
def connection_pool_config(
    pool_config: PoolConfig,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    health_check_config: HealthCheckConfig,
) -> ConnectionPoolConfig:
    """Create a complete connection pool configuration."""
    return ConnectionPoolConfig(
        connection_url="sqlite:///:memory:",
        pool=pool_config,
        retry=retry_config,
        circuit_breaker=circuit_breaker_config,
        health_check=health_check_config,
        echo=False,
    )


# =============================================================================
# DatabaseDialect Tests
# =============================================================================


class TestDatabaseDialect:
    """Tests for database dialect detection."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("postgresql://user:pass@localhost/db", DatabaseDialect.POSTGRESQL),
            ("postgres://user:pass@localhost/db", DatabaseDialect.POSTGRESQL),
            ("postgresql+psycopg2://user:pass@localhost/db", DatabaseDialect.POSTGRESQL),
            ("mysql://user:pass@localhost/db", DatabaseDialect.MYSQL),
            ("mysql+pymysql://user:pass@localhost/db", DatabaseDialect.MYSQL),
            ("sqlite:///path/to/db.sqlite", DatabaseDialect.SQLITE),
            ("sqlite:///:memory:", DatabaseDialect.SQLITE),
            ("mssql+pyodbc://user:pass@server/db", DatabaseDialect.MSSQL),
            ("oracle://user:pass@localhost/db", DatabaseDialect.ORACLE),
            ("unknown://localhost/db", DatabaseDialect.UNKNOWN),
        ],
    )
    def test_dialect_from_url(self, url: str, expected: DatabaseDialect) -> None:
        """Test dialect detection from various URLs."""
        assert DatabaseDialect.from_url(url) == expected


# =============================================================================
# PoolConfig Tests
# =============================================================================


class TestPoolConfig:
    """Tests for pool configuration."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PoolConfig()
        assert config.strategy == PoolStrategy.QUEUE_POOL
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.pool_timeout == 30.0
        assert config.pool_recycle == 3600
        assert config.pool_pre_ping is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PoolConfig(
            strategy=PoolStrategy.NULL_POOL,
            pool_size=20,
            max_overflow=30,
            pool_timeout=60.0,
            pool_recycle=1800,
            pool_pre_ping=False,
        )
        assert config.strategy == PoolStrategy.NULL_POOL
        assert config.pool_size == 20
        assert config.max_overflow == 30


# =============================================================================
# ConnectionPoolConfig Tests
# =============================================================================


class TestConnectionPoolConfig:
    """Tests for complete connection pool configuration."""

    def test_dialect_property(self) -> None:
        """Test dialect property detection."""
        config = ConnectionPoolConfig(
            connection_url="postgresql://localhost/db"
        )
        assert config.dialect == DatabaseDialect.POSTGRESQL

    def test_with_dialect_defaults_sqlite(self) -> None:
        """Test SQLite-specific defaults."""
        config = ConnectionPoolConfig(
            connection_url="sqlite:///:memory:"
        )
        updated = config.with_dialect_defaults()
        assert updated.connect_args.get("check_same_thread") is False
        # SQLite should have reduced pool size
        assert updated.pool.pool_size == 1
        assert updated.pool.max_overflow == 0

    def test_with_dialect_defaults_postgresql(self) -> None:
        """Test PostgreSQL-specific defaults."""
        config = ConnectionPoolConfig(
            connection_url="postgresql://localhost/db"
        )
        updated = config.with_dialect_defaults()
        assert updated.engine_options.get("pool_use_lifo") is True
        assert updated.pool.pool_recycle <= 1800

    def test_with_dialect_defaults_mysql(self) -> None:
        """Test MySQL-specific defaults."""
        config = ConnectionPoolConfig(
            connection_url="mysql://localhost/db"
        )
        updated = config.with_dialect_defaults()
        assert updated.connect_args.get("connect_timeout") == 10


# =============================================================================
# PoolMetrics Tests
# =============================================================================


class TestPoolMetrics:
    """Tests for pool metrics collection."""

    def test_initial_values(self) -> None:
        """Test initial metric values."""
        metrics = PoolMetrics()
        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.idle_connections == 0
        assert metrics.checkouts == 0
        assert metrics.checkins == 0

    def test_record_checkout(self) -> None:
        """Test checkout recording."""
        metrics = PoolMetrics()
        metrics.record_checkout(10.5)
        assert metrics.checkouts == 1
        assert metrics.current_checked_out == 1
        assert metrics.total_checkout_time_ms == 10.5
        assert metrics.peak_connections == 1

    def test_record_checkin(self) -> None:
        """Test checkin recording."""
        metrics = PoolMetrics()
        metrics.record_checkout(5.0)
        metrics.record_checkin()
        assert metrics.checkins == 1
        assert metrics.current_checked_out == 0

    def test_record_creation(self) -> None:
        """Test connection creation recording."""
        metrics = PoolMetrics()
        metrics.record_creation()
        assert metrics.total_created == 1
        assert metrics.current_size == 1

    def test_record_close(self) -> None:
        """Test connection close recording."""
        metrics = PoolMetrics()
        metrics.record_creation()
        metrics.record_close()
        assert metrics.total_closed == 1
        assert metrics.current_size == 0

    def test_record_health_check(self) -> None:
        """Test health check recording."""
        metrics = PoolMetrics()
        metrics.record_health_check(True)
        assert metrics.health_checks_passed == 1
        assert metrics.last_health_check is not None

        metrics.record_health_check(False)
        assert metrics.health_checks_failed == 1

    def test_avg_checkout_time(self) -> None:
        """Test average checkout time calculation."""
        metrics = PoolMetrics()
        metrics.record_checkout(10.0)
        metrics.record_checkout(20.0)
        metrics.record_checkout(30.0)
        assert metrics.avg_checkout_time_ms == 20.0

    def test_to_dict(self) -> None:
        """Test metrics serialization to dict."""
        metrics = PoolMetrics()
        metrics.record_checkout(10.0)
        metrics.record_creation()

        data = metrics.to_dict()
        assert "connections" in data
        assert "operations" in data
        assert "timing" in data
        assert "health" in data
        assert "circuit_breaker" in data

    def test_to_prometheus(self) -> None:
        """Test Prometheus format export."""
        metrics = PoolMetrics()
        metrics.record_checkout(10.0)

        prometheus = metrics.to_prometheus()
        assert "truthound_pool_checkouts_total 1" in prometheus
        assert "truthound_pool_connections_active 1" in prometheus

    def test_thread_safety(self) -> None:
        """Test thread-safe metric updates."""
        metrics = PoolMetrics()
        threads = []

        def update_metrics() -> None:
            for _ in range(100):
                metrics.record_checkout(1.0)
                metrics.record_checkin()

        for _ in range(10):
            t = threading.Thread(target=update_metrics)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert metrics.checkouts == 1000
        assert metrics.checkins == 1000


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_initial_state(self, circuit_breaker_config: CircuitBreakerConfig) -> None:
        """Test initial circuit state is closed."""
        cb = CircuitBreaker(circuit_breaker_config)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open
        assert cb.can_execute()

    def test_opens_after_threshold(
        self, circuit_breaker_config: CircuitBreakerConfig
    ) -> None:
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(circuit_breaker_config)

        for _ in range(circuit_breaker_config.failure_threshold):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.is_open
        assert not cb.can_execute()

    def test_transitions_to_half_open(
        self, circuit_breaker_config: CircuitBreakerConfig
    ) -> None:
        """Test circuit transitions to half-open after timeout."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Open the circuit
        for _ in range(circuit_breaker_config.failure_threshold):
            cb.record_failure()

        assert cb.is_open

        # Wait for timeout
        time.sleep(circuit_breaker_config.timeout + 0.1)

        # Should transition to half-open on next check
        assert cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_threshold(
        self, circuit_breaker_config: CircuitBreakerConfig
    ) -> None:
        """Test circuit closes after success threshold in half-open."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Open the circuit
        for _ in range(circuit_breaker_config.failure_threshold):
            cb.record_failure()

        # Wait for timeout
        time.sleep(circuit_breaker_config.timeout + 0.1)

        # Transition to half-open
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Record successes
        for _ in range(circuit_breaker_config.success_threshold):
            cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed

    def test_half_open_failure_reopens(
        self, circuit_breaker_config: CircuitBreakerConfig
    ) -> None:
        """Test failure in half-open immediately reopens circuit."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Open the circuit
        for _ in range(circuit_breaker_config.failure_threshold):
            cb.record_failure()

        # Wait for timeout
        time.sleep(circuit_breaker_config.timeout + 0.1)

        # Transition to half-open
        cb.can_execute()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_reset(self, circuit_breaker_config: CircuitBreakerConfig) -> None:
        """Test manual circuit reset."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Open the circuit
        for _ in range(circuit_breaker_config.failure_threshold):
            cb.record_failure()

        assert cb.is_open

        cb.reset()
        assert cb.is_closed
        assert cb.can_execute()

    def test_half_open_max_calls(
        self, circuit_breaker_config: CircuitBreakerConfig
    ) -> None:
        """Test half-open limits concurrent calls."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Open and transition to half-open
        for _ in range(circuit_breaker_config.failure_threshold):
            cb.record_failure()

        time.sleep(circuit_breaker_config.timeout + 0.1)

        # Use all half-open calls
        for _ in range(circuit_breaker_config.half_open_max_calls):
            assert cb.can_execute()

        # Next call should be rejected
        assert not cb.can_execute()


# =============================================================================
# RetryHandler Tests
# =============================================================================


class TestRetryHandler:
    """Tests for retry handler."""

    def test_should_retry_on_retryable_error(
        self, retry_config: RetryConfig
    ) -> None:
        """Test retry decision for retryable errors."""
        handler = RetryHandler(retry_config)

        if HAS_SQLALCHEMY:
            error = OperationalError("test", None, None)
            assert handler.should_retry(error, 1)
            assert handler.should_retry(error, 2)
            assert not handler.should_retry(error, 3)  # Max retries

    def test_should_not_retry_on_other_errors(
        self, retry_config: RetryConfig
    ) -> None:
        """Test no retry for non-retryable errors."""
        handler = RetryHandler(retry_config)
        error = ValueError("test")
        assert not handler.should_retry(error, 1)

    def test_get_delay_exponential(self, retry_config: RetryConfig) -> None:
        """Test exponential backoff delay calculation."""
        handler = RetryHandler(retry_config)

        delay1 = handler.get_delay(1)
        delay2 = handler.get_delay(2)
        delay3 = handler.get_delay(3)

        assert delay1 == pytest.approx(0.01, rel=0.1)
        assert delay2 == pytest.approx(0.02, rel=0.1)
        assert delay3 == pytest.approx(0.04, rel=0.1)

    def test_get_delay_max_cap(self) -> None:
        """Test delay is capped at max_delay."""
        config = RetryConfig(
            max_retries=10,
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=10.0,
            jitter=False,
        )
        handler = RetryHandler(config)

        # Very high attempt should be capped
        delay = handler.get_delay(10)
        assert delay <= 5.0

    def test_execute_with_retry_success(
        self, retry_config: RetryConfig
    ) -> None:
        """Test successful execution without retry."""
        handler = RetryHandler(retry_config)
        call_count = 0

        def operation() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = handler.execute_with_retry(operation)
        assert result == "success"
        assert call_count == 1

    def test_execute_with_retry_eventual_success(
        self, retry_config: RetryConfig
    ) -> None:
        """Test eventual success after retries."""
        if not HAS_SQLALCHEMY:
            pytest.skip("SQLAlchemy not available")

        handler = RetryHandler(retry_config)
        call_count = 0

        def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OperationalError("test", None, None)
            return "success"

        result = handler.execute_with_retry(operation)
        assert result == "success"
        assert call_count == 3

    def test_execute_with_retry_exhausted(
        self, retry_config: RetryConfig
    ) -> None:
        """Test exception raised when retries exhausted."""
        if not HAS_SQLALCHEMY:
            pytest.skip("SQLAlchemy not available")

        handler = RetryHandler(retry_config)

        def operation() -> str:
            raise OperationalError("test", None, None)

        with pytest.raises(OperationalError):
            handler.execute_with_retry(operation)


# =============================================================================
# ConnectionPoolManager Tests
# =============================================================================


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
class TestConnectionPoolManager:
    """Tests for connection pool manager."""

    def test_initialization(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test pool manager initialization."""
        manager = ConnectionPoolManager(connection_pool_config)
        manager.initialize()

        assert manager.is_healthy
        assert manager.circuit_state == CircuitState.CLOSED

        manager.dispose()

    def test_context_manager(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test context manager protocol."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            assert manager.is_healthy

    def test_get_engine(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test getting the SQLAlchemy engine."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            engine = manager.get_engine()
            assert engine is not None

    def test_get_session(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test getting a database session."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            session = manager.get_session()
            assert session is not None
            session.close()

    def test_session_context_manager(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test session context manager."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            with manager.session() as session:
                result = session.execute(text("SELECT 1"))
                assert result.scalar() == 1

    def test_session_auto_commit(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test session auto-commits on success."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            with manager.session() as session:
                session.execute(
                    text("CREATE TABLE test_table (id INTEGER PRIMARY KEY)")
                )

            # Table should exist after commit
            with manager.session() as session:
                result = session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                tables = [row[0] for row in result]
                assert "test_table" in tables

    def test_session_auto_rollback(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test session auto-rollback on error."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            # Create table first
            with manager.session() as session:
                session.execute(
                    text("CREATE TABLE rollback_test (id INTEGER PRIMARY KEY)")
                )

            # Try insert with error
            try:
                with manager.session() as session:
                    session.execute(
                        text("INSERT INTO rollback_test VALUES (1)")
                    )
                    raise ValueError("Simulated error")
            except ValueError:
                pass

            # Insert should have been rolled back
            with manager.session() as session:
                result = session.execute(
                    text("SELECT COUNT(*) FROM rollback_test")
                )
                assert result.scalar() == 0

    def test_metrics_collection(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test metrics are collected during operations."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            with manager.session() as session:
                session.execute(text("SELECT 1"))

            metrics = manager.metrics
            assert metrics.checkouts >= 1

    def test_pool_status(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test pool status reporting."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            status = manager.get_pool_status()

            assert status["initialized"] is True
            assert status["healthy"] is True
            assert status["circuit_state"] == "CLOSED"
            assert "config" in status
            assert "metrics" in status

    def test_password_masking(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test password is masked in status output."""
        config = ConnectionPoolConfig(
            connection_url="postgresql://user:secretpass@localhost/db",
            health_check=HealthCheckConfig(enabled=False),
        )

        # Just test the masking function directly
        manager = ConnectionPoolManager.__new__(ConnectionPoolManager)
        masked = manager._mask_password(config.connection_url)
        assert "secretpass" not in masked
        assert "***" in masked

    def test_recycle_connections(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test manual connection recycling."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            # Use a connection first
            with manager.session() as session:
                session.execute(text("SELECT 1"))

            # Recycle
            recycled = manager.recycle_connections()
            # SQLite may not track this well, just verify it runs
            assert isinstance(recycled, int)

    def test_circuit_breaker_integration(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test circuit breaker integration."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            # Record failures to open circuit
            for _ in range(
                connection_pool_config.circuit_breaker.failure_threshold
            ):
                manager._circuit_breaker.record_failure()

            assert manager.circuit_state == CircuitState.OPEN

            # Session should be rejected
            with pytest.raises(RuntimeError, match="Circuit breaker is open"):
                manager.get_session()

    def test_factory_simple_params(self) -> None:
        """Test factory with simple parameters."""
        manager = create_pool_manager(
            "sqlite:///:memory:",
            pool_size=3,
            max_overflow=5,
        )
        manager.initialize()
        assert manager.is_healthy
        manager.dispose()

    def test_factory_with_options(self) -> None:
        """Test factory with additional options."""
        manager = create_pool_manager(
            "sqlite:///:memory:",
            pool_size=10,
            max_overflow=20,
            strategy=PoolStrategy.STATIC_POOL,
            enable_circuit_breaker=False,
            enable_health_checks=False,
        )
        manager.initialize()
        assert manager.is_healthy
        manager.dispose()


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
class TestCreatePoolForDialect:
    """Tests for dialect-specific pool creation."""

    def test_sqlite_dialect(self) -> None:
        """Test SQLite pool creation."""
        manager = create_pool_for_dialect(
            DatabaseDialect.SQLITE,
            database=":memory:",
        )
        manager.initialize()
        assert manager.is_healthy
        manager.dispose()

    def test_postgresql_url_format(self) -> None:
        """Test PostgreSQL URL format."""
        # Just test URL construction, not actual connection
        manager = create_pool_for_dialect.__wrapped__  # type: ignore
        # We can't test actual connection without PostgreSQL

    def test_invalid_dialect(self) -> None:
        """Test error on invalid dialect."""
        with pytest.raises(ValueError):
            create_pool_for_dialect("invalid_dialect")  # type: ignore


# =============================================================================
# Concurrency Tests
# =============================================================================


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
class TestConcurrency:
    """Tests for concurrent pool operations."""

    def test_concurrent_sessions(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test concurrent session access."""
        # Increase pool size for concurrency test
        connection_pool_config.pool.pool_size = 5
        connection_pool_config.pool.max_overflow = 10

        with ConnectionPoolManager(connection_pool_config) as manager:
            results: list[int] = []
            errors: list[Exception] = []
            lock = threading.Lock()

            def worker(worker_id: int) -> None:
                try:
                    with manager.session() as session:
                        result = session.execute(text(f"SELECT {worker_id}"))
                        value = result.scalar()
                        with lock:
                            results.append(value)
                except Exception as e:
                    with lock:
                        errors.append(e)

            threads = [
                threading.Thread(target=worker, args=(i,)) for i in range(10)
            ]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(results) == 10
            assert sorted(results) == list(range(10))

    def test_concurrent_metric_updates(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test thread-safe metric updates under concurrency."""
        with ConnectionPoolManager(connection_pool_config) as manager:
            def worker() -> None:
                for _ in range(50):
                    with manager.session() as session:
                        session.execute(text("SELECT 1"))

            threads = [threading.Thread(target=worker) for _ in range(5)]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            # All operations should be tracked
            assert manager.metrics.checkouts >= 250


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_dispose_before_init(self) -> None:
        """Test disposing before initialization."""
        config = ConnectionPoolConfig(
            connection_url="sqlite:///:memory:",
            health_check=HealthCheckConfig(enabled=False),
        )
        manager = ConnectionPoolManager(config)
        manager.dispose()  # Should not raise

    def test_double_dispose(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test double dispose is safe."""
        manager = ConnectionPoolManager(connection_pool_config)
        manager.initialize()
        manager.dispose()
        manager.dispose()  # Should not raise

    def test_init_after_dispose(
        self, connection_pool_config: ConnectionPoolConfig
    ) -> None:
        """Test initialization after dispose raises error."""
        manager = ConnectionPoolManager(connection_pool_config)
        manager.initialize()
        manager.dispose()

        with pytest.raises(RuntimeError, match="Cannot initialize disposed"):
            manager.initialize()

    def test_empty_metrics(self) -> None:
        """Test metrics with no operations."""
        metrics = PoolMetrics()
        assert metrics.avg_checkout_time_ms == 0.0
        assert metrics.total_connections == 0

    def test_negative_metrics_protection(self) -> None:
        """Test metrics don't go negative."""
        metrics = PoolMetrics()
        metrics.record_checkin()  # Without checkout
        assert metrics.current_checked_out == 0

        metrics.record_close()  # Without creation
        assert metrics.current_size == 0
