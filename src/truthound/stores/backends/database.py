"""SQL Database store backend.

This module provides a store implementation that persists data to SQL databases.
Requires the sqlalchemy package.

Features:
- Enterprise-grade connection pooling with multiple strategies
- Circuit breaker pattern for fault tolerance
- Automatic retry with exponential backoff
- Comprehensive metrics and health monitoring
- Database-specific optimizations

Install with: pip install truthound[database]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

# Lazy import to avoid ImportError when sqlalchemy is not installed
try:
    from sqlalchemy import (
        Column,
        DateTime,
        Index,
        Integer,
        String,
        Text,
        UniqueConstraint,
        create_engine,
        text,
    )
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session, declarative_base, sessionmaker

    HAS_SQLALCHEMY = True
    Base = declarative_base()
except ImportError:
    HAS_SQLALCHEMY = False
    Base = object  # type: ignore
    SQLAlchemyError = Exception  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Callable

    from truthound.stores.backends._protocols import (
        SessionFactoryProtocol,
        SQLEngineProtocol,
    )
    from truthound.stores.backends.connection_pool import (
        ConnectionPoolManager,
        PoolMetrics,
    )

from truthound.stores.base import (
    StoreConfig,
    StoreConnectionError,
    StoreNotFoundError,
    StoreQuery,
    StoreReadError,
    StoreWriteError,
    ValidationStore,
)
from truthound.stores.results import ValidationResult


def _require_sqlalchemy() -> None:
    """Check if sqlalchemy is available."""
    if not HAS_SQLALCHEMY:
        raise ImportError(
            "sqlalchemy is required for DatabaseStore. "
            "Install with: pip install truthound[database]"
        )


# Define table model only if sqlalchemy is available
if HAS_SQLALCHEMY:

    class ValidationResultModel(Base):  # type: ignore
        """SQLAlchemy model for validation results."""

        __tablename__ = "validation_results"

        id = Column(Integer, primary_key=True, autoincrement=True)
        run_id = Column(String(255), nullable=False, index=True)
        data_asset = Column(String(500), nullable=False, index=True)
        run_time = Column(DateTime, nullable=False, index=True)
        status = Column(String(50), nullable=False, index=True)
        namespace = Column(String(255), nullable=False, default="default", index=True)
        tags_json = Column(Text, nullable=True)
        data_json = Column(Text, nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

        # Create composite indexes
        __table_args__ = (
            UniqueConstraint("namespace", "run_id", name="uq_validation_results_namespace_run"),
            Index("ix_namespace_data_asset", "namespace", "data_asset"),
            Index("ix_namespace_run_time", "namespace", "run_time"),
        )


def _table_model(prefix):
    """Map a fixed prefixed table without mutating the default shared model."""
    import re

    if not isinstance(prefix, str) or (
        prefix and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,23}", prefix) is None
    ):
        raise ValueError(
            "Database table prefix must be empty or a simple identifier of at most 24 characters"
        )
    if not prefix:
        return ValidationResultModel, Base.metadata
    from sqlalchemy import MetaData
    from sqlalchemy.orm import registry

    metadata = MetaData()
    table = ValidationResultModel.__table__.to_metadata(
        metadata, name=prefix + "validation_results"
    )
    for index in table.indexes:
        if index.name:
            index.name = prefix + index.name
    for constraint in table.constraints:
        if constraint.name:
            constraint.name = prefix + constraint.name
    mapped = type("PrefixedValidationResult", (), {})
    registry(metadata=metadata).map_imperatively(mapped, table)
    return mapped, metadata


@dataclass
class PoolingConfig:
    """Advanced connection pooling configuration.

    Attributes:
        pool_size: Number of connections to maintain in pool.
        max_overflow: Maximum overflow connections beyond pool_size.
        pool_timeout: Seconds to wait for available connection.
        pool_recycle: Seconds before a connection is recycled (-1 = never).
        pool_pre_ping: Whether to test connections before use.
        enable_circuit_breaker: Enable circuit breaker for fault tolerance.
        enable_health_checks: Enable background health monitoring.
        enable_retry: Enable automatic retry on transient errors.
        max_retries: Maximum retry attempts for transient errors.
        retry_base_delay: Base delay between retries in seconds.
    """

    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    enable_circuit_breaker: bool = True
    enable_health_checks: bool = True
    enable_retry: bool = True
    max_retries: int = 3
    retry_base_delay: float = 0.1


@dataclass
class DatabaseConfig(StoreConfig):
    """Configuration for database store.

    Attributes:
        connection_url: SQLAlchemy connection URL.
        table_prefix: Prefix for table names.
        pool_size: Connection pool size (deprecated, use pooling.pool_size).
        max_overflow: Maximum overflow connections (deprecated, use pooling.max_overflow).
        echo: Whether to echo SQL statements.
        create_tables: Whether to create tables on initialization.
        pooling: Advanced pooling configuration.
        use_pool_manager: Whether to use the enterprise ConnectionPoolManager.
    """

    connection_url: str = "sqlite:///.truthound/store.db"
    table_prefix: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    create_tables: bool = True
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    use_pool_manager: bool = True  # Use enterprise pool manager by default


class DatabaseStore(ValidationStore["DatabaseConfig"]):
    """SQL Database validation store.

    Stores validation results in a SQL database using SQLAlchemy.
    Supports PostgreSQL, MySQL, SQLite, and other SQLAlchemy-compatible databases.

    Features:
        - Enterprise-grade connection pooling with configurable strategies
        - Circuit breaker pattern for fault tolerance
        - Automatic retry with exponential backoff
        - Health monitoring and metrics collection
        - Database-specific optimizations (PostgreSQL, MySQL, SQLite, etc.)

    Example:
        >>> # SQLite (default)
        >>> store = DatabaseStore(
        ...     connection_url="sqlite:///validations.db"
        ... )
        >>>
        >>> # PostgreSQL with custom pooling
        >>> store = DatabaseStore(
        ...     connection_url="postgresql://user:pass@localhost/validations",
        ...     pooling=PoolingConfig(
        ...         pool_size=10,
        ...         max_overflow=20,
        ...         enable_circuit_breaker=True,
        ...     )
        ... )
        >>>
        >>> result = ValidationResult.from_report(report, "customers.csv")
        >>> run_id = store.save(result)
        >>>
        >>> # Access pool metrics
        >>> print(store.pool_metrics.to_dict())
    """

    def __init__(
        self,
        connection_url: str = "sqlite:///.truthound/store.db",
        namespace: str = "default",
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
        create_tables: bool = True,
        pooling: PoolingConfig | None = None,
        use_pool_manager: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the database store.

        Args:
            connection_url: SQLAlchemy connection URL.
            namespace: Namespace for organizing data.
            pool_size: Connection pool size.
            max_overflow: Maximum overflow connections.
            echo: Whether to echo SQL statements.
            create_tables: Whether to create tables on initialization.
            pooling: Advanced pooling configuration (overrides pool_size/max_overflow).
            use_pool_manager: Whether to use enterprise ConnectionPoolManager.
            **kwargs: Additional configuration options.

        Note:
            Dependency check is handled by the factory. Direct instantiation
            requires sqlalchemy to be installed.
        """
        _require_sqlalchemy()
        # Build pooling config from individual params if not provided
        if pooling is None:
            pooling = PoolingConfig(
                pool_size=pool_size,
                max_overflow=max_overflow,
            )

        config = DatabaseConfig(
            connection_url=connection_url,
            namespace=namespace,
            pool_size=pooling.pool_size,
            max_overflow=pooling.max_overflow,
            echo=echo,
            create_tables=create_tables,
            pooling=pooling,
            use_pool_manager=use_pool_manager,
            **{k: v for k, v in kwargs.items() if hasattr(DatabaseConfig, k)},
        )
        super().__init__(config)
        self._model, self._metadata = _table_model(config.table_prefix)

        # Connection management
        self._engine: SQLEngineProtocol | None = None
        self._session_factory: SessionFactoryProtocol | None = None
        self._pool_manager: ConnectionPoolManager | None = None

    @classmethod
    def _default_config(cls) -> DatabaseConfig:
        """Create default configuration."""
        return DatabaseConfig()

    @property
    def pool_metrics(self) -> PoolMetrics | None:
        """Get connection pool metrics.

        Returns:
            Pool metrics if using pool manager, None otherwise.
        """
        if self._pool_manager:
            return self._pool_manager.metrics
        return None

    @property
    def is_healthy(self) -> bool:
        """Check if database connection is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        if self._pool_manager:
            return self._pool_manager.is_healthy
        return self._engine is not None

    def get_pool_status(self) -> dict[str, Any]:
        """Get comprehensive pool status.

        Returns:
            Dictionary with pool status information.
        """
        if self._pool_manager:
            return self._pool_manager.get_pool_status()
        return {
            "initialized": self._initialized,
            "using_pool_manager": False,
            "config": {
                "connection_url": self._mask_password(self._config.connection_url),
                "pool_size": self._config.pool_size,
                "max_overflow": self._config.max_overflow,
            },
        }

    def _mask_password(self, url: str) -> str:
        """Mask password in connection URL."""
        import re

        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)

    def _do_initialize(self) -> None:
        """Initialize the database connection and tables."""
        try:
            if self._config.use_pool_manager:
                self._initialize_with_pool_manager()
            else:
                self._initialize_legacy()
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Database initialization and owned cleanup failed", [original, cleanup]
                ) from None
            raise

    def _initialize_with_pool_manager(self) -> None:
        """Initialize using enterprise ConnectionPoolManager."""
        try:
            from truthound.stores.backends.connection_pool import (
                CircuitBreakerConfig,
                ConnectionPoolConfig,
                ConnectionPoolManager,
                HealthCheckConfig,
                PoolConfig,
                PoolStrategy,
                RetryConfig,
            )

            pooling = self._config.pooling

            # Build pool configuration
            pool_config = PoolConfig(
                strategy=PoolStrategy.QUEUE_POOL,
                pool_size=pooling.pool_size,
                max_overflow=pooling.max_overflow,
                pool_timeout=pooling.pool_timeout,
                pool_recycle=pooling.pool_recycle,
                pool_pre_ping=pooling.pool_pre_ping,
            )

            # Build retry configuration
            retry_config = RetryConfig(
                max_retries=pooling.max_retries if pooling.enable_retry else 0,
                base_delay=pooling.retry_base_delay,
            )

            # Build circuit breaker configuration
            circuit_config = CircuitBreakerConfig()
            if not pooling.enable_circuit_breaker:
                circuit_config.failure_threshold = 999999  # Effectively disabled

            # Build health check configuration
            health_config = HealthCheckConfig(enabled=pooling.enable_health_checks)

            # Create pool manager configuration
            config = ConnectionPoolConfig(
                connection_url=self._config.connection_url,
                pool=pool_config,
                retry=retry_config,
                circuit_breaker=circuit_config,
                health_check=health_config,
                echo=self._config.echo,
            )

            # Create and initialize pool manager
            self._pool_manager = ConnectionPoolManager(config)
            self._pool_manager.initialize()

            # Get engine for table creation
            self._engine = self._pool_manager.get_engine()

            # Create tables if needed
            if self._config.create_tables:
                self._metadata.create_all(self._engine)

        except ImportError:
            # Fall back to legacy initialization
            self._initialize_legacy()
        except SQLAlchemyError:
            raise StoreConnectionError(
                "Database", "Native database initialization failed"
            ) from None

    def _initialize_legacy(self) -> None:
        """Legacy initialization without pool manager."""
        try:
            # Create engine
            connect_args = {}

            # SQLite specific settings
            if self._config.connection_url.startswith("sqlite"):
                connect_args["check_same_thread"] = False

            self._engine = create_engine(
                self._config.connection_url,
                pool_size=self._config.pool_size,
                max_overflow=self._config.max_overflow,
                echo=self._config.echo,
                connect_args=connect_args,
            )

            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine)

            # Create tables if needed
            if self._config.create_tables:
                self._metadata.create_all(self._engine)

        except SQLAlchemyError:
            raise StoreConnectionError(
                "Database", "Native database initialization failed"
            ) from None

    def _get_session(self) -> Session:
        """Get a new database session."""
        if self._pool_manager:
            return self._pool_manager.get_session()
        return self._session_factory()

    def save(self, item: ValidationResult) -> str:
        """Save a validation result to the database.

        Args:
            item: The validation result to save.

        Returns:
            The run ID of the saved result.

        Raises:
            StoreWriteError: If saving fails.
        """
        self.initialize()

        try:
            with self._get_session() as session:
                # Check if exists and update, otherwise insert
                existing = (
                    session.query(self._model)
                    .filter(
                        self._model.run_id == item.run_id,
                        self._model.namespace == self._config.namespace,
                    )
                    .first()
                )

                data_json = json.dumps(item.to_dict(), default=str)
                tags_json = json.dumps(item.tags, default=str)

                if existing:
                    existing.data_asset = item.data_asset
                    existing.run_time = item.run_time
                    existing.status = item.status.value
                    existing.namespace = self._config.namespace
                    existing.tags_json = tags_json
                    existing.data_json = data_json
                else:
                    model = self._model(
                        run_id=item.run_id,
                        data_asset=item.data_asset,
                        run_time=item.run_time,
                        status=item.status.value,
                        namespace=self._config.namespace,
                        tags_json=tags_json,
                        data_json=data_json,
                    )
                    session.add(model)

                session.commit()
                return item.run_id

        except SQLAlchemyError:
            raise StoreWriteError("Failed to save to database") from None

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result from the database.

        Args:
            item_id: The run ID of the result to retrieve.

        Returns:
            The validation result.

        Raises:
            StoreNotFoundError: If the result doesn't exist.
            StoreReadError: If reading fails.
        """
        self.initialize()

        try:
            with self._get_session() as session:
                model = (
                    session.query(self._model)
                    .filter(
                        self._model.run_id == item_id,
                        self._model.namespace == self._config.namespace,
                    )
                    .first()
                )

                if not model:
                    raise StoreNotFoundError("ValidationResult", item_id)

                data = json.loads(model.data_json)
                return ValidationResult.from_dict(data)

        except StoreNotFoundError:
            raise
        except (ValueError, KeyError, TypeError, SQLAlchemyError):
            raise StoreReadError("Failed to read from database") from None

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists.

        Args:
            item_id: The run ID to check.

        Returns:
            True if the result exists.
        """
        self.initialize()

        try:
            with self._get_session() as session:
                count = (
                    session.query(self._model)
                    .filter(
                        self._model.run_id == item_id,
                        self._model.namespace == self._config.namespace,
                    )
                    .count()
                )
                return count > 0

        except SQLAlchemyError:
            raise StoreReadError("Failed to check database record existence") from None

    def delete(self, item_id: str) -> bool:
        """Delete a validation result from the database.

        Args:
            item_id: The run ID of the result to delete.

        Returns:
            True if the result was deleted, False if it didn't exist.

        Raises:
            StoreWriteError: If deletion fails.
        """
        self.initialize()

        try:
            with self._get_session() as session:
                deleted = (
                    session.query(self._model)
                    .filter(
                        self._model.run_id == item_id,
                        self._model.namespace == self._config.namespace,
                    )
                    .delete()
                )
                session.commit()
                return deleted > 0

        except SQLAlchemyError:
            raise StoreWriteError("Failed to delete from database") from None

    def _model_query(self, session, query):
        query = query or StoreQuery()
        if query.order_by not in {"run_time", "data_asset", "status"}:
            raise ValueError("Unsupported database store ordering")
        if type(query.offset) is not int or query.offset < 0:
            raise ValueError("Database store offset must be a non-negative integer")
        if query.limit is not None and (type(query.limit) is not int or query.limit < 0):
            raise ValueError("Database store limit must be a non-negative integer")
        rows = session.query(self._model).filter(self._model.namespace == self._config.namespace)
        if query.data_asset:
            rows = rows.filter(self._model.data_asset == query.data_asset)
        if query.status:
            rows = rows.filter(self._model.status == query.status)
        if query.start_time:
            rows = rows.filter(self._model.run_time >= query.start_time)
        if query.end_time:
            rows = rows.filter(self._model.run_time <= query.end_time)
        column = getattr(self._model, query.order_by)
        return rows.order_by(
            column.asc() if query.ascending else column.desc(), self._model.run_id.asc()
        )

    def _matching_models(self, session, query):
        query = query or StoreQuery()
        rows = self._model_query(session, query)
        if query.limit == 0:
            return
        if not query.tags:
            if query.offset:
                rows = rows.offset(query.offset)
            if query.limit is not None:
                rows = rows.limit(query.limit)
            yield from rows.yield_per(1000)
            return
        # Portable tag filtering must happen before pagination. Stream native
        # rows instead of materializing the whole database for this fallback.
        skipped = returned = 0
        for row in rows.yield_per(1000):
            try:
                tags = json.loads(row.tags_json or "{}")
                if not isinstance(tags, dict):
                    raise ValueError("Invalid stored tags")
            except (ValueError, TypeError):
                raise StoreReadError("Database record tags are corrupt") from None
            if not all(tags.get(key) == value for key, value in query.tags.items()):
                continue
            if skipped < query.offset:
                skipped += 1
                continue
            yield row
            returned += 1
            if query.limit is not None and returned >= query.limit:
                return

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List namespace-local IDs, applying filters before pagination."""
        self.initialize()
        try:
            with self._get_session() as session:
                return [row.run_id for row in self._matching_models(session, query)]
        except SQLAlchemyError:
            raise StoreReadError("Failed to list database records") from None

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Read matching native results; corrupt records are explicit failures."""
        self.initialize()
        try:
            with self._get_session() as session:
                return [
                    ValidationResult.from_dict(json.loads(row.data_json))
                    for row in self._matching_models(session, query)
                ]
        except (SQLAlchemyError, ValueError, KeyError, TypeError):
            raise StoreReadError("Failed to query database records") from None

    def count(self, query: StoreQuery | None = None) -> int:
        """Count matching records in this namespace, without page boundaries."""
        self.initialize()
        query = replace(query, limit=None, offset=0) if query is not None else StoreQuery()
        try:
            with self._get_session() as session:
                if query.tags:
                    return sum(1 for _ in self._matching_models(session, query))
                return self._model_query(session, query).count()
        except SQLAlchemyError:
            raise StoreReadError("Failed to count database records") from None

    def close(self) -> None:
        """Close the database connection and release all resources."""
        if self._pool_manager:
            self._pool_manager.dispose()
            self._pool_manager = None
            self._engine = None
        elif self._engine:
            self._engine.dispose()
            self._engine = None
        self._session_factory = None
        self._initialized = False

    def recycle_connections(self) -> int:
        """Manually recycle all pool connections.

        Useful for forcing reconnection after configuration changes
        or when connections have become stale.

        Returns:
            Number of connections recycled.
        """
        if self._pool_manager:
            return self._pool_manager.recycle_connections()
        return 0

    def execute_with_retry(
        self,
        operation: Callable[[Session], Any],
    ) -> Any:
        """Execute database operation with automatic retry.

        Uses exponential backoff for transient errors.

        Args:
            operation: Callable that takes a session and returns a result.

        Returns:
            Result of the operation.

        Raises:
            SQLAlchemyError: If all retries exhausted.

        Example:
            >>> def insert_data(session):
            ...     session.execute(text("INSERT INTO ..."))
            ...     return session.execute(text("SELECT ...")).scalar()
            >>> result = store.execute_with_retry(insert_data)
        """
        if self._pool_manager:
            return self._pool_manager.execute_with_retry(operation)

        # Legacy fallback - no retry
        self.initialize()
        with self._get_session() as session:
            result = operation(session)
            session.commit()
            return result
