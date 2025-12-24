"""SQL Database store backend.

This module provides a store implementation that persists data to SQL databases.
Requires the sqlalchemy package.

Install with: pip install truthound[database]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
        create_engine,
        text,
    )
    from sqlalchemy.orm import Session, declarative_base, sessionmaker
    from sqlalchemy.exc import SQLAlchemyError

    HAS_SQLALCHEMY = True
    Base = declarative_base()
except ImportError:
    HAS_SQLALCHEMY = False
    Base = object  # type: ignore
    SQLAlchemyError = Exception  # type: ignore

if TYPE_CHECKING:
    from truthound.stores.backends._protocols import (
        SessionFactoryProtocol,
        SQLEngineProtocol,
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
        run_id = Column(String(255), unique=True, nullable=False, index=True)
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
            Index("ix_namespace_data_asset", "namespace", "data_asset"),
            Index("ix_namespace_run_time", "namespace", "run_time"),
        )


@dataclass
class DatabaseConfig(StoreConfig):
    """Configuration for database store.

    Attributes:
        connection_url: SQLAlchemy connection URL.
        table_prefix: Prefix for table names.
        pool_size: Connection pool size.
        max_overflow: Maximum overflow connections.
        echo: Whether to echo SQL statements.
        create_tables: Whether to create tables on initialization.
    """

    connection_url: str = "sqlite:///.truthound/store.db"
    table_prefix: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    create_tables: bool = True


class DatabaseStore(ValidationStore["DatabaseConfig"]):
    """SQL Database validation store.

    Stores validation results in a SQL database using SQLAlchemy.
    Supports PostgreSQL, MySQL, SQLite, and other SQLAlchemy-compatible databases.

    Example:
        >>> # SQLite (default)
        >>> store = DatabaseStore(
        ...     connection_url="sqlite:///validations.db"
        ... )
        >>>
        >>> # PostgreSQL
        >>> store = DatabaseStore(
        ...     connection_url="postgresql://user:pass@localhost/validations"
        ... )
        >>>
        >>> result = ValidationResult.from_report(report, "customers.csv")
        >>> run_id = store.save(result)
    """

    def __init__(
        self,
        connection_url: str = "sqlite:///.truthound/store.db",
        namespace: str = "default",
        pool_size: int = 5,
        echo: bool = False,
        create_tables: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the database store.

        Args:
            connection_url: SQLAlchemy connection URL.
            namespace: Namespace for organizing data.
            pool_size: Connection pool size.
            echo: Whether to echo SQL statements.
            create_tables: Whether to create tables on initialization.
            **kwargs: Additional configuration options.

        Note:
            Dependency check is handled by the factory. Direct instantiation
            requires sqlalchemy to be installed.
        """
        config = DatabaseConfig(
            connection_url=connection_url,
            namespace=namespace,
            pool_size=pool_size,
            echo=echo,
            create_tables=create_tables,
            **{k: v for k, v in kwargs.items() if hasattr(DatabaseConfig, k)},
        )
        super().__init__(config)
        self._engine: SQLEngineProtocol | None = None
        self._session_factory: SessionFactoryProtocol | None = None

    @classmethod
    def _default_config(cls) -> "DatabaseConfig":
        """Create default configuration."""
        return DatabaseConfig()

    def _do_initialize(self) -> None:
        """Initialize the database connection and tables."""
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
                Base.metadata.create_all(self._engine)

        except SQLAlchemyError as e:
            raise StoreConnectionError("Database", str(e))

    def _get_session(self) -> "Session":
        """Get a new database session."""
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
                    session.query(ValidationResultModel)
                    .filter(ValidationResultModel.run_id == item.run_id)
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
                    model = ValidationResultModel(
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

        except SQLAlchemyError as e:
            raise StoreWriteError(f"Failed to save to database: {e}")

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
                    session.query(ValidationResultModel)
                    .filter(
                        ValidationResultModel.run_id == item_id,
                        ValidationResultModel.namespace == self._config.namespace,
                    )
                    .first()
                )

                if not model:
                    raise StoreNotFoundError("ValidationResult", item_id)

                data = json.loads(model.data_json)
                return ValidationResult.from_dict(data)

        except StoreNotFoundError:
            raise
        except (json.JSONDecodeError, SQLAlchemyError) as e:
            raise StoreReadError(f"Failed to read from database: {e}")

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
                    session.query(ValidationResultModel)
                    .filter(
                        ValidationResultModel.run_id == item_id,
                        ValidationResultModel.namespace == self._config.namespace,
                    )
                    .count()
                )
                return count > 0

        except SQLAlchemyError:
            return False

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
                    session.query(ValidationResultModel)
                    .filter(
                        ValidationResultModel.run_id == item_id,
                        ValidationResultModel.namespace == self._config.namespace,
                    )
                    .delete()
                )
                session.commit()
                return deleted > 0

        except SQLAlchemyError as e:
            raise StoreWriteError(f"Failed to delete from database: {e}")

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query.

        Args:
            query: Optional query to filter results.

        Returns:
            List of matching run IDs.
        """
        self.initialize()

        try:
            with self._get_session() as session:
                q = session.query(ValidationResultModel.run_id).filter(
                    ValidationResultModel.namespace == self._config.namespace
                )

                if query:
                    if query.data_asset:
                        q = q.filter(ValidationResultModel.data_asset == query.data_asset)
                    if query.status:
                        q = q.filter(ValidationResultModel.status == query.status)
                    if query.start_time:
                        q = q.filter(ValidationResultModel.run_time >= query.start_time)
                    if query.end_time:
                        q = q.filter(ValidationResultModel.run_time <= query.end_time)

                    # Sorting
                    order_column = getattr(
                        ValidationResultModel,
                        query.order_by if query.order_by != "run_time" else "run_time",
                        ValidationResultModel.run_time,
                    )
                    if query.ascending:
                        q = q.order_by(order_column.asc())
                    else:
                        q = q.order_by(order_column.desc())

                    # Pagination
                    if query.offset:
                        q = q.offset(query.offset)
                    if query.limit:
                        q = q.limit(query.limit)
                else:
                    q = q.order_by(ValidationResultModel.run_time.desc())

                return [row[0] for row in q.all()]

        except SQLAlchemyError:
            return []

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results from the database.

        Args:
            query: Query parameters for filtering.

        Returns:
            List of matching validation results.
        """
        self.initialize()

        try:
            with self._get_session() as session:
                q = session.query(ValidationResultModel).filter(
                    ValidationResultModel.namespace == self._config.namespace
                )

                if query.data_asset:
                    q = q.filter(ValidationResultModel.data_asset == query.data_asset)
                if query.status:
                    q = q.filter(ValidationResultModel.status == query.status)
                if query.start_time:
                    q = q.filter(ValidationResultModel.run_time >= query.start_time)
                if query.end_time:
                    q = q.filter(ValidationResultModel.run_time <= query.end_time)

                # Sorting
                if query.ascending:
                    q = q.order_by(ValidationResultModel.run_time.asc())
                else:
                    q = q.order_by(ValidationResultModel.run_time.desc())

                # Pagination
                if query.offset:
                    q = q.offset(query.offset)
                if query.limit:
                    q = q.limit(query.limit)

                results: list[ValidationResult] = []
                for model in q.all():
                    try:
                        data = json.loads(model.data_json)
                        results.append(ValidationResult.from_dict(data))
                    except (json.JSONDecodeError, KeyError):
                        continue

                return results

        except SQLAlchemyError:
            return []

    def count(self, query: StoreQuery | None = None) -> int:
        """Count validation results matching the query.

        Args:
            query: Optional query to filter.

        Returns:
            Number of matching results.
        """
        self.initialize()

        try:
            with self._get_session() as session:
                q = session.query(ValidationResultModel).filter(
                    ValidationResultModel.namespace == self._config.namespace
                )

                if query:
                    if query.data_asset:
                        q = q.filter(ValidationResultModel.data_asset == query.data_asset)
                    if query.status:
                        q = q.filter(ValidationResultModel.status == query.status)
                    if query.start_time:
                        q = q.filter(ValidationResultModel.run_time >= query.start_time)
                    if query.end_time:
                        q = q.filter(ValidationResultModel.run_time <= query.end_time)

                return q.count()

        except SQLAlchemyError:
            return 0

    def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
