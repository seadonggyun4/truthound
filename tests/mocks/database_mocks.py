"""Mock implementations for database backend.

These mocks simulate SQLAlchemy behavior using in-memory storage,
matching the Protocol definitions for type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from contextlib import contextmanager


# =============================================================================
# Mock SQLAlchemy Models
# =============================================================================


@dataclass
class MockValidationResultRow:
    """In-memory representation of a validation result row."""

    id: int
    run_id: str
    data_asset: str
    run_time: datetime
    status: str
    namespace: str
    tags_json: str | None
    data_json: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class MockSQLAlchemyError(Exception):
    """Mock for sqlalchemy.exc.SQLAlchemyError."""

    pass


# =============================================================================
# Mock Query Builder
# =============================================================================


class MockQuery:
    """Mock SQLAlchemy Query object."""

    def __init__(
        self,
        session: "MockSQLSession",
        model_class: type | None = None,
        columns: list[str] | None = None,
    ):
        self._session = session
        self._model_class = model_class
        self._columns = columns
        self._filters: list[tuple[str, str, Any]] = []
        self._order_by: list[tuple[str, bool]] = []  # (column, ascending)
        self._offset_val: int | None = None
        self._limit_val: int | None = None

    def filter(self, *conditions: Any) -> "MockQuery":
        """Add filter conditions."""
        # Parse simple conditions
        for cond in conditions:
            if hasattr(cond, "_mock_filter"):
                self._filters.append(cond._mock_filter)
        return self

    def order_by(self, *columns: Any) -> "MockQuery":
        """Add ordering."""
        for col in columns:
            if hasattr(col, "_mock_order"):
                self._order_by.append(col._mock_order)
        return self

    def offset(self, value: int) -> "MockQuery":
        """Set offset."""
        self._offset_val = value
        return self

    def limit(self, value: int) -> "MockQuery":
        """Set limit."""
        self._limit_val = value
        return self

    def _apply_filters(self, rows: list[MockValidationResultRow]) -> list[MockValidationResultRow]:
        """Apply all filters to rows."""
        result = rows

        for col, op, value in self._filters:
            if op == "==":
                result = [r for r in result if getattr(r, col) == value]
            elif op == ">=":
                result = [r for r in result if getattr(r, col) >= value]
            elif op == "<=":
                result = [r for r in result if getattr(r, col) <= value]

        return result

    def _apply_order(self, rows: list[MockValidationResultRow]) -> list[MockValidationResultRow]:
        """Apply ordering."""
        for col, ascending in reversed(self._order_by):
            rows = sorted(rows, key=lambda r: getattr(r, col), reverse=not ascending)
        return rows

    def _apply_pagination(self, rows: list[MockValidationResultRow]) -> list[MockValidationResultRow]:
        """Apply offset and limit."""
        if self._offset_val:
            rows = rows[self._offset_val:]
        if self._limit_val:
            rows = rows[:self._limit_val]
        return rows

    def all(self) -> list[Any]:
        """Execute query and return all results."""
        rows = list(self._session._db._rows.values())
        rows = self._apply_filters(rows)
        rows = self._apply_order(rows)
        rows = self._apply_pagination(rows)

        if self._columns:
            # Return tuples of column values
            return [tuple(getattr(r, c) for c in self._columns) for r in rows]
        return rows

    def first(self) -> Any | None:
        """Return first result or None."""
        results = self.all()
        return results[0] if results else None

    def count(self) -> int:
        """Count matching rows."""
        rows = list(self._session._db._rows.values())
        rows = self._apply_filters(rows)
        return len(rows)

    def delete(self) -> int:
        """Delete matching rows and return count."""
        rows = list(self._session._db._rows.values())
        rows = self._apply_filters(rows)

        deleted = 0
        for row in rows:
            if row.run_id in self._session._db._rows:
                del self._session._db._rows[row.run_id]
                deleted += 1

        return deleted


# =============================================================================
# Mock Column for Filter Building
# =============================================================================


class MockColumn:
    """Mock SQLAlchemy Column for building filter expressions."""

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: Any) -> "MockFilterExpr":  # type: ignore[override]
        return MockFilterExpr(self.name, "==", other)

    def __ge__(self, other: Any) -> "MockFilterExpr":
        return MockFilterExpr(self.name, ">=", other)

    def __le__(self, other: Any) -> "MockFilterExpr":
        return MockFilterExpr(self.name, "<=", other)

    def asc(self) -> "MockOrderExpr":
        return MockOrderExpr(self.name, True)

    def desc(self) -> "MockOrderExpr":
        return MockOrderExpr(self.name, False)


class MockFilterExpr:
    """Mock filter expression."""

    def __init__(self, column: str, op: str, value: Any):
        self._mock_filter = (column, op, value)


class MockOrderExpr:
    """Mock order expression."""

    def __init__(self, column: str, ascending: bool):
        self._mock_order = (column, ascending)


# =============================================================================
# Mock Model Class
# =============================================================================


class MockValidationResultModel:
    """Mock SQLAlchemy model for ValidationResult."""

    # Class-level column accessors
    run_id = MockColumn("run_id")
    data_asset = MockColumn("data_asset")
    run_time = MockColumn("run_time")
    status = MockColumn("status")
    namespace = MockColumn("namespace")

    def __init__(self, **kwargs: Any):
        self.id: int | None = None
        self.run_id: str = kwargs.get("run_id", "")
        self.data_asset: str = kwargs.get("data_asset", "")
        self.run_time: datetime = kwargs.get("run_time", datetime.utcnow())
        self.status: str = kwargs.get("status", "")
        self.namespace: str = kwargs.get("namespace", "default")
        self.tags_json: str | None = kwargs.get("tags_json")
        self.data_json: str = kwargs.get("data_json", "{}")
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()


# =============================================================================
# Mock Session
# =============================================================================


class MockSQLSession:
    """Mock SQLAlchemy Session."""

    def __init__(self, db: "MockDatabase"):
        self._db = db
        self._pending: list[MockValidationResultRow] = []
        self._closed = False

    def query(self, *entities: Any) -> MockQuery:
        """Create a query."""
        if entities and hasattr(entities[0], "name"):
            # Querying specific columns
            columns = [e.name for e in entities]
            return MockQuery(self, columns=columns)
        return MockQuery(self)

    def add(self, instance: Any) -> None:
        """Add an object to the session."""
        if isinstance(instance, MockValidationResultModel):
            row = MockValidationResultRow(
                id=len(self._db._rows) + 1,
                run_id=instance.run_id,
                data_asset=instance.data_asset,
                run_time=instance.run_time,
                status=instance.status,
                namespace=instance.namespace,
                tags_json=instance.tags_json,
                data_json=instance.data_json,
            )
            self._pending.append(row)

    def commit(self) -> None:
        """Commit the transaction."""
        for row in self._pending:
            self._db._rows[row.run_id] = row
        self._pending.clear()

    def rollback(self) -> None:
        """Rollback the transaction."""
        self._pending.clear()

    def close(self) -> None:
        """Close the session."""
        self._closed = True

    def __enter__(self) -> "MockSQLSession":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# =============================================================================
# Mock Engine and Session Factory
# =============================================================================


class MockDatabase:
    """In-memory database storage."""

    def __init__(self):
        self._rows: dict[str, MockValidationResultRow] = {}
        self._connected = True

    def clear(self) -> None:
        """Clear all data."""
        self._rows.clear()


class MockSQLConnection:
    """Mock SQLAlchemy Connection."""

    def __init__(self, db: MockDatabase):
        self._db = db

    def execute(self, statement: Any) -> Any:
        """Execute a statement."""
        # For simple SELECT 1 test
        return MockResult([(1,)])

    def __enter__(self) -> "MockSQLConnection":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class MockResult:
    """Mock query result."""

    def __init__(self, rows: list[tuple[Any, ...]]):
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None


class MockSQLEngine:
    """Mock SQLAlchemy Engine."""

    def __init__(self, db: MockDatabase):
        self._db = db
        self._disposed = False

    def connect(self) -> MockSQLConnection:
        """Create a connection."""
        if self._disposed:
            raise MockSQLAlchemyError("Engine disposed")
        return MockSQLConnection(self._db)

    def dispose(self) -> None:
        """Dispose the engine."""
        self._disposed = True


class MockSessionFactory:
    """Mock SQLAlchemy session factory."""

    def __init__(self, db: MockDatabase):
        self._db = db

    def __call__(self) -> MockSQLSession:
        """Create a new session."""
        return MockSQLSession(self._db)


# =============================================================================
# Factory Function
# =============================================================================


def create_mock_database() -> tuple[MockSQLEngine, MockSessionFactory, MockDatabase]:
    """Create a complete mock database setup.

    Returns:
        Tuple of (engine, session_factory, database).
    """
    db = MockDatabase()
    engine = MockSQLEngine(db)
    session_factory = MockSessionFactory(db)
    return engine, session_factory, db
