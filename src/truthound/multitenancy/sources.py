"""Bound SQL reads with physical namespace verification on the reading connection."""

import re
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from truthound.datasources.base import DataSourceSizeError
from truthound.datasources.sql.sqlite import SQLiteDataSource
from truthound.multitenancy.core import IsolationLevel, TenantIsolationError


class TenantSQLSource:
    """Borrow a factory-owned SQLite source for one fixed native tenant.

    Only ordinary tables in a file-backed database are supported. Query, view,
    virtual-table and unverified remote-provider sources fail closed. The factory
    owns connection cleanup. Reads are bounded by the native source limit.
    """

    def __init__(self, source, *, tenant_id: str, namespace_database: str | None = None):
        if not isinstance(source, SQLiteDataSource) or source.is_query_mode:
            raise TenantIsolationError("A verified table source is required")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be nonempty")
        self.source = source
        self.tenant_id = tenant_id
        if namespace_database is not None and not Path(namespace_database).is_absolute():
            raise ValueError("namespace_database must be an absolute path")
        self.namespace_database = (
            Path(namespace_database).resolve() if namespace_database is not None else None
        )

    def read(self, manager, tenant_id: str):
        import polars as pl

        if tenant_id != self.tenant_id:
            raise TenantIsolationError("Source tenant does not match")
        source = self.source
        schema = source._config.schema_name or "main"
        table = source._table
        if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                   for value in (schema, table)):
            raise TenantIsolationError("Simple schema and table identifiers are required")
        quoted_schema = source._quote_identifier(schema)
        quoted_table = source._quote_identifier(table)

        def verify(conn):
            tenant = manager.require(tenant_id)
            strategy = manager.isolation_manager.get_strategy(tenant)
            strategy.validate_isolation(tenant)
            if tenant.isolation_level not in {IsolationLevel.SCHEMA, IsolationLevel.DATABASE}:
                raise TenantIsolationError("Physical isolation is required")
            expected = strategy.get_namespace(tenant)
            cursor = conn.cursor()
            try:
                cursor.execute("PRAGMA database_list")
                databases = {row[1]: row[2] for row in cursor.fetchall()}
                actual = databases.get(schema)
                if not actual:
                    raise TenantIsolationError("A file-backed namespace is required")
                if tenant.isolation_level == IsolationLevel.DATABASE:
                    if not Path(expected).is_absolute() or Path(actual).resolve() != Path(expected).resolve():
                        raise TenantIsolationError("Database namespace does not match")
                elif (
                    schema != expected
                    or self.namespace_database is None
                    or Path(actual).resolve() != self.namespace_database
                ):
                    raise TenantIsolationError("Schema namespace does not match")
                cursor.execute(f"SELECT type, sql FROM {quoted_schema}.sqlite_master WHERE name = ?", (table,))
                row = cursor.fetchone()
                if row is None or row[0] != "table" or re.search(r"\bVIRTUAL\b", row[1] or "", re.I):
                    raise TenantIsolationError("An ordinary physical table is required")
            finally:
                cursor.close()

        @contextmanager
        def scope(conn):
            # Metadata and rows share a SQLite snapshot. A nested savepoint does
            # not commit an existing caller transaction or write application data.
            token = "tenant_read_" + uuid4().hex
            conn.execute(f"SAVEPOINT {token}").close()
            try:
                verify(conn)
                yield
                verify(conn)
            finally:
                conn.execute(f"ROLLBACK TO {token}").close()
                conn.execute(f"RELEASE {token}").close()

        limit = source._materialization_limit()
        columns, rows = source._fetch_bounded_rows(
            f"SELECT * FROM {quoted_schema}.{quoted_table} LIMIT {limit + 1}",
            max_rows=limit + 1, connection_scope=scope,
        )
        if len(rows) > limit:
            raise DataSourceSizeError(len(rows), limit, "rows for tenant SQL materialization")
        return (pl.DataFrame(rows) if rows else pl.DataFrame({key: [] for key in columns})).lazy()
