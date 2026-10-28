"""Bound SQL reads with physical namespace verification on the reading connection."""

import re
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from truthound.datasources.base import DataSourceSizeError
from truthound.datasources.sql.sqlite import SQLiteDataSource
from truthound.multitenancy.core import IsolationLevel, TenantIsolationError


class TenantSQLSource:
    """Borrow a factory-owned verified SQL source for one fixed native tenant.

    Only ordinary tables in verified physical namespaces are supported. Query, view,
    virtual-table and unverified remote-provider sources fail closed. The factory
    owns connection cleanup. Reads are bounded by the native source limit.
    """

    def __init__(self, source, *, tenant_id: str, namespace_database: str | None = None):
        if not isinstance(source, SQLiteDataSource):
            from truthound.datasources.sql.postgresql import PostgreSQLDataSource
            from truthound.datasources.sql.mysql import MySQLDataSource
            from truthound.datasources.sql.oracle import OracleDataSource
            from truthound.datasources.sql.sqlserver import SQLServerDataSource
            supported = isinstance(source, (PostgreSQLDataSource, MySQLDataSource, OracleDataSource, SQLServerDataSource))
            if not supported:
                try:
                    from truthound.datasources.sql.duckdb import DuckDBDataSource
                    supported = isinstance(source, DuckDBDataSource)
                except ImportError:
                    pass
            if not supported:
                raise TenantIsolationError("A verified table source is required")
        if source.is_query_mode:
            raise TenantIsolationError("A verified table source is required")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be nonempty")
        self.source = source
        self.tenant_id = tenant_id
        self.namespace_database_name = namespace_database
        if source.source_type in {"postgresql", "mysql", "oracle", "sqlserver"}:
            if namespace_database is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", namespace_database):
                raise ValueError("namespace_database must be a simple database identifier")
            self.namespace_database = None
            return
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
        if source.source_type == "oracle":
            return self._read_oracle(manager, tenant_id)
        if source.source_type == "sqlserver":
            return self._read_sqlserver(manager, tenant_id)
        if source.source_type == "mysql":
            return self._read_mysql(manager, tenant_id)
        if source.source_type == "postgresql":
            return self._read_postgresql(manager, tenant_id)
        if source.source_type == "duckdb":
            return self._read_duckdb(manager, tenant_id)
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

    def _read_sqlserver(self, manager, tenant_id):
        import polars as pl
        from truthound.datasources.sql.base import _row_mapping

        source = self.source
        database, schema, table = source._config.database, source._schema, source._table
        if not all(isinstance(value, str) and len(value) <= 128
                   and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                   for value in (database, schema, table)):
            raise TenantIsolationError("Simple database, schema and table identifiers are required")
        if source._config.autocommit is not True:
            raise TenantIsolationError("A dedicated autocommit source is required")
        qualified = ".".join(source._quote_identifier(value) for value in (database, schema, table))
        local_name = ".".join(source._quote_identifier(value) for value in (schema, table))
        placeholder = "?" if source._driver_name == "pyodbc" else "%s"
        connection = None

        def row(cursor):
            value = cursor.fetchone()
            if value is None:
                raise TenantIsolationError("Native connection metadata is required")
            return _row_mapping(value, [description[0] for description in cursor.description])

        def verify(cursor, identity=None):
            tenant = manager.require(tenant_id)
            strategy = manager.isolation_manager.get_strategy(tenant)
            strategy.validate_isolation(tenant)
            if tenant.isolation_level not in {IsolationLevel.SCHEMA, IsolationLevel.DATABASE}:
                raise TenantIsolationError("Physical isolation is required")
            expected = strategy.get_namespace(tenant)
            cursor.execute("SELECT DB_NAME() AS database_name, @@TRANCOUNT AS transactions")
            state = row(cursor)
            if state["database_name"] != database or state["transactions"] != 1:
                raise TenantIsolationError("Database or owned transaction does not match")
            if tenant.isolation_level == IsolationLevel.DATABASE:
                if database != expected:
                    raise TenantIsolationError("Database namespace does not match")
            elif schema != expected or database != self.namespace_database_name:
                raise TenantIsolationError("Schema namespace does not match")
            cursor.execute(f"""
                SELECT t.object_id AS object_id, s.name AS schema_name, t.name AS table_name
                FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id
                WHERE s.name={placeholder} AND t.name={placeholder}
                  AND t.is_ms_shipped=0 AND t.is_memory_optimized=0
                  AND t.is_filetable=0 AND t.is_remote_data_archive_enabled=0
                  AND NOT EXISTS (SELECT 1 FROM sys.external_tables e WHERE e.object_id=t.object_id)
            """, (schema, table))
            item = row(cursor)
            if ((item["schema_name"], item["table_name"]) != (schema, table)
                    or identity is not None and item["object_id"] != identity):
                raise TenantIsolationError("An unchanged ordinary table is required")
            cursor.execute(f"SELECT permission_name FROM sys.fn_my_permissions({placeholder}, 'OBJECT')", (local_name,))
            permissions = {item["permission_name"] for item in
                           (_row_mapping(value, ["permission_name"]) for value in cursor.fetchall())}
            if "SELECT" not in permissions or permissions - {"SELECT", "VIEW DEFINITION"}:
                raise TenantIsolationError("A SELECT-only table principal is required")
            return item["object_id"]

        @contextmanager
        def capture(actual):
            nonlocal connection
            connection = actual
            try:
                yield
            finally:
                connection = None

        @contextmanager
        def scope(cursor):
            if (not isinstance(connection, source._driver.Connection)
                    or source._driver_name == "pyodbc" and connection.autocommit is not True):
                raise TenantIsolationError("A dedicated native autocommit connection is required")
            cursor.execute("SELECT @@TRANCOUNT AS transactions, @@OPTIONS AS options, "
                "@@LOCK_TIMEOUT AS lock_timeout, transaction_isolation_level AS isolation_level "
                "FROM sys.dm_exec_sessions WHERE session_id=@@SPID")
            initial = row(cursor)
            levels = {1: "READ UNCOMMITTED", 2: "READ COMMITTED", 3: "REPEATABLE READ",
                      4: "SERIALIZABLE", 5: "SNAPSHOT"}
            if (initial["transactions"] or initial["options"] & 2
                    or initial["isolation_level"] not in levels):
                raise TenantIsolationError("An idle factory-owned connection is required")
            started = False
            try:
                cursor.execute("SET LOCK_TIMEOUT 0")
                cursor.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                cursor.execute("BEGIN TRANSACTION")
                started = True
                identity = verify(cursor)
                cursor.execute(f"SELECT TOP (1) 1 FROM {qualified} WITH (TABLOCK, HOLDLOCK)")
                cursor.fetchall()
                verify(cursor, identity)
                yield
                verify(cursor, identity)
            finally:
                try:
                    if started:
                        # DB-API rollback may be a no-op in driver autocommit mode.
                        cursor.execute("IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION")
                finally:
                    cursor.execute("SET TRANSACTION ISOLATION LEVEL " + levels[initial["isolation_level"]])
                    cursor.execute("SET LOCK_TIMEOUT " + str(int(initial["lock_timeout"])))

        limit = source._materialization_limit()
        columns, rows = source._fetch_bounded_rows(
            f"SELECT TOP ({limit + 1}) * FROM {qualified}", max_rows=limit + 1,
            connection_scope=capture, cursor_scope=scope,
        )
        if len(rows) > limit:
            raise DataSourceSizeError(len(rows), limit, "rows for tenant SQL materialization")
        return (pl.DataFrame(rows) if rows else pl.DataFrame({key: [] for key in columns})).lazy()

    def _read_oracle(self, manager, tenant_id):
        import oracledb
        import polars as pl

        source = self.source
        schema, table = source._schema, source._table
        if not all(isinstance(value, str) and len(value) <= 128
                   and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                   for value in (schema, table)) or schema.upper() == "SYS":
            raise TenantIsolationError("Simple application schema and table identifiers are required")
        qualified = f"{source._quote_identifier(schema)}.{source._quote_identifier(table)}"

        def verify(cursor, identity=None):
            tenant = manager.require(tenant_id)
            strategy = manager.isolation_manager.get_strategy(tenant)
            strategy.validate_isolation(tenant)
            if tenant.isolation_level not in {IsolationLevel.SCHEMA, IsolationLevel.DATABASE}:
                raise TenantIsolationError("Physical isolation is required")
            expected = strategy.get_namespace(tenant)
            cursor.execute("SELECT SYS_CONTEXT('USERENV','CON_NAME'), "
                           "SYS_CONTEXT('USERENV','SESSION_USER'), "
                           "SYS_CONTEXT('USERENV','ISDBA') FROM DUAL")
            database, user, privileged = cursor.fetchone()
            if user == "SYS" or privileged != "FALSE":
                raise TenantIsolationError("An ordinary application connection is required")
            if tenant.isolation_level == IsolationLevel.DATABASE:
                if database != expected:
                    raise TenantIsolationError("Database namespace does not match")
            elif schema != expected or database != self.namespace_database_name:
                raise TenantIsolationError("Schema namespace does not match")
            cursor.execute("""
                SELECT o.OBJECT_ID
                FROM ALL_OBJECTS o JOIN ALL_TABLES t
                  ON t.OWNER=o.OWNER AND t.TABLE_NAME=o.OBJECT_NAME
                WHERE o.OWNER=:owner AND o.OBJECT_NAME=:table_name
                  AND o.OBJECT_TYPE='TABLE' AND o.SUBOBJECT_NAME IS NULL
                  AND o.STATUS='VALID' AND t.TEMPORARY='N'
                  AND t.NESTED='NO' AND t.SECONDARY='N'
                  AND NOT EXISTS (SELECT 1 FROM ALL_EXTERNAL_TABLES e
                    WHERE e.OWNER=t.OWNER AND e.TABLE_NAME=t.TABLE_NAME)
            """, owner=schema, table_name=table)
            rows = cursor.fetchall()
            if len(rows) != 1 or (identity is not None and rows[0][0] != identity):
                raise TenantIsolationError("An unchanged ordinary application table is required")
            return rows[0][0]

        def verify_resolution(cursor, identity):
            # Kernel name resolution is checked as well as dictionary metadata:
            # a snapshot of an old table must not authorize a replacement synonym.
            values = [cursor.var(str, 128) for _ in range(4)] + [cursor.var(int), cursor.var(int)]
            cursor.callproc("DBMS_UTILITY.NAME_RESOLVE", [qualified, 0, *values])
            owner, name, part, link, _, resolved_identity = (value.getvalue() for value in values)
            if (owner, name, part, link, resolved_identity) != (schema, table, None, None, identity):
                raise TenantIsolationError("Table resolution does not match the approved namespace")

        @contextmanager
        def scope(cursor):
            connection = cursor.connection
            if (not isinstance(connection, oracledb.Connection)
                    or connection.autocommit is not False
                    or getattr(connection, "transaction_in_progress", None) is not False):
                raise TenantIsolationError("An idle factory-owned Oracle connection is required")
            # A failed SET must not roll back a caller's existing transaction.
            try:
                cursor.execute("SET TRANSACTION READ ONLY")
            except oracledb.DatabaseError as error:
                if error.args[0].code == 1453:
                    raise TenantIsolationError("An idle factory-owned Oracle connection is required") from error
                raise
            try:
                identity = verify(cursor)
                # ROW SHARE permits ADD COLUMN on Oracle; SHARE also fences
                # definition changes that retain the same OBJECT_ID.
                cursor.execute(f"LOCK TABLE {qualified} IN SHARE MODE NOWAIT")
                verify(cursor, identity)
                verify_resolution(cursor, identity)
                yield
                verify(cursor, identity)
                verify_resolution(cursor, identity)
            finally:
                connection.rollback()

        limit = source._materialization_limit()
        columns, rows = source._fetch_bounded_rows(
            f"SELECT * FROM {qualified} WHERE ROWNUM <= {limit + 1}",
            max_rows=limit + 1, cursor_scope=scope,
        )
        if len(rows) > limit:
            raise DataSourceSizeError(len(rows), limit, "rows for tenant SQL materialization")
        return (pl.DataFrame(rows) if rows else pl.DataFrame({key: [] for key in columns})).lazy()


    def _read_duckdb(self, manager, tenant_id):
        import polars as pl

        source = self.source
        schema, table = source._config.schema_name or "main", source._table
        if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                   for value in (schema, table)):
            raise TenantIsolationError("Simple schema and table identifiers are required")

        def verify(cursor):
            tenant = manager.require(tenant_id)
            strategy = manager.isolation_manager.get_strategy(tenant)
            strategy.validate_isolation(tenant)
            if tenant.isolation_level not in {IsolationLevel.SCHEMA, IsolationLevel.DATABASE}:
                raise TenantIsolationError("Physical isolation is required")
            expected = strategy.get_namespace(tenant)
            catalog = cursor.execute("SELECT current_database()").fetchone()[0]
            databases = cursor.execute("SELECT database_name, path FROM duckdb_databases()").fetchall()
            actual = dict(databases).get(catalog)
            if not actual:
                raise TenantIsolationError("A file-backed namespace is required")
            if tenant.isolation_level == IsolationLevel.DATABASE:
                if not Path(expected).is_absolute() or Path(actual).resolve() != Path(expected).resolve():
                    raise TenantIsolationError("Database namespace does not match")
            elif schema != expected or self.namespace_database is None or Path(actual).resolve() != self.namespace_database:
                raise TenantIsolationError("Schema namespace does not match")
            if cursor.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE temporary AND schema_name = ? AND table_name = ?",
                (schema, table),
            ).fetchone()[0]:
                raise TenantIsolationError("Temporary table shadowing is forbidden")
            row = cursor.execute(
                "SELECT table_type FROM information_schema.tables WHERE table_catalog = ? AND table_schema = ? AND table_name = ?",
                (catalog, schema, table),
            ).fetchone()
            if row is None or row[0] != "BASE TABLE":
                raise TenantIsolationError("An ordinary physical table is required")
            return catalog

        @contextmanager
        def scope(cursor):
            # DuckDB cursor() creates an independent connection. Fence the actual
            # reading cursor, rather than a transaction on the parent connection.
            cursor.execute("BEGIN TRANSACTION")
            try:
                verify(cursor)
                yield
                verify(cursor)
            finally:
                cursor.execute("ROLLBACK")

        limit = source._materialization_limit()
        columns, rows = source._fetch_bounded_rows(
            f"SELECT * FROM {source._quote_identifier(schema)}.{source._quote_identifier(table)} LIMIT {limit + 1}",
            max_rows=limit + 1, cursor_scope=scope,
        )
        if len(rows) > limit:
            raise DataSourceSizeError(len(rows), limit, "rows for tenant SQL materialization")
        return (pl.DataFrame(rows) if rows else pl.DataFrame({key: [] for key in columns})).lazy()


    def _read_postgresql(self, manager, tenant_id):
        import polars as pl
        from psycopg2.extensions import TRANSACTION_STATUS_IDLE

        source = self.source
        schema, table = source._config.schema_name or "public", source._table
        if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                   for value in (schema, table)):
            raise TenantIsolationError("Simple schema and table identifiers are required")

        def verify(cursor):
            tenant = manager.require(tenant_id)
            strategy = manager.isolation_manager.get_strategy(tenant)
            strategy.validate_isolation(tenant)
            if tenant.isolation_level not in {IsolationLevel.SCHEMA, IsolationLevel.DATABASE}:
                raise TenantIsolationError("Physical isolation is required")
            expected = strategy.get_namespace(tenant)
            cursor.execute("SELECT current_database()")
            actual = cursor.fetchone()[0]
            if tenant.isolation_level == IsolationLevel.DATABASE:
                if actual != expected:
                    raise TenantIsolationError("Database namespace does not match")
            elif schema != expected or actual != self.namespace_database_name:
                raise TenantIsolationError("Schema namespace does not match")
            cursor.execute(
                "SELECT c.relkind, c.relpersistence FROM pg_catalog.pg_class c "
                "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s", (schema, table),
            )
            row = cursor.fetchone()
            # Reject views, foreign tables, temporary relations and inheritance
            # parents: their rows may originate outside the verified namespace.
            if row is None or row[0] != 'r' or row[1] == 't':
                raise TenantIsolationError("An ordinary physical table is required")
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_inherits i "
                "JOIN pg_catalog.pg_class c ON c.oid = i.inhparent "
                "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s)", (schema, table),
            )
            if cursor.fetchone()[0]:
                raise TenantIsolationError("Inherited table reads are forbidden")

        @contextmanager
        def scope(cursor):
            if cursor.connection.get_transaction_status() != TRANSACTION_STATUS_IDLE:
                raise TenantIsolationError("An idle factory-owned connection is required")
            cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                # Pin the relation before establishing the catalog snapshot so
                # concurrent DDL cannot replace it between verification and read.
                cursor.execute(f"LOCK TABLE ONLY {source._quote_identifier(schema)}.{source._quote_identifier(table)} IN ACCESS SHARE MODE")
                verify(cursor)
                yield
                verify(cursor)
            finally:
                cursor.execute("ROLLBACK")

        limit = source._materialization_limit()
        columns, rows = source._fetch_bounded_rows(
            f"SELECT * FROM ONLY {source._quote_identifier(schema)}.{source._quote_identifier(table)} LIMIT {limit + 1}",
            max_rows=limit + 1, cursor_scope=scope,
        )
        if len(rows) > limit:
            raise DataSourceSizeError(len(rows), limit, "rows for tenant SQL materialization")
        return (pl.DataFrame(rows) if rows else pl.DataFrame({key: [] for key in columns})).lazy()


    def _read_mysql(self, manager, tenant_id):
        import polars as pl

        source = self.source
        database, table = source._config.database, source._table
        if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                   for value in (database, table)):
            raise TenantIsolationError("Simple database and table identifiers are required")
        qualified = f"{source._quote_identifier(database)}.{source._quote_identifier(table)}"

        def verify(cursor):
            tenant = manager.require(tenant_id)
            strategy = manager.isolation_manager.get_strategy(tenant)
            strategy.validate_isolation(tenant)
            if tenant.isolation_level not in {IsolationLevel.SCHEMA, IsolationLevel.DATABASE}:
                raise TenantIsolationError("Physical isolation is required")
            expected = strategy.get_namespace(tenant)
            cursor.execute("SELECT DATABASE() AS namespace")
            actual = cursor.fetchone()["namespace"]
            if actual != database or actual != expected:
                raise TenantIsolationError("Database namespace does not match")
            if tenant.isolation_level == IsolationLevel.SCHEMA and self.namespace_database_name != actual:
                raise TenantIsolationError("Schema namespace does not match")
            cursor.execute(
                "SELECT TABLE_TYPE AS kind, ENGINE AS engine FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s", (database, table),
            )
            row = cursor.fetchone()
            if row is None or row["kind"] != "BASE TABLE" or row["engine"] != "InnoDB":
                raise TenantIsolationError("An ordinary InnoDB table is required")
            cursor.execute(f"SHOW CREATE TABLE {qualified}")
            row = cursor.fetchone()
            ddl = row.get("Create Table", "")
            if not re.match(r"CREATE\s+TABLE\s", ddl, re.I):
                raise TenantIsolationError("Temporary or indirect table reads are forbidden")

        @contextmanager
        def scope(cursor):
            # Factory-owned connections may not donate an existing transaction:
            # starting this snapshot must never commit caller work.
            if cursor.connection.server_status & 1:
                raise TenantIsolationError("An idle factory-owned connection is required")
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
            try:
                verify(cursor)
                cursor.execute(f"SELECT * FROM {qualified} LIMIT 0")
                verify(cursor)
                yield
                verify(cursor)
            finally:
                cursor.execute("ROLLBACK")

        limit = source._materialization_limit()
        columns, rows = source._fetch_bounded_rows(
            f"SELECT * FROM {qualified} LIMIT {limit + 1}",
            max_rows=limit + 1, cursor_scope=scope,
        )
        if len(rows) > limit:
            raise DataSourceSizeError(len(rows), limit, "rows for tenant SQL materialization")
        return (pl.DataFrame(rows) if rows else pl.DataFrame({key: [] for key in columns})).lazy()
