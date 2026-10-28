"""Physical fencing against a test-owned disposable PostgreSQL server only."""

import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def postgres_port():
    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not initdb or not pg_ctl:
        pytest.fail("PostgreSQL test binaries initdb and pg_ctl are required")
    with tempfile.TemporaryDirectory(prefix="th-pg-", dir="/tmp") as root:
        data = Path(root) / "data"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        subprocess.run([initdb, "-D", str(data), "-U", "truthound_test", "--auth=trust", "--no-locale"],
                       check=True, capture_output=True, timeout=60)
        started = False
        try:
            subprocess.run([pg_ctl, "-D", str(data), "-l", str(Path(root) / "postgres.log"),
                            "-o", f"-h 127.0.0.1 -k {root} -p {port}", "-w", "start"],
                           check=True, capture_output=True, timeout=60)
            started = True
            yield port
        finally:
            if started or (data / "postmaster.pid").exists():
                subprocess.run([pg_ctl, "-D", str(data), "-m", "fast", "-w", "stop"],
                               check=True, capture_output=True, timeout=60)


@pytest.mark.parametrize("schema_mode", [False, True])
def test_postgresql_physical_namespace_view_inheritance_and_cleanup(postgres_port, schema_mode, monkeypatch):
    import psycopg2
    from truthound.datasources.sql.postgresql import PostgreSQLDataSource, PostgreSQLDataSourceConfig
    from truthound.datasources.base import DataSourceSizeError
    from truthound.multitenancy.core import IsolationLevel, TenantIsolationError, TenantStatus, TenantSuspendedError
    from truthound.multitenancy.manager import TenantManager
    from truthound.multitenancy.sources import TenantSQLSource

    schema = "scope_schema" if schema_mode else "scope_database"
    with psycopg2.connect(host="127.0.0.1", port=postgres_port, user="truthound_test", dbname="postgres") as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
            cursor.execute(f"CREATE TABLE {schema}.records (x INTEGER)")
            cursor.execute(f"INSERT INTO {schema}.records VALUES (1), (2), (3)")
            cursor.execute(f"CREATE VIEW {schema}.indirect AS SELECT * FROM {schema}.records")
            cursor.execute(f"CREATE TABLE {schema}.parent (x INTEGER)")
            cursor.execute(f"CREATE TABLE {schema}.child () INHERITS ({schema}.parent)")
    manager = TenantManager()
    tenant = manager.create("Selected", tenant_id="selected", isolation_level=(
        IsolationLevel.SCHEMA if schema_mode else IsolationLevel.DATABASE))
    tenant.schema_name, tenant.database_name = schema, "postgres"
    for table in ("records", "indirect", "parent"):
        config = PostgreSQLDataSourceConfig(host="127.0.0.1", port=postgres_port,
            database="postgres", user="truthound_test", schema_name=schema,
            materialization_row_limit=3)
        with PostgreSQLDataSource(table=table, config=config) as source:
            bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database="postgres")
            if table != "records":
                with pytest.raises(TenantIsolationError):
                    bound.read(manager, tenant.id)
                continue
            native_read = source._read_bounded_cursor

            def fenced_read(cursor, query, max_rows, batch_size):
                cursor.execute("SHOW transaction_read_only")
                assert cursor.fetchone()[0] == "on"
                with psycopg2.connect(host="127.0.0.1", port=postgres_port, user="truthound_test", dbname="postgres") as rival:
                    with rival.cursor() as rival_cursor:
                        rival_cursor.execute("SET LOCAL lock_timeout = '50ms'")
                        with pytest.raises(psycopg2.errors.LockNotAvailable):
                            rival_cursor.execute(f"ALTER TABLE {schema}.records RENAME TO replaced")
                return native_read(cursor, query, max_rows, batch_size)

            monkeypatch.setattr(source, "_read_bounded_cursor", fenced_read)
            assert bound.read(manager, tenant.id).collect()["x"].to_list() == [1, 2, 3]
            tenant.status = TenantStatus.SUSPENDED
            with pytest.raises(TenantSuspendedError):
                bound.read(manager, tenant.id)
            tenant.status = TenantStatus.ACTIVE
            source._config.materialization_row_limit = 2
            with pytest.raises(DataSourceSizeError):
                bound.read(manager, tenant.id)
            source._config.materialization_row_limit = 3
            assert bound.read(manager, tenant.id).collect().height == 3
            if schema_mode:
                bound.namespace_database_name = "foreign_database"
            else:
                tenant.database_name = "foreign_database"
            with pytest.raises(TenantIsolationError):
                bound.read(manager, tenant.id)
            tenant.database_name = "postgres"


from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_tenancy import prepare, command
from tests.test_mcp_brokers import approved

runtime = plugins.runtime


def test_installed_postgresql_worker_reopen_and_cleanup(runtime, monkeypatch, postgres_port):
    runtime[2].with_name("tenant-source-mode").write_text(f"postgres:{postgres_port}")
    identity = prepare(runtime, monkeypatch)
    first = command(runtime, identity, "source_read", tenant_id="structured", source_id="records", limit=1)
    assert first["rows"] == [{"x": 1}]
    assert plugins.invoke(runtime[0], "resources_release", resource_id=identity)["execution_status"] == "succeeded"
    reopened = approved(runtime, "tenant_services_open", binding_id="local", idempotency_key="pg-reopen")["data"]["resource_id"]
    second = command(runtime, reopened, "source_read", key="pg-second", tenant_id="structured", source_id="records", offset=1, columns=["x"])
    assert second["rows"] == [{"x": 2}]
    assert plugins.invoke(runtime[0], "resources_release", resource_id=reopened)["execution_status"] == "succeeded"
    assert runtime[2].read_text().splitlines().count("tenant-close") == 2


def test_postgresql_query_timeout_cancels_native_statement_and_reuses_pool(postgres_port):
    import psycopg2
    from truthound.datasources.sql.postgresql import PostgreSQLDataSource, PostgreSQLDataSourceConfig

    config = PostgreSQLDataSourceConfig(host="127.0.0.1", port=postgres_port,
        database="postgres", user="truthound_test", query_timeout=0.02, pool_size=1)
    with PostgreSQLDataSource(table="unused", config=config) as source:
        with pytest.raises(psycopg2.errors.QueryCanceled):
            source.execute_scalar("SELECT pg_sleep(0.1)")
        assert source.execute_scalar("SELECT 42") == 42
