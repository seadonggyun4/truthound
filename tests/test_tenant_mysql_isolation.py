"""Actual MySQL fencing on a server initialized and owned by this fixture only."""

import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def mysql_port(tmp_path_factory):
    import pymysql

    binary = shutil.which("mysqld")
    if not binary:
        pytest.fail("A local mysqld test binary is required")
    evidence = tmp_path_factory.mktemp("mysql-runtime-evidence")
    with tempfile.TemporaryDirectory(prefix="th-mysql-", dir="/tmp") as root:
        data, unix_socket = Path(root) / "data", Path(root) / "mysql.sock"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with (evidence / "initialize.log").open("wb") as log:
            subprocess.run([binary, "--no-defaults", "--initialize-insecure", f"--datadir={data}"],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        with (evidence / "server.log").open("wb") as log:
            process = subprocess.Popen([binary, "--no-defaults", f"--datadir={data}",
                f"--socket={unix_socket}", f"--port={port}", "--bind-address=127.0.0.1",
                "--mysqlx=OFF", "--skip-log-bin", f"--pid-file={Path(root) / 'mysql.pid'}"],
                stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        pytest.fail(f"Test-owned mysqld exited; see {evidence / 'server.log'}")
                    try:
                        with pymysql.connect(unix_socket=str(unix_socket), user="root", connect_timeout=1) as connection:
                            with connection.cursor() as cursor:
                                cursor.execute("SELECT @@datadir")
                                assert Path(cursor.fetchone()[0]).resolve() == data.resolve()
                        break
                    except pymysql.Error:
                        if time.monotonic() >= deadline:
                            pytest.fail(f"Test-owned mysqld startup timed out; see {evidence / 'server.log'}")
                        time.sleep(0.1)
                yield port
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)


@pytest.mark.parametrize("schema_mode", [False, True])
def test_mysql_namespace_view_and_failure_cleanup(mysql_port, schema_mode, monkeypatch):
    import pymysql
    from truthound.datasources.sql.mysql import MySQLDataSource, MySQLDataSourceConfig
    from truthound.datasources.base import DataSourceSizeError
    from truthound.multitenancy.core import IsolationLevel, TenantIsolationError
    from truthound.multitenancy.manager import TenantManager
    from truthound.multitenancy.sources import TenantSQLSource

    database = "schema_scope" if schema_mode else "database_scope"
    with pymysql.connect(host="127.0.0.1", port=mysql_port, user="root", autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
            cursor.execute(f"CREATE TABLE {database}.records (x INTEGER) ENGINE=InnoDB")
            cursor.execute(f"INSERT INTO {database}.records VALUES (1), (2), (3)")
            cursor.execute(f"CREATE VIEW {database}.indirect AS SELECT * FROM {database}.records")
    manager = TenantManager()
    tenant = manager.create("Selected", tenant_id="selected", isolation_level=(
        IsolationLevel.SCHEMA if schema_mode else IsolationLevel.DATABASE))
    tenant.schema_name = tenant.database_name = database
    for table in ("records", "indirect"):
        config = MySQLDataSourceConfig(host="127.0.0.1", port=mysql_port, user="root",
            database=database, materialization_row_limit=3)
        with MySQLDataSource(table=table, config=config) as source:
            bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database=database)
            if table != "records":
                with pytest.raises(TenantIsolationError):
                    bound.read(manager, tenant.id)
                continue
            native_read = source._read_bounded_cursor

            def fenced_read(cursor, query, max_rows, batch_size):
                with pytest.raises(pymysql.err.OperationalError) as read_only:
                    cursor.execute(f"INSERT INTO {database}.records VALUES (999)")
                assert read_only.value.args[0] == 1792
                with pymysql.connect(host="127.0.0.1", port=mysql_port, user="root", autocommit=True) as rival:
                    with rival.cursor() as rival_cursor:
                        rival_cursor.execute("SET SESSION lock_wait_timeout = 1")
                        with pytest.raises(pymysql.err.OperationalError) as blocked:
                            rival_cursor.execute(f"ALTER TABLE {database}.records RENAME TO {database}.replaced")
                        assert blocked.value.args[0] == 1205
                return native_read(cursor, query, max_rows, batch_size)

            monkeypatch.setattr(source, "_read_bounded_cursor", fenced_read)
            assert bound.read(manager, tenant.id).collect()["x"].to_list() == [1, 2, 3]
            source._config.materialization_row_limit = 2
            with pytest.raises(DataSourceSizeError):
                bound.read(manager, tenant.id)
            source._config.materialization_row_limit = 3
            assert bound.read(manager, tenant.id).collect().height == 3
            tenant.database_name = tenant.schema_name = "foreign_database"
            with pytest.raises(TenantIsolationError):
                bound.read(manager, tenant.id)
            tenant.database_name = tenant.schema_name = database


from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_tenancy import prepare, command
from tests.test_mcp_brokers import approved

runtime = plugins.runtime


def test_installed_mysql_worker_reopen_and_cleanup(runtime, monkeypatch, mysql_port):
    runtime[2].with_name("tenant-source-mode").write_text(f"mysql:{mysql_port}")
    identity = prepare(runtime, monkeypatch)
    first = command(runtime, identity, "source_read", tenant_id="structured", source_id="records", limit=1)
    assert first["rows"] == [{"x": 1}]
    assert plugins.invoke(runtime[0], "resources_release", resource_id=identity)["execution_status"] == "succeeded"
    reopened = approved(runtime, "tenant_services_open", binding_id="local", idempotency_key="mysql-reopen")["data"]["resource_id"]
    second = command(runtime, reopened, "source_read", key="mysql-second", tenant_id="structured", source_id="records", offset=1, columns=["x"])
    assert second["rows"] == [{"x": 2}]
    assert plugins.invoke(runtime[0], "resources_release", resource_id=reopened)["execution_status"] == "succeeded"
    assert runtime[2].read_text().splitlines().count("tenant-close") == 2


def test_mysql_temporary_shadow_and_existing_transaction_fail_closed(mysql_port, monkeypatch):
    from contextlib import contextmanager
    import pymysql
    from truthound.datasources.sql.mysql import MySQLDataSource
    from truthound.multitenancy.core import IsolationLevel, TenantIsolationError
    from truthound.multitenancy.manager import TenantManager
    from truthound.multitenancy.sources import TenantSQLSource

    with pymysql.connect(host="127.0.0.1", port=mysql_port, user="root", autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE DATABASE guarded_scope")
            cursor.execute("CREATE TABLE guarded_scope.records (x INTEGER) ENGINE=InnoDB")
            cursor.execute("INSERT INTO guarded_scope.records VALUES (1)")
            cursor.execute("CREATE TABLE guarded_scope.indirect_engine (x INTEGER) ENGINE=MyISAM")
    manager = TenantManager()
    tenant = manager.create("Guarded", tenant_id="guarded", isolation_level=IsolationLevel.DATABASE)
    tenant.database_name = "guarded_scope"
    with MySQLDataSource(table="indirect_engine", host="127.0.0.1", port=mysql_port, user="root", database="guarded_scope") as source:
        with pytest.raises(TenantIsolationError):
            TenantSQLSource(source, tenant_id=tenant.id).read(manager, tenant.id)
    with MySQLDataSource(table="records", host="127.0.0.1", port=mysql_port, user="root", database="guarded_scope") as source:
        bound = TenantSQLSource(source, tenant_id=tenant.id)
        with source._get_connection() as connection:
            @contextmanager
            def borrowed():
                yield connection
            monkeypatch.setattr(source, "_get_connection", borrowed)
            with connection.cursor() as cursor:
                cursor.execute("CREATE TEMPORARY TABLE guarded_scope.records (x INTEGER)")
                cursor.execute("INSERT INTO guarded_scope.records VALUES (999)")
                with pytest.raises(TenantIsolationError):
                    bound.read(manager, tenant.id)
                cursor.execute("DROP TEMPORARY TABLE guarded_scope.records")
                assert bound.read(manager, tenant.id).collect()["x"].to_list() == [1]
                cursor.execute("START TRANSACTION")
                cursor.execute("INSERT INTO guarded_scope.records VALUES (2)")
                with pytest.raises(TenantIsolationError):
                    bound.read(manager, tenant.id)
                assert connection.server_status & 1
                connection.rollback()
                assert bound.read(manager, tenant.id).collect()["x"].to_list() == [1]
