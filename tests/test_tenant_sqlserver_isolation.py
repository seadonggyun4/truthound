"""Real SQL Server, fixed synthetic credentials and a fixture-owned container."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import time
import tempfile
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration
IMAGE = "mcr.microsoft.com/mssql/server@sha256:cf8f0ab2ebe61feec2eed2629061a07d247f791c2a8ddda74a4ac9f06a291718"
LABEL = "owned-sqlserver-isolation"


@dataclass(frozen=True)
class SQLServerRuntime:
    port: int
    password: str = field(repr=False)
    user: str = "th_reader"
    database: str = "th_tenantdb"
    schema: str = "th_scope"

    def connect(self, *, administrative=False, database=None):
        import pymssql
        return pymssql.connect(server="127.0.0.1", port=self.port,
            user="sa" if administrative else self.user, password=self.password,
            database=database or self.database, autocommit=True, login_timeout=2, timeout=5)


@pytest.fixture(scope="module")
def sqlserver_runtime(tmp_path_factory):
    import pymssql
    if not shutil.which("docker"):
        pytest.fail("Docker and the pinned SQL Server image are required")
    evidence_root = os.environ.get("TRUTHOUND_SQLSERVER_TEST_EVIDENCE")
    if evidence_root:
        root = Path(evidence_root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        evidence = Path(tempfile.mkdtemp(prefix="sqlserver-", dir=root))
    else:
        evidence = tmp_path_factory.mktemp("sqlserver-runtime-evidence")
    config = evidence / "mssql.conf"
    config.write_text("[telemetry]\ncustomerfeedback = false\n[memory]\nmemorylimitmb = 2048\n")
    config.chmod(0o644)
    password = "Th_" + secrets.token_hex(16)
    environment = dict(os.environ, ACCEPT_EULA="Y", MSSQL_PID="Developer", MSSQL_SA_PASSWORD=password)
    name, container = "truthound-sqlserver-" + uuid4().hex[:16], None

    def docker(*args, timeout=30):
        return subprocess.run(["docker", *args], check=True, text=True, capture_output=True,
                              timeout=timeout, env=environment).stdout.strip()

    try:
        architecture = docker("info", "--format", "{{.Architecture}}")
        native = architecture in {"amd64", "x86_64"}
        if os.environ.get("TRUTHOUND_SQLSERVER_REQUIRE_NATIVE") == "1":
            assert native, "The required SQL Server CI gate needs a native x86-64 engine"
        (evidence / "platform.json").write_text(json.dumps({
            "architecture": architecture, "native_x86": native}) + "\n")
        docker("image", "inspect", "--format", "{{.Id}}", IMAGE)
        container = docker("run", "--detach", "--rm", "--pull=never", "--platform", "linux/amd64",
            "--name", name, "--label", "truthound.fixture=" + LABEL,
            "--cpus=2", "--memory=3g", "--env", "ACCEPT_EULA", "--env", "MSSQL_PID",
            "--env", "MSSQL_SA_PASSWORD", "--publish", "127.0.0.1::1433",
            "--mount", f"type=bind,src={config.resolve()},dst=/var/opt/mssql/mssql.conf,readonly",
            IMAGE, timeout=60)
        port = int(docker("port", container, "1433/tcp").rsplit(":", 1)[1])
        runtime = SQLServerRuntime(port, password)
        deadline = time.monotonic() + 180
        while True:
            try:
                admin = runtime.connect(administrative=True, database="master")
                break
            except pymssql.Error:
                if time.monotonic() >= deadline:
                    pytest.fail(f"Test-owned SQL Server startup timed out; evidence: {evidence}")
                if docker("inspect", "--format", "{{.State.Running}}", container) != "true":
                    pytest.fail(f"Test-owned SQL Server exited; evidence: {evidence}")
                time.sleep(2)
        with admin:
            with admin.cursor() as cursor:
                cursor.execute("CREATE DATABASE th_tenantdb")
                cursor.execute("CREATE LOGIN th_reader WITH PASSWORD=%s, CHECK_POLICY=OFF", (password,))
                cursor.execute("USE th_tenantdb")
                cursor.execute("CREATE SCHEMA th_scope")
                cursor.execute("CREATE TABLE th_scope.records (x INTEGER)")
                cursor.execute("INSERT INTO th_scope.records VALUES (1), (2), (3)")
                cursor.execute("CREATE VIEW th_scope.indirect AS SELECT * FROM th_scope.records")
                cursor.execute("CREATE SYNONYM th_scope.shortcut FOR th_scope.records")
                cursor.execute("CREATE USER th_reader FOR LOGIN th_reader")
                for target in ("records", "indirect", "shortcut"):
                    cursor.execute(f"GRANT SELECT, VIEW DEFINITION ON OBJECT::th_scope.{target} TO th_reader")
        yield runtime
    finally:
        if container is None:
            candidate = subprocess.run(["docker", "inspect", "--format", "{{.Id}}", name],
                                      text=True, capture_output=True, timeout=15)
            if candidate.returncode == 0:
                container = candidate.stdout.strip()
        if container:
            try:
                logs = subprocess.run(["docker", "logs", "--tail", "100", container],
                                      text=True, capture_output=True, timeout=10)
                (evidence / "startup.log").write_text(
                    (logs.stdout + logs.stderr).replace(password, "<redacted>"))
            except subprocess.TimeoutExpired:
                (evidence / "startup.log").write_text("Log capture timed out\n")
            assert docker("inspect", "--format", '{{index .Config.Labels "truthound.fixture"}}', container) == LABEL
            removal = subprocess.run(["docker", "rm", "--force", container],
                                    text=True, capture_output=True, timeout=60)
            check = subprocess.run(["docker", "inspect", container], text=True,
                                   capture_output=True, timeout=15)
            gone = check.returncode != 0 and any(token in check.stderr.lower()
                for token in ("no such object", "no such container"))
            (evidence / "cleanup.json").write_text(json.dumps({
                "remove_returncode": removal.returncode, "container_removed": gone}) + "\n")
            assert gone, f"Test-owned SQL Server cleanup failed; evidence: {evidence}"


def sqlserver_source(runtime, table="records", *, administrative=False, autocommit=True):
    from truthound.datasources.sql.sqlserver import SQLServerConfig, SQLServerDataSource
    return SQLServerDataSource(table=table, host="127.0.0.1", port=runtime.port,
        database=runtime.database, user="sa" if administrative else runtime.user,
        password=runtime.password, schema=runtime.schema,
        config=SQLServerConfig(dbapi_driver="pymssql", autocommit=autocommit,
                               materialization_row_limit=3))


def selected_tenant(runtime, *, schema_mode=True):
    from truthound.multitenancy.core import IsolationLevel
    from truthound.multitenancy.manager import TenantManager
    manager = TenantManager()
    tenant = manager.create("Selected", tenant_id="selected", isolation_level=(
        IsolationLevel.SCHEMA if schema_mode else IsolationLevel.DATABASE))
    tenant.database_name, tenant.schema_name = runtime.database, runtime.schema
    return manager, tenant


@pytest.mark.parametrize("schema_mode", [False, True])
def test_sqlserver_namespace_limits_native_ddl_fence_and_cleanup(sqlserver_runtime, schema_mode, monkeypatch):
    import pymssql
    from truthound.datasources.base import DataSourceSizeError
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource

    runtime = sqlserver_runtime
    manager, tenant = selected_tenant(runtime, schema_mode=schema_mode)
    with sqlserver_source(runtime) as direct:
        expected = direct.to_polars_lazyframe().collect()
    with sqlserver_source(runtime) as source:
        bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database=runtime.database)
        native = source._read_bounded_cursor
        def fenced_read(cursor, query, max_rows, batch_size):
            with runtime.connect(administrative=True) as rival:
                with rival.cursor() as ddl:
                    ddl.execute("SET LOCK_TIMEOUT 0")
                    with pytest.raises(pymssql.Error) as locked:
                        ddl.execute("ALTER TABLE th_scope.records ADD blocked INTEGER")
                    assert locked.value.args[0] == 1222
            return native(cursor, query, max_rows, batch_size)
        monkeypatch.setattr(source, "_read_bounded_cursor", fenced_read)
        assert bound.read(manager, tenant.id).collect().equals(expected)
        source._config.materialization_row_limit = 2
        with pytest.raises(DataSourceSizeError):
            bound.read(manager, tenant.id)
        source._config.materialization_row_limit = 3
        assert bound.read(manager, tenant.id).collect().equals(expected)
        tenant.database_name = tenant.schema_name = "foreign_namespace"
        with pytest.raises(TenantIsolationError):
            bound.read(manager, tenant.id)
        tenant.database_name, tenant.schema_name = runtime.database, runtime.schema
        assert bound.read(manager, tenant.id).collect().equals(expected)


@pytest.mark.parametrize("table", ["indirect", "shortcut", "#temporary"])
def test_sqlserver_indirect_sources_never_fetch(sqlserver_runtime, table, monkeypatch):
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource
    manager, tenant = selected_tenant(sqlserver_runtime)
    with sqlserver_source(sqlserver_runtime, table) as source:
        monkeypatch.setattr(source, "_read_bounded_cursor", lambda *args: pytest.fail("An indirect source reached data fetch"))
        with pytest.raises(TenantIsolationError):
            TenantSQLSource(source, tenant_id=tenant.id,
                namespace_database=sqlserver_runtime.database).read(manager, tenant.id)


def test_sqlserver_rejects_writable_principal_and_preserves_default_driver_mode(sqlserver_runtime):
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource
    manager, tenant = selected_tenant(sqlserver_runtime)
    with sqlserver_source(sqlserver_runtime, administrative=True) as source:
        with pytest.raises(TenantIsolationError):
            TenantSQLSource(source, tenant_id=tenant.id,
                namespace_database=sqlserver_runtime.database).read(manager, tenant.id)
    with sqlserver_source(sqlserver_runtime, autocommit=False) as source:
        with source._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT @@TRANCOUNT")
                assert cursor.fetchone()[0] == 1
                with pytest.raises(TenantIsolationError):
                    TenantSQLSource(source, tenant_id=tenant.id,
                        namespace_database=sqlserver_runtime.database).read(manager, tenant.id)
                cursor.execute("SELECT @@TRANCOUNT")
                assert cursor.fetchone()[0] == 1


def test_sqlserver_restores_settings_without_rolling_back_caller_work(sqlserver_runtime, monkeypatch):
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource
    runtime = sqlserver_runtime
    manager, tenant = selected_tenant(runtime)
    with sqlserver_source(runtime) as source, runtime.connect() as connection:
        @contextmanager
        def borrowed():
            yield connection
        monkeypatch.setattr(source, "_get_connection", borrowed)
        bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database=runtime.database)
        with connection.cursor() as cursor:
            cursor.execute("SET LOCK_TIMEOUT 157")
            cursor.execute("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
            assert bound.read(manager, tenant.id).collect().height == 3

            cursor.execute("SELECT @@TRANCOUNT, @@LOCK_TIMEOUT, transaction_isolation_level FROM sys.dm_exec_sessions WHERE session_id=@@SPID")
            assert cursor.fetchone() == (0, 157, 1)
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("CREATE TABLE #caller_work (x INTEGER)")
            cursor.execute("INSERT INTO #caller_work VALUES (99)")
            with pytest.raises(TenantIsolationError):
                bound.read(manager, tenant.id)
            cursor.execute("SELECT @@TRANCOUNT")
            assert cursor.fetchone()[0] == 1
            cursor.execute("SELECT x FROM #caller_work")
            assert cursor.fetchall() == [(99,)]
            cursor.execute("ROLLBACK TRANSACTION")
            assert bound.read(manager, tenant.id).collect().height == 3


from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_tenancy import prepare, command

runtime = plugins.runtime


def configure_sqlserver_binding(runtime, sqlserver_runtime):
    runtime[2].with_name("tenant-source-mode").write_text(f"sqlserver:{sqlserver_runtime.port}")
    settings = runtime[2].with_name("tenant-sqlserver-settings.json")
    settings.write_text(json.dumps({"password": sqlserver_runtime.password}))
    settings.chmod(0o600)
    path = runtime[1] / "bindings.json"
    document = json.loads(path.read_text())
    document["plugins"]["fixture"]["capabilities"]["tenant-factory"] = "tenant-factory"
    document["tenant_services"] = {"local": "tenant-factory"}
    path.write_text(json.dumps(document))


def app_session_count(runtime):
    with runtime.connect(administrative=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM sys.dm_exec_sessions WHERE login_name=%s AND is_user_process=1", (runtime.user,))
            return cursor.fetchone()[0]


def assert_app_sessions_closed(runtime):
    deadline = time.monotonic() + 2
    while app_session_count(runtime):
        assert time.monotonic() < deadline, "The released source retained a SQL Server session"
        time.sleep(0.05)


def test_installed_sqlserver_worker_reopens_and_closes_its_source(runtime, monkeypatch, sqlserver_runtime):
    configure_sqlserver_binding(runtime, sqlserver_runtime)
    identity = prepare(runtime, monkeypatch)
    first = command(runtime, identity, "source_read", tenant_id="structured", source_id="records", limit=1)
    assert first["rows"] == [{"x": 1}]
    assert app_session_count(sqlserver_runtime) > 0
    assert plugins.invoke(runtime[0], "resources_release", resource_id=identity)["execution_status"] == "succeeded"
    assert_app_sessions_closed(sqlserver_runtime)
    reopened = approved(runtime, "tenant_services_open", binding_id="local", idempotency_key="sqlserver-reopen")["data"]["resource_id"]
    second = command(runtime, reopened, "source_read", key="sqlserver-second", tenant_id="structured", source_id="records", offset=1, limit=1, columns=["x"])
    assert second["rows"] == [{"x": 2}]
    assert plugins.invoke(runtime[0], "resources_release", resource_id=reopened)["execution_status"] == "succeeded"
    assert_app_sessions_closed(sqlserver_runtime)
    assert runtime[2].read_text().splitlines().count("tenant-close") == 2


def test_real_sdk_sqlserver_approval_read_and_frontend_shutdown(runtime, sqlserver_runtime):
    from tests.mcp_sql_client import read_tenant_source_over_stdio

    configure_sqlserver_binding(runtime, sqlserver_runtime)
    def live():
        assert app_session_count(sqlserver_runtime) > 0
    read_tenant_source_over_stdio(runtime, on_live=live)
    assert_app_sessions_closed(sqlserver_runtime)
    assert runtime[2].read_text().splitlines().count("tenant-close") == 1
