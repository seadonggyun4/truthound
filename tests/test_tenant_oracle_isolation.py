"""Native Oracle fencing on a disposable, synthetic, fixture-owned PDB only."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration
IMAGE = "ghcr.io/gvenzl/oracle-free@sha256:6d61d267a3b978c24c5ac1790e62e927416a0aec446bd86e4b3a1527562757bd"
LABEL = "owned-oracle-isolation"


@dataclass(frozen=True)
class OracleRuntime:
    port: int
    password: str = field(repr=False)
    user: str = "TH_TENANT"
    database: str = "FREEPDB1"

    def connect(self, *, administrative=False):
        import oracledb

        return oracledb.connect(user="SYS" if administrative else self.user,
            password=self.password, dsn=f"127.0.0.1:{self.port}/{self.database}",
            mode=oracledb.AUTH_MODE_SYSDBA if administrative else oracledb.AUTH_MODE_DEFAULT,
            tcp_connect_timeout=2)


def stable_fixture_ddl(connection):
    """Seeded DDL must precede Oracle's time-based read-only snapshot boundary.

    No product retry/delay is introduced: recent/concurrent DDL still fails closed.
    """
    deadline = time.monotonic() + 5
    with connection.cursor() as cursor:
        while True:
            cursor.execute("SELECT MAX(CREATED), SYSDATE FROM USER_OBJECTS")
            created, current = cursor.fetchone()
            if created is not None and (current - created).total_seconds() >= 3:
                break
            if time.monotonic() >= deadline:
                pytest.fail("The test-owned Oracle DDL clock did not stabilize")
            time.sleep(0.05)
    connection.rollback()


@pytest.fixture(scope="module")
def oracle_runtime(tmp_path_factory):
    import oracledb

    if not shutil.which("docker"):
        pytest.fail("A Docker engine and the pinned Oracle image are required")
    evidence_root = os.environ.get("TRUTHOUND_ORACLE_TEST_EVIDENCE")
    if evidence_root:
        from pathlib import Path
        root = Path(evidence_root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        evidence = Path(tempfile.mkdtemp(prefix="oracle-", dir=root))
    else:
        evidence = tmp_path_factory.mktemp("oracle-runtime-evidence")
    password = "Th_" + secrets.token_hex(16)
    name = "truthound-oracle-" + uuid4().hex[:16]
    environment = dict(os.environ, ORACLE_PASSWORD=password, APP_USER="TH_TENANT",
                       APP_USER_PASSWORD=password)

    def docker(*args, timeout=30):
        return subprocess.run(["docker", *args], text=True, capture_output=True,
                              check=True, timeout=timeout, env=environment).stdout.strip()

    container = None
    try:
        docker("info", "--format", "{{.ServerVersion}}")
        docker("image", "inspect", "--format", "{{.Id}}", IMAGE)
        container = docker("run", "--detach", "--rm", "--pull=never", "--name", name,
            "--label", "truthound.fixture=" + LABEL, "--cpus=2", "--memory=3g",
            "--env", "ORACLE_PASSWORD", "--env", "APP_USER", "--env", "APP_USER_PASSWORD",
            "--publish", "127.0.0.1::1521", IMAGE, timeout=60)
        port = int(docker("port", container, "1521/tcp").rsplit(":", 1)[1])
        runtime = OracleRuntime(port, password)
        deadline = time.monotonic() + 240
        while True:
            try:
                connection = runtime.connect()
                break
            except oracledb.Error:
                if time.monotonic() >= deadline:
                    pytest.fail(f"Test-owned Oracle startup timed out; evidence: {evidence}")
                if docker("inspect", "--format", "{{.State.Running}}", container) != "true":
                    pytest.fail(f"Test-owned Oracle exited; evidence: {evidence}")
                time.sleep(2)
        connection.close()
        with runtime.connect(administrative=True) as admin:
            with admin.cursor() as cursor:
                cursor.execute("GRANT CREATE VIEW, CREATE SYNONYM TO TH_TENANT")
                cursor.execute("CREATE DIRECTORY TH_TENANT_NOFILE AS '/tmp/truthound-unused-external'")
                cursor.execute("GRANT READ ON DIRECTORY TH_TENANT_NOFILE TO TH_TENANT")
        with runtime.connect() as connection:
            with connection.cursor() as cursor:
                for table in ("RECORDS", "SWAPPABLE"):
                    cursor.execute(f'CREATE TABLE {table} ("x" INTEGER)')
                    cursor.executemany(f"INSERT INTO {table} VALUES (:1)", [(1,), (2,), (3,)])
                cursor.execute("CREATE VIEW INDIRECT AS SELECT * FROM RECORDS")
                cursor.execute("CREATE SYNONYM SHORTCUT FOR RECORDS")
                cursor.execute("CREATE SYNONYM REMOTE_ALIAS FOR OTHER_OWNER.RECORDS@UNCONFIGURED_LINK")
                cursor.execute('CREATE GLOBAL TEMPORARY TABLE TEMP_RECORDS ("x" INTEGER)')
                cursor.execute('''CREATE TABLE EXTERNAL_RECORDS ("x" INTEGER)
                    ORGANIZATION EXTERNAL (TYPE ORACLE_LOADER
                    DEFAULT DIRECTORY TH_TENANT_NOFILE
                    ACCESS PARAMETERS (RECORDS DELIMITED BY NEWLINE FIELDS TERMINATED BY ',')
                    LOCATION ('must-not-be-opened.csv')) REJECT LIMIT 0''')
            connection.commit()
            stable_fixture_ddl(connection)
        yield runtime
    finally:
        # Resolve an uncertain create by its exact unique name before cleanup.
        if container is None:
            candidate = subprocess.run(["docker", "inspect", "--format", "{{.Id}}", name],
                text=True, capture_output=True, timeout=15)
            if candidate.returncode == 0:
                container = candidate.stdout.strip()
        if container:
            try:
                logs = subprocess.run(["docker", "logs", "--tail", "200", container],
                    text=True, capture_output=True, timeout=10)
                (evidence / "startup.log").write_text(
                    (logs.stdout + logs.stderr).replace(password, "<redacted>"))
            except subprocess.TimeoutExpired:
                (evidence / "startup.log").write_text("Log capture timed out\n")
            assert docker("inspect", "--format", '{{index .Config.Labels "truthound.fixture"}}', container) == LABEL
            removed = subprocess.run(["docker", "rm", "--force", container],
                text=True, capture_output=True, timeout=60)
            check = subprocess.run(["docker", "inspect", container], text=True,
                                   capture_output=True, timeout=15)
            gone = check.returncode != 0 and any(token in check.stderr.lower()
                for token in ("no such object", "no such container"))
            (evidence / "cleanup.json").write_text(json.dumps({
                "remove_returncode": removed.returncode, "container_removed": gone}) + "\n")
            assert gone, f"Test-owned Oracle cleanup failed; evidence: {evidence}"


def oracle_source(runtime, table="RECORDS"):
    from truthound.datasources.sql.oracle import OracleConfig, OracleDataSource

    return OracleDataSource(table=table, host="127.0.0.1", port=runtime.port,
        service_name=runtime.database, user=runtime.user, password=runtime.password,
        schema=runtime.user, config=OracleConfig(materialization_row_limit=3))


def selected_tenant(runtime, *, schema_mode=True):
    from truthound.multitenancy.core import IsolationLevel
    from truthound.multitenancy.manager import TenantManager

    manager = TenantManager()
    tenant = manager.create("Selected", tenant_id="selected", isolation_level=(
        IsolationLevel.SCHEMA if schema_mode else IsolationLevel.DATABASE))
    tenant.schema_name, tenant.database_name = runtime.user, runtime.database
    return manager, tenant


@pytest.mark.parametrize("schema_mode", [False, True])
def test_oracle_native_namespace_readonly_lock_and_failure_cleanup(oracle_runtime, schema_mode, monkeypatch):
    import oracledb
    from truthound.datasources.base import DataSourceSizeError
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource

    runtime = oracle_runtime
    manager, tenant = selected_tenant(runtime, schema_mode=schema_mode)
    with oracle_source(runtime) as direct:
        expected = direct.to_polars_lazyframe().collect()
    with oracle_source(runtime) as source:
        bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database=runtime.database)
        native = source._read_bounded_cursor

        def fenced_read(cursor, query, max_rows, batch_size):
            with pytest.raises(oracledb.DatabaseError) as write:
                cursor.execute("INSERT INTO TH_TENANT.RECORDS VALUES (999)")
            assert write.value.args[0].code == 1456
            with runtime.connect() as rival:
                with rival.cursor() as ddl:
                    with pytest.raises(oracledb.DatabaseError) as locked:
                        ddl.execute('ALTER TABLE RECORDS ADD ("blocked" INTEGER)')
                    assert locked.value.args[0].code == 54
            return native(cursor, query, max_rows, batch_size)

        monkeypatch.setattr(source, "_read_bounded_cursor", fenced_read)
        assert bound.read(manager, tenant.id).collect().equals(expected)
        source._config.materialization_row_limit = 2
        with pytest.raises(DataSourceSizeError):
            bound.read(manager, tenant.id)
        source._config.materialization_row_limit = 3
        assert bound.read(manager, tenant.id).collect().equals(expected)
        tenant.schema_name = tenant.database_name = "FOREIGN_NAMESPACE"
        with pytest.raises(TenantIsolationError):
            bound.read(manager, tenant.id)
        tenant.schema_name, tenant.database_name = runtime.user, runtime.database
        assert bound.read(manager, tenant.id).collect().equals(expected)


@pytest.mark.parametrize("table", ["INDIRECT", "SHORTCUT", "REMOTE_ALIAS", "TEMP_RECORDS", "EXTERNAL_RECORDS"])
def test_oracle_indirect_and_temporary_sources_never_read_rows(oracle_runtime, table, monkeypatch):
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource

    manager, tenant = selected_tenant(oracle_runtime)
    with oracle_source(oracle_runtime, table) as source:
        monkeypatch.setattr(source, "_read_bounded_cursor", lambda *args: pytest.fail("An indirect source reached data fetch"))
        with pytest.raises(TenantIsolationError):
            TenantSQLSource(source, tenant_id=tenant.id,
                namespace_database=oracle_runtime.database).read(manager, tenant.id)


def test_oracle_preserves_caller_work_and_rejects_administrative_connections(oracle_runtime, monkeypatch):
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource

    runtime = oracle_runtime
    manager, tenant = selected_tenant(runtime)
    with oracle_source(runtime) as source:
        bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database=runtime.database)
        with runtime.connect() as connection:
            @contextmanager
            def borrowed():
                yield connection
            monkeypatch.setattr(source, "_get_connection", borrowed)
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO RECORDS VALUES (999)")
                with pytest.raises(TenantIsolationError):
                    bound.read(manager, tenant.id)
                assert connection.transaction_in_progress
                cursor.execute('SELECT COUNT(*) FROM RECORDS WHERE "x"=999')
                assert cursor.fetchone()[0] == 1
                connection.rollback()
                assert bound.read(manager, tenant.id).collect().height == 3
        with runtime.connect(administrative=True) as administrative:
            @contextmanager
            def privileged():
                yield administrative
            monkeypatch.setattr(source, "_get_connection", privileged)
            with pytest.raises(TenantIsolationError):
                bound.read(manager, tenant.id)


def test_oracle_preserves_an_existing_readonly_snapshot(oracle_runtime, monkeypatch):
    import oracledb
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource

    runtime = oracle_runtime
    manager, tenant = selected_tenant(runtime)
    with oracle_source(runtime) as source, runtime.connect() as connection:
        @contextmanager
        def borrowed():
            yield connection
        monkeypatch.setattr(source, "_get_connection", borrowed)
        bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database=runtime.database)
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SELECT COUNT(*) FROM RECORDS")
            assert cursor.fetchone()[0] == 3
            try:
                with runtime.connect() as rival:
                    with rival.cursor() as writer:
                        writer.execute("INSERT INTO RECORDS VALUES (999)")
                    rival.commit()
                with pytest.raises(TenantIsolationError):
                    bound.read(manager, tenant.id)
                cursor.execute("SELECT COUNT(*) FROM RECORDS")
                assert cursor.fetchone()[0] == 3
                with pytest.raises(oracledb.DatabaseError) as write:
                    cursor.execute("INSERT INTO RECORDS VALUES (998)")
                assert write.value.args[0].code == 1456
            finally:
                connection.rollback()
                with runtime.connect() as rival:
                    with rival.cursor() as writer:
                        writer.execute('DELETE FROM RECORDS WHERE "x"=999')
                    rival.commit()


def test_oracle_revocation_after_fetch_discards_rows_and_releases_ownership(oracle_runtime, monkeypatch):
    from truthound.multitenancy.core import TenantStatus, TenantSuspendedError
    from truthound.multitenancy.sources import TenantSQLSource

    manager, tenant = selected_tenant(oracle_runtime)
    with oracle_source(oracle_runtime) as source:
        bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database=oracle_runtime.database)
        native = source._read_bounded_cursor
        def revoked(*args):
            value = native(*args)
            tenant.status = TenantStatus.SUSPENDED
            return value
        monkeypatch.setattr(source, "_read_bounded_cursor", revoked)
        with pytest.raises(TenantSuspendedError):
            bound.read(manager, tenant.id)
        tenant.status = TenantStatus.ACTIVE
        monkeypatch.setattr(source, "_read_bounded_cursor", native)
        assert bound.read(manager, tenant.id).collect().height == 3


def test_oracle_replacement_between_metadata_and_lock_never_reaches_fetch(oracle_runtime, monkeypatch):
    import oracledb
    from truthound.multitenancy.core import TenantIsolationError
    from truthound.multitenancy.sources import TenantSQLSource

    manager, tenant = selected_tenant(oracle_runtime)
    original = oracledb.Cursor.execute
    replaced = False
    def replace_before_lock(cursor, statement, parameters=None, **kwargs):
        nonlocal replaced
        if statement.startswith('LOCK TABLE "TH_TENANT"."SWAPPABLE" ') and not replaced:
            replaced = True
            with oracle_runtime.connect() as rival:
                with rival.cursor() as ddl:
                    ddl.execute("DROP TABLE SWAPPABLE PURGE")
                    ddl.execute('CREATE TABLE SWAPPABLE ("x" INTEGER)')
                    ddl.execute("INSERT INTO SWAPPABLE VALUES (999)")
                rival.commit()
        return original(cursor, statement, parameters, **kwargs)
    monkeypatch.setattr(oracledb.Cursor, "execute", replace_before_lock)
    with oracle_source(oracle_runtime, "SWAPPABLE") as source:
        monkeypatch.setattr(source, "_read_bounded_cursor", lambda *args: pytest.fail("A replaced table reached data fetch"))
        with pytest.raises(TenantIsolationError):
            TenantSQLSource(source, tenant_id=tenant.id,
                namespace_database=oracle_runtime.database).read(manager, tenant.id)
    assert replaced


from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_tenancy import prepare, command

runtime = plugins.runtime


def configure_oracle_binding(runtime, oracle_runtime):
    runtime[2].with_name("tenant-source-mode").write_text(f"oracle:{oracle_runtime.port}")
    settings = runtime[2].with_name("tenant-oracle-settings.json")
    settings.write_text(json.dumps({"password": oracle_runtime.password}))
    settings.chmod(0o600)
    path = runtime[1] / "bindings.json"
    document = json.loads(path.read_text())
    document["plugins"]["fixture"]["capabilities"]["tenant-factory"] = "tenant-factory"
    document["tenant_services"] = {"local": "tenant-factory"}
    path.write_text(json.dumps(document))


def app_session_count(runtime):
    with runtime.connect(administrative=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM V$SESSION WHERE USERNAME=:1", [runtime.user])
            return cursor.fetchone()[0]


def assert_app_sessions_closed(runtime):
    deadline = time.monotonic() + 2
    while app_session_count(runtime):
        assert time.monotonic() < deadline, "The released source retained an Oracle server session"
        time.sleep(0.05)


def test_installed_oracle_worker_reopens_and_closes_its_source(runtime, monkeypatch, oracle_runtime):
    configure_oracle_binding(runtime, oracle_runtime)
    identity = prepare(runtime, monkeypatch)
    first = command(runtime, identity, "source_read", tenant_id="structured", source_id="records", limit=1)
    assert first["rows"] == [{"x": 1}]
    assert app_session_count(oracle_runtime) > 0
    assert plugins.invoke(runtime[0], "resources_release", resource_id=identity)["execution_status"] == "succeeded"
    assert_app_sessions_closed(oracle_runtime)
    reopened = approved(runtime, "tenant_services_open", binding_id="local", idempotency_key="oracle-reopen")["data"]["resource_id"]
    second = command(runtime, reopened, "source_read", key="oracle-second", tenant_id="structured", source_id="records", offset=1, limit=1, columns=["x"])
    assert second["rows"] == [{"x": 2}]
    assert plugins.invoke(runtime[0], "resources_release", resource_id=reopened)["execution_status"] == "succeeded"
    assert_app_sessions_closed(oracle_runtime)
    assert runtime[2].read_text().splitlines().count("tenant-close") == 2


def test_real_sdk_oracle_approval_read_and_frontend_shutdown(runtime, oracle_runtime):
    from tests.mcp_sql_client import read_tenant_source_over_stdio

    configure_oracle_binding(runtime, oracle_runtime)
    def live():
        assert app_session_count(oracle_runtime) > 0
    read_tenant_source_over_stdio(runtime, on_live=live)
    assert_app_sessions_closed(oracle_runtime)
    assert runtime[2].read_text().splitlines().count("tenant-close") == 1


def test_oracle_query_timeout_interrupts_round_trip_and_connection_recovers(oracle_runtime):
    import oracledb
    source = oracle_source(oracle_runtime)
    source._config.query_timeout = 0.02
    source._config.pool_size = 1
    with source:
        with source._get_connection() as connection:
            with connection.cursor() as cursor:
                with pytest.raises(oracledb.DatabaseError) as error:
                    cursor.execute("BEGIN DBMS_SESSION.SLEEP(0.2); END;")
                assert error.value.args[0].full_code in {"DPY-4024", "DPI-1067", "ORA-01013", "DPY-4011", "DPI-1080"}
        assert source.execute_scalar("SELECT 42 FROM DUAL") == 42
