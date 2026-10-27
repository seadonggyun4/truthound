"""Native row policies must run after tenant fencing and before row limits."""

import polars as pl
import pytest

from truthound.multitenancy.core import (
    Tenant,
    TenantIsolationError,
    TenantStatus,
    TenantSuspendedError,
)
from truthound.multitenancy.isolation import RowLevelIsolation, RowLevelPolicy

pytestmark = pytest.mark.fault


def test_row_policies_filter_before_limit_and_never_receive_other_tenant_rows():
    tenant = Tenant("first", "First")
    strategy = RowLevelIsolation()
    seen = []

    def positive(row, active):
        seen.append(row["tenant_id"])
        return row["owner"] == active.owner_id and row["x"] > 0

    tenant.owner_id = "owner"
    strategy.add_policy(RowLevelPolicy("positive", positive))
    frame = pl.DataFrame({"tenant_id": ["other", "first", "first", "first"],
                          "owner": ["owner"] * 4, "x": [99, -1, 1, 2]}).lazy()
    result = strategy.apply_filter(frame, tenant).select("x").limit(1).collect()
    assert result["x"].to_list() == [1]
    assert seen and set(seen) == {"first"}


@pytest.mark.parametrize("value", ["False", None, 1])
def test_non_boolean_row_policy_result_cannot_allow_access(value):
    strategy = RowLevelIsolation()
    strategy.add_policy(RowLevelPolicy("invalid", lambda row, tenant: value))
    result = strategy.apply_filter(pl.DataFrame({"tenant_id": ["first"], "x": [1]}).lazy(), Tenant("first", "First"))
    assert result.collect().is_empty()


@pytest.mark.parametrize("fail_closed", [True, False])
def test_policy_error_denies_or_fails_without_private_error_body(fail_closed):
    strategy = RowLevelIsolation(fail_closed=fail_closed)

    def failed(row, tenant):
        raise RuntimeError("synthetic-private-policy-detail")

    strategy.add_policy(RowLevelPolicy("failure", failed))
    result = strategy.apply_filter(pl.DataFrame({"tenant_id": ["first"], "x": [1]}).lazy(), Tenant("first", "First"))
    if fail_closed:
        assert result.collect().is_empty()
    else:
        with pytest.raises(TenantIsolationError) as error:
            result.collect()
        assert "synthetic-private-policy-detail" not in str(error.value)


def test_suspended_tenant_is_rechecked_when_lazy_rows_are_collected():
    tenant = Tenant("first", "First")
    frame = RowLevelIsolation().apply_filter(pl.DataFrame({"tenant_id": ["first"]}).lazy(), tenant)
    tenant.status = TenantStatus.SUSPENDED
    with pytest.raises(TenantSuspendedError):
        frame.collect()


def test_sql_source_checks_actual_database_and_rejects_view(tmp_path):
    import sqlite3

    from truthound.datasources.sql.sqlite import SQLiteDataSource
    from truthound.multitenancy.core import IsolationLevel
    from truthound.multitenancy.manager import TenantManager
    from truthound.multitenancy.sources import TenantSQLSource

    paths = [tmp_path / 'first.db', tmp_path / 'other.db']
    for index, path in enumerate(paths):
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE records (x INTEGER)')
            conn.execute('INSERT INTO records VALUES (?)', (index,))
            conn.execute('CREATE VIEW indirect AS SELECT * FROM records')
    manager = TenantManager()
    tenant = manager.create(name='First', tenant_id='first', isolation_level=IsolationLevel.DATABASE)
    tenant.database_name = str(paths[0])
    for path, table, allowed in [(paths[0], 'records', True), (paths[1], 'records', False), (paths[0], 'indirect', False)]:
        with SQLiteDataSource(table=table, database=str(path)) as source:
            bound = TenantSQLSource(source, tenant_id='first')
            if allowed:
                assert bound.read(manager, tenant.id).collect()['x'].to_list() == [0]
                tenant.status = TenantStatus.SUSPENDED
                with pytest.raises(TenantSuspendedError):
                    bound.read(manager, tenant.id)
                tenant.status = TenantStatus.ACTIVE
            else:
                with pytest.raises(TenantIsolationError):
                    bound.read(manager, tenant.id)


def test_attached_schema_is_verified_and_limit_failure_releases_snapshot(tmp_path):
    import sqlite3

    from truthound.datasources.base import DataSourceSizeError
    from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig
    from truthound.multitenancy.core import IsolationLevel
    from truthound.multitenancy.manager import TenantManager
    from truthound.multitenancy.sources import TenantSQLSource

    main, attached = tmp_path / 'main.db', tmp_path / 'attached.db'
    with sqlite3.connect(main) as conn:
        conn.execute('CREATE TABLE records (wrong TEXT)')
    with sqlite3.connect(attached) as conn:
        conn.execute('CREATE TABLE records (x INTEGER)')
        conn.executemany('INSERT INTO records VALUES (?)', [(1,), (2,)])

    class AttachedSource(SQLiteDataSource):
        def _create_connection(self):
            connection = super()._create_connection()
            connection.execute('ATTACH DATABASE ? AS selected', (str(attached),)).close()
            return connection

    manager = TenantManager()
    tenant = manager.create(name='First', tenant_id='first', isolation_level=IsolationLevel.SCHEMA)
    tenant.schema_name = 'selected'
    config = SQLiteDataSourceConfig(schema_name='selected', materialization_row_limit=1, pool_size=1)
    with AttachedSource(table='records', database=str(main), config=config) as source:
        assert source.columns == ['x']
        bound = TenantSQLSource(source, tenant_id=tenant.id, namespace_database=str(attached))
        with pytest.raises(DataSourceSizeError):
            bound.read(manager, tenant.id)
        with source._get_connection() as connection:
            assert not connection.in_transaction
        config.materialization_row_limit = 2
        assert bound.read(manager, tenant.id).collect()['x'].to_list() == [1, 2]
        tenant.schema_name = 'main'
        with pytest.raises(TenantIsolationError):
            bound.read(manager, tenant.id)
        with source._get_connection() as connection:
            assert not connection.in_transaction
        with pytest.raises(TenantIsolationError):
            bound.read(manager, 'outside')
    with (
        SQLiteDataSource(database=str(main), query='SELECT * FROM records') as source,
        pytest.raises(TenantIsolationError),
    ):
        TenantSQLSource(source, tenant_id=tenant.id)
