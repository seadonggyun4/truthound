"""Native SQLite query cancellation must leave its pooled connection reusable."""
import sqlite3
import pytest
from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig


def test_query_timeout_interrupts_native_work_and_allows_next_query(tmp_path):
    path = tmp_path / 'timeout.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE records (x INTEGER)')
    source = SQLiteDataSource(database=str(path), table='records',
        config=SQLiteDataSourceConfig(query_timeout=0.001, pool_size=1))
    try:
        with pytest.raises(sqlite3.OperationalError, match='interrupted'):
            source.execute_scalar('WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n WHERE x<1000000) SELECT sum(x) FROM n')
        assert source.execute_scalar('SELECT 42') == 42
    finally:
        source._disconnect()
