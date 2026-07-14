from __future__ import annotations

import sqlite3
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import truthound as th
from truthound.datasources.base import DataSourceSchemaError, DataSourceSizeError
from truthound.datasources.sql.base import BaseSQLDataSource, SQLDataSourceConfig
from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig


@dataclass
class MappingProxyRow:
    _mapping: dict[str, Any]


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.description: list[tuple[Any, ...]] = []
        self._rows: list[Any] = []
        self._offset = 0
        self.closed = False

    def execute(self, query: str, params: Any = None) -> None:
        self.connection.queries.append((query, params))
        if "COUNT" in query.upper():
            self.description = [("count", None)]
            self._rows = [self.connection.scalar_row]
        else:
            self.description = [(column, None) for column in self.connection.columns]
            self._rows = list(self.connection.rows)
        self._offset = 0

    def fetchone(self) -> Any:
        if not self._rows:
            return None
        return self._rows[0]

    def fetchall(self) -> list[Any]:
        self.connection.fetchall_calls += 1
        return list(self._rows)

    def fetchmany(self, size: int) -> list[Any]:
        self.connection.fetchmany_sizes.append(size)
        batch = self._rows[self._offset : self._offset + size]
        self._offset += len(batch)
        return batch

    def close(self) -> None:
        self.closed = True
        self.connection.closed_cursors += 1


class FakeConnection:
    def __init__(
        self,
        *,
        columns: list[str],
        rows: list[Any],
        scalar_row: Any,
    ) -> None:
        self.columns = columns
        self.rows = rows
        self.scalar_row = scalar_row
        self.queries: list[tuple[str, Any]] = []
        self.fetchmany_sizes: list[int] = []
        self.fetchall_calls = 0
        self.closed_cursors = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        pass


class FakeSQLDataSource(BaseSQLDataSource):
    source_type = "fake"

    def __init__(
        self,
        connection: FakeConnection,
        *,
        config: SQLDataSourceConfig | None = None,
        query: str | None = None,
    ) -> None:
        self.connection = connection
        super().__init__(
            table=None if query is not None else "records",
            query=query,
            config=config or SQLDataSourceConfig(),
        )

    def _create_connection(self) -> FakeConnection:
        return self.connection

    def _get_table_schema_query(self) -> str:
        return "SELECT column_name, data_type FROM fake_schema"

    def _get_row_count_query(self) -> str:
        return "SELECT COUNT(*) FROM records"

    def _quote_identifier(self, identifier: str) -> str:
        return f'"{identifier}"'


@pytest.mark.parametrize(
    ("scalar_row", "expected"),
    [
        ((5,), 5),
        ({"count": 5}, 5),
        (MappingProxyRow({"count": 5}), 5),
    ],
)
def test_row_count_accepts_tuple_mapping_and_mapping_proxy(
    scalar_row: Any,
    expected: int,
) -> None:
    connection = FakeConnection(columns=["id"], rows=[], scalar_row=scalar_row)
    source = FakeSQLDataSource(connection)

    assert source.row_count == expected
    assert connection.closed_cursors == 1


@pytest.mark.parametrize(
    "rows",
    [
        [(1, "alpha"), (2, "beta")],
        [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}],
        [
            MappingProxyRow({"id": 1, "name": "alpha"}),
            MappingProxyRow({"id": 2, "name": "beta"}),
        ],
    ],
)
def test_execute_query_normalizes_supported_row_shapes(rows: list[Any]) -> None:
    connection = FakeConnection(
        columns=["id", "name"],
        rows=rows,
        scalar_row=(len(rows),),
    )
    source = FakeSQLDataSource(connection)

    assert source.execute_query("SELECT id, name FROM records") == [
        {"id": 1, "name": "alpha"},
        {"id": 2, "name": "beta"},
    ]


def test_execute_query_rejects_duplicate_columns_and_closes_cursor() -> None:
    connection = FakeConnection(
        columns=["id", "id"],
        rows=[(1, 2)],
        scalar_row=(1,),
    )
    source = FakeSQLDataSource(connection)

    with pytest.raises(DataSourceSchemaError, match="duplicate column"):
        source.execute_query("SELECT left_id AS id, right_id AS id FROM records")

    assert connection.closed_cursors == 1


def test_polars_materialization_uses_bounded_fetchmany() -> None:
    rows = [{"id": index, "name": f"row-{index}"} for index in range(5)]
    connection = FakeConnection(
        columns=["id", "name"],
        rows=rows,
        scalar_row={"count": len(rows)},
    )
    source = FakeSQLDataSource(
        connection,
        config=SQLDataSourceConfig(
            fetch_size=2,
            max_rows=100,
            materialization_row_limit=5,
        ),
    )

    frame = source.to_polars_lazyframe().collect()

    assert frame.to_dicts() == rows
    assert connection.fetchall_calls == 0
    assert connection.fetchmany_sizes == [2, 2, 2, 1]
    assert any("LIMIT 6" in query for query, _ in connection.queries)


def test_polars_materialization_rejects_known_oversize_before_read() -> None:
    connection = FakeConnection(
        columns=["id"],
        rows=[{"id": index} for index in range(6)],
        scalar_row={"count": 6},
    )
    source = FakeSQLDataSource(
        connection,
        config=SQLDataSourceConfig(materialization_row_limit=5),
    )

    with pytest.raises(DataSourceSizeError, match="materialization"):
        source.to_polars_lazyframe()

    assert all("SELECT *" not in query for query, _ in connection.queries)


def test_polars_materialization_rejects_concurrent_growth_without_truncation() -> None:
    connection = FakeConnection(
        columns=["id"],
        rows=[{"id": index} for index in range(6)],
        scalar_row={"count": 5},
    )
    source = FakeSQLDataSource(
        connection,
        config=SQLDataSourceConfig(fetch_size=3, materialization_row_limit=5),
    )

    with pytest.raises(DataSourceSizeError, match="materialization"):
        source.to_polars_lazyframe()

    assert connection.fetchall_calls == 0


@pytest.mark.parametrize(
    ("dialect", "fragment"),
    [
        ("limit", "LIMIT 11"),
        ("top", "TOP (11)"),
        ("rownum", "ROWNUM <= 11"),
    ],
)
def test_materialization_query_uses_provider_dialect(
    dialect: str,
    fragment: str,
) -> None:
    connection = FakeConnection(columns=["id"], rows=[], scalar_row=(0,))
    source = FakeSQLDataSource(connection)
    source.materialization_dialect = dialect

    assert fragment in source._build_bounded_materialization_query(11)


def test_sampled_source_uses_parent_subquery_alias_dialect() -> None:
    connection = FakeConnection(columns=["id"], rows=[], scalar_row=(0,))
    source = FakeSQLDataSource(connection)
    source.materialization_dialect = "rownum"
    source.subquery_alias_keyword = ""

    sampled = source.sample(10)

    assert sampled.full_table_name.endswith(") sampled")
    assert ") AS sampled" not in sampled.full_table_name
    assert "ROWNUM <= 10" in sampled.full_table_name


def test_sqlite_mapping_rows_and_query_mode_sample_are_materialized_safely(
    tmp_path: Path,
) -> None:
    database = tmp_path / "provider-contract.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE records (id INTEGER, name TEXT)")
        connection.executemany(
            "INSERT INTO records VALUES (?, ?)",
            [(index, f"row-{index}") for index in range(5)],
        )
        connection.commit()
    finally:
        connection.close()

    config = SQLiteDataSourceConfig(
        database=str(database),
        fetch_size=2,
        materialization_row_limit=10,
    )
    source = SQLiteDataSource(
        database=str(database),
        query="SELECT id, name FROM records ORDER BY id;",
        config=config,
    )

    assert source.row_count == 5
    assert source.to_polars_lazyframe().collect().height == 5
    assert source.sample(2).to_polars_lazyframe().collect().height == 2


def test_sqlite_source_completes_public_check_and_profile_contract(
    tmp_path: Path,
) -> None:
    database = tmp_path / "public-api-contract.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE records (id INTEGER, name TEXT)")
        connection.executemany(
            "INSERT INTO records VALUES (?, ?)",
            [(1, "alpha"), (2, "beta")],
        )
        connection.commit()
    finally:
        connection.close()

    source = SQLiteDataSource(table="records", database=str(database))

    validation = th.check(source=source)
    profile = th.profile(source=source)

    assert type(validation).__name__ == "ValidationRunResult"
    assert profile.row_count == 2
    assert profile.column_count == 2


def test_provider_extras_cover_the_supported_sql_matrix() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]

    expected = {
        "postgresql",
        "mysql",
        "duckdb",
        "snowflake",
        "bigquery",
        "redshift",
        "databricks",
        "oracle",
        "sqlserver",
        "sql-connectors",
    }
    assert expected <= set(extras)
