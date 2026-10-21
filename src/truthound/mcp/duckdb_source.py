"""Native DuckDB computation with an explicit isolated read-only connection policy."""

import re
from importlib.metadata import PackageNotFoundError, version

from truthound.mcp.policy import MCPError


def supported_release(value):
    release = re.fullmatch(r"1\.(\d+)\.(\d+)", value)
    return release is not None and int(release.group(1)) >= 4


def dependency_available():
    """Metadata only: discovery must not initialize a native SQL runtime."""
    try:
        return supported_release(version("duckdb"))
    except PackageNotFoundError:
        return False


def open_source(database, table, config):
    # Optional providers must not be imported by catalog discovery.
    import duckdb

    from truthound.datasources.sql.duckdb import DuckDBDataSource, DuckDBDataSourceConfig

    if not supported_release(duckdb.__version__):
        raise MCPError("unsupported_dependency")

    class ReadOnlySource(DuckDBDataSource):
        # Only connection policy is adapted. Schema, SQL engine, validation,
        # bounded materialization and sampling remain native Core behavior.
        def _create_connection(self):
            cfg = self.config
            return duckdb.connect(
                database=cfg.database,
                read_only=True,
                config={
                    "enable_external_access": "false",
                    "autoload_known_extensions": "false",
                    "autoinstall_known_extensions": "false",
                    "allow_community_extensions": "false",
                    "allow_unsigned_extensions": "false",
                    "threads": str(cfg.threads or 1),
                    "memory_limit": cfg.memory_limit or f"{cfg.max_memory_mb}MB",
                    "lock_configuration": "true",
                },
            )

    cfg = DuckDBDataSourceConfig(**config)
    schema = cfg.schema_name or "main"

    def quote(value):
        return '"' + value.replace('"', '""') + '"'

    source = ReadOnlySource(
        database=database,
        read_only=True,
        query=f"SELECT * FROM {quote(schema)}.{quote(table)}",
        config=cfg,
    )
    source.__enter__()
    try:
        # Views can encode filesystem/network actions. Verify a real physical
        # table before Core inspects or executes the selected relation.
        with source._get_connection() as connection:
            relations = connection.execute(
                "SELECT table_type FROM information_schema.tables "
                "WHERE table_schema = ? AND table_name = ?",
                [schema, table],
            ).fetchall()
        if relations != [("BASE TABLE",)]:
            raise MCPError("unsupported_table")
        _ = source.schema
    except BaseException:
        source.__exit__(None, None, None)
        raise
    return source
