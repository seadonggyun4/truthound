"""Bounded native SQLite backup shared by materialized and resident adapters."""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from truthound.mcp.policy import MCPError


def backup(source_path, snapshot_path, budget, table):
    # SQLite read-only WAL connections may write shared-memory bookkeeping.
    # Admit rollback-journal databases only; never silently switch modes.
    with os.fdopen(
        os.open(source_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb"
    ) as stream:
        header = stream.read(100)
    if len(header) != 100 or header[:16] != b"SQLite format 3\x00":
        raise MCPError("unsupported_format")
    if header[18] != 1 or header[19] != 1:
        raise MCPError("operator_configuration_required")
    with closing(sqlite3.connect(Path(source_path).as_uri() + "?mode=ro", uri=True)) as original:
        original.execute("PRAGMA query_only = ON")
        page_size = original.execute("PRAGMA page_size").fetchone()[0]

        def check_backup(status, remaining, total):
            if total * page_size > budget:
                raise MCPError("file_too_large")

        with closing(sqlite3.connect(snapshot_path)) as snapshot:
            original.backup(snapshot, pages=128, progress=check_backup)
            if (
                snapshot.execute(
                    "SELECT 1 FROM sqlite_schema WHERE name = ? AND type = 'table'", (table,)
                ).fetchone()
                is None
            ):
                raise MCPError("invalid_arguments")


def table_query(table):
    # Single literal identifier, never agent-supplied SQL or an ATTACH target.
    return 'SELECT * FROM "' + table.replace('"', '""') + '"'
