"""Owned native datasource runtime; transport selects only explicit operations."""

import contextlib
import json
import os
import sys
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


def main():
    source = None
    wire = sys.stdout
    try:
        with open(os.devnull, "w") as sink:
            while line := sys.stdin.buffer.readline(131073):
                response = {"ok": False, "code": "worker_error"}
                validation = None
                try:
                    if len(line) > 131072 or not line.endswith(b"\n"):
                        raise MCPError("resource_exhausted")
                    request = json.loads(line)
                    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                        operation = request["operation"]
                        if operation == "sources_open_polars":
                            if source is not None:
                                raise MCPError("invalid_arguments")
                            import polars as pl

                            from truthound.datasources.polars_source import (
                                PolarsDataSource,
                                PolarsDataSourceConfig,
                            )

                            source = PolarsDataSource(
                                pl.scan_ipc(request["asset_path"], memory_map=False),
                                config=PolarsDataSourceConfig(**request["config"]),
                            )
                            source.__enter__()
                        elif operation == "sources_open_sqlite":
                            if source is not None:
                                raise MCPError("invalid_arguments")
                            from truthound.datasources.sql.sqlite import (
                                SQLiteDataSource,
                                SQLiteDataSourceConfig,
                            )
                            from truthound.mcp.sqlite_snapshot import backup, table_query

                            backup(
                                request["source_path"],
                                request["snapshot_path"],
                                request["snapshot_budget"],
                                request["table"],
                            )
                            source = SQLiteDataSource(
                                database=request["snapshot_path"],
                                query=table_query(request["table"]),
                                config=SQLiteDataSourceConfig(**request["config"]),
                            )
                            source.__enter__()
                        elif operation == "sources_open_duckdb":
                            if source is not None:
                                raise MCPError("invalid_arguments")
                            from truthound.mcp.duckdb_source import open_source

                            source = open_source(
                                request["snapshot_path"], request["table"], request["config"]
                            )
                        elif source is None:
                            raise MCPError("invalid_handle")
                        elif operation == "sources_validate":
                            from truthound.mcp.sources import SourceValidation, effective_validation
                            from truthound.mcp.validation_worker import execute

                            arguments = SourceValidation.model_validate(request["arguments"])
                            validation = execute(
                                request,
                                effective_validation(arguments, request["native_asset_id"]),
                                None,
                                source=source,
                            )
                        elif operation == "sources_sample":
                            arguments = request["arguments"]
                            sampled = source.sample(arguments["n"], seed=arguments["seed"])
                            try:
                                frame = sampled.to_polars_lazyframe().collect()
                                with BoundedWriter(
                                    Path(request["output_path"]), request["output_budget"]
                                ) as sink_file:
                                    frame.write_ipc(sink_file)
                            finally:
                                if sampled is not source:
                                    sampled.__exit__(None, None, None)
                        else:
                            if operation == "sources_inspect":
                                from dataclasses import asdict

                                result = {
                                    "name": source.name,
                                    "config": asdict(source.config),
                                    "schema": {k: v.value for k, v in source.schema.items()},
                                    "columns": source.columns,
                                    "row_count": source.row_count,
                                    "capabilities": sorted(v.value for v in source.capabilities),
                                    "needs_sampling": source.needs_sampling(),
                                    "numeric_columns": source.get_numeric_columns(),
                                    "string_columns": source.get_string_columns(),
                                    "datetime_columns": source.get_datetime_columns(),
                                }
                            elif operation == "sources_validate_connection":
                                result = {"connected": source.validate_connection()}
                            elif operation == "sources_validate_columns":
                                result = {
                                    "missing_columns": source.validate_columns(
                                        request["arguments"]["required_columns"]
                                    )
                                }
                            elif operation == "sources_check_size_limits":
                                source.check_size_limits()
                                result = {"within_limits": True}
                            else:
                                raise MCPError("unsupported_operation")
                            with BoundedWriter(
                                Path(request["output_path"]), request["output_budget"]
                            ) as sink_file:
                                sink_file.write(
                                    json.dumps(encode(result), allow_nan=False).encode()
                                )
                    response = {"ok": True, "completed": True}
                    if validation is not None:
                        response["result"] = validation
                except MCPError as exc:
                    response = {"ok": False, "code": exc.code}
                except ImportError:
                    response = {"ok": False, "code": "missing_dependency"}
                except Exception:
                    response = {"ok": False, "code": "native_source_error"}
                wire.write(json.dumps(response) + "\n")
                wire.flush()
    finally:
        if source is not None:
            source.__exit__(None, None, None)


if __name__ == "__main__":
    main()
