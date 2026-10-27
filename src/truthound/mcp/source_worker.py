"""Owned native datasource runtime; transport selects only explicit operations."""

import contextlib
import json
import os
import sys
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.native_stdio import native_worker_wire
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter
from truthound.mcp.source_facades import OUTPUTS


def main():
    source = None
    children = {}
    plugins = contextlib.ExitStack()
    bound = False
    wire = native_worker_wire()
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            while line := sys.stdin.buffer.readline(131073):
                response = {"ok": False, "code": "worker_error"}
                validation = None
                try:
                    if len(line) > 131072 or not line.endswith(b"\n"):
                        raise MCPError("resource_exhausted")
                    request = json.loads(line)
                    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                        operation = request["operation"]
                        key = request.get("source_key", "root")
                        active = source if key == "root" else children.get(key)
                        if operation == "sources_open_binding":
                            if source is not None or bound:
                                raise MCPError("invalid_arguments")
                            from truthound.datasources.adapters import AsyncToSyncAdapter
                            from truthound.datasources.async_base import AsyncBaseDataSource
                            from truthound.datasources.base import BaseDataSource
                            from truthound.mcp.plugin_bindings import approved_capabilities

                            bound = True
                            capabilities = plugins.enter_context(
                                approved_capabilities(request["plugins"])
                            )
                            factory = capabilities[request["capability"]]
                            if not callable(factory):
                                raise MCPError("invalid_binding")
                            source = factory()
                            if isinstance(source, AsyncBaseDataSource):
                                source = AsyncToSyncAdapter(source)
                            if not isinstance(source, BaseDataSource):
                                source = None
                                raise MCPError("invalid_binding")
                            if source.name != request["public_name"]:
                                raise MCPError("invalid_binding")
                            source.__enter__()
                        elif operation == "sources_open_polars":
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
                        elif operation == "sources_close_child":
                            if key not in children:
                                raise MCPError("invalid_handle")
                            child = children[key]
                            if child is not source and all(
                                child is not item
                                for alias, item in children.items()
                                if alias != key
                            ):
                                child.__exit__(None, None, None)
                            del children[key]
                        elif active is None:
                            raise MCPError("invalid_handle")
                        elif operation == "sources_validate":
                            from truthound.mcp.sources import SourceValidation, effective_validation
                            from truthound.mcp.validation_worker import execute

                            arguments = SourceValidation.model_validate(request["arguments"])
                            validation = execute(
                                request,
                                effective_validation(arguments, request["native_asset_id"]),
                                None,
                                source=active,
                            )
                        elif operation in OUTPUTS:
                            from truthound.mcp.source_facades import execute

                            execute(
                                operation, active, request["arguments"],
                                request["output_path"], request["output_budget"],
                            )
                        elif operation == "sources_sample_source":
                            if len(children) >= 16 or request["sample_key"] in children:
                                raise MCPError("resource_exhausted")
                            from truthound.datasources.base import BaseDataSource

                            arguments = request["arguments"]
                            sampled = active.sample(arguments["n"], seed=arguments["seed"])
                            if not isinstance(sampled, BaseDataSource):
                                raise MCPError("unsupported_value")
                            fresh = sampled is not source and all(
                                sampled is not value for value in children.values()
                            )
                            if fresh:
                                try:
                                    sampled.__enter__()
                                except BaseException:
                                    with contextlib.suppress(Exception):
                                        sampled.__exit__(None, None, None)
                                    raise
                            children[request["sample_key"]] = sampled
                        elif operation == "sources_sample":
                            arguments = request["arguments"]
                            sampled = active.sample(arguments["n"], seed=arguments["seed"])
                            fresh = sampled is not source and all(
                                sampled is not value for value in children.values()
                            )
                            try:
                                if fresh:
                                    sampled.__enter__()
                                frame = sampled.to_polars_lazyframe().collect()
                                with BoundedWriter(
                                    Path(request["output_path"]), request["output_budget"]
                                ) as sink_file:
                                    frame.write_ipc(sink_file)
                            finally:
                                if fresh:
                                    sampled.__exit__(None, None, None)
                        else:
                            if operation == "sources_inspect":
                                from dataclasses import asdict

                                result = {
                                    "name": active.name,
                                    "config": None if bound else asdict(active.config),
                                    "schema": {k: v.value for k, v in active.schema.items()},
                                    "columns": active.columns,
                                    "row_count": active.row_count,
                                    "capabilities": sorted(v.value for v in active.capabilities),
                                    "needs_sampling": active.needs_sampling(),
                                    "numeric_columns": active.get_numeric_columns(),
                                    "string_columns": active.get_string_columns(),
                                    "datetime_columns": active.get_datetime_columns(),
                                }
                            elif operation == "sources_validate_connection":
                                result = {"connected": active.validate_connection()}
                            elif operation == "sources_validate_columns":
                                result = {
                                    "missing_columns": active.validate_columns(
                                        request["arguments"]["required_columns"]
                                    )
                                }
                            elif operation == "sources_check_size_limits":
                                active.check_size_limits()
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
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
            contextlib.ExitStack() as cleanup,
        ):
            cleanup.callback(plugins.close)
            seen = set()
            if source is not None:
                cleanup.callback(source.__exit__, None, None, None)
                seen.add(id(source))
            for child in children.values():
                if id(child) not in seen:
                    cleanup.callback(child.__exit__, None, None, None)
                    seen.add(id(child))


if __name__ == "__main__":
    main()
