"""Named retained-data bindings executed only in the owned disposable worker."""

from __future__ import annotations

import json
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


def execute(request: dict) -> dict:
    import polars as pl

    import truthound as th
    from truthound.mcp.catalog import CATALOG
    from truthound.mcp.tables import inline_table, page

    operation = request["operation"]
    arguments = CATALOG[operation].input_type.model_validate(request["arguments"])
    if operation == "schedules_preview":
        from truthound.mcp.scheduling import execute as schedule_execute

        return schedule_execute(operation, arguments)
    if operation.startswith("schema_audit_"):
        from truthound.mcp.schema_audit import execute as audit_execute

        return audit_execute(request, arguments)
    if operation == "contexts_execute":
        from truthound.mcp.contexts import execute as execute_context

        return execute_context(request, arguments)
    if operation.startswith("lineage_"):
        from truthound.mcp.lineage_worker import execute as lineage_execute

        return lineage_execute(request, arguments)
    if operation in {"drift_compare", "drift_detect"}:
        from truthound.drift import compare

        baseline = pl.read_ipc(request["baseline_path"], memory_map=False)
        current = pl.read_ipc(request["current_path"], memory_map=False)
        if operation == "drift_compare":
            result = compare(
                baseline, current, **arguments.model_dump(exclude={"baseline_id", "current_id"})
            )
        else:
            from truthound.mcp.validator_wire import _known_type, contracts, fields

            row = contracts()["drift_detectors"][arguments.detector]
            detector = _known_type(row["target"])(
                **fields(row["fields"], arguments.settings, build=True)
            )
            result = detector.detect(
                baseline.get_column(arguments.baseline_column),
                current.get_column(arguments.current_column or arguments.baseline_column),
            )
        with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
            sink.write(json.dumps(encode(result.to_dict()), allow_nan=False).encode())
        return {
            "resource_created": True,
            "media_type": "application/vnd.truthound.drift+json",
            "row_count": current.height,
            "column_count": current.width,
        }
    if operation == "contexts_initialize":
        from truthound.context import TruthoundContext, TruthoundContextConfig

        context = TruthoundContext(
            Path(request["context"]["workspace_root"]),
            config=TruthoundContextConfig(**request["context"]["config"]),
        )
        context.ensure_workspace()
        return {"workspace_initialized": True}
    frame = None
    if "asset_path" in request:
        frame = pl.read_ipc(request["asset_path"], memory_map=False)
    if operation in {"assets_validate", "suites_validate"}:
        from truthound.mcp.validation_worker import execute as validate

        return validate(request, arguments, frame)
    produced = None
    media_type = "application/vnd.apache.arrow.file"
    if operation in {"schemas_create", "schemas_export"}:
        from truthound.mcp.tables import decode
        from truthound.schema import Schema

        schema = Schema.from_dict(
            arguments.schema_spec.model_dump(exclude_none=True)
            if operation == "schemas_create"
            else decode(json.loads(Path(request["schema_path"]).read_text(encoding="utf-8")))
        )
        if operation == "schemas_export" and arguments.format == "yaml":
            import tempfile

            with tempfile.TemporaryDirectory(prefix="truthound-mcp-schema-") as directory:
                path = Path(directory) / "schema.yaml"
                schema.save(path)
                with (
                    BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink,
                    path.open("rb") as source,
                ):
                    while chunk := source.read(65536):
                        sink.write(chunk)
            media_type = "application/yaml"
        else:
            with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
                sink.write(json.dumps(encode(schema.to_dict()), allow_nan=False).encode())
            media_type = "application/vnd.truthound.schema+json"
        return {
            "resource_created": True,
            "media_type": media_type,
            "row_count": schema.row_count,
            "column_count": len(schema.columns),
        }
    if operation in {"artifacts_render", "artifacts_datadocs"}:
        from truthound.core.results import ValidationRunResult
        from truthound.mcp.tables import decode

        run = ValidationRunResult.from_dict(
            decode(json.loads(Path(request["run_path"]).read_text(encoding="utf-8")))
        )
        if operation == "artifacts_render":
            from truthound.mcp.validator_wire import contracts, fields, record_fields
            from truthound.reporters import get_reporter

            if arguments.format == "html":
                from importlib.util import find_spec

                if find_spec("jinja2") is None:
                    raise MCPError("missing_dependency")
            config = fields(
                record_fields(contracts()["reporters"][arguments.format]),
                arguments.settings,
                build=True,
            )
            reporter = get_reporter(arguments.format, **config)
            text, media_type = reporter.render(run), reporter.content_type
        else:
            from truthound.datadocs import generate_validation_report

            text = generate_validation_report(run, **arguments.model_dump(exclude={"resource_id"}))
            media_type = "text/html"
        with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
            sink.write(text.encode("utf-8"))
        return {
            "resource_created": True,
            "media_type": media_type,
            "row_count": run.row_count,
            "column_count": run.column_count,
        }
    if operation in {"assets_profile", "assets_scan", "profilers_profile"}:
        if operation == "profilers_profile":
            from truthound.mcp.profiling import execute as profile

            result = profile(arguments, frame)
        else:
            result = th.profile(frame) if operation == "assets_profile" else th.scan(frame)
        with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
            sink.write(json.dumps(encode(result.to_dict()), allow_nan=False).encode())
        media_type = (
            "application/vnd.truthound."
            + (
                "advanced-profile"
                if operation == "profilers_profile"
                else "profile"
                if operation == "assets_profile"
                else "pii"
            )
            + "+json"
        )
    elif operation == "assets_sqlite":
        from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig
        from truthound.mcp.sqlite_snapshot import backup, table_query

        backup(
            request["source_path"],
            request["snapshot_path"],
            request["snapshot_budget"],
            arguments.table,
        )
        source = SQLiteDataSource(
            database=request["snapshot_path"],
            query=table_query(arguments.table),
            config=SQLiteDataSourceConfig(
                materialization_row_limit=arguments.materialization_row_limit,
                fetch_size=arguments.fetch_size,
            ),
        )
        with source:
            if arguments.sample_size is not None:
                with source.sample(arguments.sample_size, seed=arguments.sample_seed) as sampled:
                    produced = th.read(sampled)
            else:
                produced = th.read(source)
    elif operation == "assets_capture":
        from truthound.datasources.polars_source import FileDataSource, FileDataSourceConfig

        frame = th.read(
            FileDataSource(
                request["path"], config=FileDataSourceConfig(**arguments.options.model_dump())
            ),
            sample_size=arguments.sample_size,
        )
        produced = frame
    elif operation == "assets_inline":
        produced = inline_table(arguments.columns, maximum_bytes=request["output_budget"])
    elif operation == "assets_sample":
        from truthound.datasources import PolarsDataSource

        produced = (
            PolarsDataSource(frame)
            .sample(arguments.n, seed=arguments.seed)
            .to_polars_lazyframe()
            .collect()
        )
    elif operation == "assets_mask":
        produced = th.mask(
            frame, columns=arguments.columns, strategy=arguments.strategy, strict=arguments.strict
        )
    elif operation == "assets_page":
        bounded = page(frame, arguments.offset, arguments.limit)
        end = arguments.offset + len(bounded)
        return envelope(
            operation,
            data={
                "encoding": "tagged-v1",
                "rows": bounded,
                "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
                "total_rows": frame.height,
                "next_offset": end if end < frame.height else None,
            },
            coverage={
                "rows": frame.height,
                "returned_rows": len(bounded),
                "truncated": arguments.offset > 0 or end < frame.height,
            },
            provenance=request["provenance"],
        )
    elif operation == "schemas_learn":
        schema = th.learn(
            frame,
            infer_constraints=arguments.infer_constraints,
            categorical_threshold=arguments.categorical_threshold,
        )
        with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
            sink.write(json.dumps(encode(schema.to_dict()), allow_nan=False).encode())
        media_type = "application/vnd.truthound.schema+json"
    elif operation == "artifacts_export":
        if arguments.format == "ipc":
            produced = frame
        else:
            media_type = {
                "csv": "text/csv",
                "json": "application/json",
                "ndjson": "application/x-ndjson",
                "parquet": "application/vnd.apache.parquet",
            }[arguments.format]
            with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
                if arguments.format == "parquet":
                    frame.write_parquet(sink)
                elif arguments.format == "csv":
                    sink.write(frame.write_csv().encode())
                elif arguments.format == "json":
                    sink.write(frame.write_json().encode())
                else:
                    sink.write(frame.write_ndjson().encode())
    else:
        raise MCPError("unknown_operation")
    if produced is not None:
        with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
            produced.write_ipc(sink)
        frame = produced
    # Private IPC metadata is consumed by the parent and never emitted as a tool result.
    return {
        "resource_created": True,
        "media_type": media_type,
        "row_count": frame.height,
        "column_count": frame.width,
    }
