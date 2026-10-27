"""Consume native registered ports in their actual approved manager owner."""

import json
from pathlib import Path
from uuid import uuid4

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.tables import decode


def hook_rows(rows):
    return [
        {
            "hook_type": row.hook_type,
            "source": row.source,
            "success": row.success,
            "result": row.result,
            "error_type": row.error_type,
        }
        for row in rows
    ]


def hook_bundle(bundle):
    return {
        "hooks": {name: hook_rows(rows) for name, rows in bundle.hooks.items()},
        "hooks_failed": bundle.hooks_failed,
    }


async def execute(owner, args, budget, *, inputs=None):
    from truthound.core.results import ValidationRunResult

    manager, command = owner.runtime.manager, args["command"]
    if command == "provider_inspect":
        from truthound.plugins.workflows import validation_asset

        asset = validation_asset(manager, asset_provider=args["alias"])
        return {
            "provider_id": args["alias"],
            "row_count": asset.row_count,
            "column_count": asset.column_count,
            "backend": asset.backend_name,
        }
    if command == "validate":
        if len(owner.native_runs) >= 16:
            raise MCPError("resource_exhausted")
        data = None
        if args.get("asset_id"):
            import polars as pl

            if not inputs or set(inputs) != {"asset_path"}:
                raise MCPError("invalid_handle")
            data = pl.scan_ipc(inputs["asset_path"], memory_map=False)
        bundle = await manager.validate_async(
            args["check_aliases"],
            data,
            asset_provider=args.get("provider_id"),
            name=args.get("suite_name", "plugin-suite"),
            result_format=args.get("result_format", "summary"),
            parallel=args.get("parallel", False),
            max_workers=args.get("max_workers"),
            pushdown=args.get("pushdown"),
        )
        payload = {"run": bundle.run.to_dict(), **hook_bundle(bundle)}
        size = len(json.dumps(encode(payload), allow_nan=False).encode())
        if size + sum(row[2] for row in owner.native_runs.values()) > budget:
            raise MCPError("resource_exhausted")
        identity = "plugin_run_" + uuid4().hex
        owner.native_runs[identity] = (bundle.run, payload, size)
        return {"run_id": identity, **payload}
    if command in {"run_get", "run_export", "run_release"}:
        identity = args["run_id"]
        if identity not in owner.native_runs:
            raise MCPError("invalid_handle")
        if command == "run_release":
            del owner.native_runs[identity]
            return {"released": True}
        run, payload, _ = owner.native_runs[identity]
        if command == "run_export":
            return run.to_dict()
        return {"run_id": identity, **payload, "run": run.to_dict()}
    if command == "render":
        if args.get("run_id"):
            row = owner.native_runs.get(args["run_id"])
            if row is None:
                raise MCPError("invalid_handle")
            run = row[0]
        else:
            if not inputs or set(inputs) != {"run_path"}:
                raise MCPError("invalid_handle")
            run = ValidationRunResult.from_dict(
                decode(json.loads(Path(inputs["run_path"]).read_text(encoding="utf-8")))
            )
        result = await manager.render_async(args["alias"], run)
        return {"reporter_id": args["alias"], "content": result.content, **hook_bundle(result)}
    raise MCPError("invalid_arguments")
