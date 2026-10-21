"""Native retained validation and approved installed-extension lifetimes."""

import json
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.plugin_bindings import approved_capabilities
from truthound.mcp.resources import BoundedWriter


def execute(request, arguments, frame, *, source=None):
    import truthound as th
    from truthound.context import TruthoundContext, TruthoundContextConfig
    from truthound.mcp.operations import _context
    from truthound.mcp.policy import MCPError
    from truthound.mcp.projection import project
    from truthound.mcp.tables import decode
    from truthound.mcp.validator_wire import constructor, settings, spec
    from truthound.schema import Schema

    suite = None
    if request["operation"] == "suites_validate":
        from truthound.mcp.suites import SuiteSpec, effective_arguments

        suite = SuiteSpec.model_validate(request["kernel_suite"])
        arguments = effective_arguments(arguments, suite)

    with approved_capabilities(request.get("plugin_bindings", {})) as callbacks:
        configured = request.get("constructor_settings")
        if configured is None:
            configured = [check.settings for check in arguments.checks or ()]
        parameters = request.get("parameter_objects", [{} for _ in arguments.checks or ()])

        def build(index, check):
            objects = parameters[index]
            try:
                kwargs = settings(
                    check.validator,
                    configured[index],
                    frames=request.get("frames", {}),
                    build=True,
                    trusted=index in request.get("trusted_indices", []),
                    callbacks=callbacks,
                    partial=bool(objects),
                )
            except (ValueError, TypeError):
                # Native nested config records reject invalid semantic values
                # before the validator constructor itself is entered.
                raise MCPError("validator_configuration_error") from None
            for key, identity in objects.items():
                if (
                    key in kwargs
                    or key not in spec(check.validator)["fields"]
                    or identity not in callbacks
                ):
                    raise MCPError("invalid_binding")
                kwargs[key] = callbacks[identity]
            native = constructor(check.validator)
            try:
                return native(**kwargs)
            except (ValueError, TypeError):
                # Stable failure phase without constructor messages that may
                # contain sensitive settings. No algorithm was run.
                raise MCPError("validator_configuration_error") from None

        validators = (
            None
            if arguments.checks is None
            else [build(index, check) for index, check in enumerate(arguments.checks)]
        )
        schema = None
        if "schema_path" in request:
            schema = Schema.from_dict(
                decode(json.loads(Path(request["schema_path"]).read_text(encoding="utf-8")))
            )
        elif arguments.schema_spec is not None:
            schema = Schema.from_dict(arguments.schema_spec.model_dump(exclude_none=True))
        context = (
            TruthoundContext(
                Path(request["context"]["workspace_root"]),
                config=TruthoundContextConfig(**request["context"]["config"]),
            )
            if "context" in request
            else _context(request["scratch_path"])
        )
        options = arguments.model_dump(
            exclude={
                "asset_id",
                "path",
                "checks",
                "schema_spec",
                "schema_id",
                "context_id",
                "idempotency_key",
                "evidence",
            }
        )
        if suite is not None:
            from truthound.mcp.suites import run_suite

            result = run_suite(suite, validators, frame, arguments, context, schema)
        else:
            result = th.check(
                frame,
                source=source,
                validators=validators,
                schema=schema,
                context=context,
                **options,
            )
        summary, truncated = project("check", result, request["max_items"])
        # Registry discovery does not establish target/skip accounting for all
        # algorithms. Never infer a quality pass from absent Core issues alone.
        coverage = {
            "rows": result.row_count,
            "columns": result.column_count,
            "total_checks": len(result.checks),
            "returned_checks": len(summary["checks"]),
            "truncated": truncated,
            "execution_issue_count": len(result.execution_issues),
            "accounting_complete": False,
        }
        with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
            sink.write(json.dumps(encode(result.to_dict()), allow_nan=False).encode())
        return {
            "validation_created": True,
            "summary": summary,
            "coverage": coverage,
            "core_execution_failed": bool(result.execution_issues),
        }
