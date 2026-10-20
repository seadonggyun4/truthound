"""Disposable Core execution process; launched only by the local adapter."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path

from truthound.mcp.projection import project, response


def normalize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def execute(request: dict) -> dict:
    import truthound as th

    operation, path = request["operation"], request["path"]
    if operation == "check":
        from truthound.context import TruthoundContext, TruthoundContextConfig

        context = TruthoundContext(
            Path(path).parent,
            config=TruthoundContextConfig(
                persist_runs=False,
                persist_docs=False,
                auto_create_baseline=False,
                auto_create_workspace=False,
            ),
        )
        result = th.check(
            path,
            context=context,
            validators=["null", "duplicate"],
            result_format="boolean_only",
            catch_exceptions=False,
        )
    elif operation == "profile":
        result = th.profile(path)
    elif operation == "scan":
        result = th.scan(path)
    else:
        return response(operation, error="unknown_operation")
    data, truncated = project(operation, result, request["max_items"])
    return response(operation, data=normalize(data), truncated=truncated)


def main() -> None:
    request = json.loads(sys.stdin.read(16384))
    # Third-party/Core console output must not contaminate protocol output.
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        try:
            result = execute(request)
            encoded = json.dumps(result, ensure_ascii=True, allow_nan=False)
            if len(encoded.encode()) > request["max_response_bytes"]:
                result = response(request["operation"], error="response_too_large")
        except Exception:
            result = response(request.get("operation", "unknown"), error="core_error")
    sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
