"""Bounded child results decoded only through Core's data-only value codec."""

import json
import os
import stat
from pathlib import Path

from truthound.ml.serialization import decode, reject_constant, unique_object
from truthound.plugins.security.exceptions import SandboxError, SandboxResourceError


def read_result(path, budget, *, plugin_id=None, sandbox_id=None, memory_limit=0):
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if path.is_symlink() or not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Invalid result file")
            content = stream.read(budget + 1)
        if len(content) > budget:
            raise SandboxResourceError("Worker result exceeds IPC budget", resource_type="result")
        result = json.loads(
            content, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
        if type(result) is not dict or type(result.get("success")) is not bool:
            raise ValueError("Invalid result envelope")
        if result["success"]:
            if set(result) != {"success", "result"}:
                raise ValueError("Invalid result fields")
            return decode(result["result"])
        if (
            set(result) != {"success", "error", "stage"}
            or type(result["error"]) is not str
            or type(result["stage"]) is not str
            or result["stage"] not in {"limits", "operation", "result"}
        ):
            raise ValueError("Invalid failure envelope")
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        raise SandboxError(
            "Invalid data-only worker result", plugin_id=plugin_id, sandbox_id=sandbox_id
        ) from None
    if result["error"] == "MemoryError":
        raise SandboxResourceError(
            "Worker exhausted memory limit", resource_type="memory", limit=memory_limit
        )
    safe_types = {
        "ValueError",
        "TypeError",
        "MemoryError",
        "ImportError",
        "OSError",
        "RuntimeError",
    }
    raise SandboxError(
        "Selected worker operation failed",
        plugin_id=plugin_id,
        sandbox_id=sandbox_id,
        details={
            "stage": result["stage"],
            "error_type": result["error"] if result["error"] in safe_types else "WorkerError",
        },
    )
