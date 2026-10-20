"""Synthetic long worker, only launched by the private test server fixture."""

import contextlib
import json
import os
import sys
import time
from pathlib import Path

request = json.load(sys.stdin)
source = Path(request["path"])
if source.read_text(encoding="utf-8").splitlines()[0] == "hold":
    state = Path(sys.argv[1]) / "ready.json"
    state.write_text(
        json.dumps({"pid": os.getpid(), "directory": str(source.parent)}), encoding="utf-8"
    )
    while True:
        time.sleep(0.1)
else:
    from truthound.mcp.worker import execute

    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        result = execute(request)
    print(json.dumps(result, allow_nan=False), flush=True)
