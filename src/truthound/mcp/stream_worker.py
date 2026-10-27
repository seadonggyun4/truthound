"""Fixed native stream bindings; no model-selected class or network source."""

import contextlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


def main():
    source = iterator = None
    wire = sys.stdout
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            while line := sys.stdin.buffer.readline(131073):
                try:
                    if len(line) > 131072 or not line.endswith(b"\n"):
                        raise MCPError("resource_exhausted")
                    request = json.loads(line)
                    result = None
                    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                        operation = request["operation"]
                        if operation == "streams_open":
                            if source is not None:
                                raise MCPError("invalid_arguments")
                            from truthound.validators.streaming.sources import (
                                ArrowIPCStreamingSource,
                                CSVStreamingSource,
                                JSONLStreamingSource,
                                ParquetStreamingSource,
                            )

                            # Static native bindings, not arbitrary imports.
                            factory = {
                                "csv": CSVStreamingSource,
                                "jsonl": JSONLStreamingSource,
                                "parquet": ParquetStreamingSource,
                                "arrow": ArrowIPCStreamingSource,
                            }[request["format"]]
                            source = factory(request["path"], **request["config"])
                            source.__enter__()
                            iterator = iter(source)
                        elif source is None:
                            raise MCPError("invalid_handle")
                        elif operation == "streams_next":
                            frame = next(iterator, None)
                            result = {"exhausted": frame is None}
                            if frame is not None:
                                with BoundedWriter(
                                    Path(request["output_path"]), request["output_budget"]
                                ) as output:
                                    frame.write_ipc(output)
                        elif operation == "streams_inspect":
                            with BoundedWriter(
                                Path(request["output_path"]), request["output_budget"]
                            ) as output:
                                output.write(
                                    json.dumps(
                                        encode(
                                            {
                                                "config": asdict(source.config),
                                                "rows_read": source.rows_read,
                                                "is_open": source.is_open,
                                                "length": len(source),
                                            }
                                        ),
                                        allow_nan=False,
                                    ).encode()
                                )
                        else:
                            raise MCPError("unsupported_operation")
                    response = {"ok": True, "completed": True}
                    if result is not None:
                        response["result"] = result
                except MCPError as exc:
                    response = {"ok": False, "code": exc.code}
                except ImportError:
                    response = {"ok": False, "code": "missing_dependency"}
                except Exception:
                    response = {"ok": False, "code": "native_stream_error"}
                wire.write(json.dumps(response) + "\n")
                wire.flush()
    finally:
        if source is not None:
            source.__exit__(None, None, None)


if __name__ == "__main__":
    main()
