"""Public Core facades applied to the actual resident datasource."""

import json
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter

OUTPUTS = {
    "sources_read": ("asset", "application/vnd.apache.arrow.file"),
    "sources_mask": ("asset", "application/vnd.apache.arrow.file"),
    "sources_profile": ("artifact", "application/vnd.truthound.profile+json"),
    "sources_scan": ("artifact", "application/vnd.truthound.pii+json"),
    "sources_learn": ("schema", "application/vnd.truthound.schema+json"),
}


def execute(operation, source, arguments, output_path, output_budget):
    import truthound as th

    if operation == "sources_read":
        result = th.read(source, sample_size=arguments["sample_size"])
    elif operation == "sources_mask":
        result = th.mask(
            source=source,
            columns=arguments["columns"],
            strategy=arguments["strategy"],
            strict=arguments["strict"],
        )
    elif operation == "sources_profile":
        result = th.profile(source=source)
    elif operation == "sources_scan":
        result = th.scan(source=source)
    elif operation == "sources_learn":
        result = th.learn(
            source=source,
            infer_constraints=arguments["infer_constraints"],
            categorical_threshold=arguments["categorical_threshold"],
        )
    else:
        raise MCPError("unsupported_operation")
    with BoundedWriter(Path(output_path), output_budget) as sink:
        if OUTPUTS[operation][0] == "asset":
            result.write_ipc(sink)
        else:
            sink.write(json.dumps(encode(result.to_dict()), allow_nan=False).encode())
