"""Native HTML reports from complete retained profiles and fixed private slots."""

import hashlib

from truthound.mcp.broker_worker import NativeServiceBytes
from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def execute(owner, args, budget):
    from truthound.profiler.visualization import ProfileDataConverter, ReportExporter

    profile = selected(owner.profiles, args["profile_id"])
    if not profile.complete:
        raise MCPError("native_profile_incomplete")
    generator = selected(owner.runtime.visualizers, args["visualizer_id"])
    owner.guard()
    data = ProfileDataConverter.from_dict(profile.to_dict())
    if args.get("export_format", "html") == "pdf":
        content = ReportExporter(generator.config).render_pdf(data)
    else:
        text = generator.generate(data)
        content = text.encode("utf-8")
    if len(content) > budget:
        raise MCPError("resource_exhausted")
    owner.guard()
    if args["command"] == "visualization_export":
        return NativeServiceBytes(content)
    target = selected(owner.runtime.visualization_slots, args["visualization_slot_id"])
    from truthound.stores._filesystem_io import publish
    target.parent.mkdir(parents=True, exist_ok=True)
    publish(target, content)
    owner.guard()
    return {"saved": True, "visualization_slot_id": args["visualization_slot_id"],
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
