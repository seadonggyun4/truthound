"""Closed native file-history snapshots; never extract caller archive paths."""

import json
import re
from pathlib import Path

from truthound.mcp.policy import MCPError

VERSION_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def validate_files(files, maximum):
    from truthound.profiler.evolution.breaking_alerts import BreakingChangeAlert
    from truthound.profiler.evolution.history import SchemaVersion

    if not isinstance(files, dict) or any(not isinstance(v, str) for v in files.values()):
        raise MCPError("invalid_control_state")
    index = json.loads(files["index.json"])
    if (
        not isinstance(index, list)
        or len(index) > maximum
        or any(not isinstance(v, str) or not VERSION_ID.fullmatch(v) for v in index)
        or len(set(index)) != len(index)
    ):
        raise MCPError("invalid_control_state")
    expected = {"index.json", "alerts.json"} | {f"versions/{v}.json" for v in index}
    if set(files) != expected:
        raise MCPError("invalid_control_state")
    for identity in index:
        data = json.loads(files[f"versions/{identity}.json"])
        if data["version_id"] != identity:
            raise MCPError("invalid_control_state")
        SchemaVersion.from_dict(data)
    alerts = json.loads(files["alerts.json"])
    if not isinstance(alerts, list) or len(alerts) > maximum:
        raise MCPError("invalid_control_state")
    for alert in alerts:
        BreakingChangeAlert.from_dict(alert)
    return files


def snapshot(root, configuration):
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise MCPError("invalid_control_state")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    files.setdefault("index.json", "[]")
    files.setdefault("alerts.json", "[]")
    validate_files(files, configuration.max_versions)
    return {"format": 1, "configuration": configuration.model_dump(), "files": files}


def restore(path, root, configuration):
    from truthound.mcp.tables import decode

    data = decode(json.loads(Path(path).read_text(encoding="utf-8")))
    if (
        set(data) != {"format", "configuration", "files"}
        or data["format"] != 1
        or data["configuration"] != configuration.model_dump()
        or configuration.backend != "file"
    ):
        raise MCPError("invalid_control_state")
    files = validate_files(data["files"], configuration.max_versions)
    for name, value in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
