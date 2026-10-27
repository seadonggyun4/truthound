"""Deterministic, allowlisted MCPB packaging from the installed Core release."""

from __future__ import annotations

import importlib.metadata
import json
import zipfile
from email.parser import BytesParser
from pathlib import Path


def _license() -> str:
    dist = importlib.metadata.distribution("truthound")
    for file in dist.files or ():
        if file.name == "LICENSE" and "dist-info" in str(file):
            return dist.locate_file(file).read_text(encoding="utf-8")
    # Source/editable build. Never recursively include the checkout.
    candidate = Path(__file__).resolve().parents[3] / "LICENSE"
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    raise ValueError("Truthound distribution license was not found")


def build_bundle(output: Path, *, wheel: Path | None = None, profile: str = "summary") -> Path:
    from truthound import __version__

    if profile not in {"summary", "core"}:
        raise ValueError("Invalid MCP profile")
    output = Path(output)
    if output.suffix != ".mcpb":
        raise ValueError("Output must have .mcpb suffix")
    files: dict[str, bytes] = {}
    source_config = ""
    if wheel is not None:
        wheel = Path(wheel)
        # Validate exact package identity without extracting archive paths.
        with zipfile.ZipFile(wheel) as z:
            metadata = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("Expected one wheel metadata record")
            identity = BytesParser().parsebytes(z.read(metadata[0]))
            if identity["Name"] != "truthound" or identity["Version"] != __version__:
                raise ValueError("Wheel identity must match installed Truthound")
        if not wheel.name.startswith("truthound-") or not wheel.name.endswith(".whl"):
            raise ValueError("Invalid wheel filename")
        relative = "wheels/" + wheel.name
        files[relative] = wheel.read_bytes()
        source_config = "\n[tool.uv.sources]\ntruthound = { path = " + json.dumps(relative) + " }\n"
    manifest = {
        "manifest_version": "0.4",
        "name": "truthound",
        "display_name": "Truthound",
        "version": __version__,
        "description": "Local data quality checks, aggregate profiles and sampled PII detection.",
        "author": {"name": "Truthound contributors"},
        "license": "Apache-2.0",
        "homepage": "https://github.com/seadonggyun4/truthound",
        "server": {
            "type": "uv",
            "entry_point": "server.py",
            "mcp_config": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    "${__dirname}",
                    "server.py",
                    "serve",
                    "--allow-dir",
                    "${user_config.allowed_directory}",
                ],
            },
        },
        "user_config": {
            "allowed_directory": {
                "type": "directory",
                "title": "Allowed data directory",
                "description": "Choose the directory whose local data files Truthound may read.",
                "required": True,
            }
        },
        "tools": [
            {"name": "check_file", "description": "Check nulls and duplicate rows"},
            {"name": "profile_file", "description": "Return aggregate column metadata"},
            {"name": "scan_pii", "description": "Find sampled PII estimates"},
        ],
        "compatibility": {"platforms": ["darwin", "win32"], "runtimes": {"python": ">=3.11,<3.14"}},
    }
    launcher = (
        b'from truthound.mcp.__main__ import main\n\nif __name__ == "__main__":\n    main()\n'
    )
    if profile == "core":
        from truthound.mcp.catalog import CATALOG

        manifest["display_name"] = "Truthound Core"
        manifest["description"] = (
            "Local Core operations with explicit data, workspace and disclosure grants."
        )
        manifest["tools"].extend(
            {"name": item.operation_id, "description": item.description}
            for item in CATALOG.values()
        )
        manifest["user_config"].update(
            {
                "workspace_directory": {
                    "type": "directory",
                    "title": "Optional writable workspace",
                    "description": "Grant writes only within this existing directory; blank denies workspace access.",
                    "required": False,
                    "default": "",
                },
                "control_directory": {
                    "type": "directory",
                    "title": "Optional private operator control directory",
                    "description": "Existing private directory separate from data and workspace; approvals remain operator-only.",
                    "required": False,
                    "default": "",
                },
                "sensitive_results": {
                    "type": "boolean",
                    "title": "Allow sensitive Core evidence",
                    "description": "Explicitly permit raw values and full evidence from approved data.",
                    "required": False,
                    "default": False,
                },
            }
        )
        manifest["server"]["mcp_config"]["args"] = [
            "run",
            "--directory",
            "${__dirname}",
            "server.py",
            "--allowed-directory",
            "${user_config.allowed_directory}",
            "--workspace-directory",
            "${user_config.workspace_directory}",
            "--control-directory",
            "${user_config.control_directory}",
            "--sensitive-results",
            "${user_config.sensitive_results}",
        ]
        launcher = (
            b'from truthound.mcp.desktop import main\n\nif __name__ == "__main__":\n    main()\n'
        )
    project = (
        f'[project]\nname = "truthound-desktop"\nversion = "{__version__}"\nrequires-python = ">=3.11,<3.14"\ndependencies = ["truthound[mcp]=={__version__}"]\n'
        + source_config
    )
    files.update(
        {
            "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
            "pyproject.toml": project.encode(),
            "server.py": launcher,
            "LICENSE": _license().encode(),
        }
    )
    # Exclusive create protects an existing release artifact from replacement.
    with (
        output.open("xb") as stream,
        zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z,
    ):
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, data)
    return output
