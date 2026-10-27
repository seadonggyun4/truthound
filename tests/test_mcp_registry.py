"""Registry schema, ownership and capability consent must match the package."""

from __future__ import annotations

import copy
import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
SPEC = importlib.util.spec_from_file_location("mcp_registry", ROOT / "verification/mcp_registry.py")
assert SPEC and SPEC.loader
registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-version",
        "wrong-owner",
        "missing-extra",
        "implicit-directory",
        "wrong-transport",
        "wrong-command",
        "missing-readme",
    ],
)
def test_registry_rejects_install_identity_or_consent_regressions(mutation):
    document = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    readmes = [(ROOT / p).read_text(encoding="utf-8") for p in ("README.md", "README.en.md")]
    package = document["packages"][0]
    if mutation == "wrong-version":
        package["version"] = "3.1.21"
    if mutation == "wrong-owner":
        document["name"] = "io.github.other/truthound"
    if mutation == "missing-extra":
        package["runtimeArguments"] = []
    if mutation == "implicit-directory":
        package["packageArguments"][2]["default"] = "/"
    if mutation == "wrong-transport":
        package["transport"] = {"type": "streamable-http", "url": "https://example.invalid/mcp"}
    if mutation == "wrong-command":
        package["packageArguments"][1]["value"] = "bundle"
    if mutation == "missing-readme":
        readmes = ["No ownership marker"]
    with pytest.raises(ValueError):
        registry.validate_metadata(document, VERSION, readmes)


def test_registry_schema_and_explicit_local_command_match_release(tmp_path):
    document = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    readmes = [(ROOT / p).read_text(encoding="utf-8") for p in ("README.md", "README.en.md")]
    registry.validate_metadata(copy.deepcopy(document), VERSION, readmes)
    assert registry.runtime_argv(document, tmp_path) == [
        "uvx",
        "--with",
        f"truthound[mcp]=={VERSION}",
        f"truthound@{VERSION}",
        "mcp",
        "serve",
        "--allow-dir",
        str(tmp_path),
    ]
    with pytest.raises(ValueError):
        registry.runtime_argv(document, Path("relative"))
