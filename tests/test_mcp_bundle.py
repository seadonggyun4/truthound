from __future__ import annotations

import hashlib
import json
import tomllib
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

def test_bundle_is_deterministic_versioned_and_private_free(tmp_path):
    from truthound.mcp.bundle import build_bundle
    first = build_bundle(tmp_path / "first.mcpb")
    second = build_bundle(tmp_path / "second.mcpb")
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    with zipfile.ZipFile(first) as z:
        assert set(z.namelist()) == {"manifest.json", "pyproject.toml", "server.py", "LICENSE", "assets/icon.png"}
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["server"]["type"] == "uv"
        icon = z.read(manifest["icon"])
        assert icon.startswith(b"\x89PNG\r\n\x1a\n")
        assert icon == (Path(__file__).resolve().parents[1] / "plugins/truthound/assets/icon.png").read_bytes()
        project = tomllib.loads(z.read("pyproject.toml").decode())
        assert project["project"]["version"] == manifest["version"]
        assert "tool" not in project  # PyPI mode must not carry a local wheel source.
        assert any(d.startswith("truthound[mcp]==") for d in project["project"]["dependencies"])
        assert manifest["user_config"]["allowed_directory"]["required"]
        assert "default" not in manifest["user_config"]["allowed_directory"]

def test_documentation_and_optional_dependency_contract():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(d.startswith("mcp>=") for d in project["project"]["optional-dependencies"]["mcp"])
    assert not any(d.startswith("mcp") for d in project["project"]["dependencies"])
    for path in ("docs/guides/mcp.md", "docs/locales/ko/guides/mcp.md"):
        text = (root / path).read_text(encoding="utf-8")
        for term in ("truthound[mcp]", "--allow-dir", "check_file", "profile_file", "scan_pii", ".mcpb"):
            assert term in text
    for path in ("mkdocs.yml", "mkdocs.public.yml"):
        assert "guides/mcp.md" in (root / path).read_text(encoding="utf-8")


def test_core_bundle_exposes_the_catalog_and_denies_blank_optional_grants(tmp_path):
    from truthound.mcp.bundle import build_bundle
    from truthound.mcp.catalog import CATALOG
    from truthound.mcp.desktop import arguments

    bundle = build_bundle(tmp_path / "core.mcpb", profile="core")
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert {tool["name"] for tool in manifest["tools"]} == set(CATALOG) | {
            "check_file", "profile_file", "scan_pii"
        }
        command = manifest["server"]["mcp_config"]["args"]
        substituted = [
            argument.replace("${user_config.allowed_directory}", str(tmp_path))
            .replace("${user_config.workspace_directory}", "")
            .replace("${user_config.control_directory}", "")
            .replace("${user_config.sensitive_results}", "false")
            for argument in command[4:]
        ]
        assert arguments(substituted) == ["serve", "--profile", "core", "--allow-dir", str(tmp_path)]
        assert b"truthound.mcp.desktop" in archive.read("server.py")
        with pytest.raises(SystemExit):
            arguments(["--allowed-directory", ""])
        with pytest.raises(SystemExit):
            arguments(["--allowed-directory", str(tmp_path), "--sensitive-results", "yes"])
    with pytest.raises(ValueError):
        build_bundle(tmp_path / "invalid.mcpb", profile="unknown")
