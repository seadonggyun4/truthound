"""Consent regressions for the repository-distributed local OpenAI plugin."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "plugins/truthound/permissions.py"))


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"schema_version": True, "allowed_directories": []},
        {"schema_version": 1, "allowed_directories": ["relative"]},
        {"schema_version": 1, "allowed_directories": [False]},
        {"schema_version": 1, "allowed_directories": [], "extra": "bad"},
    ],
)
def test_corrupt_consent_never_becomes_directory_authority(tmp_path, value):
    path = tmp_path / "permissions.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        API["read_directories"](path)


def test_consent_requires_confirmation_and_revokes_without_data_changes(tmp_path):
    data = tmp_path / "approved"
    data.mkdir()
    source = data / "input.csv"
    source.write_bytes(b"x\n1\n")
    config = tmp_path / "state" / "permissions.json"
    base = [sys.executable, str(ROOT / "plugins/truthound/permissions.py"), "--config", str(config)]
    denied = subprocess.run(base + ["--allow-dir", str(data)], capture_output=True)
    assert denied.returncode != 0 and not config.exists()
    subprocess.run(base + ["--allow-dir", str(data), "--confirm"], check=True, capture_output=True)
    assert API["read_directories"](config) == (data.resolve(),)
    subprocess.run(base + ["--revoke"], check=True, capture_output=True)
    assert API["read_directories"](config) == ()
    assert source.read_bytes() == b"x\n1\n" and list(data.iterdir()) == [source]


@pytest.mark.parametrize("kind", ["relative", "missing", "file", "root"])
def test_invalid_scope_does_not_replace_existing_consent(tmp_path, kind):
    existing = tmp_path / "state" / "permissions.json"
    API["write_directories"](existing, [str(tmp_path)])
    before = existing.read_bytes()
    value = {
        "relative": "relative",
        "missing": str(tmp_path / "missing"),
        "file": str(existing),
        "root": str(Path(tmp_path.anchor)),
    }[kind]
    with pytest.raises((ValueError, OSError)):
        API["write_directories"](existing, [value])
    assert existing.read_bytes() == before


def test_oversized_config_is_rejected_before_parsing(tmp_path):
    config = tmp_path / "permissions.json"
    config.write_bytes(b" " * 65537)
    with pytest.raises(ValueError):
        API["read_directories"](config)


def test_package_builder_never_sweeps_generated_or_private_files(tmp_path):
    api = runpy.run_path(str(ROOT / "verification/openai_plugin.py"))
    plugin = tmp_path / "plugin"
    import shutil

    shutil.copytree(
        ROOT / "plugins/truthound", plugin, ignore=shutil.ignore_patterns(".venv", "__pycache__")
    )
    (plugin / "permissions.json").write_text("private", encoding="utf-8")
    output = tmp_path / "plugin.zip"
    api["build_zip"](plugin, output)
    import zipfile

    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(api["FILES"])
        assert "permissions.json" not in archive.namelist()
    with pytest.raises(FileExistsError):
        api["build_zip"](plugin, output)


def test_unsafe_permission_file_is_rejected_on_the_platform(tmp_path):
    path = tmp_path / "permissions.json"
    API["write_directories"](path, [str(tmp_path)])
    if os.name == "nt":
        path.write_text('{"schema_version": true, "allowed_directories": []}', encoding="utf-8")
    else:
        path.chmod(0o666)
    with pytest.raises(ValueError):
        API["read_directories"](path)


def test_cli_build_checks_manifest_lock_and_relocatable_command(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "verification/openai_plugin.py"),
            "--output",
            str(tmp_path / "dist"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    proof = json.loads(result.stdout)
    import tomllib
    import zipfile

    with zipfile.ZipFile(next((tmp_path / "dist").glob("*.zip"))) as archive:
        manifest = json.loads(archive.read("plugin.json"))
        lock = tomllib.loads(archive.read("uv.lock").decode("utf-8"))
        assert (
            next(p for p in lock["package"] if p["name"] == "truthound-openai-plugin")["version"]
            == manifest["version"]
            == proof["plugin_version"]
        )
        mcp = json.loads(archive.read("mcp.json"))["mcpServers"]["truthound"]
        assert mcp["cwd"] == "." and not any("${" in a for a in mcp["args"])


def test_windows_npm_shim_uses_node_without_shell_evaluation(tmp_path, monkeypatch):
    api = runpy.run_path(str(ROOT / "verification/openai_plugin.py"))
    prefix = tmp_path / "npm with spaces"
    entry = prefix / "node_modules/@openai/codex/bin/codex.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// synthetic npm layout", encoding="utf-8")
    command = api["codex_command"]
    monkeypatch.setattr(
        command.__globals__["shutil"],
        "which",
        lambda name: str(prefix / "codex.cmd") if name == "codex" else str(prefix / "node.exe"),
    )
    assert command() == [str(prefix / "node.exe"), str(entry)]
    entry.unlink()
    with pytest.raises(RuntimeError, match="unsupported_codex_npm_layout"):
        command()
