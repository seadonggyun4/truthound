"""Core distribution must keep client manifests and the frozen project aligned."""

import json
import runpy
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[1]


def api(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "verification"))
    return runpy.run_path(str(ROOT / "verification/core_plugin.py"))


def test_core_archive_preserves_client_launchers_and_excludes_private_state(tmp_path):
    out = tmp_path / "dist"
    subprocess.run(
        [sys.executable, str(ROOT / "verification/core_plugin.py"), "--output", str(out)],
        check=True,
        capture_output=True,
    )
    with zipfile.ZipFile(next(out.glob("*.zip"))) as z:
        names = z.namelist()
        assert ".claude-plugin/plugin.json" in names and "uv.lock" in names
        assert not any(".env" in n or "permissions.json" in n or ".venv/" in n for n in names)
        portable = json.loads(z.read("plugin.json"))
        claude = json.loads(z.read(".claude-plugin/plugin.json"))
        assert portable["version"] == claude["version"]
        args = claude["mcpServers"]["truthound-core"]["args"]
        assert args[args.index("--directory") + 1] == "${CLAUDE_PLUGIN_ROOT}"
        assert "--frozen" in args
        lock = tomllib.loads(z.read("uv.lock").decode())
        assert next(p for p in lock["package"] if p["name"] == "truthound")["version"] == "3.3.1"


@pytest.mark.parametrize(
    "file", [".claude-plugin/plugin.json", ".codex-plugin/plugin.json", "uv.lock"]
)
def test_client_or_lock_version_drift_rejected(tmp_path, monkeypatch, file):
    plugin = tmp_path / "plugin"
    shutil.copytree(
        ROOT / "plugins/truthound-core",
        plugin,
        ignore=shutil.ignore_patterns(".venv", "__pycache__"),
    )
    path = plugin / file
    if file.endswith(".json"):
        value = json.loads(path.read_text())
        value["version"] = "999.0.0"
        path.write_text(json.dumps(value))
    else:
        path.write_text(path.read_text().replace('version = "0.2.2"', 'version = "999.0.0"'))
    with pytest.raises(AssertionError):
        api(monkeypatch)["versions"](plugin)
