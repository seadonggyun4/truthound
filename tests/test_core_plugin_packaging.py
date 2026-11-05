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
        assert next(p for p in lock["package"] if p["name"] == "truthound")["version"] == "3.3.4"


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
        path.write_text(path.read_text().replace('version = "0.2.6"', 'version = "999.0.0"'))
    with pytest.raises(AssertionError):
        api(monkeypatch)["versions"](plugin)


def test_development_catalog_copies_only_packaged_files(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "verification"))
    from plugin_marketplace import build_marketplace

    first = build_marketplace(tmp_path / "first")
    (first / "plugins/truthound/permissions.json").write_text("private")
    (first / "plugins/truthound-core/.env").write_text("private")
    second = build_marketplace(tmp_path / "second", first / "plugins")
    assert not (second / "plugins/truthound/permissions.json").exists()
    assert not (second / "plugins/truthound-core/.env").exists()
    catalog = json.loads((second / ".agents/plugins/marketplace.json").read_text())
    assert catalog["name"] == "truthound-development"
    for row in catalog["plugins"]:
        source = ROOT / "plugins" / row["name"]
        for member in (second / row["source"]["path"]).rglob("*"):
            if member.is_file():
                relative = member.relative_to(second / row["source"]["path"])
                assert member.read_bytes() == (source / relative).read_bytes()
    with pytest.raises(FileExistsError):
        build_marketplace(second)


def test_product_catalog_is_a_retired_compatibility_marker():
    catalog = json.loads((ROOT / ".agents/plugins/marketplace.json").read_text())
    assert catalog["name"] == "truthound" and catalog["plugins"] == []


@pytest.mark.parametrize('metadata', [None, 'Name: other\nVersion: 3.3.2\n', 'Name: truthound\nVersion: untrusted\n'])
def test_candidate_fixture_rejects_wrong_distribution_before_install(tmp_path, monkeypatch, metadata):
    wheel = tmp_path / 'candidate.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        if metadata is not None:
            archive.writestr('candidate.dist-info/METADATA', metadata)
    output = tmp_path / 'plugin'
    with pytest.raises(AssertionError):
        api(monkeypatch)['candidate_plugin'](ROOT / 'plugins/truthound-core', wheel, output)
    assert not output.exists()
