"""Build an allowlisted plugin ZIP and test its actual frozen UV stdio command."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "plugin.json",
    "mcp.json",
    ".codex-plugin/plugin.json",
    ".mcp.json",
    "pyproject.toml",
    "uv.lock",
    "permissions.py",
    "server.py",
    "LICENSE",
    "assets/icon.png",
    "skills/truthound-setup/SKILL.md",
)


def build_zip(plugin: Path, output: Path) -> None:
    with (
        output.open("xb") as stream,
        zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for name in sorted(FILES):
            source = plugin / name
            if source.is_symlink() or not source.resolve().is_relative_to(plugin.resolve()):
                raise ValueError("unsafe_plugin_member")
            entry = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, source.read_bytes())


async def smoke(plugin: Path, work: Path) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    config = work / "consent" / "permissions.json"
    environment = work / "runtime"
    env = {
        "TRUTHOUND_OPENAI_CONFIG": str(config),
        "UV_CACHE_DIR": str(work / "uv-cache"),
        "UV_PROJECT_ENVIRONMENT": str(environment),
        "POLARS_MAX_THREADS": "1",
    }
    connection = json.loads((plugin / "mcp.json").read_text(encoding="utf-8"))["mcpServers"][
        "truthound"
    ]
    assert (
        connection["type"] == "stdio" and connection["command"] == "uv" and "url" not in connection
    )
    args = [a.replace("${PLUGIN_ROOT}", str(plugin)) for a in connection["args"]]
    assert not any("${" in a for a in args)
    data = work / "approved"
    data.mkdir()
    source = data / "input.csv"
    source.write_text("email,x\na@example.com,1\na@example.com,1\n,2\n", encoding="utf-8")
    original = source.read_bytes()
    outside = work / "outside.csv"
    outside.write_text("x\n9\n", encoding="utf-8")
    async with (
        stdio_client(StdioServerParameters(command=shutil.which("uv"), args=args, env=env)) as (
            read,
            write,
        ),
        ClientSession(read, write) as client,
    ):
        initialized = await client.initialize()
        assert (
            initialized.serverInfo.version == "3.2.0"
            and initialized.protocolVersion == "2025-11-25"
        )
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {"check_file", "profile_file", "scan_pii"}
        denied = await client.call_tool("check_file", {"path": str(source)})
        assert denied.isError and denied.structuredContent["error"]["code"] == "access_denied"
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        configure = [str(python), str(plugin / "permissions.py"), "--config", str(config)]
        subprocess.run(
            configure + ["--allow-dir", str(data), "--confirm"], check=True, capture_output=True
        )
        for name in sorted(tools):
            result = await client.call_tool(name, {"path": str(source)})
            assert not result.isError and result.structuredContent["data"]["row_count"] == 3
            assert "a@example.com" not in result.model_dump_json()
        for name in ["check_file", "profile_file"]:
            result = await client.call_tool(name, {"path": str(outside)})
            assert result.isError and result.structuredContent["error"]["code"] == "access_denied"
        config.write_text("broken", encoding="utf-8")
        result = await client.call_tool("scan_pii", {"path": str(source)})
        assert result.isError and result.structuredContent["error"]["code"] == "access_denied"
        subprocess.run(configure + ["--revoke"], check=True, capture_output=True)
        result = await client.call_tool("profile_file", {"path": str(source)})
        assert result.isError and result.structuredContent["error"]["code"] == "access_denied"
    assert source.read_bytes() == original and not (data / ".truthound").exists()
    version = subprocess.run(
        [str(python), "-I", "-c", "import importlib.metadata as m; print(m.version('truthound'))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert version == "3.2.0"
    return {
        "core_version": version,
        "protocol": initialized.protocolVersion,
        "tools": sorted(tools),
        "actual_manifest_uv_launch": True,
        "fresh_cache": True,
        "unconfigured_denied": True,
        "outside_denied": True,
        "corrupt_config_denied": True,
        "live_revoke_denied": True,
        "raw_value_redacted": True,
        "input_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-codex", action="store_true")
    parser.add_argument("--plugin", type=Path, default=ROOT / "plugins/truthound")
    args = parser.parse_args()
    plugin = args.plugin.resolve(strict=True)
    manifest = json.loads((plugin / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "truthound" and manifest["version"] == "0.1.0"
    extension = manifest["extensions"]["com.openai"]
    compat = json.loads((plugin / ".codex-plugin/plugin.json").read_text(encoding="utf-8"))
    assert compat["interface"] == extension["interface"]
    assert compat["name"] == manifest["name"] and compat["version"] == manifest["version"]
    assert (
        json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
        == json.loads((plugin / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    )
    assert extension["requires_local_executor"] is True and not any(
        k in extension for k in ["apps", "hooks"]
    )
    assert extension["onboardingSkill"] == "./skills/truthound-setup/SKILL.md"
    lock = tomllib.loads((plugin / "uv.lock").read_text(encoding="utf-8"))
    core = next(p for p in lock["package"] if p["name"] == "truthound")
    assert core["version"] == "3.2.0" and core["source"] == {"registry": "https://pypi.org/simple"}
    assert (
        core["wheels"][0]["hash"]
        == "sha256:6e743beb2daadb09a7c06859271675cb2fd794b7a140cccb0074cdd28b914265"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / ("truthound-openai-plugin-" + manifest["version"] + ".zip")
    build_zip(plugin, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (args.output / "SHA256SUMS").write_text(digest + "  " + output.name + "\n", encoding="utf-8")
    result = {
        "plugin_version": manifest["version"],
        "zip_sha256": digest,
        "files": list(FILES),
        "openai_public_directory_published": False,
        "smoke": None,
    }
    if args.check_codex:
        cli = subprocess.run(
            ["codex", "mcp", "list", "--json"], check=True, capture_output=True, text=True
        )
        targets = [s for s in json.loads(cli.stdout) if s["name"] == "truthound"]
        assert len(targets) == 1 and targets[0]["enabled"]
        transport = targets[0]["transport"]
        expected = json.loads((plugin / "mcp.json").read_text(encoding="utf-8"))["mcpServers"][
            "truthound"
        ]
        assert transport["type"] == "stdio" and transport["command"] == "uv"
        assert transport["args"] == expected["args"] and not any(
            "${" in a for a in transport["args"]
        )
        installed = Path(transport["cwd"])
        assert installed.is_absolute() and (installed / "server.py").is_file()
        assert (
            json.loads((installed / ".codex-plugin/plugin.json").read_text(encoding="utf-8"))[
                "version"
            ]
            == manifest["version"]
        )
        for name in FILES:
            assert (installed / name).read_bytes() == (plugin / name).read_bytes(), name
        result["actual_codex_install_and_runtime_config_verified"] = True
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="truthound-plugin-") as td:
            work = Path(td)
            extracted = work / "plugin with spaces 한글"
            with zipfile.ZipFile(output) as archive:
                assert set(archive.namelist()) == set(FILES)
                archive.extractall(extracted)
            result["smoke"] = asyncio.run(smoke(extracted, work))
    (args.output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
