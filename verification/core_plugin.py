"""Package the Core plugin and verify its installed stdio command, never publish."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

from openai_plugin import codex_command, validate_portable

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "plugin.json",
    ".codex-plugin/plugin.json",
    "mcp.json",
    ".mcp.json",
    "pyproject.toml",
    "LICENSE",
    "assets/icon.png",
    "skills/truthound-core-setup/SKILL.md",
)


async def smoke(plugin, wheel, work):
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.shared.exceptions import McpError

    connection = json.loads((plugin / "mcp.json").read_text())["mcpServers"]["truthound-core"]
    environment = {
        **os.environ,
        "TRUTHOUND_CORE_CONFIG": str(work / "consent" / "permissions.json"),
        "UV_PROJECT_ENVIRONMENT": str(work / "venv"),
        "UV_NO_SOURCES": "true",
        "POLARS_MAX_THREADS": "1",
    }
    if wheel:
        environment["UV_FIND_LINKS"] = str(wheel.parent)
    data = work / "data"
    data.mkdir()
    source = data / "input.csv"
    source.write_text("x\n1\n1\n", encoding="utf-8")
    command = [connection["command"], *connection["args"][:-1]]
    # Replace serve with the trusted local operator command, never an MCP tool.
    subprocess.run(
        command + ["configure", "--allow-dir", str(data), "--confirm"],
        cwd=plugin,
        env=environment,
        check=True,
        capture_output=True,
    )
    parameters = StdioServerParameters(
        command=connection["command"], args=connection["args"], cwd=str(plugin), env=environment
    )
    if wheel:
        subprocess.run(
            [
                connection["command"],
                "run",
                "--no-config",
                "python",
                "-I",
                "-c",
                "import importlib.metadata,sys,zipfile; from pathlib import Path; "
                "d=importlib.metadata.distribution('truthound'); assert d.version=='3.3.0'; "
                "z=zipfile.ZipFile(sys.argv[1]); names=[n for n in z.namelist() "
                "if n.startswith('truthound/') and n.endswith(('.py','.json'))]; "
                "assert names and all(Path(d.locate_file(n)).read_bytes()==z.read(n) for n in names)",
                str(wheel),
            ],
            cwd=plugin,
            env=environment,
            check=True,
            capture_output=True,
        )
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
        initialized = await client.initialize()
        assert initialized.serverInfo.version == "3.3.0"
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert {"capabilities_list", "assets_capture", "execution_services_open"} <= names
        result = await client.call_tool("check_file", {"path": str(source)})
        assert not result.isError and result.structuredContent["data"]["row_count"] == 2
        outside = work / "outside.csv"
        outside.write_text("x\n9\n")
        denied = await client.call_tool("check_file", {"path": str(outside)})
        assert denied.isError
        denied = await client.call_tool("data_read", {"path": str(source), "limit": 1})
        assert denied.isError
        result = await client.call_tool("capabilities_list", {})
        assert not result.isError and result.structuredContent["data"]["full_core_parity"] is False
    subprocess.run(
        command + ["configure", "--allow-dir", str(data), "--allow-sensitive-results", "--confirm"],
        cwd=plugin,
        env=environment,
        check=True,
        capture_output=True,
    )
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        retained = await client.call_tool("assets_capture", {"path": str(source)})
        assert not retained.isError
        await asyncio.to_thread(
            subprocess.run,
            command + ["revoke"],
            cwd=plugin,
            env=environment,
            check=True,
            capture_output=True,
        )
        await asyncio.sleep(0.3)
        try:
            await asyncio.wait_for(client.list_tools(), timeout=5)
        except (McpError, anyio.ClosedResourceError, anyio.BrokenResourceError):
            pass
        else:
            raise AssertionError("Revoked Core session remained connected")
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        denied = await client.call_tool("check_file", {"path": str(source)})
        assert denied.isError
    assert source.read_text() == "x\n1\n1\n"
    return {
        "tools": len(names),
        "read_boundary": True,
        "sensitive_denied": True,
        "revoked": True,
        "live_session_revoked": True,
        "installed_wheel_bytes_verified": wheel is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-codex", action="store_true")
    args = parser.parse_args()
    plugin = ROOT / "plugins/truthound-core"
    validate_portable(plugin)
    manifest = json.loads((plugin / "plugin.json").read_text())
    assert manifest["name"] == "truthound-core" and manifest["version"] == "0.2.0"
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "truthound-core-openai-plugin-0.2.0.zip"
    with output.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(FILES):
            source = plugin / name
            assert not source.is_symlink() and source.resolve().is_relative_to(plugin.resolve())
            entry = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, source.read_bytes())
    proof = {
        "plugin_version": "0.2.0",
        "core_version": "3.3.0",
        "zip_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "openai_public_directory_published": False,
    }
    if args.check_codex:
        result = subprocess.run(
            [*codex_command(), "mcp", "list", "--json"], check=True, capture_output=True, text=True
        )
        (target,) = [item for item in json.loads(result.stdout) if item["name"] == "truthound-core"]
        assert target["enabled"]
        transport = target["transport"]
        connection = json.loads((plugin / "mcp.json").read_text())["mcpServers"]["truthound-core"]
        assert (
            transport["command"] == connection["command"]
            and transport["args"] == connection["args"]
        )
        installed = Path(transport["cwd"])
        assert installed.is_absolute()
        for name in FILES:
            assert (installed / name).read_bytes() == (plugin / name).read_bytes()
        proof["actual_codex_install_verified"] = True
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="truthound-core-plugin-") as directory:
            # macOS exposes its temporary root through /var -> /private/var.
            # Use the canonical owned fixture path; consent still rejects aliases.
            work = Path(directory).resolve(strict=True)
            extracted = work / "plugin with spaces 한글"
            with zipfile.ZipFile(output) as archive:
                assert set(archive.namelist()) == set(FILES)
                archive.extractall(extracted)
            proof["smoke"] = asyncio.run(
                smoke(extracted, args.wheel.resolve() if args.wheel else None, work)
            )
    (args.output / "verification.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(proof["zip_sha256"] + "  " + output.name + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
