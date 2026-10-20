"""Build/inspect distributions and smoke-test fresh installs; never publish."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

SMOKE = """
import asyncio, json, sys, tempfile
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
async def run():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"input.csv"
        p.write_text("email,x\\na@example.com,1\\na@example.com,1\\n,2\\n")
        params=StdioServerParameters(command=sys.executable,args=["-I","-m","truthound.mcp","serve","--allow-dir",td])
        async with stdio_client(params) as (r,w):
            async with ClientSession(r,w) as client:
                await client.initialize()
                assert len((await client.list_tools()).tools)==3
                for name in ("check_file","profile_file","scan_pii"):
                    result=await client.call_tool(name,{"path":str(p)})
                    assert not result.isError, result
                    assert result.structuredContent["data"]["row_count"]==3
                    assert "a@example.com" not in result.model_dump_json()
                assert not (Path(td)/".truthound").exists()
asyncio.run(run())
print("fresh-install-stdio-pass")
"""


def run(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required")
    run([uv, "build", "--out-dir", output], cwd=root)
    wheel = next(output.glob("*.whl"))
    sdist = next(output.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as z, tarfile.open(sdist) as t:
        inventory = z.namelist() + t.getnames()
        forbidden = {
            ".golem",
            "AGENTS.md",
            "CLAUDE.md",
            ".env",
            ".venv",
            "node_modules",
            "ai-context-docs",
            ".claude",
            ".codex",
        }
        assert not [n for n in inventory if forbidden.intersection(Path(n).parts)], (
            "Private distribution inventory"
        )
        assert any(n.endswith("truthound/mcp/server.py") for n in z.namelist())
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    python = (
        root / ".venv" / ("Scripts/python.exe" if __import__("os").name == "nt" else "bin/python")
    )
    run(
        [python, "-m", "truthound.mcp", "bundle", "--output", output / f"truthound-{version}.mcpb"],
        cwd=root,
    )
    run(
        [
            python,
            "-m",
            "truthound.mcp",
            "bundle",
            "--wheel",
            wheel,
            "--output",
            output / f"truthound-{version}-local.mcpb",
        ],
        cwd=root,
    )
    with zipfile.ZipFile(output / f"truthound-{version}-local.mcpb") as z:
        # This is our own allowlisted generated archive, not an uploaded bundle.
        assert set(z.namelist()) == {
            "manifest.json",
            "pyproject.toml",
            "server.py",
            "LICENSE",
            "wheels/" + wheel.name,
        }
        z.extractall(output / "extension")
    if args.smoke:
        for artifact in (wheel, sdist):
            with tempfile.TemporaryDirectory(prefix="truthound-install-") as td:
                venv = Path(td) / "venv"
                run([uv, "venv", "--python", "3.11", venv])
                executable = venv / (
                    "Scripts/python.exe" if __import__("os").name == "nt" else "bin/python"
                )
                run([uv, "pip", "install", "--python", executable, str(artifact) + "[mcp]"])
                run([executable, "-I", "-c", SMOKE], cwd=td)
        # Exercise the actual extracted launcher and UV source mapping before PyPI publishing.
        with tempfile.TemporaryDirectory(prefix="truthound-bundle-data-") as td:
            launcher_code = SMOKE.replace(
                '["-I","-m","truthound.mcp","serve","--allow-dir",td]',
                '[str(Path.cwd()/"server.py"),"serve","--allow-dir",td]',
            )
            run(
                [uv, "run", "--directory", output / "extension", "python", "-c", launcher_code],
                cwd=output / "extension",
            )
    inventory = [
        p for p in output.iterdir() if p.is_file() and p.suffix in {".whl", ".gz", ".mcpb"}
    ]
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(inventory)
        )
    )
    print(
        json.dumps(
            {
                "version": version,
                "artifacts": [p.name for p in inventory],
                "fresh_installs_verified": args.smoke,
            }
        )
    )


if __name__ == "__main__":
    main()
