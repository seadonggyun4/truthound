"""Build/inspect distributions and smoke-test fresh installs; never publish."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

BASE_SMOKE = """
import importlib.util, tempfile
from pathlib import Path
import truthound as th
assert importlib.util.find_spec("mcp") is None
with tempfile.TemporaryDirectory() as td:
    p=Path(td)/"input.csv"
    p.write_text("email,x\\na@example.com,1\\na@example.com,1\\n,2\\n",encoding="utf-8")
    assert th.scan(str(p)).row_count==3
    assert th.profile(str(p)).row_count==3
    result=th.check(str(p),validators=["null","duplicate"])
    assert result.row_count==3 and not result.success
print("fresh-base-core-without-sdk-pass")
"""

SMOKE = """
import asyncio, hashlib, json, os, re, subprocess, sys, tempfile
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
                initialized=await client.initialize()
                assert initialized.instructions and "No raw values are returned" in initialized.instructions
                assert initialized.protocolVersion=="2025-11-25"
                assert len((await client.list_tools()).tools)==3
                for name in ("check_file","profile_file","scan_pii"):
                    result=await client.call_tool(name,{"path":str(p)})
                    if result.isError and result.structuredContent["error"]["code"] == "worker_error":
                        # Metadata-only diagnostics for a fresh runtime failure. Never print
                        # raw stderr, data rows, environment values or absolute paths.
                        env={k:v for k,v in os.environ.items() if k in {"PATH","SYSTEMROOT","WINDIR","TEMP","TMP","PROCESSOR_ARCHITECTURE","PROCESSOR_ARCHITEW6432"}}
                        env.update(POLARS_MAX_THREADS="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
                        request={"operation":"check","path":str(p),"max_items":100,"max_response_bytes":65536}
                        probe=subprocess.run([sys.executable,"-I","-m","truthound.mcp.worker"],input=json.dumps(request),capture_output=True,text=True,cwd=td,env=env,timeout=30)
                        kinds=re.findall(r"([A-Za-z]+(?:Error|Exception)):",probe.stderr)
                        missing=re.findall(r"No module named '([A-Za-z0-9_.]+)'",probe.stderr)
                        frames=[{"file":Path(file).name,"line":int(line),"function":func} for file,line,func in re.findall(r'File "([^"]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)',probe.stderr)]
                        known={code: phrase in probe.stderr for code,phrase in {"missing_home":"Could not determine home directory","lost_stdin":"lost sys.stdin","lost_stdout":"lost sys.stdout","cpu_feature":"CPU feature"}.items()}
                        flags=re.findall(r"unknown feature flag: '([A-Za-z0-9_.-]+)'",probe.stderr)
                        print(json.dumps({"worker_returncode":probe.returncode,"exception_types":kinds,"missing_modules":missing,"frames":frames,"known_causes":known,"unknown_cpu_flags":flags,"stderr_sha256":hashlib.sha256(probe.stderr.encode()).hexdigest()}),flush=True)
                    assert not result.isError, result
                    assert result.structuredContent["data"]["row_count"]==3
                    assert "a@example.com" not in result.model_dump_json()
                # Parameterized dtypes themselves can encode raw category values.
                import polars as pl
                marker="synthetic_private_category_9f31"
                enum_file=Path(td)/"enum.parquet"
                pl.DataFrame({"kind":pl.Series([marker],dtype=pl.Enum([marker]))}).write_parquet(enum_file)
                result=await client.call_tool("profile_file",{"path":str(enum_file)})
                assert not result.isError
                assert result.structuredContent["data"]["columns"][0]["dtype"]=="Enum"
                assert marker not in result.model_dump_json()
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
    parser.add_argument(
        "--expected-python", default=f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    args = parser.parse_args()
    if args.expected_python != f"{sys.version_info.major}.{sys.version_info.minor}":
        parser.error("Source interpreter does not match intended matrix Python")
    proofs = []

    def smoke(command, code, artifact, stage, cwd):
        # Assertions and metadata execute in the installed interpreter, not its parent.
        prefix = (
            "import sys; assert str(sys.version_info.major)+'.'+str(sys.version_info.minor)=="
            + repr(args.expected_python)
            + ";\n"
        )
        suffix = """
import importlib.metadata as metadata, importlib.util, json, platform
print("TRUTHOUND_RUNTIME_PROOF:"+json.dumps({
    "python": platform.python_version(),
    "mcp": metadata.version("mcp") if importlib.util.find_spec("mcp") else None,
    "polars": metadata.version("polars"), "truthound": metadata.version("truthound")
}))
"""
        result = subprocess.run(
            list(map(str, command)) + [prefix + code + suffix],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("INSTALLED_RUNTIME_SMOKE_FAILED:" + stage)
        lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("TRUTHOUND_RUNTIME_PROOF:")
        ]
        assert len(lines) == 1
        proof = json.loads(lines[0].split(":", 1)[1])
        proof.update(
            stage=stage,
            artifact=artifact.name,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            expected_python=args.expected_python,
        )
        proofs.append(proof)
        print(json.dumps(proof), flush=True)

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
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    python = sys.executable
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
                run([uv, "venv", "--python", args.expected_python, venv])
                executable = venv / (
                    "Scripts/python.exe" if __import__("os").name == "nt" else "bin/python"
                )
                run([uv, "pip", "install", "--python", executable, artifact])
                smoke([executable, "-I", "-c"], BASE_SMOKE, artifact, "base-install", td)
                run([uv, "pip", "install", "--python", executable, str(artifact) + "[mcp]"])
                smoke([executable, "-I", "-c"], SMOKE, artifact, "mcp-install", td)
        # Exercise the actual extracted launcher and UV source mapping before PyPI publishing.
        with tempfile.TemporaryDirectory(prefix="truthound-bundle-data-") as td:
            launcher_code = SMOKE.replace(
                '["-I","-m","truthound.mcp","serve","--allow-dir",td]',
                '[str(Path.cwd()/"server.py"),"serve","--allow-dir",td]',
            )
            smoke(
                [
                    uv,
                    "run",
                    "--directory",
                    output / "extension",
                    "--python",
                    args.expected_python,
                    "python",
                    "-c",
                ],
                launcher_code,
                output / f"truthound-{version}-local.mcpb",
                "uv-launcher",
                output / "extension",
            )
    (output / "runtime-proofs.json").write_text(json.dumps(proofs, indent=2), encoding="utf-8")
    inventory = [
        p for p in output.iterdir() if p.is_file() and p.suffix in {".whl", ".gz", ".mcpb"}
    ]
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(inventory)
        ),
        encoding="utf-8",
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
