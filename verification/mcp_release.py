"""Build/inspect distributions and smoke-test fresh installs; never publish."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
import truthound.mcp.service
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
from contextlib import closing
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
async def core_profile():
    with tempfile.TemporaryDirectory() as td:
        data,workspace,control=(Path(td)/name for name in ("data","workspace","control"))
        for root in (data,workspace,control): root.mkdir(mode=0o700)
        p=data/"input.csv"
        p.write_text("email,age\\na@example.com,10\\na@example.com,150\\n",encoding="utf-8")
        params=StdioServerParameters(command=sys.executable,args=["-I","-m","truthound.mcp","serve","--allow-dir",str(data),"--workspace-dir",str(workspace),"--control-dir",str(control),"--profile","core","--allow-sensitive-results"])
        async with stdio_client(params) as (r,w):
            async with ClientSession(r,w) as client:
                await client.initialize()
                names={tool.name for tool in (await client.list_tools()).tools}
                required={"capabilities_list","capability_describe","validation_check","schema_learn","data_read","data_mask","data_profile","privacy_scan"}
                required|={"assets_capture","assets_inline","assets_page","assets_sample","assets_mask","schemas_learn","schemas_inspect","artifacts_export","artifacts_read","resources_describe","resources_release","contexts_open","contexts_inspect","contexts_initialize","plans_status","validators_list","validators_describe","assets_validate","assets_sqlite","assets_profile","assets_scan","artifacts_render","artifacts_datadocs"}
                required.add("drift_compare")
                required|={"drift_catalog","drift_detect","profilers_catalog","profilers_profile","schemas_create","schemas_export"}
                required|={"suites_create","suites_inspect","suites_validate","contexts_execute"}
                required|={"sources_open_polars","sources_inspect","sources_sample","sources_validate_connection","sources_validate_columns","sources_check_size_limits","sources_validate"}
                required|={"sources_open_sqlite","sources_open_duckdb","streams_open","streams_next","streams_inspect"}
                required|={"lineage_catalog","lineage_create","lineage_edit","lineage_query","lineage_render","lineage_impact","lineage_critical_paths","lineage_what_if_delete","lineage_schema_impact","lineage_dependency_chain"}
                required|={"models_catalog","models_create","models_inspect","models_fit","models_partial_fit","models_predict","models_score","models_rules","models_fit_predict","models_learn_rules","models_detect","models_diagnostics"}
                assert names==required|{"check_file","profile_file","scan_pii"}
                import importlib.util
                if importlib.util.find_spec("duckdb") is None:
                    optional=data/"optional.duckdb"
                    optional.write_bytes(b"synthetic optional-provider probe")
                    failed=await client.call_tool("sources_open_duckdb",{"path":str(optional),"table":"facts"})
                    assert failed.isError and failed.structuredContent["issues"]==[{"code":"missing_dependency"}]
                result=await client.call_tool("capabilities_list",{})
                assert not result.isError and result.structuredContent["data"]["full_core_parity"] is False
                calls={
                    "validation_check":{"checks":[{"validator":"between","columns":["age"],"max_value":120}]},
                    "schema_learn":{},"data_read":{"limit":1},
                    "data_mask":{"columns":["email"],"strategy":"hash","limit":1},
                    "data_profile":{"evidence":"full"},"privacy_scan":{},
                }
                for name,extra in calls.items():
                    result=await client.call_tool(name,{"path":str(p),**extra})
                    assert not result.isError,(name,result)
                    body=result.structuredContent
                    assert body["provenance"]["input_sha256"]==hashlib.sha256(p.read_bytes()).hexdigest()
                    if name=="validation_check":
                        assert body["quality_status"]=="fail"
                    elif name=="data_read":
                        assert "a@example.com" in result.model_dump_json()
                    elif name=="data_mask":
                        assert "a@example.com" not in result.model_dump_json()
                assert not (Path(td)/".truthound").exists()
                async def invoke(name,arguments):
                    result=await client.call_tool(name,arguments)
                    assert not result.isError,(name,result)
                    return result.structuredContent["data"]
                graph=(await invoke("lineage_create",{"nodes":[{"id":"a","name":"input","node_type":"source"},{"id":"b","name":"output","node_type":"transformation"}],"edges":[{"source":"a","target":"b","edge_type":"derived_from"}]}))["resource_id"]
                import base64
                from truthound.lineage import LineageGraph
                from truthound.lineage.visualization import RenderConfig,get_renderer
                from truthound.mcp.tables import decode
                async def artifact_bytes(identity):
                    chunks,offset=[],0
                    while offset is not None:
                        part=await invoke("artifacts_read",{"resource_id":identity,"offset":offset})
                        chunks.append(base64.b64decode(part["base64"]))
                        offset=part["next_offset"]
                    return b"".join(chunks)
                snapshot=await invoke("lineage_query",{"query":{"graph_id":graph,"view":"graph"}})
                native_graph=LineageGraph.from_dict(decode(json.loads(await artifact_bytes(snapshot["resource_id"]))))
                for renderer in ("d3","cytoscape","graphviz","mermaid"):
                    for subgraph in (False,True):
                        config=RenderConfig(theme="dark",max_depth=1)
                        target=get_renderer(renderer)
                        expected=target.render_subgraph(native_graph,"a",max_depth=1,config=config) if subgraph else target.render(native_graph,config=config)
                        rendered=await invoke("lineage_render",{"graph_id":graph,"renderer":renderer,"settings":{"theme":"dark","max_depth":1},**({"root_node_id":"a"} if subgraph else {})})
                        assert await artifact_bytes(rendered["resource_id"])==expected.encode("utf-8")
                await invoke("resources_release",{"resource_id":graph})
                asset=(await invoke("assets_capture",{"path":str(p)}))["resource_id"]
                # The lean MCP extra does not install statistical/ML extras.
                # Known optional constructors fail as missing dependencies,
                # not unknown-validator or invalid-configuration success.
                import importlib.util
                if importlib.util.find_spec("numpy") is None:
                    absent=await client.call_tool("assets_validate",{"asset_id":asset,"checks":[{"validator":"ks_test","settings":{"column":"age","reference_data":{"asset_id":asset}}}]})
                    assert absent.isError and absent.structuredContent["issues"]==[{"code":"missing_dependency"}]
                rows=await invoke("assets_page",{"asset_id":asset,"limit":1})
                assert rows["next_offset"]==1 and rows["total_rows"]==2
                native_source=(await invoke("sources_open_polars",{"asset_id":asset,"config":{"name":"installed-native"}}))["resource_id"]
                assert (await invoke("sources_inspect",{"source_id":native_source}))["bytes"]>0
                native_run=await invoke("sources_validate",{"source_id":native_source,"checks":[{"validator":"mean_between","settings":{"columns":["age"],"max_value":1}}]})
                assert native_run["summary"]["success"] is False
                suite=(await invoke("suites_create",{"spec":{"checks":[{"check_id":"native-null","validator":"null"}]}}))["resource_id"]
                assert (await invoke("suites_validate",{"asset_id":asset,"suite_id":suite}))["artifact"]["bytes"]>0
                import sqlite3
                database=data/"native.db"
                with closing(sqlite3.connect(database)) as connection, connection:
                    connection.execute("CREATE TABLE items (age INTEGER)")
                    connection.executemany("INSERT INTO items VALUES (?)",[(10,),(None,)])
                sql=(await invoke("sources_open_sqlite",{"path":str(database),"table":"items","config":{"name":"installed-sql"}}))["resource_id"]
                assert (await invoke("sources_validate",{"source_id":sql,"checks":[{"validator":"null","settings":{"columns":["age"]}}]}))["summary"]["success"] is False
                await invoke("resources_release",{"resource_id":sql})
                stream=(await invoke("streams_open",{"path":str(p),"source":{"format":"csv","config":{"chunk_size":1}}}))["resource_id"]
                for index,valid in enumerate((True,False)):
                    args={"stream_id":stream,"idempotency_key":"installed-batch-"+str(index)}
                    batch=await invoke("streams_next",args)
                    identity=batch["artifact"]["resource_id"]
                    assert not batch["exhausted"]
                    assert (await invoke("streams_next",args))["artifact"]["resource_id"]==identity
                    assert (await invoke("assets_validate",{"asset_id":identity,"checks":[{"validator":"between","settings":{"columns":["age"],"max_value":120}}]}))["summary"]["success"] is valid
                assert (await invoke("streams_next",{"stream_id":stream,"idempotency_key":"installed-eof"}))["exhausted"]
                await invoke("resources_release",{"resource_id":stream})
                validation=await invoke("assets_validate",{"asset_id":asset,"checks":[{"validator":"between","settings":{"columns":["age"],"max_value":120}}]})
                assert validation["summary"]["success"] is False
                run=validation["artifact"]["resource_id"]
                rendered=await invoke("artifacts_render",{"resource_id":run,"format":"json"})
                chunk=await invoke("artifacts_read",{"resource_id":rendered["resource_id"]})
                assert chunk["total_bytes"]>0 and len(chunk["sha256"])==64
                for name in ("assets_profile","assets_scan"):
                    assert (await invoke(name,{"asset_id":asset}))["bytes"]>0
                training=(await invoke("assets_inline",{"columns":[{"name":"x","dtype":{"kind":"Float64"},"values":[float(i) for i in range(100)]}]}))["resource_id"]
                model=await invoke("models_create",{"algorithm":"ZScoreAnomalyDetector"})
                fitted=await invoke("models_fit",{"model_id":model["model_id"],"asset_id":training,"expected_revision":0})
                assert fitted["trained"] and fitted["revision"]==1
                prediction=await invoke("models_predict",{"model_id":model["model_id"],"asset_id":training,"expected_revision":1})
                assert prediction["bytes"]>0
                schema=(await invoke("schemas_learn",{"asset_id":asset}))["resource_id"]
                assert (await invoke("schemas_inspect",{"resource_id":schema}))["encoding"]=="tagged-v1"
                context=(await invoke("contexts_open",{"path":str(workspace)}))["context_id"]
                args={"context_id":context,"idempotency_key":"installed-initialize-1"}
                pending=await client.call_tool("contexts_initialize",args)
                assert pending.isError and pending.structuredContent["issues"]==[{"code":"approval_required"}]
                plan=pending.structuredContent["data"]["plan_id"]
                subprocess.run([sys.executable,"-I","-m","truthound.mcp","review-plan","--control-dir",str(control),"--plan-id",plan],check=True,capture_output=True)
                subprocess.run([sys.executable,"-I","-m","truthound.mcp","approve-plan","--control-dir",str(control),"--plan-id",plan],check=True,capture_output=True)
                completed=await invoke("contexts_initialize",args)
                assert completed["workspace_exists"] and not completed["replayed"]
                assert (await invoke("contexts_initialize",args))["replayed"]
                effect={"context_id":context,"idempotency_key":"installed-native-metric","operation":{"action":"metric_put","key":"installed-proof","value":{"synthetic":True}}}
                pending=await client.call_tool("contexts_execute",effect)
                assert pending.isError and pending.structuredContent["issues"]==[{"code":"approval_required"}]
                subprocess.run([sys.executable,"-I","-m","truthound.mcp","approve-plan","--control-dir",str(control),"--plan-id",pending.structuredContent["data"]["plan_id"]],check=True,capture_output=True)
                assert (await invoke("contexts_execute",effect))["artifact"]["bytes"]>0
                assert (await invoke("contexts_execute",effect))["replayed"]
                await invoke("resources_release",{"resource_id":asset})
asyncio.run(core_profile())
print("fresh-install-stdio-pass")
"""


def run(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def runtime_failure_diagnostic(result):
    """Bounded metadata only; never disclose captured output or data values."""
    kinds = {
        "AssertionError",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "ImportError",
        "ModuleNotFoundError",
        "UnicodeDecodeError",
        "ExceptionGroup",
        "OSError",
        "PermissionError",
        "FileNotFoundError",
    }
    diagnostic = {
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        "exception_types": sorted(kind for kind in kinds if kind in result.stderr),
        "smoke_lines": [int(n) for n in re.findall(r'File "<string>", line (\d+)', result.stderr)][
            :16
        ],
        "worker_error": "worker_error" in result.stderr,
    }
    allowed_causes = {"missing_home", "lost_stdin", "lost_stdout", "cpu_feature"}
    for line in result.stdout.splitlines()[:16]:
        if len(line) > 16384:
            continue
        try:
            probe = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(probe, dict) and isinstance(probe.get("known_causes"), dict):
            diagnostic["worker_known_causes"] = {
                key: probe["known_causes"].get(key) is True for key in sorted(allowed_causes)
            }
    return diagnostic


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
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            print(
                json.dumps({"stage": stage, "runtime_failure": runtime_failure_diagnostic(result)}),
                flush=True,
            )
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
            "__pycache__",
        }
        assert not [n for n in inventory if forbidden.intersection(Path(n).parts)], (
            "Private distribution inventory"
        )
        assert not [n for n in inventory if Path(n).suffix in {".pyc", ".pyo"}]
        assert any(n.endswith("truthound/mcp/server.py") for n in z.namelist())
        metadata_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        assert "mcp-name: io.github.seadonggyun4/truthound" in z.read(metadata_name).decode("utf-8")
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
                registry_smoke = SMOKE.replace(
                    '["-I","-m","truthound.mcp","serve","--allow-dir",td]',
                    '["-I","-m","truthound.cli","mcp","serve","--allow-dir",td]',
                )
                assert registry_smoke != SMOKE
                smoke(
                    [executable, "-I", "-c"], registry_smoke, artifact, "registry-cli-install", td
                )
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
