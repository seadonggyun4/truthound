"""Regressions for actual release blockers; no service or credential access."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification/ci"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_runtime_failure_preserves_safe_diagnostics_not_captured_secrets():
    import json

    release = load("release_failure_diagnostic_test", "verification/mcp_release.py")
    secret = "synthetic-private-output-never-publish"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1]); raise AssertionError(sys.argv[1])",
            secret,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert child.returncode != 0
    child.stdout += json.dumps({"known_causes": {"missing_home": True, secret: secret}}) + "\n"
    observed = release.runtime_failure_diagnostic(child)
    assert observed["exception_types"] == ["AssertionError"]
    assert observed["smoke_lines"] == [1]
    assert observed["worker_known_causes"]["missing_home"] is True
    assert secret not in json.dumps(observed)
    assert len(observed["stderr_sha256"]) == len(observed["stdout_sha256"]) == 64
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/mcp-runtime-diagnostic.yml").read_text(encoding="utf-8")
    )
    assert set(workflow.get("on") or workflow.get(True)) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["windows-installed-runtime"]["runs-on"] == "windows-latest"
    assert "not a release gate" in workflow["name"]


@pytest.mark.parametrize("script", ["BASE_SMOKE", "SMOKE"])
def test_installed_smoke_scripts_resolve_names_in_their_own_scopes(script):
    release = load("release_smoke_scope_test", "verification/mcp_release.py")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--isolated", "--select", "F821",
         "--stdin-filename", "installed_smoke.py", "-"],
        input=getattr(release, script), capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_sqlite_smoke_closes_its_actual_fixture_connection(tmp_path):
    import ast
    import contextlib
    import sqlite3

    release = load("release_sqlite_fixture_test", "verification/mcp_release.py")
    statements = [
        node
        for node in ast.walk(ast.parse(release.SMOKE))
        if isinstance(node, ast.With)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "connect"
            for item in node.items
            for child in ast.walk(item.context_expr)
        )
    ]
    assert len(statements) == 1
    connections = []

    def connect(*args, **kwargs):
        connection = sqlite3.connect(*args, **kwargs)
        connections.append(connection)
        return connection

    database = tmp_path / "native-smoke.sqlite"
    try:
        exec(
            compile(
                ast.Module(body=statements, type_ignores=[]), "<installed-sql-fixture>", "exec"
            ),
            {
                "sqlite3": SimpleNamespace(connect=connect),
                "closing": contextlib.closing,
                "database": database,
            },
        )
        assert len(connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connections[0].execute("SELECT 1")
        with contextlib.closing(sqlite3.connect(database)) as independent:
            assert independent.execute("SELECT age FROM items ORDER BY rowid").fetchall() == [
                (10,),
                (None,),
            ]
    finally:
        for connection in connections:
            connection.close()


def test_optional_metadata_does_not_erase_results_but_full_still_requires_sdk(monkeypatch):
    nightly = load("nightly_diagnostic_test", "verification/ci/nightly_plan.py")
    original = nightly.importlib.metadata.version

    def missing(name):
        if name == "mcp":
            raise nightly.importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(nightly.importlib.metadata, "version", missing)
    observed = nightly.environment()
    assert observed["mcp"] is None and observed["pytest"]
    with pytest.raises(ValueError, match="missing_required"):
        nightly.require_environment(observed)


def test_collected_module_alias_is_rejected():
    nightly = load("nightly_source_test", "verification/ci/nightly_plan.py")
    item = SimpleNamespace(
        path=ROOT / "tests/test_nightly_ci.py",
        nodeid="tests/test_nightly_ci.py::test_x",
        module=SimpleNamespace(__file__=str(ROOT / "tests/test_mcp_registry.py")),
    )
    with pytest.raises(ValueError, match="source_mismatch"):
        nightly.source_identity(item)


@pytest.mark.parametrize(
    "mutation",
    [None, "missing", "failure", "cancelled", "pending", "sha", "integration", "path", "event"],
)
def test_release_gate_uses_workflow_identity_not_mutable_titles(mutation):
    gate = load("release_gate_test", "verification/mcp_publish_gate.py")
    sha = "a" * 40
    runs = [
        dict(
            path=path,
            head_sha=sha,
            status="completed",
            conclusion="success",
            name="custom title",
            display_title="Nightly / manual-full",
            event="workflow_dispatch",
        )
        for path in sorted(gate.REQUIRED_WORKFLOWS)
    ]
    full = next(r for r in runs if r["path"].endswith("/release-verify.yml"))
    if mutation == "missing":
        runs.remove(full)
    elif mutation in {"failure", "cancelled"}:
        full["conclusion"] = mutation
    elif mutation == "pending":
        full["status"] = "in_progress"
    elif mutation == "sha":
        full["head_sha"] = "b" * 40
    elif mutation == "integration":
        # Discovery is independent of titles; actual full mode/flags/executions
        # are subsequently required from the immutable artifact, not this label.
        full["display_title"] = "arbitrary renamed nightly"
    elif mutation == "path":
        full["path"] = ".github/workflows/impostor.yml"
    elif mutation == "event":
        full["event"] = "schedule"
    if mutation in {None, "integration"}:
        assert gate.validate_workflows(runs, sha) == [full]
    else:
        with pytest.raises(ValueError):
            gate.validate_workflows(runs, sha)


def test_scale_duplicate_oracle_is_equivalent_and_bounded():
    from tests.test_100m_scale import _duplicate_candidates

    seen = set(range(2500))
    assert _duplicate_candidates(seen) == tuple(list(seen)[:1000])

    class Bounded:
        def __iter__(self):
            yield from range(1000)
            raise AssertionError("consumed beyond duplicate candidate bound")

    assert _duplicate_candidates(Bounded()) == tuple(range(1000))


def test_stress_summary_and_all_reports_complete_with_concurrent_recording(tmp_path):
    code = """
import json, sys
from concurrent.futures import ThreadPoolExecutor
from tests.stress.metrics import StressMetricsCollector
from tests.stress.framework import StressTestConfig, StressTestResult
from tests.stress.reports import ReportGenerator
m = StressMetricsCollector(name="bounded")
m.start()
def record(i):
    m.record_operation(success=i % 2 == 0, latency_ms=2.0)
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(record, i) for i in range(200)]
    for _ in range(20):
        s = m.get_summary()
        assert s["operations"]["total"] == s["operations"]["successful"] + s["operations"]["failed"]
    for f in futures:
        f.result()
m.stop()
s = m.get_summary()
assert s["operations"]["total"] == 200
assert s["rates"]["success_rate"] == s["rates"]["failure_rate"] == 0.5
r = StressTestResult(config=StressTestConfig(name="bounded"), total_operations=200, successful_operations=100, failed_operations=100)
files = ReportGenerator(sys.argv[1]).generate(r, metrics=m, format="all")
assert {p.suffix for p in files} == {".json", ".md", ".html"}
data = json.loads(next(p for p in files if p.suffix == ".json").read_text(encoding="utf-8"))
assert data["detailed_metrics"]["operations"]["total"] == 200
print("bounded-summary-and-reports-pass")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "bounded-summary-and-reports-pass" in result.stdout


def test_missing_lz4_fails_explicitly_without_fallback(monkeypatch):
    import builtins

    from truthound.stores.compression.base import UnsupportedAlgorithmError
    from truthound.stores.compression.providers import LZ4Compressor

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "lz4.frame":
            raise ModuleNotFoundError("synthetic missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(LZ4Compressor, "_lz4", None)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(UnsupportedAlgorithmError):
        LZ4Compressor._get_lz4()


def test_missing_argon2_is_an_explicit_error_not_a_weaker_kdf(monkeypatch):
    import builtins

    from truthound.stores.encryption.base import KeyDerivationError
    from truthound.stores.encryption.keys import Argon2KeyDeriver

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "argon2.low_level":
            raise ModuleNotFoundError("synthetic missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(KeyDerivationError, match="argon2-cffi"):
        Argon2KeyDeriver().derive("synthetic", b"x" * 16, 32)


def test_paired_sampling_budget_still_detects_actual_slow_path():
    import time

    from tests.fixtures.sampling_timing import measure_sampling_pair

    calls = []

    def sampled():
        calls.append("sampled")
        time.sleep(0.005)

    def full():
        calls.append("full")

    sampled_time, full_time = measure_sampling_pair(sampled, full)
    assert calls.count("sampled") == calls.count("full") == 8
    assert calls[2:6] == ["sampled", "full", "full", "sampled"]
    assert sampled_time > full_time * 5


@pytest.mark.asyncio
async def test_stress_failure_rate_matches_actual_early_stop():
    import asyncio

    from tests.stress.framework import LoadProfile, StressTestConfig, StressTestRunner

    async def fail(operation_id):
        return False

    profile = LoadProfile(
        warmup_seconds=0,
        ramp_up_seconds=0,
        steady_seconds=5,
        ramp_down_seconds=0,
        cooldown_seconds=0,
        target_rate=500,
    )
    runner = StressTestRunner(
        StressTestConfig(
            name="bounded-early-stop", load_profile=profile, concurrency=4, stop_on_failure_rate=0.3
        ),
        fail,
    )
    result = await asyncio.wait_for(runner.run(), timeout=3)
    assert result.total_operations > 100
    assert result.failed_operations == result.total_operations
    assert result.failure_rate == 1.0 and result.success_rate == 0.0
    assert not result.passed


def test_release_smoke_can_run_on_an_exact_final_ref_even_without_path_changes():
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/benchmarks-pr.yml").read_text(encoding="utf-8")
    )
    trigger = workflow.get("on", workflow.get(True))
    assert "workflow_dispatch" in trigger
    assert workflow["jobs"]["benchmark-smoke"]["steps"]
