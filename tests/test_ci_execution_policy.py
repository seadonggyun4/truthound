"""Protect execution scope and canonical artifact identity during CI reduction."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_universe_is_preserved_and_boundary_cannot_impersonate_it():
    gate = load("ci_policy_mcp", "verification/mcp_ci_gate.py")
    assert len(gate.TEST_FILES) == len(set(gate.TEST_FILES)) >= 40
    assert set(gate.BOUNDARY_FILES) < set(gate.TEST_FILES)
    assert "tests/test_mcp_algorithm_parity.py" in gate.TEST_FILES
    assert "tests/test_mcp_model_persistence.py" in gate.BOUNDARY_FILES
    assert "tests/test_mcp_lifecycle.py" in gate.BOUNDARY_FILES
    for name in ("tests-pr", "docs", "benchmarks-pr", "openai-plugin", "mcp"):
        workflow = yaml.load(
            (ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader
        )
        assert workflow["on"]["push"]["branches"] == ["main"]
        assert "pull_request" in workflow["on"] and "merge_group" in workflow["on"]
    workflow = yaml.load((ROOT / ".github/workflows/mcp.yml").read_text(), Loader=yaml.BaseLoader)
    matrix = workflow["jobs"]["compatibility"]["strategy"]["matrix"]["include"]
    full, development = [json.loads(value) for value in matrix.split("'") if value.startswith("[{")]
    assert {(r["os"], r["python"]) for r in full} == {
        (os, python)
        for os in ("ubuntu-latest", "macos-latest", "windows-latest")
        for python in ("3.11", "3.12", "3.13")
    }
    assert len(full) == 9 and len(development) == 4
    jobs = workflow["jobs"]
    boundary = jobs["platform-contracts"]
    boundary_full, boundary_dev = [
        json.loads(value)
        for value in boundary["strategy"]["matrix"]["include"].split("'")
        if value.startswith("[{")
    ]
    for installs, contracts in ((full, boundary_full), (development, boundary_dev)):
        expected = {(row["os"], row["python"]) for row in installs} - {("ubuntu-latest", "3.11")}
        assert {(row["os"], row["python"]) for row in contracts} == expected
        for os_name, python in expected:
            rows = [r for r in contracts if (r["os"], r["python"]) == (os_name, python)]
            count = 2 if os_name == "windows-latest" else 1
            assert len(rows) == count
            assert {r["shard"] for r in rows} == set(range(count))
            assert all(r["count"] == count for r in rows)
    assert "needs" not in boundary
    assert jobs["compatibility"]["needs"] == ["build", "platform-contracts"]
    assert set(jobs["mcp-gate"]["needs"]) == {
        "mcp-tests",
        "build",
        "platform-contracts",
        "compatibility",
    }
    assert boundary["timeout-minutes"] == jobs["compatibility"]["timeout-minutes"] == "15"
    assert sum("--scope boundary" in step.get("run", "") for step in boundary["steps"]) == 1
    aggregates = [
        step["run"]
        for step in jobs["compatibility"]["steps"]
        if "--scope boundary" in step.get("run", "")
    ]
    assert len(aggregates) == 1 and "--aggregate-dir" in aggregates[0]
    assert any(
        step.get("with", {}).get("pattern")
        == "truthound-mcp-boundary-${{ matrix.os }}-${{ matrix.python }}-*"
        for step in jobs["compatibility"]["steps"]
    )
    assert workflow["jobs"]["mcp-tests"]["strategy"]["matrix"] == {"shard": ["0", "1", "2", "3"]}


@pytest.mark.parametrize("mutation", ["hash", "missing", "duplicate", "escape"])
def test_canonical_artifacts_reject_corruption_before_copy(tmp_path, mutation):
    release = load("ci_policy_release", "verification/mcp_release.py")
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    names = [
        "truthound-1-py3-none-any.whl",
        "truthound-1.tar.gz",
        "truthound-1.mcpb",
        "truthound-1-local.mcpb",
    ]
    lines = []
    for name in names:
        (source / name).write_bytes(b"test artifact")
        lines.append(hashlib.sha256(b"test artifact").hexdigest() + "  " + name)
    if mutation == "hash":
        (source / names[0]).write_bytes(b"corrupt")
    elif mutation == "missing":
        lines.pop()
    elif mutation == "duplicate":
        lines.append(lines[0])
    else:
        lines[0] = lines[0].replace(names[0], "../" + names[0])
    (source / "SHA256SUMS").write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError):
        release.copy_verified_artifacts(source, output)
    assert not list(output.iterdir())


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "b" * 40),
        ("path", ".github/workflows/tests-pr.yml"),
        ("event", "pull_request"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("conclusion", "cancelled"),
        ("conclusion", "skipped"),
    ],
)
def test_publication_rejects_wrong_or_incomplete_verification(field, value):
    candidate = load("ci_policy_candidate", "verification/ci/release_candidate.py")
    run = dict(
        head_sha="a" * 40,
        path=".github/workflows/release-verify.yml",
        event="workflow_dispatch",
        status="completed",
        conclusion="success",
    )
    candidate.verify(run, "a" * 40)
    run[field] = value
    with pytest.raises(ValueError):
        candidate.verify(run, "a" * 40)


def test_release_preserves_full_execution_and_canonical_publication():
    def workflow(name):
        return yaml.load(
            (ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader
        )

    jobs = workflow("release-verify")["jobs"]
    assert jobs["full"]["with"]["run_full_suite"] == "true"
    assert jobs["mcp"]["with"]["full_compatibility"] == "true"
    assert set(jobs["release-ready"]["needs"]) == set(jobs) - {"release-ready"}
    publish = workflow("release-pypi")
    assert set(publish["on"]) == {"workflow_dispatch"}
    steps = publish["jobs"]["publish"]["steps"]
    commands = [step.get("run", "") for step in steps]
    assert any("release_candidate.py" in command for command in commands)
    assert any("--from-artifacts canonical" in command for command in commands)
    assert not any("uv build" in command or "mcp_ci_gate.py" in command for command in commands)
    download = next(
        step for step in steps if step.get("uses", "").startswith("actions/download-artifact")
    )
    assert download["with"]["name"] == "truthound-mcp-canonical"
    assert download["with"]["run-id"] == "${{ inputs.verification_run_id }}"
