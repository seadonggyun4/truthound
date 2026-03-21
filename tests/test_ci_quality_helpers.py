from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml


def _load_module(script_name: str):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "verification"
        / "ci"
        / script_name
    )
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.contract
def test_build_quality_shards_writes_balanced_manifests(tmp_path: Path):
    module = _load_module("build_quality_shards.py")
    contract_collect = tmp_path / "contract.txt"
    fault_collect = tmp_path / "fault-e2e.txt"
    e2e_collect = tmp_path / "e2e.txt"
    output_dir = tmp_path / "quality-shards"
    summary_path = tmp_path / "quality-summary.json"

    contract_collect.write_text(
        "\n".join(
            [
                "tests/unit/checkpoint/test_alpha.py::test_one",
                "tests/unit/checkpoint/test_alpha.py::test_two",
                "tests/unit/execution/test_beta.py::test_three",
                "tests/unit/execution/test_beta.py::test_four",
                "tests/unit/execution/test_beta.py::test_five",
                "tests/unit/profiler/test_gamma.py::test_six",
                "=========================== short test summary info ============================",
            ]
        ),
        encoding="utf-8",
    )
    fault_collect.write_text(
        "\n".join(
            [
                "tests/test_process_timeout.py::test_fast_timeout",
                "tests/test_checkpoint.py::test_fault_path",
            ]
        ),
        encoding="utf-8",
    )
    e2e_collect.write_text("", encoding="utf-8")

    exit_code = module.main(
        [
            "--contract-nodeids",
            str(contract_collect),
            "--fault-nodeids",
            str(fault_collect),
            "--e2e-nodeids",
            str(e2e_collect),
            "--output-dir",
            str(output_dir),
            "--summary-out",
            str(summary_path),
            "--contract-shards",
            "2",
        ]
    )

    assert exit_code == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["contract_selected"] == 6
    assert summary["fault_e2e_selected"] == 2
    assert summary["e2e_selected"] == 0
    assert summary["contract_shard_count"] == 2

    manifests = {
        path.name: path.read_text(encoding="utf-8").splitlines()
        for path in sorted(output_dir.glob("*.txt"))
    }
    assert set(manifests) == {"contract-0.txt", "contract-1.txt", "fault-e2e.txt"}

    assigned_contract = set(manifests["contract-0.txt"]) | set(manifests["contract-1.txt"])
    assert assigned_contract == {
        "tests/unit/checkpoint/test_alpha.py",
        "tests/unit/execution/test_beta.py",
        "tests/unit/profiler/test_gamma.py",
    }
    assert manifests["fault-e2e.txt"] == [
        "tests/test_checkpoint.py",
        "tests/test_process_timeout.py",
    ]

    alpha_shards = [
        name for name, nodeids in manifests.items()
        if "tests/unit/checkpoint/test_alpha.py" in nodeids
    ]
    beta_shards = [
        name for name, nodeids in manifests.items()
        if "tests/unit/execution/test_beta.py" in nodeids
    ]
    assert alpha_shards == ["contract-1.txt"] or alpha_shards == ["contract-0.txt"]
    assert beta_shards == ["contract-1.txt"] or beta_shards == ["contract-0.txt"]
    assert alpha_shards != beta_shards


@pytest.mark.contract
def test_build_quality_shards_rejects_invalid_contract_shard_count(tmp_path: Path):
    module = _load_module("build_quality_shards.py")
    contract_collect = tmp_path / "contract.txt"
    fault_collect = tmp_path / "fault.txt"
    contract_collect.write_text(
        "tests/unit/checkpoint/test_alpha.py::test_one\n",
        encoding="utf-8",
    )
    fault_collect.write_text(
        "tests/test_process_timeout.py::test_fast_timeout\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--contract-nodeids",
            str(contract_collect),
            "--fault-nodeids",
            str(fault_collect),
            "--output-dir",
            str(tmp_path / "quality-shards"),
            "--summary-out",
            str(tmp_path / "quality-summary.json"),
            "--contract-shards",
            "0",
        ]
    )

    assert exit_code == 1


@pytest.mark.contract
def test_build_quality_shards_dedupes_overlap_into_fault_lane(tmp_path: Path):
    module = _load_module("build_quality_shards.py")
    contract_collect = tmp_path / "contract.txt"
    fault_collect = tmp_path / "fault.txt"
    output_dir = tmp_path / "quality-shards"
    summary_path = tmp_path / "quality-summary.json"

    contract_collect.write_text(
        "\n".join(
            [
                "tests/unit/security/test_overlap.py::test_shared",
                "tests/unit/security/test_overlap.py::test_contract_only",
                "tests/unit/security/test_other.py::test_other",
            ]
        ),
        encoding="utf-8",
    )
    fault_collect.write_text(
        "\n".join(
            [
                "tests/unit/security/test_overlap.py::test_shared",
                "tests/unit/security/test_fault.py::test_fault_only",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--contract-nodeids",
            str(contract_collect),
            "--fault-nodeids",
            str(fault_collect),
            "--output-dir",
            str(output_dir),
            "--summary-out",
            str(summary_path),
            "--contract-shards",
            "2",
        ]
    )

    assert exit_code == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_selected"] == 4
    assert summary["contract_selected"] == 2
    assert summary["fault_e2e_selected"] == 2
    assert summary["overlap_selected"] == 1

    contract_manifests = {
        path.name: path.read_text(encoding="utf-8").splitlines()
        for path in sorted(output_dir.glob("contract-*.txt"))
    }
    fault_manifest = (output_dir / "fault-e2e.txt").read_text(encoding="utf-8").splitlines()

    assert fault_manifest == [
        "tests/unit/security/test_fault.py",
        "tests/unit/security/test_overlap.py",
    ]
    assert any("tests/unit/security/test_overlap.py" in entries for entries in contract_manifests.values())


@pytest.mark.contract
def test_run_pytest_manifest_executes_selected_nodeids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "verification"
        / "ci"
        / "run_pytest_manifest.py"
    )
    test_file = tmp_path / "test_sample.py"
    manifest = tmp_path / "manifest.txt"
    junit_path = tmp_path / "manifest-junit.xml"

    test_file.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "@pytest.mark.contract",
                "def test_selected():",
                "    assert True",
                "",
                "@pytest.mark.fault",
                "def test_unselected():",
                "    assert False",
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text("test_sample.py\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--manifest",
            str(manifest),
            "--junitxml",
            str(junit_path),
            "--pytest-arg=-m",
            "--pytest-arg=contract and not (fault or e2e)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert junit_path.exists()
    assert "1 targets from 1 files" in result.stdout


@pytest.mark.contract
def test_run_pytest_manifest_rejects_empty_manifest(tmp_path: Path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "verification"
        / "ci"
        / "run_pytest_manifest.py"
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--manifest",
            str(manifest),
            "--junitxml",
            str(tmp_path / "empty-junit.xml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "does not contain any pytest targets" in result.stderr


@pytest.mark.contract
def test_tests_pr_workflow_uses_sharded_quality_gate():
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "tests-pr.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert workflow["name"] == "Tests PR"
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "quality-collect" in workflow["jobs"]
    assert "quality-contract" in workflow["jobs"]
    assert "quality-fault-e2e" in workflow["jobs"]
    assert "quality-gate" in workflow["jobs"]
    assert workflow["jobs"]["quality-contract"]["strategy"]["matrix"]["shard_id"] == [0, 1, 2, 3]
    assert workflow["jobs"]["quality-gate"]["needs"] == ["quality-contract", "quality-fault-e2e"]
    collect_steps = workflow["jobs"]["quality-collect"]["steps"]
    e2e_step = next(step for step in collect_steps if step.get("name") == "Collect e2e nodes")
    assert '[ "$status" -ne 5 ]' in e2e_step["run"]
    contract_steps = workflow["jobs"]["quality-contract"]["steps"]
    fault_steps = workflow["jobs"]["quality-fault-e2e"]["steps"]
    contract_download = next(
        step for step in contract_steps if step.get("name") == "Download quality shard artifacts"
    )
    fault_download = next(
        step for step in fault_steps if step.get("name") == "Download quality shard artifacts"
    )
    contract_run = next(step for step in contract_steps if step.get("name") == "Run contract shard")
    fault_run = next(step for step in fault_steps if step.get("name") == "Run fault and e2e manifest")
    assert contract_download["with"]["path"] == "test-artifacts"
    assert fault_download["with"]["path"] == "test-artifacts"
    assert "--pytest-arg=-m" in contract_run["run"]
    assert "contract and not (fault or e2e)" in contract_run["run"]
    assert "--pytest-arg=-m" in fault_run["run"]
    assert "fault or e2e" in fault_run["run"]


@pytest.mark.contract
def test_tests_nightly_workflow_publishes_collect_summary():
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "tests-nightly.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["nightly-chaos"]["steps"]
    step_names = [step.get("name", "") for step in steps]

    assert "Collect nightly lane summary" in step_names
    assert "Write nightly selection summary" in step_names
    assert "Upload nightly test artifacts" in step_names


@pytest.mark.contract
def test_dev_and_streaming_extras_cover_quality_gate_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    dev = optional["dev"]
    streaming = optional["streaming"]
    all_extra = optional["all"]
    reports = optional["reports"]

    assert any(dep.startswith("jinja2") for dep in reports)
    assert any(dep.startswith("jinja2") for dep in dev)
    assert any(dep.startswith("pyarrow") for dep in streaming)
    assert any(dep.startswith("pyarrow") for dep in dev)
    assert any(dep.startswith("pyarrow") for dep in all_extra)
