"""The required integration lane may not silently pass missing/optional coverage."""

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# The manifest runner invokes pytest.main with verification/ci on sys.path,
# unlike python -m pytest. Load this repository-owned helper by its file identity.
spec = importlib.util.spec_from_file_location(
    "truthound_mcp_ci_gate", Path(__file__).resolve().parents[1] / "verification/mcp_ci_gate.py"
)
assert spec is not None and spec.loader is not None
mcp_ci_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_ci_gate)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "body",
    [
        "<testsuites/>",
        '<testsuite><testcase classname="test_required"><skipped/></testcase></testsuite>',
        '<testsuite><testcase classname="test_other"/></testsuite>',
    ],
)
def test_required_gate_rejects_zero_skipped_or_missing_tests(tmp_path, body):
    report = tmp_path / "report.xml"
    report.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        mcp_ci_gate.validate_report(report, ("tests/test_required.py",))


def test_required_gate_rejects_missing_sdk_before_pytest(monkeypatch):
    def missing(_name):
        raise importlib.metadata.PackageNotFoundError("mcp")

    monkeypatch.setattr(mcp_ci_gate.importlib.metadata, "version", missing)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        mcp_ci_gate.verify_runtime()


def test_required_gate_rejects_wrong_python_before_pytest():
    with pytest.raises(ValueError, match="MCP_PYTHON_MISMATCH"):
        mcp_ci_gate.verify_runtime("0.0")


def test_contract_generator_keeps_paths_blocked_across_python_implementation_moves():
    import json

    root = Path(__file__).resolve().parents[1]
    descriptor = importlib.util.spec_from_file_location(
        "truthound_mcp_contract_generator",
        root / "verification/mcp_generate_validator_contracts.py",
    )
    assert descriptor is not None and descriptor.loader is not None
    generator = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(generator)
    expected = {"kind": "blocked", "reason": "<class 'pathlib.Path'>"}
    assert generator.describe(Path) == expected
    document = json.loads(
        (root / "src/truthound/mcp/validator_contracts.json").read_text(encoding="utf-8")
    )
    alternatives = document["validators"]["polygon"]["fields"]["geojson_file"]["type"]["items"]
    assert expected in alternatives


@pytest.fixture(params=[("full", 4), ("full", 8), ("boundary", 2)])
def shard_reports(tmp_path, monkeypatch, request):
    import xml.etree.ElementTree as ET

    helpers = mcp_ci_gate.proof_helpers()
    scope, count = request.param
    required = mcp_ci_gate.TEST_FILES if scope == "full" else mcp_ci_gate.BOUNDARY_FILES
    full = {file + "::test_expected": "source-hash" for file in required}
    identity = {
        "package_sources": {"native.py": "hash"},
        "test_and_verification_sources": {"test.py": "hash"},
        "project_sources": {"uv.lock": "hash"},
    }
    monkeypatch.setattr(
        mcp_ci_gate,
        "proof_helpers",
        lambda: SimpleNamespace(
            partition=helpers.partition,
            sources=lambda: identity["package_sources"],
            supporting_sources=lambda: identity["test_and_verification_sources"],
            project_sources=lambda: identity["project_sources"],
        ),
    )
    union = {}
    for index in range(count):
        selected = helpers.partition(full, count, index)
        assert not union.keys() & selected.keys()
        union.update(selected)
        proof = {
            "valid": True,
            "full_collection": full,
            "collection": selected,
            "shard_count": count,
            "shard_index": index,
            "tests": len(selected),
            "execution_phases": 3 * len(selected),
            **identity,
        }
        (tmp_path / f"mcp-{index}.sources.json").write_text(json.dumps(proof), encoding="utf-8")
        suites = ET.Element("testsuites")
        suite = ET.SubElement(suites, "testsuite")
        for node in selected:
            ET.SubElement(
                suite,
                "testcase",
                classname=node.split("::")[0].removesuffix(".py").replace("/", "."),
                name="test_expected",
            )
        ET.ElementTree(suites).write(tmp_path / f"mcp-{index}.xml", encoding="utf-8")
    assert union == full
    return tmp_path, count, required


def test_required_shard_gate_assembles_only_complete_disjoint_execution(shard_reports):
    shard_reports, count, required = shard_reports
    destination = shard_reports / "complete.xml"
    result = mcp_ci_gate.aggregate(shard_reports, count, destination, required)
    assert result["tests"] == len(required)
    proof = json.loads(destination.with_suffix(".sources.json").read_text(encoding="utf-8"))
    assert proof["valid"] and proof["execution_phases"] == 3 * result["tests"]


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "failed",
        "duplicate",
        "selection",
        "phases",
        "sources",
        "universe",
        "skip",
        "junit_count",
        "junit_duplicate",
    ],
)
def test_required_shard_gate_rejects_incomplete_or_altered_evidence(shard_reports, damage):
    shard_reports, count, required = shard_reports
    path = shard_reports / "mcp-0.sources.json"
    proof = json.loads(path.read_text(encoding="utf-8"))
    if damage == "missing":
        path.unlink()
    elif damage in {"skip", "junit_count", "junit_duplicate"}:
        import xml.etree.ElementTree as ET

        report = shard_reports / "mcp-0.xml"
        tree = ET.parse(report)
        case = next(tree.getroot().iter("testcase"))
        if damage == "skip":
            ET.SubElement(case, "skipped")
        elif damage == "junit_count":
            tree.getroot()[0].remove(case)
        else:
            cases = list(tree.getroot().iter("testcase"))
            cases[1].set("classname", case.get("classname"))
            cases[1].set("name", case.get("name"))
        tree.write(report, encoding="utf-8")
    else:
        if damage == "failed":
            proof["valid"] = False
        elif damage == "duplicate":
            proof["shard_index"] = 1
        elif damage == "selection":
            proof["collection"].pop(next(iter(proof["collection"])))
        elif damage == "phases":
            proof["execution_phases"] -= 1
        elif damage == "sources":
            proof["package_sources"]["native.py"] = "changed"
        elif damage == "universe":
            proof["full_collection"]["tests/test_other.py::test_hidden"] = "changed"
        path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError):
        mcp_ci_gate.aggregate(shard_reports, count, shard_reports / "complete.xml", required)
    assert not (shard_reports / "complete.sources.json").exists()


def test_required_shard_gate_rejects_different_scope(shard_reports):
    directory, count, required = shard_reports
    with pytest.raises(ValueError, match="MCP_PARTITION_SCOPE_MISMATCH"):
        mcp_ci_gate.aggregate(directory, count, directory / "wrong.xml", required[:-1])
    assert not (directory / "wrong.sources.json").exists()
