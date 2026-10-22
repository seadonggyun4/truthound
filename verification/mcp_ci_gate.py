"""Mandatory MCP lane: real SDK, intended Python, nonempty complete no-skip JUnit."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

TEST_FILES = (
    "tests/test_mcp.py",
    "tests/test_mcp_bundle.py",
    "tests/test_mcp_faults.py",
    "tests/test_mcp_environment.py",
    "tests/test_polars_dependency_contract.py",
    "tests/test_mcp_lifecycle.py",
    "tests/test_mcp_ci_gate.py",
    "tests/test_mcp_dtype_privacy.py",
    "tests/test_mcp_registry.py",
    "tests/test_mcp_core_profile.py",
    "tests/test_mcp_resources.py",
    "tests/test_mcp_approval.py",
    "tests/test_mcp_validator_transport.py",
    "tests/test_mcp_core_artifacts.py",
    "tests/test_mcp_advanced.py",
    "tests/test_mcp_lineage.py",
    "tests/test_mcp_bindings.py",
    "tests/test_mcp_plugin_bindings.py",
    "tests/test_mcp_schema_lifecycle.py",
    "tests/test_core_execution_accounting.py",
    "tests/test_mcp_profiling.py",
    "tests/test_mcp_suites.py",
    "tests/test_mcp_context_lifecycle.py",
    "tests/test_mcp_algorithm_parity.py",
    "tests/test_mcp_sources.py",
    "tests/test_core_referential_empty.py",
    "tests/test_mcp_streams.py",
    "tests/test_mcp_drift_tails.py",
    "tests/test_core_online_svm_native.py",
    "tests/test_reference_cache.py",
    "tests/test_validators_drift.py",
    "tests/test_core_frequency_order.py",
    "tests/test_mcp_duckdb.py",
    "tests/test_core_result_reentry.py",
    "tests/test_core_suite_identity.py",
    "tests/test_core_resource_limits.py",
    "tests/test_mcp_suite_identity.py",
    "tests/unit/validators/test_pushdown_support.py",
)


def verify_runtime(expected_python: str | None = None) -> dict:
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if expected_python is not None and actual != expected_python:
        raise ValueError("MCP_PYTHON_MISMATCH")
    sdk = importlib.metadata.version("mcp")
    from packaging.version import Version

    if (
        not Version("1.30") <= Version(sdk) < Version("2")
        or importlib.util.find_spec("mcp") is None
    ):
        raise ValueError("MCP_SDK_REQUIRED")
    return {
        "python": platform.python_version(),
        "mcp": sdk,
        "polars": importlib.metadata.version("polars"),
    }


def validate_report(path: Path, required_files=TEST_FILES) -> dict:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if not cases:
        raise ValueError("MCP_NO_TESTS")
    for tag in ("skipped", "failure", "error"):
        if any(case.find(tag) is not None for case in cases):
            raise ValueError("MCP_TEST_" + tag.upper())
    covered = {part for case in cases for part in case.get("classname", "").split(".")}
    if any(Path(file).stem not in covered for file in required_files):
        raise ValueError("MCP_MISSING_TEST_FILE")
    return {"tests": len(cases), "skipped": 0, "required_files": len(required_files)}


def proof_helpers():
    spec = importlib.util.spec_from_file_location(
        "truthound_required_mcp_proof", Path(__file__).with_name("mcp_collection_proof.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def junit_key(node):
    path, _, target = node.partition("::")
    name, bracket, parameters = target.partition("[")
    parts = name.split("::")
    return (
        ".".join([path.removesuffix(".py").replace("/", "."), *parts[:-1]]),
        parts[-1] + bracket + parameters,
    )


def aggregate(directory: Path, count: int, destination: Path):
    helpers = proof_helpers()
    files = sorted(directory.glob("mcp-*.sources.json"))
    if len(files) != count or not 1 <= count <= 16:
        raise ValueError("MCP_MISSING_SHARD")
    combined, full, reference, indices, phases = {}, None, None, set(), 0
    suites = ET.Element("testsuites")
    for path in files:
        proof = json.loads(path.read_text(encoding="utf-8"))
        index = proof["shard_index"]
        if proof.get("valid") is not True or proof["shard_count"] != count or index in indices:
            raise ValueError("MCP_INVALID_SHARD_PROOF")
        indices.add(index)
        expected = helpers.partition(proof["full_collection"], count, index)
        if not expected or proof["collection"] != expected or combined.keys() & expected.keys():
            raise ValueError("MCP_SHARD_COLLECTION_MISMATCH")
        if proof["tests"] != len(expected) or proof["execution_phases"] != 3 * len(expected):
            raise ValueError("MCP_MISSING_SHARD_PHASES")
        identity = {
            key: proof[key]
            for key in ("package_sources", "test_and_verification_sources", "project_sources")
        }
        if reference is None:
            reference, full = identity, proof["full_collection"]
        if identity != reference or full != proof["full_collection"]:
            raise ValueError("MCP_SHARD_SOURCE_MISMATCH")
        report = path.with_suffix("").with_suffix(".xml")
        validate_report(report, tuple({node.split("::")[0] for node in expected}))
        cases = list(ET.parse(report).getroot().iter("testcase"))
        # Proof establishes exact pytest identity; JUnit must independently
        # account for the same number of actual successful executions.
        if Counter((case.get("classname"), case.get("name")) for case in cases) != Counter(
            junit_key(node) for node in expected
        ):
            raise ValueError("MCP_SHARD_REPORT_MISMATCH")
        for suite in ET.parse(report).getroot():
            suites.append(suite)
        combined.update(expected)
        phases += proof["execution_phases"]
    if indices != set(range(count)) or combined != full:
        raise ValueError("MCP_INCOMPLETE_PARTITION")
    if reference != {
        "package_sources": helpers.sources(),
        "test_and_verification_sources": helpers.supporting_sources(),
        "project_sources": helpers.project_sources(),
    }:
        raise ValueError("MCP_CURRENT_SOURCE_MISMATCH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suites).write(destination, encoding="utf-8")
    result = validate_report(destination)
    destination.with_suffix(".sources.json").write_text(
        json.dumps(
            {
                "valid": True,
                "collection": combined,
                **reference,
                "tests": len(combined),
                "execution_phases": phases,
                "aggregate_shard_count": count,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-python")
    parser.add_argument("--junitxml", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--aggregate-dir", type=Path)
    args = parser.parse_args()
    try:
        runtime = verify_runtime(args.expected_python)
        if args.aggregate_dir is not None:
            result = aggregate(args.aggregate_dir, args.shard_count, args.junitxml)
            print(json.dumps({"stage": "complete-mcp-partition", **runtime, **result}))
            return 0
        args.junitxml.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *TEST_FILES,
                "-p",
                "no:cacheprovider",
                "-p",
                "verification.mcp_collection_proof",
                "--mcp-source-proof=" + str(args.junitxml.with_suffix(".sources.json")),
                "--mcp-shard-count=" + str(args.shard_count),
                "--mcp-shard-index=" + str(args.shard_index),
                f"--junitxml={args.junitxml}",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        if result.returncode:
            return result.returncode
        print(
            json.dumps(
                {
                    "stage": "required-mcp-integration",
                    **runtime,
                    **validate_report(
                        args.junitxml,
                        tuple(
                            {
                                node.split("::")[0]
                                for node in json.loads(
                                    args.junitxml.with_suffix(".sources.json").read_text(
                                        encoding="utf-8"
                                    )
                                )["collection"]
                            }
                        )
                        if args.shard_count != 1
                        else TEST_FILES,
                    ),
                }
            )
        )
        return 0
    except (
        ValueError,
        KeyError,
        TypeError,
        importlib.metadata.PackageNotFoundError,
        ET.ParseError,
        OSError,
    ):
        print("MCP_REQUIRED_GATE_FAILED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
