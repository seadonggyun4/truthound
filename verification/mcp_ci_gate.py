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
from pathlib import Path

TEST_FILES = (
    "tests/test_mcp.py",
    "tests/test_mcp_bundle.py",
    "tests/test_mcp_faults.py",
    "tests/test_mcp_environment.py",
    "tests/test_polars_dependency_contract.py",
    "tests/test_mcp_lifecycle.py",
    "tests/test_mcp_ci_gate.py",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-python")
    parser.add_argument("--junitxml", type=Path, required=True)
    args = parser.parse_args()
    try:
        runtime = verify_runtime(args.expected_python)
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
                f"--junitxml={args.junitxml}",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        if result.returncode:
            return result.returncode
        print(
            json.dumps(
                {"stage": "required-mcp-integration", **runtime, **validate_report(args.junitxml)}
            )
        )
        return 0
    except (ValueError, importlib.metadata.PackageNotFoundError, ET.ParseError, OSError):
        print("MCP_REQUIRED_GATE_FAILED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
