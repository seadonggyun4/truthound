"""Fail-closed algorithm coverage report from an actual mandatory MCP proof.

Discovery/constructor coverage is not execution parity. Report-only mode emits
gaps without treating them as successful full coverage.
"""

import argparse
import hashlib
import json
from pathlib import Path

from truthound.mcp.validator_wire import contracts


def coverage(proof):
    root = Path(__file__).resolve().parents[1]
    if proof.get("valid") is not True:
        raise ValueError("MCP_PARITY_EXECUTION_PROOF_FAILED")
    supporting = proof.get("test_and_verification_sources", {})
    for name in ("pyproject.toml", "uv.lock"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != proof.get(
            "project_sources", {}
        ).get(name):
            raise ValueError("MCP_PARITY_PROJECT_MISMATCH")
    for relative in (
        "tests/mcp_remaining_cases.py",
        "tests/mcp_error_cases.py",
        "tests/test_mcp_plugin_bindings.py",
        "tests/mcp_validator_cases.py",
        "tests/mcp_additional_cases.py",
        "tests/test_mcp_algorithm_parity.py",
    ):
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != supporting.get(relative):
            raise ValueError("MCP_PARITY_SOURCE_MISMATCH")
    import truthound

    package = Path(truthound.__file__).resolve().parent
    current = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    if current != proof.get("package_sources"):
        raise ValueError("MCP_PARITY_PACKAGE_MISMATCH")
    registered = set(contracts()["validators"])
    exercised = {polarity: set() for polarity in ("positive", "negative", "error", "caught_error")}
    prefixes = (
        "tests/test_mcp_algorithm_parity.py::test_actual_algorithm_matches_core_with_full_canonical_evidence[",
        "tests/test_mcp_plugin_bindings.py::test_row_level_algorithm_matches_core_with_full_canonical_evidence[",
    )
    constructor_prefix = "tests/test_mcp_algorithm_parity.py::test_actual_constructor_configuration_error_is_not_execution_coverage["
    constructor_errors = set()
    swallowed_errors = set()
    swallowed_prefix = "tests/test_mcp_algorithm_parity.py::test_swallowed_native_error_matches_core_with_observed_exception["
    uncertain_constructors = set()
    uncertain_prefix = "tests/test_mcp_algorithm_parity.py::test_approved_constructor_failure_preserves_uncertain_effects["
    for node in proof.get("collection", {}):
        if node.startswith(swallowed_prefix) and node.endswith("]"):
            name = node[len(swallowed_prefix) : -1]
            if name not in registered or name in swallowed_errors:
                raise ValueError("MCP_PARITY_INVALID_SWALLOWED_ERROR")
            swallowed_errors.add(name)
        if (
            node
            == "tests/test_mcp_plugin_bindings.py::test_row_level_constructor_failure_is_effect_unknown"
        ):
            uncertain_constructors.add("row_level")
        if node.startswith(uncertain_prefix) and node.endswith("]"):
            name = node[len(uncertain_prefix) : -1]
            if name not in registered or name in uncertain_constructors:
                raise ValueError("MCP_PARITY_INVALID_UNCERTAIN_CONSTRUCTOR")
            uncertain_constructors.add(name)
        if node.startswith(constructor_prefix) and node.endswith("]"):
            name = node[len(constructor_prefix) : -1]
            if name not in registered or name in constructor_errors:
                raise ValueError("MCP_PARITY_INVALID_CONSTRUCTOR_SCENARIO")
            constructor_errors.add(name)
        prefix = next((value for value in prefixes if node.startswith(value)), None)
        if prefix is not None and node.endswith("]"):
            polarity, separator, name = node[len(prefix) : -1].partition("-")
            if not separator or polarity not in exercised or name not in registered:
                raise ValueError("MCP_PARITY_UNKNOWN_SCENARIO")
            if name in exercised[polarity]:
                raise ValueError("MCP_PARITY_DUPLICATE_SCENARIO")
            exercised[polarity].add(name)
    paired = exercised["positive"] & exercised["negative"]
    runtime_errors = exercised["error"] | exercised["caught_error"] | swallowed_errors
    complete = paired & runtime_errors
    return {
        "registered_algorithms": len(registered),
        "positive_negative_verified": len(paired),
        "error_path_verified": len(exercised["error"]),
        "caught_algorithm_error_verified_separately": len(exercised["caught_error"]),
        "swallowed_algorithm_exception_observed_separately": len(swallowed_errors),
        "runtime_error_boundary_verified": len(runtime_errors),
        "constructor_configuration_error_verified_separately": len(constructor_errors),
        "native_constructor_rejection_with_uncertain_trusted_effects": len(uncertain_constructors),
        "full_constructor_failure_boundary_verified": (constructor_errors | uncertain_constructors)
        == registered,
        "missing_positive_negative": sorted(registered - paired),
        "missing_error_path": sorted(registered - runtime_errors),
        "full_validator_parity": bool(registered)
        and complete == registered
        and (constructor_errors | uncertain_constructors) == registered,
        "full_core_parity": False,
        "limits": "Validator parity cannot establish datasource or advanced-domain parity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    try:
        result = coverage(json.loads(args.proof.read_text(encoding="utf-8")))
    except (ValueError, OSError, KeyError, TypeError):
        print("MCP_PARITY_PROOF_REJECTED")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if args.report_only or result["full_validator_parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
