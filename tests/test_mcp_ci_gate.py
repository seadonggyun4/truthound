"""The required integration lane may not silently pass missing/optional coverage."""

import importlib.metadata
import importlib.util
from pathlib import Path

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
