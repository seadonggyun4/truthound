"""Default installation must not resolve a major that breaks existing Core APIs."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.contract


def test_default_dependency_supports_actual_core_pii_scan(tmp_path):
    import truthound as th
    from truthound.types import PIIType

    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    requirement = next(
        Requirement(item) for item in project["dependencies"] if Requirement(item).name == "polars"
    )
    assert "1.37.1" in requirement.specifier
    assert "2.0.0" not in requirement.specifier
    source = tmp_path / "input.csv"
    source.write_text("email,x\nsynthetic@example.com,1\n,2\n", encoding="utf-8")
    result = th.scan(str(source))
    assert result.row_count == 2
    assert any(
        item["column"] == "email" and item["pii_type"] == PIIType.EMAIL.value
        for item in result.findings
    )
