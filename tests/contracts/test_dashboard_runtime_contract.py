from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = REPO_ROOT.parent / "truthound-dashboard"


def _load_truthound_pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _load_dashboard_contract() -> dict[str, object]:
    contract_path = (
        DASHBOARD_ROOT
        / "deployment"
        / "private-runtime"
        / "dashboard-runtime.contract.toml"
    )
    if not contract_path.exists():
        pytest.skip("truthound-dashboard sibling repository is not checked out")
    return tomllib.loads(contract_path.read_text(encoding="utf-8"))


def test_internal_dashboard_runtime_manifest_exists_and_matches_optional_dependencies() -> None:
    pyproject = _load_truthound_pyproject()
    optional = pyproject["project"]["optional-dependencies"]
    manifest = pyproject["tool"]["truthound"]["internal_runtime"]["dashboard_runtime"]

    included = manifest["included_extras"]
    excluded = manifest["excluded_extras"]

    assert included == ["ai", "drift", "anomaly", "reports"]
    assert set(included) <= set(optional)
    assert set(excluded) <= set(optional)
    assert "dashboard-runtime" not in optional
    assert "connector extras" in manifest["rationale"].lower()
    assert "dashboard release line" in manifest["version_policy"].lower()


def test_all_extra_is_a_superset_of_internal_dashboard_runtime_dependencies() -> None:
    pyproject = _load_truthound_pyproject()
    optional = pyproject["project"]["optional-dependencies"]
    manifest = pyproject["tool"]["truthound"]["internal_runtime"]["dashboard_runtime"]

    all_extra = set(optional["all"])
    included_deps = {
        dep
        for extra in manifest["included_extras"]
        for dep in optional[extra]
    }

    assert included_deps <= all_extra


def test_dashboard_runtime_contract_matches_dashboard_repository_snapshot_when_available() -> None:
    pyproject = _load_truthound_pyproject()
    manifest = pyproject["tool"]["truthound"]["internal_runtime"]["dashboard_runtime"]
    dashboard_contract = _load_dashboard_contract()["dashboard_runtime"]

    assert dashboard_contract["contract_name"] == "dashboard-runtime"
    assert dashboard_contract["truthound_version"] == pyproject["project"]["version"]
    assert dashboard_contract["included_extras"] == manifest["included_extras"]
    assert dashboard_contract["excluded_extras"] == manifest["excluded_extras"]


def test_dashboard_runtime_contract_is_not_exposed_in_public_truthound_docs() -> None:
    disallowed = "dashboard-runtime"
    surfaces = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
        REPO_ROOT / "mkdocs.yml",
        REPO_ROOT / "mkdocs.public.yml",
        REPO_ROOT / "docs" / "public_docs.yml",
    ]

    for path in surfaces:
        assert disallowed not in path.read_text(encoding="utf-8"), str(path)
