from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import truthound
from truthound import __version__
from truthound._version import (
    get_source_checkout_version,
    resolve_truthound_version,
)
from truthound.benchmark.parity import TruthoundAdapter
from truthound.cli_modules.scaffolding.plugins import (
    DEFAULT_TRUTHOUND_PLUGIN_MIN_VERSION as SCAFFOLDING_PLUGIN_MIN_VERSION,
)
from truthound.plugins.cli import (
    DEFAULT_TRUTHOUND_PLUGIN_MIN_VERSION as CLI_PLUGIN_MIN_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["project"]["version"]


@pytest.mark.contract
def test_runtime_version_matches_pyproject_version() -> None:
    assert __version__ == _project_version()
    assert truthound.__version__ == _project_version()


@pytest.mark.contract
def test_version_helper_uses_pyproject_when_metadata_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import truthound._version as version_module

    def _raise_package_not_found(_: str) -> str:
        raise version_module.importlib_metadata.PackageNotFoundError()

    monkeypatch.setattr(version_module.importlib_metadata, "version", _raise_package_not_found)

    assert get_source_checkout_version(REPO_ROOT / "src" / "truthound") == _project_version()
    assert resolve_truthound_version(REPO_ROOT / "src" / "truthound") == _project_version()


@pytest.mark.contract
def test_plugin_default_min_versions_follow_unified_core_version() -> None:
    expected = resolve_truthound_version(REPO_ROOT / "src" / "truthound")

    assert expected == CLI_PLUGIN_MIN_VERSION
    assert expected == SCAFFOLDING_PLUGIN_MIN_VERSION


@pytest.mark.contract
def test_benchmark_truthound_adapter_uses_unified_version_surface() -> None:
    assert TruthoundAdapter().framework_version() == resolve_truthound_version(
        REPO_ROOT / "src" / "truthound"
    )


@pytest.mark.contract
def test_root_module_docstring_drops_stale_hardcoded_minor_line() -> None:
    init_text = (REPO_ROOT / "src" / "truthound" / "__init__.py").read_text(encoding="utf-8")

    assert "Truthound 3.0 keeps the root package intentionally small" not in init_text
    assert "Truthound keeps the root package intentionally small" in init_text
