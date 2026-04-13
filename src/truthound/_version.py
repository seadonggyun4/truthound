"""Private helpers for resolving the canonical Truthound core version."""

from __future__ import annotations

import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path

TRUTHOUND_DISTRIBUTION_NAME = "truthound"
TRUTHOUND_SAFE_FALLBACK_VERSION = "0.0.0.dev"


def _read_project_version(pyproject_path: Path) -> str | None:
    """Read ``[project].version`` from a pyproject file."""

    try:
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return None

    project = payload.get("project", {})
    version = project.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return None


def _nearest_pyproject(start: Path | None = None) -> Path | None:
    """Return the nearest ``pyproject.toml`` visible from ``start``."""

    origin = (start or Path(__file__)).resolve()
    search_root = origin if origin.is_dir() else origin.parent

    for directory in (search_root, *search_root.parents):
        pyproject_path = directory / "pyproject.toml"
        if pyproject_path.is_file():
            return pyproject_path
    return None


def get_installed_truthound_version() -> str | None:
    """Return the installed distribution version when package metadata exists."""

    try:
        return importlib_metadata.version(TRUTHOUND_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError:
        return None


def get_source_checkout_version(start: Path | None = None) -> str | None:
    """Return the nearest source-checkout version from ``pyproject.toml``."""

    pyproject_path = _nearest_pyproject(start)
    if pyproject_path is None:
        return None
    return _read_project_version(pyproject_path)


def resolve_truthound_version(start: Path | None = None) -> str:
    """Resolve the canonical Truthound core version.

    Resolution order:
    1. Installed package metadata
    2. Source checkout ``pyproject.toml``
    3. Safe development fallback
    """

    return (
        get_installed_truthound_version()
        or get_source_checkout_version(start)
        or TRUTHOUND_SAFE_FALLBACK_VERSION
    )
