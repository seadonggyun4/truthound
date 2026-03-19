from __future__ import annotations

from pathlib import Path

import pytest


_LANE_MARKERS = {"contract", "fault", "integration", "soak", "manual_verification"}
_COMPAT_MARKERS = {"slow", "e2e", "scale_100m", "expensive", "stress"}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("truthound-test-lanes")
    group.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services or backends.",
    )
    group.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="Run slow or expensive tests in addition to the fast default lane.",
    )
    group.addoption(
        "--run-soak",
        action="store_true",
        default=False,
        help="Run soak, stress, and large-scale chaos tests.",
    )
    group.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag. Slow tests are skipped by default.",
    )
    group.addoption(
        "--skip-expensive",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag. Expensive tests are skipped by default.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "contract: deterministic public-contract regression coverage",
    )
    config.addinivalue_line(
        "markers",
        "fault: deterministic failure injection, timeout, corruption, and concurrency coverage",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests requiring external services or backend integration",
    )
    config.addinivalue_line(
        "markers",
        "soak: long-running or probabilistic chaos coverage intended for nightly execution",
    )
    config.addinivalue_line(
        "markers",
        "manual_verification: manual verification artifacts that must not run under pytest by default",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_integration = config.getoption("--run-integration")
    run_expensive = config.getoption("--run-expensive")
    run_soak = config.getoption("--run-soak")
    skip_slow = config.getoption("--skip-slow")
    skip_expensive = config.getoption("--skip-expensive")

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        _apply_default_lane_markers(item)

        item_markers = {mark.name for mark in item.iter_markers()}

        if "manual_verification" in item_markers:
            deselected.append(item)
            continue
        if "integration" in item_markers and not run_integration:
            deselected.append(item)
            continue
        if ("soak" in item_markers or "stress" in item_markers or "scale_100m" in item_markers) and not run_soak:
            deselected.append(item)
            continue
        if "expensive" in item_markers and (skip_expensive or not run_expensive):
            deselected.append(item)
            continue
        if "slow" in item_markers and (skip_slow or not (run_expensive or run_integration or run_soak)):
            deselected.append(item)
            continue

        selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _apply_default_lane_markers(item: pytest.Item) -> None:
    path = Path(str(item.fspath))
    parts = set(path.parts)
    existing_markers = {mark.name for mark in item.iter_markers()}

    if "integration" not in existing_markers and "integration" in parts:
        item.add_marker(pytest.mark.integration)
        existing_markers.add("integration")

    if "soak" not in existing_markers and "stress" in parts:
        item.add_marker(pytest.mark.soak)
        existing_markers.add("soak")

    if "manual_verification" not in existing_markers and "phase6_verification" in parts:
        item.add_marker(pytest.mark.manual_verification)
        existing_markers.add("manual_verification")

    lane_like_markers = _LANE_MARKERS | _COMPAT_MARKERS
    if existing_markers.isdisjoint(lane_like_markers):
        item.add_marker(pytest.mark.contract)
