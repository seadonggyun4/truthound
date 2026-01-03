"""Common pytest configuration for integration tests.

This module provides shared pytest fixtures, markers, and options
that are used across all integration test modules.

Usage:
    pytest tests/integration/ -v
    pytest tests/integration/ -v --skip-expensive
    pytest tests/integration/ -v --skip-slow
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add common command line options for integration tests.

    Note: These options are defined here to avoid duplication across
    multiple conftest.py files (cloud_dw, external, stores).
    """
    # Only add if not already present (handles pytest plugin reloading)
    try:
        parser.addoption(
            "--skip-expensive",
            action="store_true",
            default=False,
            help="Skip expensive tests (cloud costs, long duration)",
        )
    except ValueError:
        # Option already added
        pass

    try:
        parser.addoption(
            "--skip-slow",
            action="store_true",
            default=False,
            help="Skip slow tests (> 30 seconds)",
        )
    except ValueError:
        # Option already added
        pass


def pytest_configure(config: pytest.Config) -> None:
    """Configure common pytest markers for integration tests."""
    config.addinivalue_line(
        "markers",
        "expensive: marks test as expensive (cloud costs, long duration)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks test as slow (> 30 seconds)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks test as requiring external services",
    )
