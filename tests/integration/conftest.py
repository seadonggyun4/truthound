"""Common pytest configuration for integration tests.

This module provides shared pytest fixtures, markers, and options
that are used across all integration test modules.

Usage:
    pytest tests/integration/ -v --run-integration
    pytest tests/integration/ -v --run-integration --run-expensive
"""

from __future__ import annotations

import pytest

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
