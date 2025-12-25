"""End-to-End Test Framework for Truthound.

This package provides a comprehensive E2E testing infrastructure with:
- Test fixtures and data generators
- Scenario-based test organization
- CLI integration testing utilities
- Test utilities and helpers

Modules:
    fixtures: Reusable test fixtures and data generators
    utils: Test utilities and helpers
    test_core_paths: Core path E2E tests

Example:
    # Run all E2E tests
    pytest tests/e2e/ -v

    # Run specific test class
    pytest tests/e2e/test_core_paths.py::TestDataProfilingPath -v

    # Run with markers
    pytest tests/e2e/ -v -m "not slow"
"""

__all__ = [
    # Submodules are imported on demand to avoid circular imports
]
