"""Deterministic fault-injection helpers for fast regression tests."""

from __future__ import annotations

from tests.stress.chaos import FailureConfig, FailureInjector, FailureType


DEFAULT_FAULT_SEED = 42


def deterministic_injector(*configs: FailureConfig) -> FailureInjector:
    """Build a reproducible injector for PR-friendly fault scenarios."""
    injector = FailureInjector(enabled=True, seed=DEFAULT_FAULT_SEED)
    for config in configs:
        injector.add_failure(config)
    return injector


def always_fail(
    failure_type: FailureType,
    *,
    target: str = "default",
    duration_ms: float = 0.0,
    message: str | None = None,
) -> FailureConfig:
    """Create a deterministic failure configuration that always fires."""
    metadata = {"message": message} if message is not None else {}
    return FailureConfig(
        failure_type=failure_type,
        probability=1.0,
        duration_ms=duration_ms,
        targets=[target],
        metadata=metadata,
    )
