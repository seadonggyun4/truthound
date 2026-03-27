from __future__ import annotations

from enum import Enum


class PlannedExecutionMode(str, Enum):
    """Planner-selected coarse execution strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PUSHDOWN = "pushdown"


class RuntimeExecutionMode(str, Enum):
    """Actual runtime execution modes observed during execution."""

    SEQUENTIAL = "sequential"
    THREADPOOL = "threadpool"
    PARALLEL = "parallel"
    PUSHDOWN = "pushdown"


def normalize_runtime_execution_mode(value: str | RuntimeExecutionMode | None) -> str:
    """Normalize persisted or user-provided runtime execution mode values."""

    raw = value.value if isinstance(value, RuntimeExecutionMode) else str(value or "").lower()
    if raw in {
        RuntimeExecutionMode.SEQUENTIAL.value,
        RuntimeExecutionMode.THREADPOOL.value,
        RuntimeExecutionMode.PARALLEL.value,
        RuntimeExecutionMode.PUSHDOWN.value,
    }:
        return raw
    return RuntimeExecutionMode.SEQUENTIAL.value


def coarse_planned_execution_mode(value: str | RuntimeExecutionMode | PlannedExecutionMode | None) -> str:
    """Map runtime or mixed-mode values to the planner's coarse execution mode."""

    raw = value.value if isinstance(value, Enum) else str(value or "").lower()
    if raw == RuntimeExecutionMode.THREADPOOL.value:
        return PlannedExecutionMode.SEQUENTIAL.value
    if raw in {
        PlannedExecutionMode.SEQUENTIAL.value,
        PlannedExecutionMode.PARALLEL.value,
        PlannedExecutionMode.PUSHDOWN.value,
    }:
        return raw
    return PlannedExecutionMode.SEQUENTIAL.value


def normalize_planned_execution_mode(value: str | PlannedExecutionMode | None) -> str:
    """Normalize planner-selected execution mode values."""

    raw = value.value if isinstance(value, PlannedExecutionMode) else str(value or "").lower()
    if raw in {
        PlannedExecutionMode.SEQUENTIAL.value,
        PlannedExecutionMode.PARALLEL.value,
        PlannedExecutionMode.PUSHDOWN.value,
    }:
        return raw
    return coarse_planned_execution_mode(raw)
