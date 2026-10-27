"""Lossless reconstruction of native benchmark measurements, with strict JSON input."""

import json
import math
from datetime import datetime
from pathlib import Path

from truthound.benchmark.base import (
    BenchmarkCategory,
    BenchmarkMetrics,
    BenchmarkResult,
    EnvironmentInfo,
    MetricUnit,
    MetricValue,
)


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate benchmark JSON member")
        value[key] = item
    return value


def _constant(value):
    raise ValueError("Nonfinite benchmark JSON value")


def _float(value):
    return _number(float(value))


def _number(value, *, nonnegative=False, integer=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError("Invalid benchmark measurement type")
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("Invalid benchmark measurement value")
    return value


def _environment(value):
    platform = value["platform"]
    return EnvironmentInfo(
        python_version=value["python_version"],
        platform_system=platform["system"],
        platform_release=platform["release"],
        platform_machine=platform["machine"],
        cpu_count=_number(value["cpu_count"], nonnegative=True, integer=True),
        polars_version=value["polars_version"],
        truthound_version=value["truthound_version"],
        timestamp=datetime.fromisoformat(value["timestamp"]),
    )


def _result(value):
    metrics = value["metrics"]
    timing, memory, throughput = metrics["timing"], metrics["memory"], metrics["throughput"]
    if "durations_seconds" not in timing:
        raise ValueError("Legacy benchmark summary lacks raw measurements; rerun before comparison")
    durations = timing["durations_seconds"]
    if not isinstance(durations, list):
        raise ValueError("Invalid benchmark measurements")
    durations = [_number(item, nonnegative=True) for item in durations]
    if type(value["success"]) is not bool or (value["success"] and not durations):
        raise ValueError("Successful benchmark requires measurements")
    if _number(timing["iterations"], nonnegative=True, integer=True) != len(durations):
        raise ValueError("Benchmark iteration count does not match measurements")
    custom = []
    for metric in metrics.get("custom", []):
        if type(metric["is_lower_better"]) is not bool:
            raise ValueError("Invalid custom metric ordering")
        custom.append(
            MetricValue(
                name=metric["name"],
                value=_number(metric["value"]),
                unit=MetricUnit(metric["unit"]),
                is_lower_better=metric["is_lower_better"],
                description=metric.get("description", ""),
                tags=metric.get("tags", {}),
            )
        )
    return BenchmarkResult(
        benchmark_name=value["benchmark_name"],
        category=BenchmarkCategory(value["category"]),
        success=value["success"],
        metrics=BenchmarkMetrics(
            durations=durations,
            peak_memory_bytes=_number(memory["peak_bytes"], nonnegative=True, integer=True),
            memory_delta_bytes=_number(memory["delta_bytes"], integer=True),
            rows_processed=_number(throughput["rows_processed"], nonnegative=True, integer=True),
            bytes_processed=_number(throughput["bytes_processed"], nonnegative=True, integer=True),
            custom_metrics=custom,
        ),
        environment=_environment(value["environment"]),
        parameters=value.get("parameters", {}),
        error=value.get("error"),
        started_at=datetime.fromisoformat(value["started_at"]),
        completed_at=datetime.fromisoformat(value["completed_at"])
        if value.get("completed_at")
        else None,
        tags=value.get("tags", {}),
    )


def load_suite(path):
    from truthound.benchmark.runner import SuiteResult

    with Path(path).open("rb") as stream:
        payload = stream.read(64 * 1024 * 1024 + 1)
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError("Benchmark evidence exceeds 64 MiB")
    value = json.loads(
        payload, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float
    )
    try:
        return SuiteResult(
            suite_name=value["suite_name"],
            results=[_result(item) for item in value["results"]],
            environment=_environment(value["environment"]),
            started_at=datetime.fromisoformat(value["started_at"]),
            completed_at=datetime.fromisoformat(value["completed_at"])
            if value.get("completed_at")
            else None,
            config=value.get("config", {}),
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid benchmark evidence structure") from error
