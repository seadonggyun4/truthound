"""Runtime capture and benchmarking helpers for reporter SDK tests."""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

from truthound.reporters.sdk._testing_models import MockValidationResult, create_mock_result


@dataclass
class CapturedOutput:
    """Container for captured reporter output."""

    content: Any
    duration_ms: float
    memory_bytes: int
    exception: Exception | None = None


def capture_output(
    reporter: Any,
    result: MockValidationResult,
    method: str = "render",
) -> CapturedOutput:
    """Capture reporter output with timing and memory information."""
    tracemalloc.start()
    start_time = time.perf_counter()
    exception: Exception | None = None
    content: Any = None

    try:
        render_func = getattr(reporter, method)
        content = render_func(result)
    except Exception as exc:
        exception = exc

    end_time = time.perf_counter()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return CapturedOutput(
        content=content,
        duration_ms=(end_time - start_time) * 1000,
        memory_bytes=peak,
        exception=exception,
    )


@dataclass
class BenchmarkResult:
    """Result of a reporter benchmark."""

    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    peak_memory_bytes: int
    output_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "p50_time_ms": self.p50_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "p99_time_ms": self.p99_time_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "output_size_bytes": self.output_size_bytes,
        }

    def __repr__(self) -> str:
        return (
            "BenchmarkResult(\n"
            f"  iterations={self.iterations},\n"
            f"  avg_time={self.avg_time_ms:.2f}ms,\n"
            f"  p95_time={self.p95_time_ms:.2f}ms,\n"
            f"  peak_memory={self.peak_memory_bytes / 1024:.2f}KB,\n"
            f"  output_size={self.output_size_bytes / 1024:.2f}KB\n"
            ")"
        )


def benchmark_reporter(
    reporter: Any,
    result: MockValidationResult | None = None,
    iterations: int = 100,
    warmup: int = 5,
    method: str = "render",
) -> BenchmarkResult:
    """Benchmark reporter performance."""
    if result is None:
        result = create_mock_result(passed=10, failed=2)

    render_func = getattr(reporter, method)

    for _ in range(warmup):
        render_func(result)

    times_ms: list[float] = []
    peak_memory = 0
    output_size = 0
    tracemalloc.start()

    for _ in range(iterations):
        start = time.perf_counter()
        output = render_func(result)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

        _current, peak = tracemalloc.get_traced_memory()
        peak_memory = max(peak_memory, peak)

        if output_size == 0:
            if isinstance(output, str):
                output_size = len(output.encode("utf-8"))
            elif isinstance(output, bytes):
                output_size = len(output)
            elif isinstance(output, dict):
                output_size = len(json.dumps(output).encode("utf-8"))

    tracemalloc.stop()
    times_ms.sort()

    return BenchmarkResult(
        iterations=iterations,
        total_time_ms=sum(times_ms),
        avg_time_ms=statistics.mean(times_ms),
        min_time_ms=min(times_ms),
        max_time_ms=max(times_ms),
        p50_time_ms=times_ms[len(times_ms) // 2],
        p95_time_ms=times_ms[int(len(times_ms) * 0.95)],
        p99_time_ms=times_ms[int(len(times_ms) * 0.99)],
        peak_memory_bytes=peak_memory,
        output_size_bytes=output_size,
    )


__all__ = [
    "BenchmarkResult",
    "CapturedOutput",
    "benchmark_reporter",
    "capture_output",
]
