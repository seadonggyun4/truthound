"""Cancellation cannot abandon a native benchmark thread owned by the service."""

import asyncio
import threading

import pytest

from truthound.benchmark import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkRunner,
    RunnerConfig,
    benchmark_registry,
)
from truthound.mcp.benchmark_services import BenchmarkRuntime, BenchmarkWorkload
from truthound.mcp.benchmark_worker import BenchmarkOwner


@pytest.mark.asyncio
async def test_cancelled_benchmark_waiter_joins_native_work_and_teardown():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class WaitingBenchmark(Benchmark):
        name = "synthetic-owned-wait"

        def run_iteration(self):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("synthetic test release missing")
            return 1

        def teardown(self):
            finished.set()

    benchmark_registry.register(WaitingBenchmark)
    runner = BenchmarkRunner(
        RunnerConfig(save_results=False),
        BenchmarkConfig(warmup_iterations=0, measure_iterations=1, cooldown_seconds=0),
    )
    owner = BenchmarkOwner(
        BenchmarkRuntime({"local": BenchmarkWorkload(runner, "single", WaitingBenchmark.name)}), {}
    )
    task = asyncio.create_task(
        owner.execute({"command": "run", "workload_id": "local"}, 1024 * 1024)
    )
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        asyncio.get_running_loop().call_later(0.05, release.set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        await owner.close()
        assert not owner.results
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 5)
        benchmark_registry.unregister(WaitingBenchmark.name)


@pytest.mark.asyncio
async def test_empty_suite_is_incomplete_and_failed_reports_redact_private_error():
    from truthound.benchmark import BenchmarkSuite
    from truthound.mcp.broker_worker import NativeServiceResult

    class FailedBenchmark(Benchmark):
        name = "synthetic-redacted-report"

        def run_iteration(self):
            from truthound.benchmark.base import MetricUnit, MetricValue

            self._metrics.add_custom_metric(
                MetricValue("count", 1, MetricUnit.COUNT, tags={"private": "private-metric-secret"})
            )
            raise RuntimeError("private-benchmark-secret")

    benchmark_registry.register(FailedBenchmark)
    runner = BenchmarkRunner(
        RunnerConfig(save_results=False),
        BenchmarkConfig(warmup_iterations=0, measure_iterations=1, cooldown_seconds=0),
    )
    owner = BenchmarkOwner(
        BenchmarkRuntime(
            {
                "empty": BenchmarkWorkload(runner, "suite", BenchmarkSuite("empty")),
                "failed": BenchmarkWorkload(runner, "single", FailedBenchmark.name),
            }
        ),
        {},
    )
    try:
        empty = await owner.execute({"command": "run", "workload_id": "empty"}, 1024 * 1024)
        assert isinstance(empty, NativeServiceResult)
        failed = await owner.execute({"command": "run", "workload_id": "failed"}, 1024 * 1024)
        assert isinstance(failed, NativeServiceResult)
        assert "private-metric-secret" not in str(failed.payload)
        for format in ("markdown", "html"):
            report = await owner.execute(
                {
                    "command": "report",
                    "result_id": failed.payload["result_id"],
                    "format": format,
                },
                1024 * 1024,
            )
            assert b"private-benchmark-secret" not in report.content
            assert (
                b"native_benchmark_failed" if format == "markdown" else b"FAIL"
            ) in report.content
    finally:
        await owner.close()
        benchmark_registry.unregister(FailedBenchmark.name)


@pytest.mark.asyncio
async def test_suite_comparison_retains_incomplete_verdict_for_missing_measurements():
    from truthound.benchmark.base import BenchmarkMetrics, BenchmarkResult
    from truthound.benchmark.comparison import BenchmarkComparator
    from truthound.benchmark.runner import SuiteResult
    from truthound.mcp.broker_worker import NativeServiceResult

    owner = BenchmarkOwner(BenchmarkRuntime({}, comparators={"local": BenchmarkComparator()}), {})
    try:
        for key, name in [("baseline", "before"), ("current", "after")]:
            owner.results[key] = (
                SuiteResult(
                    suite_name=key,
                    environment=None,
                    results=[
                        BenchmarkResult(
                            benchmark_name=name,
                            category="synthetic",
                            success=True,
                            environment=None,
                            metrics=BenchmarkMetrics(durations=[1.0]),
                        )
                    ],
                ),
                0,
            )
        result = await owner.execute(
            {
                "command": "compare",
                "result_id": "current",
                "baseline_id": "baseline",
                "comparator_id": "local",
            },
            1024 * 1024,
        )
        assert isinstance(result, NativeServiceResult)
        assert result.payload == {"comparable": False}
        assert result.failure_code == "native_benchmark_incomplete"
    finally:
        await owner.close()
