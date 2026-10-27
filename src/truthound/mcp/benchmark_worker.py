"""Native benchmark owner with retained results and explicit failure artifacts."""

import asyncio
import io
import json
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

import polars as pl

from truthound.mcp.benchmark_services import BenchmarkRuntime, BenchmarkWorkload
from truthound.mcp.broker_worker import NativeServiceBytes, NativeServiceResult
from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError


def projection(result):
    """Keep measurements and verdicts; never expose bound kwargs or tracebacks."""
    from truthound.benchmark.parity import ParityResult
    from truthound.benchmark.runner import SuiteResult

    if isinstance(result, pl.DataFrame):
        return {
            "kind": "generated_data",
            "rows": result.height,
            "columns": [
                {"name": name, "dtype": str(dtype)} for name, dtype in result.schema.items()
            ],
        }
    if isinstance(result, SuiteResult):
        value = result.to_dict()
        value.pop("config", None)
        value["results"] = [projection(item) for item in result.results]
        return value
    value = result.to_dict()
    for key in ("parameters", "tags", "error_traceback", "artifact_root", "metadata"):
        value.pop(key, None)
    if isinstance(result, ParityResult):
        # Detailed parity evidence remains in the operator-selected artifact root.
        value.pop("observations", None)
        value.pop("environment", None)
        value["assertions"] = [
            {"name": item.name, "passed": item.passed, "severity": item.severity}
            for item in result.assertions
        ]
    else:
        for metric in value.get("metrics", {}).get("custom", []):
            metric.pop("tags", None)
        if value.get("error"):
            value["error"] = "native_benchmark_failed"
    return value


class BenchmarkOwner:
    def __init__(self, runtime, request):
        from truthound.benchmark import BenchmarkRunner, ParityRunner
        from truthound.benchmark.comparison import (
            BenchmarkComparator,
            PerformanceThreshold,
            RegressionDetector,
        )
        from truthound.benchmark.generators import DataGenerator, GeneratorConfig

        if not isinstance(runtime, BenchmarkRuntime):
            raise MCPError("invalid_binding")
        if any(
            not isinstance(item, BenchmarkWorkload)
            or item.kind
            not in {"single", "suite", "all", "category", "parity", "generator", "generator_lazy"}
            or not isinstance(
                item.runner,
                DataGenerator
                if item.kind.startswith("generator")
                else ParityRunner
                if item.kind == "parity"
                else BenchmarkRunner,
            )
            for item in runtime.workloads.values()
        ):
            raise MCPError("invalid_binding")
        if any(
            not isinstance(item, BenchmarkComparator) for item in runtime.comparators.values()
        ) or any(
            not isinstance(item, PerformanceThreshold) for item in runtime.thresholds.values()
        ):
            raise MCPError("invalid_binding")
        if any(not isinstance(item, RegressionDetector) for item in runtime.histories.values()):
            raise MCPError("invalid_binding")
        if any(
            item.kind.startswith("generator")
            and (not isinstance(item.target, GeneratorConfig) or item.parameters)
            for item in runtime.workloads.values()
        ):
            raise MCPError("invalid_binding")
        self.histories = dict(runtime.histories)
        self.workloads = {
            key: replace(item, target=deepcopy(item.target), parameters=deepcopy(item.parameters))
            for key, item in runtime.workloads.items()
        }
        self.comparators = dict(runtime.comparators)
        self.thresholds = deepcopy(runtime.thresholds)
        self.results = {}
        self._pending = None
        self._closed = False
        self.shutdown_requested = False

    @staticmethod
    def selected(mapping, key):
        try:
            return mapping[key]
        except KeyError:
            raise MCPError("invalid_handle") from None

    def execute_native(self, args, budget):
        from truthound.benchmark.base import BenchmarkResult
        from truthound.benchmark.parity import ParityResult
        from truthound.benchmark.runner import SuiteResult

        command = args["command"]
        if command == "inspect":
            return {
                "workloads": sorted(self.workloads),
                "comparators": sorted(self.comparators),
                "thresholds": sorted(self.thresholds),
                "histories": sorted(self.histories),
                "results": list(self.results),
            }
        if command == "shutdown":
            self.shutdown_requested = True
            return {"closed": True}
        if command == "run":
            if len(self.results) >= 16:
                raise MCPError("resource_exhausted")
            item = self.selected(self.workloads, args["workload_id"])
            options = deepcopy(item.parameters)
            if item.kind == "generator":
                result = item.runner.generate(deepcopy(item.target))
            elif item.kind == "generator_lazy":
                result = item.runner.generate_lazy(deepcopy(item.target)).collect()
            elif item.kind == "single":
                result = item.runner.run(item.target, **options)
            elif item.kind in {"suite", "parity"}:
                result = item.runner.run_suite(item.target, **options)
            elif item.kind == "category":
                result = item.runner.run_category(item.target, **options)
            else:
                result = item.runner.run_all(**options)
            if not isinstance(result, (BenchmarkResult, SuiteResult, ParityResult, pl.DataFrame)):
                raise MCPError("unsupported_value")
            return self.retain(result, budget)
        if command.startswith("history_"):
            return self.history(args, budget)
        result = self.selected(self.results, args["result_id"])[0]
        if command == "result":
            return {"result_id": args["result_id"], "result": projection(result)}
        if command == "release":
            del self.results[args["result_id"]]
            return {"released": True}
        if command == "data_page":
            if not isinstance(result, pl.DataFrame):
                raise MCPError("invalid_arguments")
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            end = min(result.height, offset + limit)
            return {
                "rows": result.slice(offset, limit).to_dicts(),
                "next_offset": end if limit and end < result.height else None,
            }
        if command == "compare":
            baseline = self.selected(self.results, args["baseline_id"])[0]
            comparator = self.selected(self.comparators, args["comparator_id"])
            measurements = []
            for candidate in (baseline, result):
                if isinstance(candidate, SuiteResult):
                    measurements.extend(candidate.results)
                    if not candidate.results:
                        return NativeServiceResult(
                            {"comparable": False}, "native_benchmark_incomplete"
                        )
                elif isinstance(candidate, BenchmarkResult):
                    measurements.append(candidate)
            if any(not item.success or not item.metrics.durations for item in measurements):
                return NativeServiceResult({"comparable": False}, "native_benchmark_incomplete")
            if isinstance(result, SuiteResult) and isinstance(baseline, SuiteResult):
                names = [
                    [item.benchmark_name for item in suite.results] for suite in (baseline, result)
                ]
                if any(len(group) != len(set(group)) for group in names) or set(names[0]) != set(
                    names[1]
                ):
                    return NativeServiceResult({"comparable": False}, "native_benchmark_incomplete")
                compared = comparator.compare_suites(baseline, result)
                return {"comparisons": [entry.to_dict() for entry in compared]}
            if isinstance(result, BenchmarkResult) and isinstance(baseline, BenchmarkResult):
                return {"comparison": comparator.compare(baseline, result).to_dict()}
            raise MCPError("invalid_arguments")
        if command == "threshold":
            if not isinstance(result, BenchmarkResult):
                raise MCPError("invalid_arguments")
            baseline = (
                self.selected(self.results, args["baseline_id"])[0]
                if args.get("baseline_id")
                else None
            )
            if baseline is not None and not isinstance(baseline, BenchmarkResult):
                raise MCPError("invalid_arguments")
            passed, violations = self.selected(self.thresholds, args["threshold_id"]).check(
                result, baseline
            )
            return {"passed": passed, "violations": violations}
        if command == "report":
            if args.get("format") == "ipc":
                if not isinstance(result, pl.DataFrame):
                    raise MCPError("invalid_arguments")
                output = io.BytesIO()
                result.write_ipc(output)
                if output.tell() > budget:
                    raise MCPError("resource_exhausted")
                return NativeServiceBytes(output.getvalue())
            if isinstance(result, pl.DataFrame) and args.get("format", "json") != "json":
                raise MCPError("invalid_arguments")
            if args.get("format", "json") == "json":
                return NativeServiceBytes(
                    json.dumps(encode(projection(result)), allow_nan=False).encode()
                )
            if isinstance(result, ParityResult):
                raise MCPError("invalid_arguments")
            from truthound.benchmark.reporters import HTMLReporter, MarkdownReporter

            reporter = HTMLReporter() if args["format"] == "html" else MarkdownReporter()

            def redacted(value):
                return replace(
                    value,
                    parameters={},
                    tags={},
                    metrics=replace(
                        value.metrics,
                        custom_metrics=[
                            replace(metric, tags={}) for metric in value.metrics.custom_metrics
                        ],
                    ),
                    error="native_benchmark_failed" if value.error else None,
                    error_traceback=None,
                )

            if isinstance(result, SuiteResult):
                content = reporter.report_suite(
                    replace(result, config={}, results=[redacted(item) for item in result.results])
                )
            else:
                content = reporter.report_result(redacted(result))
            return NativeServiceBytes(content.encode())
        raise MCPError("invalid_arguments")

    def history_bytes(self, exclude=None):
        return sum(
            len(json.dumps(encode(item.to_dict()), allow_nan=False).encode())
            for key, detector in self.histories.items()
            if key != exclude
            for item in detector._history
        )

    def retain(self, result, budget):
        from truthound.benchmark.base import BenchmarkResult
        from truthound.benchmark.runner import SuiteResult

        if len(self.results) >= 16:
            raise MCPError("resource_exhausted")
        payload = projection(result)
        size = (
            result.estimated_size() + len(json.dumps(payload).encode())
            if isinstance(result, pl.DataFrame)
            else len(json.dumps(encode(result.to_dict()), allow_nan=False).encode())
        )
        if size + sum(entry[1] for entry in self.results.values()) + self.history_bytes() > budget:
            raise MCPError("resource_exhausted")
        identity = "benchmark_" + uuid4().hex
        self.results[identity] = (result, size)
        output = {"result_id": identity, "result": payload}
        failed = (
            False
            if isinstance(result, pl.DataFrame)
            else not result.success or not result.metrics.durations
            if isinstance(result, BenchmarkResult)
            else not result.results
            or any(not entry.success or not entry.metrics.durations for entry in result.results)
            if isinstance(result, SuiteResult)
            else result.has_blocking_failures
        )
        return NativeServiceResult(output, "native_benchmark_incomplete") if failed else output

    def history(self, args, budget):
        from truthound.benchmark.runner import SuiteResult

        detector = self.selected(self.histories, args["history_id"])
        command = args["command"]
        if command == "history_load":
            detector.load_history()
            size = sum(
                len(json.dumps(encode(item.to_dict()), allow_nan=False).encode())
                for item in detector._history
            )
            if (
                size
                + sum(entry[1] for entry in self.results.values())
                + self.history_bytes(args["history_id"])
                > budget
            ):
                detector._history.clear()
                raise MCPError("resource_exhausted")
            return {"loaded": len(detector._history)}
        if command == "history_inspect":
            return {"loaded": len(detector._history), "thresholds": sorted(detector.thresholds)}
        if command == "history_baseline":
            baseline = detector.get_baseline()
            return {"available": False} if baseline is None else self.retain(baseline, budget)
        if command == "history_threshold":
            detector.add_threshold(deepcopy(self.selected(self.thresholds, args["threshold_id"])))
            return {"configured": True}
        result = self.selected(self.results, args["result_id"])[0]
        if not isinstance(result, SuiteResult):
            raise MCPError("invalid_arguments")
        if command == "history_save":
            detector.save_baseline(result)
            return {"saved": True}
        if command == "history_thresholds":
            return {"checks": detector.check_thresholds(result)}
        baseline = (
            self.selected(self.results, args["baseline_id"])[0]
            if args.get("baseline_id")
            else detector.get_baseline()
        )
        if baseline is None:
            return NativeServiceResult({"baseline_available": False}, "native_benchmark_incomplete")
        if not isinstance(baseline, SuiteResult):
            raise MCPError("invalid_arguments")
        if (
            len({item.benchmark_name for item in baseline.results}) != len(baseline.results)
            or len({item.benchmark_name for item in result.results}) != len(result.results)
            or not baseline.results
            or not result.results
            or any(
                not item.success or not item.metrics.durations
                for item in [*baseline.results, *result.results]
            )
            or {item.benchmark_name for item in baseline.results}
            != {item.benchmark_name for item in result.results}
        ):
            return NativeServiceResult({"comparable": False}, "native_benchmark_incomplete")
        if command == "history_check":
            return {"regressions": [item.to_dict() for item in detector.check(result, baseline)]}
        if command == "history_report":
            return NativeServiceBytes(detector.generate_report(result, baseline).encode())
        raise MCPError("invalid_arguments")

    async def execute(self, args, budget):
        if self._closed or self._pending is not None:
            raise MCPError("invalid_handle")
        task = asyncio.create_task(asyncio.to_thread(self.execute_native, args, budget))
        self._pending = task
        task.add_done_callback(
            lambda completed: None if completed.cancelled() else completed.exception()
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with suppress(Exception):
                await asyncio.shield(task)
            raise
        finally:
            if task.done():
                self._pending = None

    async def close(self):
        self._closed = True
        if self._pending is not None:
            with suppress(Exception):
                await asyncio.shield(self._pending)
            self._pending = None
        self.results.clear()


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(BenchmarkOwner)
