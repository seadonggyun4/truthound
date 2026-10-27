"""Pinned native benchmark workloads; no arbitrary code, paths or runner kwargs on wire."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.session_contracts import Handle


@dataclass
class BenchmarkWorkload:
    runner: Any
    kind: Literal["single", "suite", "all", "category", "parity", "generator", "generator_lazy"]
    target: Any = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRuntime:
    workloads: dict[str, BenchmarkWorkload]
    comparators: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    histories: dict[str, Any] = field(default_factory=dict)


class BenchmarkCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "data_page",
        "inspect",
        "run",
        "result",
        "compare",
        "threshold",
        "report",
        "release",
        "shutdown",
        "history_load",
        "history_inspect",
        "history_baseline",
        "history_save",
        "history_check",
        "history_report",
        "history_threshold",
        "history_thresholds",
    ]
    history_id: Name | None = None
    workload_id: Name | None = None
    result_id: Name | None = None
    baseline_id: Name | None = None
    comparator_id: Name | None = None
    threshold_id: Name | None = None
    format: Literal["json", "markdown", "html", "ipc"] = "json"
    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=100, ge=0, le=1000, strict=True)

    @model_validator(mode="after")
    def shape(self):
        required = {
            "workload_id": {"run"},
            "result_id": {
                "data_page",
                "result",
                "compare",
                "threshold",
                "report",
                "release",
                "history_save",
                "history_check",
                "history_report",
                "history_thresholds",
            },
            "history_id": {
                "history_load",
                "history_inspect",
                "history_baseline",
                "history_save",
                "history_check",
                "history_report",
                "history_threshold",
                "history_thresholds",
            },
            "comparator_id": {"compare"},
            "threshold_id": {"threshold", "history_threshold"},
        }
        for name, commands in required.items():
            if (self.command in commands) != (getattr(self, name) is not None):
                raise ValueError("Field does not match benchmark command")
        if self.command == "compare" and self.baseline_id is None:
            raise ValueError("Comparison requires a baseline")
        if self.baseline_id is not None and self.command not in {
            "compare",
            "threshold",
            "history_check",
            "history_report",
        }:
            raise ValueError("Baseline does not match benchmark command")
        if {"offset", "limit"} & self.model_fields_set and self.command != "data_page":
            raise ValueError("Pagination applies only to generated data")
        if "format" in self.model_fields_set and self.command != "report":
            raise ValueError("Format applies only to reports")
        return self


class BenchmarkProcess(ServiceProcess):
    worker_module = "truthound.mcp.benchmark_worker"

    def validate_reply(self, result):
        if result == {"ok": True, "native_failed": "native_benchmark_incomplete"}:
            return result
        return super().validate_reply(result)


class BenchmarkServices(BoundServiceSession):
    group = "benchmark_services"
    process_type = BenchmarkProcess
    effects = (
        "installed_extension_execute",
        "operator_selected_native_benchmark_compute_and_callbacks",
        "operator_selected_benchmark_artifact_writes_and_optional_framework_execution",
        "sensitive_environment_and_measurement_artifact",
    )

    def artifact_media_type(self, arguments):
        if arguments.command == "history_report":
            return "text/plain"
        if arguments.command == "report":
            return {
                "json": "application/json",
                "markdown": "text/markdown",
                "html": "text/html",
                "ipc": "application/vnd.apache.arrow.file",
            }[arguments.format]
        return super().artifact_media_type(arguments)

    def reply_error(self, arguments, reply):
        return reply.get("native_failed")
