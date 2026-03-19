"""Truthound 3.0 parity runner for release-grade benchmark gating."""

from __future__ import annotations

import html
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from truthound.benchmark.base import EnvironmentInfo
from truthound.benchmark.workloads import (
    ParityWorkload,
    WorkloadClass,
    load_suite_workloads,
)
from truthound.context import get_context

FIXED_RUNNER_CLASS = "self-hosted-fixed"
RUNNER_CLASS_ENV = "TRUTHOUND_BENCHMARK_RUNNER_CLASS"
RUNNER_LABELS_ENV = "TRUTHOUND_BENCHMARK_RUNNER_LABELS"
RUNNER_STORAGE_CLASS_ENV = "TRUTHOUND_BENCHMARK_STORAGE_CLASS"
RUNNER_CPU_MODEL_ENV = "TRUTHOUND_BENCHMARK_CPU_MODEL"
RUNNER_CPU_PHYSICAL_CORES_ENV = "TRUTHOUND_BENCHMARK_CPU_PHYSICAL_CORES"
RUNNER_CPU_LOGICAL_CORES_ENV = "TRUTHOUND_BENCHMARK_CPU_LOGICAL_CORES"
RUNNER_RAM_BYTES_ENV = "TRUTHOUND_BENCHMARK_RAM_BYTES"
RELEASE_VERDICT_ENV = "TRUTHOUND_BENCHMARK_RELEASE_VERDICT"
RELEASE_ARTIFACT_WORKLOAD_COUNT = 8


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Protocol for framework-specific benchmark execution."""

    name: str

    def framework_version(self) -> str:
        """Return the framework version string."""

    def is_available(self) -> tuple[bool, str | None]:
        """Return availability and optional reason."""


@dataclass(frozen=True)
class FrameworkObservation:
    """Measured benchmark observation for one framework/workload pair."""

    framework: str
    framework_version: str
    workload_id: str
    dataset_fingerprint: str
    backend: str
    exactness: str
    cold_start_seconds: float
    warm_median_seconds: float
    peak_rss_bytes: int
    correctness_passed: bool
    expected_issue_count: int
    observed_issue_count: int
    artifact_paths: dict[str, str] = field(default_factory=dict)
    available: bool = True
    status: str = "ok"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return int(self.metadata.get("row_count", 0))

    @property
    def throughput_rows_per_second(self) -> float:
        if self.row_count <= 0 or self.warm_median_seconds <= 0:
            return 0.0
        return self.row_count / self.warm_median_seconds

    @classmethod
    def unavailable(
        cls,
        *,
        framework: str,
        workload: ParityWorkload,
        reason: str,
    ) -> FrameworkObservation:
        return cls(
            framework=framework,
            framework_version="unavailable",
            workload_id=workload.id,
            dataset_fingerprint=workload.dataset_fingerprint,
            backend=workload.backend.value,
            exactness=workload.exactness,
            cold_start_seconds=0.0,
            warm_median_seconds=0.0,
            peak_rss_bytes=0,
            correctness_passed=False,
            expected_issue_count=workload.expected.issue_count,
            observed_issue_count=0,
            artifact_paths={},
            available=False,
            status="unavailable",
            error=reason,
            metadata={"row_count": workload.row_count},
        )

    @classmethod
    def error_observation(
        cls,
        *,
        framework: str,
        workload: ParityWorkload,
        error: str,
    ) -> FrameworkObservation:
        return cls(
            framework=framework,
            framework_version="error",
            workload_id=workload.id,
            dataset_fingerprint=workload.dataset_fingerprint,
            backend=workload.backend.value,
            exactness=workload.exactness,
            cold_start_seconds=0.0,
            warm_median_seconds=0.0,
            peak_rss_bytes=0,
            correctness_passed=False,
            expected_issue_count=workload.expected.issue_count,
            observed_issue_count=0,
            artifact_paths={},
            available=True,
            status="error",
            error=error,
            metadata={"row_count": workload.row_count},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "workload_id": self.workload_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "backend": self.backend,
            "exactness": self.exactness,
            "cold_start_seconds": self.cold_start_seconds,
            "warm_median_seconds": self.warm_median_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "correctness_passed": self.correctness_passed,
            "expected_issue_count": self.expected_issue_count,
            "observed_issue_count": self.observed_issue_count,
            "throughput_rows_per_second": self.throughput_rows_per_second,
            "artifact_paths": self.artifact_paths,
            "available": self.available,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameworkObservation:
        return cls(
            framework=str(data["framework"]),
            framework_version=str(data["framework_version"]),
            workload_id=str(data["workload_id"]),
            dataset_fingerprint=str(data["dataset_fingerprint"]),
            backend=str(data["backend"]),
            exactness=str(data.get("exactness", "exact")),
            cold_start_seconds=float(data.get("cold_start_seconds", 0.0)),
            warm_median_seconds=float(data.get("warm_median_seconds", 0.0)),
            peak_rss_bytes=int(data.get("peak_rss_bytes", 0)),
            correctness_passed=bool(data.get("correctness_passed", False)),
            expected_issue_count=int(data.get("expected_issue_count", 0)),
            observed_issue_count=int(data.get("observed_issue_count", 0)),
            artifact_paths={str(k): str(v) for k, v in dict(data.get("artifact_paths", {})).items()},
            available=bool(data.get("available", True)),
            status=str(data.get("status", "ok")),
            error=data.get("error"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ParityAssertion:
    """A release-gate assertion derived from parity observations."""

    name: str
    passed: bool
    message: str
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParityAssertion:
        return cls(
            name=str(data["name"]),
            passed=bool(data.get("passed", False)),
            message=str(data.get("message", "")),
            severity=str(data.get("severity", "error")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BenchmarkMethodology:
    """Documented measurement methodology for parity runs."""

    name: str = "truthound-3.0-parity-gate"
    cold_iterations: int = 1
    warm_iterations: int = 2
    runner_class: str = "hybrid"
    exactness_policy: str = "exact-by-default"
    official_claim_policy: str = "self-hosted-fixed-runner-only"
    local_speedup_threshold: float = 1.5
    sql_speedup_threshold: float = 1.0
    sql_speedup_target: float = 1.2
    local_memory_ratio_threshold: float = 0.60
    truthound_baseline_regression_percent: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cold_iterations": self.cold_iterations,
            "warm_iterations": self.warm_iterations,
            "runner_class": self.runner_class,
            "exactness_policy": self.exactness_policy,
            "official_claim_policy": self.official_claim_policy,
            "local_speedup_threshold": self.local_speedup_threshold,
            "sql_speedup_threshold": self.sql_speedup_threshold,
            "sql_speedup_target": self.sql_speedup_target,
            "local_memory_ratio_threshold": self.local_memory_ratio_threshold,
            "truthound_baseline_regression_percent": self.truthound_baseline_regression_percent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkMethodology:
        return cls(
            name=str(data.get("name", "truthound-3.0-parity-gate")),
            cold_iterations=int(data.get("cold_iterations", 1)),
            warm_iterations=int(data.get("warm_iterations", 2)),
            runner_class=str(data.get("runner_class", "hybrid")),
            exactness_policy=str(data.get("exactness_policy", "exact-by-default")),
            official_claim_policy=str(
                data.get("official_claim_policy", "self-hosted-fixed-runner-only")
            ),
            local_speedup_threshold=float(data.get("local_speedup_threshold", 1.5)),
            sql_speedup_threshold=float(data.get("sql_speedup_threshold", 1.0)),
            sql_speedup_target=float(data.get("sql_speedup_target", 1.2)),
            local_memory_ratio_threshold=float(data.get("local_memory_ratio_threshold", 0.60)),
            truthound_baseline_regression_percent=float(
                data.get("truthound_baseline_regression_percent", 10.0)
            ),
        )


@dataclass(frozen=True)
class ParityResult:
    """Cross-framework benchmark result for a parity suite."""

    suite_name: str
    observations: tuple[FrameworkObservation, ...]
    assertions: tuple[ParityAssertion, ...]
    methodology: BenchmarkMethodology
    environment: dict[str, Any]
    artifact_root: str
    started_at: datetime
    completed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking_failures(self) -> tuple[ParityAssertion, ...]:
        return tuple(
            assertion
            for assertion in self.assertions
            if assertion.severity == "error" and not assertion.passed
        )

    @property
    def has_blocking_failures(self) -> bool:
        return bool(self.blocking_failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "summary": {
                "observation_count": len(self.observations),
                "assertion_count": len(self.assertions),
                "blocking_failures": len(self.blocking_failures),
                "passed": not self.has_blocking_failures,
            },
            "methodology": self.methodology.to_dict(),
            "environment": self.environment,
            "artifact_root": self.artifact_root,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "metadata": self.metadata,
            "observations": [observation.to_dict() for observation in self.observations],
            "assertions": [assertion.to_dict() for assertion in self.assertions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParityResult:
        return cls(
            suite_name=str(data["suite_name"]),
            observations=tuple(
                FrameworkObservation.from_dict(item) for item in data.get("observations", ())
            ),
            assertions=tuple(
                ParityAssertion.from_dict(item) for item in data.get("assertions", ())
            ),
            methodology=BenchmarkMethodology.from_dict(dict(data.get("methodology", {}))),
            environment=dict(data.get("environment", {})),
            artifact_root=str(data.get("artifact_root", "")),
            started_at=datetime.fromisoformat(str(data["started_at"])),
            completed_at=datetime.fromisoformat(str(data["completed_at"])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# Benchmark Parity: {self.suite_name}",
            "",
            f"- Passed: {'yes' if not self.has_blocking_failures else 'no'}",
            f"- Observations: {len(self.observations)}",
            f"- Assertions: {len(self.assertions)}",
            f"- Artifact root: `{self.artifact_root}`",
            "",
            "## Observations",
            "",
            "| Workload | Framework | Backend | Correctness | Warm Median (s) | Peak RSS (MB) | Status |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
        for observation in self.observations:
            lines.append(
                "| "
                f"{observation.workload_id} | "
                f"{observation.framework} | "
                f"{observation.backend} | "
                f"{'pass' if observation.correctness_passed else 'fail'} | "
                f"{observation.warm_median_seconds:.6f} | "
                f"{observation.peak_rss_bytes / (1024 * 1024):.2f} | "
                f"{observation.status} |"
            )

        lines.extend(["", "## Assertions", ""])
        for assertion in self.assertions:
            status = "PASS" if assertion.passed else "FAIL"
            lines.append(f"- [{status}] `{assertion.name}`: {assertion.message}")
        return "\n".join(lines)

    def to_html(self) -> str:
        body = html.escape(self.to_markdown())
        return (
            "<html><head><meta charset='utf-8'><title>Truthound Benchmark Parity</title>"
            "<style>body{font-family:ui-monospace,Menlo,monospace;padding:2rem;line-height:1.5;}"
            "pre{white-space:pre-wrap;}</style></head><body><pre>"
            f"{body}</pre></body></html>"
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(), encoding="utf-8")


def classify_release_blockers(result: ParityResult) -> dict[str, Any]:
    """Classify blocking failures into release follow-up buckets."""

    categories: set[str] = set()

    for assertion in result.blocking_failures:
        name = assertion.name
        if name.startswith("release-ga:") or name.endswith(":present") or name.endswith(":available"):
            categories.add("environment")
            continue
        if name.endswith(":correctness") or name.endswith(":issue-parity"):
            categories.add("correctness")
            continue
        if (
            name.endswith(":local-speedup")
            or name.endswith(":local-memory-ratio")
            or name.endswith(":sql-speedup")
            or name.endswith(":baseline-regression")
        ):
            categories.add("performance")

    for observation in result.observations:
        if observation.framework != "gx":
            continue
        if observation.status == "error":
            categories.add("gx_drift")
            continue
        if not observation.available:
            error_message = (observation.error or "").lower()
            if "not installed" in error_message:
                categories.add("environment")
            else:
                categories.add("gx_drift")

    ordered_categories = tuple(
        category
        for category in ("environment", "gx_drift", "correctness", "performance")
        if category in categories
    )
    primary = ordered_categories[0] if ordered_categories else None
    return {
        "primary": primary,
        "categories": ordered_categories,
        "has_blockers": bool(ordered_categories),
    }


def benchmark_artifact_root(start_path: str | Path | None = None) -> Path:
    """Return the stable benchmark artifact root under `.truthound/benchmarks/`."""

    override = os.environ.get("TRUTHOUND_BENCHMARK_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        context = get_context(start_path)
        root = context.workspace_dir / "benchmarks"
    for name in ("results", "baselines", "artifacts", "release"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _physical_core_count() -> int:
    override = _env_int(RUNNER_CPU_PHYSICAL_CORES_ENV)
    if override is not None:
        return override
    try:
        import psutil

        return int(psutil.cpu_count(logical=False) or 0)
    except Exception:
        return 0


def _logical_core_count(default: int) -> int:
    override = _env_int(RUNNER_CPU_LOGICAL_CORES_ENV)
    if override is not None:
        return override
    return default


def _total_memory_bytes() -> int:
    override = _env_int(RUNNER_RAM_BYTES_ENV)
    if override is not None:
        return override
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def capture_parity_environment() -> dict[str, Any]:
    """Capture the environment metadata used for parity artifacts and release gating."""

    payload = EnvironmentInfo.capture().to_dict()
    platform_payload = dict(payload.get("platform", {}))
    platform_payload["platform_string"] = platform.platform()
    platform_payload["python_minor"] = f"{sys.version_info.major}.{sys.version_info.minor}"

    runner_class = os.environ.get(RUNNER_CLASS_ENV, "ad-hoc-local").strip() or "ad-hoc-local"
    runner_labels = tuple(
        label.strip()
        for label in os.environ.get(RUNNER_LABELS_ENV, "").split(",")
        if label.strip()
    )
    storage_class = os.environ.get(RUNNER_STORAGE_CLASS_ENV, "").strip() or "unspecified"
    cpu_model = os.environ.get(RUNNER_CPU_MODEL_ENV, "").strip() or platform.processor().strip()
    if not cpu_model:
        cpu_model = platform_payload.get("machine", "unknown")

    cpu_logical_cores = _logical_core_count(int(payload.get("cpu_count", 0)))
    cpu_physical_cores = _physical_core_count()
    ram_bytes = _total_memory_bytes()
    payload["platform"] = platform_payload
    payload["machine"] = {
        "cpu_model": cpu_model,
        "cpu_physical_cores": cpu_physical_cores,
        "cpu_logical_cores": cpu_logical_cores,
        "ram_bytes": ram_bytes,
        "ram_gb": round(ram_bytes / (1024**3), 2) if ram_bytes else 0.0,
    }
    payload["runner"] = {
        "class": runner_class,
        "labels": runner_labels,
        "storage_class": storage_class,
        "release_verdict": _env_bool(RELEASE_VERDICT_ENV, default=False),
        "official_claim_allowed": (
            runner_class == FIXED_RUNNER_CLASS and _env_bool(RELEASE_VERDICT_ENV, default=False)
        ),
    }
    return payload


def default_output_path(
    suite_name: str,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Return the default JSON output path for a parity suite run."""

    root = Path(artifact_root) if artifact_root is not None else benchmark_artifact_root()
    bucket = "release" if suite_name == "release-ga" else "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / bucket / f"{suite_name}_{timestamp}.json"


def default_baseline_path(
    suite_name: str,
    *,
    backend: str | None = None,
    frameworks: str = "both",
    artifact_root: str | Path | None = None,
) -> Path:
    """Return the canonical baseline path for a parity suite selection."""

    root = Path(artifact_root) if artifact_root is not None else benchmark_artifact_root()
    backend_key = backend or "all"
    return root / "baselines" / f"{suite_name}_{frameworks}_{backend_key}.json"


def default_env_manifest_path(
    output_path: str | Path,
) -> Path:
    """Return the canonical environment-manifest location for a parity output."""

    return Path(output_path).resolve().parent / "env-manifest.json"


def default_release_summary_path(
    output_path: str | Path,
) -> Path:
    """Return the canonical release-summary location for a release parity output."""

    return Path(output_path).resolve().parent / "latest-benchmark-summary.md"


class TruthoundAdapter:
    """Truthound public-path benchmark adapter."""

    name = "truthound"

    def framework_version(self) -> str:
        try:
            return importlib_metadata.version("truthound")
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover - local editable path
            return "0.0.0.dev"

    def is_available(self) -> tuple[bool, str | None]:
        return True, None


class GreatExpectationsAdapter:
    """Great Expectations parity adapter."""

    name = "gx"

    def framework_version(self) -> str:
        try:
            return importlib_metadata.version("great-expectations")
        except importlib_metadata.PackageNotFoundError:
            return "unavailable"

    def is_available(self) -> tuple[bool, str | None]:
        try:
            import great_expectations  # noqa: F401
        except ImportError:
            return False, "great-expectations is not installed. Install with truthound[benchmarks]."
        return True, None


def evaluate_parity_assertions(
    workloads: list[ParityWorkload],
    observations: list[FrameworkObservation],
    methodology: BenchmarkMethodology,
    *,
    requested_frameworks: tuple[str, ...],
    baseline_result: ParityResult | None = None,
) -> list[ParityAssertion]:
    """Evaluate correctness, parity, threshold, and baseline assertions."""

    grouped: dict[str, dict[str, FrameworkObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.workload_id, {})[observation.framework] = observation

    assertions: list[ParityAssertion] = []
    for workload in workloads:
        workload_observations = grouped.get(workload.id, {})
        supported_requested = tuple(
            framework for framework in requested_frameworks if workload.supports_framework(framework)
        )

        for framework in supported_requested:
            observation = workload_observations.get(framework)
            if observation is None:
                assertions.append(
                    ParityAssertion(
                        name=f"{workload.id}:{framework}:present",
                        passed=False,
                        message=f"Missing observation for requested framework '{framework}'.",
                    )
                )
                continue
            if not observation.available:
                assertions.append(
                    ParityAssertion(
                        name=f"{workload.id}:{framework}:available",
                        passed=False,
                        message=observation.error or f"Framework '{framework}' is unavailable.",
                    )
                )
                continue
            assertions.append(
                ParityAssertion(
                    name=f"{workload.id}:{framework}:correctness",
                    passed=observation.correctness_passed,
                    message=(
                        f"Observed {observation.observed_issue_count} issues; "
                        f"expected {observation.expected_issue_count}."
                    ),
                    details={
                        "status": observation.status,
                        "workload": workload.id,
                        "framework": framework,
                    },
                )
            )

        truthound_observation = workload_observations.get("truthound")
        gx_observation = workload_observations.get("gx")
        if (
            truthound_observation is not None
            and gx_observation is not None
            and truthound_observation.available
            and gx_observation.available
        ):
            parity_passed = (
                truthound_observation.observed_issue_count == gx_observation.observed_issue_count
                and truthound_observation.correctness_passed
                and gx_observation.correctness_passed
            )
            assertions.append(
                ParityAssertion(
                    name=f"{workload.id}:issue-parity",
                    passed=parity_passed,
                    message=(
                        f"Truthound observed {truthound_observation.observed_issue_count}; "
                        f"GX observed {gx_observation.observed_issue_count}."
                    ),
                )
            )

            if (
                truthound_observation.warm_median_seconds > 0
                and gx_observation.warm_median_seconds > 0
            ):
                speedup = (
                    gx_observation.warm_median_seconds / truthound_observation.warm_median_seconds
                )
                if workload.benchmark_class == WorkloadClass.LOCAL_EXACT:
                    assertions.append(
                        ParityAssertion(
                            name=f"{workload.id}:local-speedup",
                            passed=speedup >= methodology.local_speedup_threshold,
                            message=(
                                f"Truthound speedup {speedup:.2f}x vs GX "
                                f"(target {methodology.local_speedup_threshold:.2f}x)."
                            ),
                            details={"speedup": speedup},
                        )
                    )
                    if gx_observation.peak_rss_bytes > 0 and truthound_observation.peak_rss_bytes > 0:
                        memory_ratio = (
                            truthound_observation.peak_rss_bytes / gx_observation.peak_rss_bytes
                        )
                        assertions.append(
                            ParityAssertion(
                                name=f"{workload.id}:local-memory-ratio",
                                passed=memory_ratio <= methodology.local_memory_ratio_threshold,
                                message=(
                                    f"Truthound RSS ratio {memory_ratio:.2%} vs GX "
                                    f"(max {methodology.local_memory_ratio_threshold:.0%})."
                                ),
                                details={"memory_ratio": memory_ratio},
                            )
                        )
                    else:
                        assertions.append(
                            ParityAssertion(
                                name=f"{workload.id}:local-memory-ratio",
                                passed=True,
                                severity="warning",
                                message="Memory ratio not evaluated because peak RSS was unavailable.",
                            )
                        )
                elif workload.benchmark_class == WorkloadClass.SQL_EXACT:
                    assertions.append(
                        ParityAssertion(
                            name=f"{workload.id}:sql-speedup",
                            passed=speedup >= methodology.sql_speedup_threshold,
                            message=(
                                f"Truthound speedup {speedup:.2f}x vs GX "
                                f"(floor {methodology.sql_speedup_threshold:.2f}x, "
                                f"target {methodology.sql_speedup_target:.2f}x)."
                            ),
                            details={"speedup": speedup},
                        )
                    )

        if baseline_result is not None and truthound_observation is not None and truthound_observation.available:
            baseline_lookup = {
                (item.workload_id, item.framework): item for item in baseline_result.observations
            }
            baseline_observation = baseline_lookup.get((workload.id, "truthound"))
            if baseline_observation is None:
                assertions.append(
                    ParityAssertion(
                        name=f"{workload.id}:baseline-present",
                        passed=False,
                        message="Missing Truthound baseline observation for regression comparison.",
                    )
                )
            elif (
                baseline_observation.warm_median_seconds > 0
                and truthound_observation.warm_median_seconds > 0
            ):
                regression = (
                    (truthound_observation.warm_median_seconds - baseline_observation.warm_median_seconds)
                    / baseline_observation.warm_median_seconds
                ) * 100
                assertions.append(
                    ParityAssertion(
                        name=f"{workload.id}:baseline-regression",
                        passed=regression <= methodology.truthound_baseline_regression_percent,
                        message=(
                            f"Truthound regression {regression:.2f}% vs baseline "
                            f"(max {methodology.truthound_baseline_regression_percent:.2f}%)."
                        ),
                        details={"regression_percent": regression},
                    )
                )

    return assertions


def evaluate_release_environment_assertions(
    *,
    suite_name: str,
    requested_frameworks: tuple[str, ...],
    backend_filter: str | None,
    workload_count: int,
    environment: dict[str, Any],
) -> list[ParityAssertion]:
    """Evaluate release-only assertions for fixed-runner GA verdicts."""

    if suite_name != "release-ga":
        return []

    runner = dict(environment.get("runner", {}))
    machine = dict(environment.get("machine", {}))
    platform_payload = dict(environment.get("platform", {}))
    labels = {str(label).strip() for label in runner.get("labels", ()) if str(label).strip()}

    runner_ok = (
        runner.get("class") == FIXED_RUNNER_CLASS
        and "self-hosted" in labels
        and "benchmark-fixed" in labels
        and bool(runner.get("release_verdict"))
    )
    metadata_complete = (
        bool(str(machine.get("cpu_model", "")).strip())
        and int(machine.get("cpu_logical_cores", 0) or 0) > 0
        and int(machine.get("ram_bytes", 0) or 0) > 0
        and bool(str(platform_payload.get("python_minor", "")).strip())
        and bool(str(runner.get("storage_class", "")).strip())
        and str(runner.get("storage_class")) != "unspecified"
    )

    return [
        ParityAssertion(
            name="release-ga:runner-policy",
            passed=runner_ok,
            message=(
                "Release verdicts require a fixed self-hosted runner with "
                "`self-hosted,benchmark-fixed` labels and "
                "`TRUTHOUND_BENCHMARK_RELEASE_VERDICT=true`."
            ),
            details={
                "runner_class": runner.get("class"),
                "runner_labels": sorted(labels),
                "release_verdict": bool(runner.get("release_verdict")),
            },
        ),
        ParityAssertion(
            name="release-ga:runner-metadata",
            passed=metadata_complete,
            message=(
                "Release artifacts must document CPU model, logical cores, RAM, "
                "Python minor, and storage class."
            ),
            details={
                "cpu_model": machine.get("cpu_model"),
                "cpu_logical_cores": machine.get("cpu_logical_cores"),
                "ram_bytes": machine.get("ram_bytes"),
                "python_minor": platform_payload.get("python_minor"),
                "storage_class": runner.get("storage_class"),
            },
        ),
        ParityAssertion(
            name="release-ga:framework-selector",
            passed=requested_frameworks == ("truthound", "gx"),
            message="The release-ga suite must run both Truthound and GX.",
            details={"requested_frameworks": requested_frameworks},
        ),
        ParityAssertion(
            name="release-ga:backend-filter",
            passed=backend_filter is None,
            message="The release-ga suite must execute the full local + SQLite catalog without a backend filter.",
            details={"backend_filter": backend_filter},
        ),
        ParityAssertion(
            name="release-ga:catalog-size",
            passed=workload_count == RELEASE_ARTIFACT_WORKLOAD_COUNT,
            message=(
                f"The release-ga suite must contain exactly {RELEASE_ARTIFACT_WORKLOAD_COUNT} "
                "repo-tracked tier-1 workloads."
            ),
            details={"workload_count": workload_count},
        ),
    ]


class ParityRunner:
    """Run repo-tracked benchmark workloads across Truthound and GX."""

    _SUPPORTED_FRAMEWORKS = ("truthound", "gx")

    def __init__(self, methodology: BenchmarkMethodology | None = None) -> None:
        self.methodology = methodology or BenchmarkMethodology()

    def run_suite(
        self,
        suite_name: str,
        *,
        frameworks: str = "both",
        backend: str | None = None,
        artifact_root: str | Path | None = None,
        baseline_result: ParityResult | None = None,
    ) -> ParityResult:
        workloads = load_suite_workloads(suite_name, backend=backend)
        requested_frameworks = self._resolve_requested_frameworks(frameworks)
        root = Path(artifact_root) if artifact_root is not None else benchmark_artifact_root()
        started_at = datetime.now()
        observations: list[FrameworkObservation] = []

        for workload in workloads:
            for framework in requested_frameworks:
                if not workload.supports_framework(framework):
                    continue
                observation = self._run_worker(
                    workload=workload,
                    framework=framework,
                    artifact_root=root,
                )
                observations.append(observation)

        assertions = evaluate_parity_assertions(
            workloads,
            observations,
            self.methodology,
            requested_frameworks=requested_frameworks,
            baseline_result=baseline_result,
        )
        environment = capture_parity_environment()
        assertions.extend(
            evaluate_release_environment_assertions(
                suite_name=suite_name,
                requested_frameworks=requested_frameworks,
                backend_filter=backend,
                workload_count=len(workloads),
                environment=environment,
            )
        )
        completed_at = datetime.now()

        return ParityResult(
            suite_name=suite_name,
            observations=tuple(observations),
            assertions=tuple(assertions),
            methodology=self.methodology,
            environment=environment,
            artifact_root=str(root),
            started_at=started_at,
            completed_at=completed_at,
            metadata={
                "requested_frameworks": requested_frameworks,
                "backend_filter": backend,
                "suite_catalog_size": len(workloads),
                "release_claim_ready": suite_name == "release-ga" and not any(
                    assertion.severity == "error" and not assertion.passed for assertion in assertions
                ),
                "release_blockers": classify_release_blockers(
                    ParityResult(
                        suite_name=suite_name,
                        observations=tuple(observations),
                        assertions=tuple(assertions),
                        methodology=self.methodology,
                        environment=environment,
                        artifact_root=str(root),
                        started_at=started_at,
                        completed_at=completed_at,
                    )
                ),
            },
        )

    def _resolve_requested_frameworks(self, frameworks: str) -> tuple[str, ...]:
        if frameworks == "both":
            return self._SUPPORTED_FRAMEWORKS
        if frameworks not in self._SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unknown framework selector '{frameworks}'. "
                f"Expected one of: truthound, gx, both."
            )
        return (frameworks,)

    def _run_worker(
        self,
        *,
        workload: ParityWorkload,
        framework: str,
        artifact_root: Path,
    ) -> FrameworkObservation:
        artifact_dir = (
            artifact_root
            / "artifacts"
            / workload.id
            / framework
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload_path = artifact_dir / "observation.json"

        env = os.environ.copy()
        src_root = str(Path(__file__).resolve().parents[2])
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            src_root if not current_pythonpath else f"{src_root}{os.pathsep}{current_pythonpath}"
        )

        command = [
            sys.executable,
            "-m",
            "truthound.benchmark._parity_worker",
            "--manifest",
            str(workload.manifest_path),
            "--framework",
            framework,
            "--artifact-dir",
            str(artifact_dir),
            "--output",
            str(payload_path),
            "--warm-iterations",
            str(self.methodology.warm_iterations),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            error_message = completed.stderr.strip() or completed.stdout.strip() or (
                f"Worker exited with code {completed.returncode}."
            )
            return FrameworkObservation.error_observation(
                framework=framework,
                workload=workload,
                error=error_message,
            )
        if not payload_path.exists():
            return FrameworkObservation.error_observation(
                framework=framework,
                workload=workload,
                error="Worker finished without writing an observation payload.",
            )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        return FrameworkObservation.from_dict(payload)


def write_parity_artifacts(result: ParityResult, output_path: str | Path) -> tuple[Path, Path, Path]:
    """Write JSON, Markdown, and HTML parity artifacts."""

    json_path = Path(output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(result.to_json(), encoding="utf-8")
    markdown_path = json_path.with_suffix(".md")
    markdown_path.write_text(result.to_markdown(), encoding="utf-8")
    html_path = json_path.with_suffix(".html")
    html_path.write_text(result.to_html(), encoding="utf-8")
    return json_path, markdown_path, html_path


def write_environment_manifest(result: ParityResult, path: str | Path) -> Path:
    """Write a canonical environment manifest alongside parity artifacts."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite_name": result.suite_name,
        "summary": result.to_dict()["summary"],
        "methodology": result.methodology.to_dict(),
        "environment": result.environment,
        "artifact_root": result.artifact_root,
        "started_at": result.started_at.isoformat(),
        "completed_at": result.completed_at.isoformat(),
        "metadata": result.metadata,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def write_release_summary(
    result: ParityResult,
    path: str | Path,
    *,
    env_manifest_path: str | Path | None = None,
) -> Path:
    """Write a release-oriented Markdown summary for a release-ga artifact set."""

    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, FrameworkObservation]] = {}
    for observation in result.observations:
        grouped.setdefault(observation.workload_id, {})[observation.framework] = observation

    runner = dict(result.environment.get("runner", {}))
    machine = dict(result.environment.get("machine", {}))
    env_manifest_label = Path(env_manifest_path).name if env_manifest_path is not None else None
    release_claim_ready = bool(result.metadata.get("release_claim_ready"))
    release_blockers = dict(result.metadata.get("release_blockers", classify_release_blockers(result)))
    blocker_categories = tuple(release_blockers.get("categories", ()))
    primary_blocker = release_blockers.get("primary")

    lines = [
        "# Latest Benchmark Summary",
        "",
        "## Status",
        "",
        (
            "This artifact set passed the fixed-runner 3.0 GA benchmark gate. "
            "It is eligible to back official benchmark claims."
            if release_claim_ready
            else "This artifact set did not clear the full 3.0 GA benchmark gate. "
            "Do not publish official performance claims from this result."
        ),
        "",
        f"- Suite: `{result.suite_name}`",
        f"- Passed: `{'yes' if not result.has_blocking_failures else 'no'}`",
        f"- Runner class: `{runner.get('class', 'unknown')}`",
        f"- Storage class: `{runner.get('storage_class', 'unspecified')}`",
        f"- CPU model: `{machine.get('cpu_model', 'unknown')}`",
        f"- Logical cores: `{machine.get('cpu_logical_cores', 0)}`",
        f"- RAM (bytes): `{machine.get('ram_bytes', 0)}`",
    ]
    if primary_blocker:
        lines.append(f"- Primary blocker: `{primary_blocker}`")
    if blocker_categories:
        lines.append(f"- Blocker categories: `{', '.join(blocker_categories)}`")
    if env_manifest_label is not None:
        lines.append(f"- Environment manifest: `{env_manifest_label}`")

    lines.extend(
        [
            "",
            "## Comparable Workloads",
            "",
            "| Workload | Truthound Warm (s) | GX Warm (s) | Speedup | Memory Ratio | Correctness |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for workload_id in sorted(grouped):
        truthound_observation = grouped[workload_id].get("truthound")
        gx_observation = grouped[workload_id].get("gx")
        speedup = "n/a"
        memory_ratio = "n/a"
        correctness = "incomplete"
        truthound_warm = (
            f"{truthound_observation.warm_median_seconds:.6f}"
            if truthound_observation is not None
            else "n/a"
        )
        gx_warm = (
            f"{gx_observation.warm_median_seconds:.6f}"
            if gx_observation is not None
            else "n/a"
        )
        if truthound_observation is not None and gx_observation is not None:
            if truthound_observation.warm_median_seconds > 0 and gx_observation.warm_median_seconds > 0:
                speedup = f"{gx_observation.warm_median_seconds / truthound_observation.warm_median_seconds:.2f}x"
            if truthound_observation.peak_rss_bytes > 0 and gx_observation.peak_rss_bytes > 0:
                memory_ratio = (
                    f"{truthound_observation.peak_rss_bytes / gx_observation.peak_rss_bytes:.2%}"
                )
            correctness = (
                "pass"
                if truthound_observation.correctness_passed and gx_observation.correctness_passed
                else "fail"
            )
        lines.append(
            "| "
            f"{workload_id} | "
            f"{truthound_warm} | "
            f"{gx_warm} | "
            f"{speedup} | "
            f"{memory_ratio} | "
            f"{correctness} |"
        )

    lines.extend(["", "## Assertions", ""])
    for assertion in result.assertions:
        status = "PASS" if assertion.passed else "FAIL"
        lines.append(f"- [{status}] `{assertion.name}`: {assertion.message}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
