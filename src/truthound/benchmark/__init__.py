"""Performance benchmarking system for Truthound.

This module provides a comprehensive, extensible benchmarking framework for
measuring and tracking performance across different operations.

Key features:
- Pluggable data generators for various scenarios
- Extensible benchmark scenarios with warmup/cooldown
- Multiple result exporters (JSON, Markdown, HTML)
- CI/CD integration support
- Historical comparison and regression detection

Example:
    from truthound.benchmark import (
        BenchmarkRunner,
        BenchmarkConfig,
        ProfileBenchmark,
        TabularDataGenerator,
    )

    # Run quick benchmarks
    runner = BenchmarkRunner()
    results = runner.run_all()
    print(results.to_markdown())

    # Run specific benchmark
    benchmark = ProfileBenchmark()
    result = benchmark.run(row_count=1_000_000)
"""

from truthound.benchmark.base import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkSize,
    EnvironmentInfo,
    benchmark_registry,
    register_benchmark,
)
from truthound.benchmark.comparison import (
    BenchmarkComparator,
    PerformanceThreshold,
    RegressionDetector,
)
from truthound.benchmark.generators import (
    DataGenerator,
    FinancialDataGenerator,
    GeneratorConfig,
    GeneratorRegistry,
    TabularDataGenerator,
    TextDataGenerator,
    TimeSeriesDataGenerator,
    generator_registry,
    register_generator,
)
from truthound.benchmark.parity import (
    BenchmarkMethodology,
    FrameworkAdapter,
    FrameworkObservation,
    ParityAssertion,
    ParityResult,
    ParityRunner,
    benchmark_artifact_root,
    capture_parity_environment,
    classify_release_blockers,
    default_baseline_path,
    default_env_manifest_path,
    default_output_path,
    default_release_summary_path,
    evaluate_parity_assertions,
    evaluate_release_environment_assertions,
    write_environment_manifest,
    write_parity_artifacts,
    write_release_summary,
)
from truthound.benchmark.reporters import (
    BenchmarkReporter,
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
)
from truthound.benchmark.runner import BenchmarkRunner, BenchmarkSuite, RunnerConfig
from truthound.benchmark.scenarios import (
    DriftBenchmark,
    E2EBenchmark,
    LearnBenchmark,
    ProfileBenchmark,
    ScanBenchmark,
    ValidationBenchmark,
)
from truthound.benchmark.workloads import (
    PARITY_SUITES,
    ExpectationSpec,
    GXWorkloadSpec,
    LoaderSpec,
    ParityWorkload,
    TruthoundWorkloadSpec,
    WorkloadBackend,
    WorkloadClass,
    WorkloadExpectation,
    discover_workloads,
    load_suite_workloads,
    load_workload,
    workload_root,
)

__all__ = [
    # Base
    "Benchmark",
    "BenchmarkCategory",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkMetrics",
    "BenchmarkRegistry",
    "BenchmarkSize",
    "EnvironmentInfo",
    "benchmark_registry",
    "register_benchmark",
    # Generators
    "DataGenerator",
    "GeneratorConfig",
    "GeneratorRegistry",
    "generator_registry",
    "TabularDataGenerator",
    "TimeSeriesDataGenerator",
    "FinancialDataGenerator",
    "TextDataGenerator",
    "register_generator",
    # Scenarios
    "ProfileBenchmark",
    "ValidationBenchmark",
    "DriftBenchmark",
    "LearnBenchmark",
    "ScanBenchmark",
    "E2EBenchmark",
    # Runner
    "BenchmarkRunner",
    "BenchmarkSuite",
    "RunnerConfig",
    # Reporters
    "BenchmarkReporter",
    "JSONReporter",
    "MarkdownReporter",
    "HTMLReporter",
    "ConsoleReporter",
    # Comparison
    "BenchmarkComparator",
    "RegressionDetector",
    "PerformanceThreshold",
    # Parity
    "FrameworkAdapter",
    "FrameworkObservation",
    "ParityAssertion",
    "ParityResult",
    "BenchmarkMethodology",
    "ParityRunner",
    "benchmark_artifact_root",
    "capture_parity_environment",
    "classify_release_blockers",
    "default_baseline_path",
    "default_env_manifest_path",
    "default_output_path",
    "default_release_summary_path",
    "evaluate_parity_assertions",
    "evaluate_release_environment_assertions",
    "write_parity_artifacts",
    "write_environment_manifest",
    "write_release_summary",
    # Workloads
    "PARITY_SUITES",
    "ExpectationSpec",
    "GXWorkloadSpec",
    "LoaderSpec",
    "ParityWorkload",
    "TruthoundWorkloadSpec",
    "WorkloadBackend",
    "WorkloadClass",
    "WorkloadExpectation",
    "discover_workloads",
    "load_suite_workloads",
    "load_workload",
    "workload_root",
]
