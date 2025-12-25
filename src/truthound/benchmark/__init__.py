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
    BenchmarkResult,
    BenchmarkMetrics,
    BenchmarkRegistry,
    BenchmarkSize,
    EnvironmentInfo,
    benchmark_registry,
    register_benchmark,
)
from truthound.benchmark.generators import (
    DataGenerator,
    GeneratorConfig,
    GeneratorRegistry,
    generator_registry,
    TabularDataGenerator,
    TimeSeriesDataGenerator,
    FinancialDataGenerator,
    TextDataGenerator,
    register_generator,
)
from truthound.benchmark.scenarios import (
    ProfileBenchmark,
    ValidationBenchmark,
    DriftBenchmark,
    LearnBenchmark,
    ScanBenchmark,
    E2EBenchmark,
)
from truthound.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkSuite,
    RunnerConfig,
)
from truthound.benchmark.reporters import (
    BenchmarkReporter,
    JSONReporter,
    MarkdownReporter,
    HTMLReporter,
    ConsoleReporter,
)
from truthound.benchmark.comparison import (
    BenchmarkComparator,
    RegressionDetector,
    PerformanceThreshold,
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
]
