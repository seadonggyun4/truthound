"""Stress Testing Framework for Truthound Distributed Systems.

Provides comprehensive stress testing for distributed saga transactions,
including load generation, failure injection, and performance analysis.
"""

from tests.stress.framework import (
    StressTestRunner,
    StressTestConfig,
    StressTestResult,
    LoadProfile,
    LoadPhase,
)
from tests.stress.generators import (
    LoadGenerator,
    ConstantLoadGenerator,
    RampUpLoadGenerator,
    SpikeLoadGenerator,
    WaveLoadGenerator,
)
from tests.stress.chaos import (
    ChaosEngine,
    FailureInjector,
    FailureType,
    FailureConfig,
)
from tests.stress.metrics import (
    StressMetricsCollector,
    LatencyHistogram,
    ThroughputTracker,
)
from tests.stress.reports import (
    StressTestReport,
    ReportGenerator,
)

__all__ = [
    # Framework
    "StressTestRunner",
    "StressTestConfig",
    "StressTestResult",
    "LoadProfile",
    "LoadPhase",
    # Generators
    "LoadGenerator",
    "ConstantLoadGenerator",
    "RampUpLoadGenerator",
    "SpikeLoadGenerator",
    "WaveLoadGenerator",
    # Chaos
    "ChaosEngine",
    "FailureInjector",
    "FailureType",
    "FailureConfig",
    # Metrics
    "StressMetricsCollector",
    "LatencyHistogram",
    "ThroughputTracker",
    # Reports
    "StressTestReport",
    "ReportGenerator",
]
