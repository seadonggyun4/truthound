"""A/B Testing Framework for threshold tuning validation.

This package provides comprehensive A/B testing capabilities for:
- Comparing different threshold configurations
- Validating auto-tuned thresholds
- Statistical significance testing
- Experiment tracking and reporting

Example:
    from truthound.profiler.ab_testing import (
        ABExperiment,
        ExperimentConfig,
        run_ab_test,
        analyze_results,
    )

    # Create experiment
    experiment = ABExperiment(
        name="null_threshold_test",
        control=ThresholdConfig(null_threshold=0.1),
        treatment=ThresholdConfig(null_threshold=0.05),
    )

    # Run on data
    results = experiment.run(validation_data)

    # Analyze
    analysis = analyze_results(results)
    print(f"Winner: {analysis.winner} (p-value: {analysis.p_value:.4f})")
"""

from truthound.profiler.ab_testing.base import (
    ExperimentStatus,
    MetricType,
    ThresholdVariant,
    ExperimentConfig,
    ExperimentResult,
    MetricResult,
    StatisticalAnalysis,
)
from truthound.profiler.ab_testing.experiment import (
    ABExperiment,
    MultiVariantExperiment,
    ExperimentRunner,
)
from truthound.profiler.ab_testing.analysis import (
    StatisticalAnalyzer,
    analyze_results,
    calculate_sample_size,
    is_significant,
)
from truthound.profiler.ab_testing.tracking import (
    ExperimentTracker,
    ExperimentStore,
    get_tracker,
)

__all__ = [
    # Types
    "ExperimentStatus",
    "MetricType",
    "ThresholdVariant",
    "ExperimentConfig",
    "ExperimentResult",
    "MetricResult",
    "StatisticalAnalysis",
    # Experiments
    "ABExperiment",
    "MultiVariantExperiment",
    "ExperimentRunner",
    # Analysis
    "StatisticalAnalyzer",
    "analyze_results",
    "calculate_sample_size",
    "is_significant",
    # Tracking
    "ExperimentTracker",
    "ExperimentStore",
    "get_tracker",
]
