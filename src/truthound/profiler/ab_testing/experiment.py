"""A/B Experiment implementation.

Provides experiment classes for running threshold comparison tests.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import polars as pl

from truthound.profiler.ab_testing.base import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    MetricResult,
    MetricType,
    StatisticalAnalysis,
    ThresholdVariant,
    VariantType,
)


logger = logging.getLogger(__name__)


class ABExperiment:
    """A/B experiment for comparing two threshold configurations.

    Example:
        control = ThresholdVariant(
            name="current",
            variant_type=VariantType.CONTROL,
            thresholds={"null_threshold": 0.1},
        )
        treatment = ThresholdVariant(
            name="strict",
            variant_type=VariantType.TREATMENT,
            thresholds={"null_threshold": 0.05},
        )

        experiment = ABExperiment(
            config=ExperimentConfig(
                name="null_threshold_optimization",
                control=control,
                treatments=[treatment],
            )
        )

        results = experiment.run(data)
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.experiment_id = str(uuid.uuid4())[:8]
        self.status = ExperimentStatus.DRAFT

        self._control_samples: List[Dict[str, Any]] = []
        self._treatment_samples: Dict[str, List[Dict[str, Any]]] = {}
        self._start_time: Optional[datetime] = None
        self._validators: Dict[str, Callable] = {}

    @property
    def is_running(self) -> bool:
        return self.status == ExperimentStatus.RUNNING

    def set_validator(
        self,
        variant_name: str,
        validator: Callable[[pl.DataFrame, Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Set a custom validator function for a variant.

        Args:
            variant_name: Variant name
            validator: Function that takes (data, thresholds) and returns metrics dict
        """
        self._validators[variant_name] = validator

    def run(
        self,
        data: pl.DataFrame,
        ground_truth: Optional[Dict[str, bool]] = None,
        batch_size: int = 1000,
    ) -> ExperimentResult:
        """Run the A/B experiment on data.

        Args:
            data: Data to validate
            ground_truth: Optional ground truth for false positive calculation
            batch_size: Batch size for processing

        Returns:
            Experiment results
        """
        self.status = ExperimentStatus.RUNNING
        self._start_time = datetime.now()

        logger.info(f"Starting experiment '{self.config.name}' (ID: {self.experiment_id})")

        try:
            # Run control
            control_metrics = self._run_variant(
                data,
                self.config.control,
                ground_truth,
            )
            self._control_samples.append(control_metrics)

            # Run treatments
            treatment_metrics: Dict[str, Dict[MetricType, MetricResult]] = {}
            for treatment in self.config.treatments:
                metrics = self._run_variant(data, treatment, ground_truth)
                treatment_metrics[treatment.name] = self._metrics_to_results(
                    metrics, treatment.name, len(data)
                )
                self._treatment_samples.setdefault(treatment.name, []).append(metrics)

            # Build result
            result = ExperimentResult(
                experiment_id=self.experiment_id,
                experiment_name=self.config.name,
                status=ExperimentStatus.COMPLETED,
                start_time=self._start_time,
                end_time=datetime.now(),
                control_metrics=self._metrics_to_results(
                    control_metrics, "control", len(data)
                ),
                treatment_metrics=treatment_metrics,
                total_samples=len(data),
            )

            # Analyze results
            from truthound.profiler.ab_testing.analysis import StatisticalAnalyzer

            analyzer = StatisticalAnalyzer(self.config)
            result = analyzer.analyze(result)

            self.status = ExperimentStatus.COMPLETED
            logger.info(f"Experiment completed. Winner: {result.winner}")

            return result

        except Exception as e:
            self.status = ExperimentStatus.FAILED
            logger.error(f"Experiment failed: {e}")
            raise

    def _run_variant(
        self,
        data: pl.DataFrame,
        variant: ThresholdVariant,
        ground_truth: Optional[Dict[str, bool]],
    ) -> Dict[str, float]:
        """Run a single variant.

        Args:
            data: Data to validate
            variant: Variant configuration
            ground_truth: Optional ground truth

        Returns:
            Metrics dictionary
        """
        start_time = time.time()

        # Use custom validator if provided
        if variant.name in self._validators:
            metrics = self._validators[variant.name](data, variant.thresholds)
        else:
            metrics = self._default_validation(data, variant.thresholds, ground_truth)

        # Add execution time
        metrics["execution_time"] = time.time() - start_time

        return metrics

    def _default_validation(
        self,
        data: pl.DataFrame,
        thresholds: Dict[str, Any],
        ground_truth: Optional[Dict[str, bool]],
    ) -> Dict[str, float]:
        """Default validation logic.

        Args:
            data: Data to validate
            thresholds: Threshold configuration
            ground_truth: Optional ground truth

        Returns:
            Metrics dictionary
        """
        metrics = {
            "violation_count": 0,
            "violation_rate": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        }

        total_violations = 0
        total_checks = 0

        for col_name in data.columns:
            col = data.get_column(col_name)

            # Check null threshold
            null_threshold = thresholds.get("null_threshold", 0.1)
            if isinstance(thresholds, dict) and col_name in thresholds:
                col_config = thresholds[col_name]
                if isinstance(col_config, dict):
                    null_threshold = col_config.get("null_threshold", null_threshold)

            null_ratio = col.null_count() / len(col) if len(col) > 0 else 0
            if null_ratio > null_threshold:
                total_violations += 1
            total_checks += 1

            # Check range thresholds if numeric
            if col.dtype.is_numeric():
                min_val = thresholds.get("min_value")
                max_val = thresholds.get("max_value")

                if min_val is not None:
                    below_min = (col < min_val).sum()
                    if below_min > 0:
                        total_violations += 1
                    total_checks += 1

                if max_val is not None:
                    above_max = (col > max_val).sum()
                    if above_max > 0:
                        total_violations += 1
                    total_checks += 1

        # Calculate rates
        if total_checks > 0:
            metrics["violation_count"] = total_violations
            metrics["violation_rate"] = total_violations / total_checks

        # If ground truth provided, calculate precision/recall
        if ground_truth:
            # Simplified calculation
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            true_negatives = 0

            # This would need actual violation tracking per row
            # For now, use placeholder values
            pass

        return metrics

    def _metrics_to_results(
        self,
        metrics: Dict[str, float],
        variant_name: str,
        sample_size: int,
    ) -> Dict[MetricType, MetricResult]:
        """Convert metrics dict to MetricResult objects."""
        results = {}

        metric_mapping = {
            "violation_count": MetricType.VIOLATION_COUNT,
            "violation_rate": MetricType.VIOLATION_RATE,
            "false_positive_rate": MetricType.FALSE_POSITIVE_RATE,
            "false_negative_rate": MetricType.FALSE_NEGATIVE_RATE,
            "precision": MetricType.PRECISION,
            "recall": MetricType.RECALL,
            "f1_score": MetricType.F1_SCORE,
            "execution_time": MetricType.EXECUTION_TIME,
            "data_quality_score": MetricType.DATA_QUALITY_SCORE,
        }

        for key, value in metrics.items():
            metric_type = metric_mapping.get(key)
            if metric_type:
                results[metric_type] = MetricResult(
                    metric_type=metric_type,
                    variant_name=variant_name,
                    value=value,
                    sample_size=sample_size,
                )

        return results


class MultiVariantExperiment(ABExperiment):
    """Experiment with multiple treatment variants.

    Supports comparing control against multiple treatments simultaneously.
    """

    def run(
        self,
        data: pl.DataFrame,
        ground_truth: Optional[Dict[str, bool]] = None,
        batch_size: int = 1000,
    ) -> ExperimentResult:
        """Run multi-variant experiment."""
        logger.info(
            f"Running multi-variant experiment with "
            f"{len(self.config.treatments)} treatments"
        )

        return super().run(data, ground_truth, batch_size)


class ExperimentRunner:
    """Runner for executing multiple experiments.

    Provides utilities for running experiments in sequence or parallel.
    """

    def __init__(self):
        self._experiments: List[ABExperiment] = []
        self._results: List[ExperimentResult] = []

    def add_experiment(self, experiment: ABExperiment) -> None:
        """Add an experiment to the queue."""
        self._experiments.append(experiment)

    def run_all(
        self,
        data: pl.DataFrame,
        ground_truth: Optional[Dict[str, bool]] = None,
    ) -> List[ExperimentResult]:
        """Run all experiments sequentially.

        Args:
            data: Data to validate
            ground_truth: Optional ground truth

        Returns:
            List of experiment results
        """
        self._results = []

        for experiment in self._experiments:
            try:
                result = experiment.run(data, ground_truth)
                self._results.append(result)
            except Exception as e:
                logger.error(f"Experiment {experiment.experiment_id} failed: {e}")

        return self._results

    def get_best_variant(
        self,
        metric: MetricType = MetricType.VIOLATION_RATE,
        minimize: bool = True,
    ) -> Optional[tuple[str, str, float]]:
        """Get the best performing variant across all experiments.

        Args:
            metric: Metric to compare
            minimize: If True, lower is better

        Returns:
            Tuple of (experiment_id, variant_name, metric_value)
        """
        best = None
        best_value = float("inf") if minimize else float("-inf")

        for result in self._results:
            # Check control
            if metric in result.control_metrics:
                value = result.control_metrics[metric].value
                if (minimize and value < best_value) or (not minimize and value > best_value):
                    best_value = value
                    best = (result.experiment_id, "control", value)

            # Check treatments
            for name, metrics in result.treatment_metrics.items():
                if metric in metrics:
                    value = metrics[metric].value
                    if (minimize and value < best_value) or (not minimize and value > best_value):
                        best_value = value
                        best = (result.experiment_id, name, value)

        return best
