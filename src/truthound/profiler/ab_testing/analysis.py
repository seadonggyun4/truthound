"""Statistical analysis for A/B testing.

Provides statistical tests and analysis for experiment results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from truthound.profiler.ab_testing.base import (
    ExperimentConfig,
    ExperimentResult,
    MetricResult,
    MetricType,
    StatisticalAnalysis,
)


class StatisticalAnalyzer:
    """Analyze A/B experiment results.

    Provides statistical significance testing, confidence intervals,
    and effect size calculations.
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize analyzer.

        Args:
            config: Experiment configuration
        """
        self.config = config

    def analyze(self, result: ExperimentResult) -> ExperimentResult:
        """Perform statistical analysis on experiment results.

        Args:
            result: Raw experiment result

        Returns:
            Result with statistical analysis added
        """
        # Analyze primary metric
        primary_metric = self.config.primary_metric

        if primary_metric in result.control_metrics:
            control_result = result.control_metrics[primary_metric]

            # Analyze each treatment
            for treatment_name, treatment_metrics in result.treatment_metrics.items():
                if primary_metric in treatment_metrics:
                    treatment_result = treatment_metrics[primary_metric]

                    analysis = self._analyze_metric(
                        control_result,
                        treatment_result,
                        primary_metric,
                    )

                    if result.primary_analysis is None:
                        result.primary_analysis = analysis
                    else:
                        result.secondary_analyses.append(analysis)

        # Determine winner
        if result.primary_analysis:
            result.winner = result.primary_analysis.winner
            result.recommendation = result.primary_analysis.recommendation

        # Check guardrails
        result.guardrail_violations = self._check_guardrails(result)

        return result

    def _analyze_metric(
        self,
        control: MetricResult,
        treatment: MetricResult,
        metric_type: MetricType,
    ) -> StatisticalAnalysis:
        """Analyze a single metric comparison.

        Args:
            control: Control group result
            treatment: Treatment group result
            metric_type: Type of metric

        Returns:
            Statistical analysis
        """
        # Calculate differences
        control_mean = control.value
        treatment_mean = treatment.value
        absolute_diff = treatment_mean - control_mean

        if control_mean != 0:
            relative_diff = absolute_diff / control_mean
        else:
            relative_diff = 1.0 if treatment_mean > 0 else 0.0

        # Calculate pooled standard deviation
        pooled_std = math.sqrt(
            (control.std_dev ** 2 + treatment.std_dev ** 2) / 2
        ) if control.std_dev > 0 or treatment.std_dev > 0 else 0.1

        # Effect size (Cohen's d)
        effect_size = absolute_diff / pooled_std if pooled_std > 0 else 0

        # P-value using z-test approximation
        p_value = self._calculate_p_value(
            control_mean, treatment_mean,
            control.std_dev, treatment.std_dev,
            control.sample_size, treatment.sample_size,
        )

        # Is significant?
        alpha = 1 - self.config.confidence_level
        is_significant = p_value < alpha

        # Confidence interval for difference
        ci = self._confidence_interval(
            absolute_diff,
            pooled_std,
            control.sample_size + treatment.sample_size,
            self.config.confidence_level,
        )

        # Power calculation
        power = self._calculate_power(
            effect_size,
            control.sample_size,
            treatment.sample_size,
            alpha,
        )

        # Determine winner
        winner = None
        recommendation = ""

        if is_significant:
            # Lower is better for violation-related metrics
            lower_is_better = metric_type in [
                MetricType.VIOLATION_COUNT,
                MetricType.VIOLATION_RATE,
                MetricType.FALSE_POSITIVE_RATE,
                MetricType.FALSE_NEGATIVE_RATE,
                MetricType.EXECUTION_TIME,
            ]

            if lower_is_better:
                winner = treatment.variant_name if treatment_mean < control_mean else "control"
            else:
                winner = treatment.variant_name if treatment_mean > control_mean else "control"

            recommendation = (
                f"Statistically significant difference detected (p={p_value:.4f}). "
                f"Recommend adopting '{winner}' configuration."
            )
        else:
            recommendation = (
                f"No statistically significant difference (p={p_value:.4f}). "
                f"Consider running experiment longer or with more samples."
            )

        return StatisticalAnalysis(
            metric_type=metric_type,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_difference=absolute_diff,
            relative_difference=relative_diff,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.config.confidence_level,
            confidence_interval=ci,
            effect_size=effect_size,
            power=power,
            sample_size_control=control.sample_size,
            sample_size_treatment=treatment.sample_size,
            winner=winner,
            recommendation=recommendation,
        )

    def _calculate_p_value(
        self,
        mean1: float,
        mean2: float,
        std1: float,
        std2: float,
        n1: int,
        n2: int,
    ) -> float:
        """Calculate p-value using Welch's t-test approximation.

        Args:
            mean1, mean2: Group means
            std1, std2: Group standard deviations
            n1, n2: Sample sizes

        Returns:
            Two-tailed p-value
        """
        if n1 == 0 or n2 == 0:
            return 1.0

        # Use pooled variance if individual std not available
        if std1 == 0 and std2 == 0:
            # Estimate from proportion variance
            p_pooled = (mean1 * n1 + mean2 * n2) / (n1 + n2)
            if p_pooled > 0 and p_pooled < 1:
                std1 = math.sqrt(p_pooled * (1 - p_pooled))
                std2 = std1
            else:
                return 1.0

        # Standard error of difference
        se = math.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)

        if se == 0:
            return 1.0

        # Z-score
        z = abs(mean2 - mean1) / se

        # Two-tailed p-value (normal approximation)
        p_value = 2 * (1 - self._normal_cdf(z))

        return p_value

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function approximation."""
        # Using Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        d = 0.3989423 * math.exp(-x * x / 2)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))

        if x > 0:
            return 1.0 - p
        return p

    def _confidence_interval(
        self,
        diff: float,
        std: float,
        n: int,
        confidence: float,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference.

        Args:
            diff: Difference between means
            std: Pooled standard deviation
            n: Total sample size
            confidence: Confidence level (0-1)

        Returns:
            (lower, upper) bounds
        """
        if n <= 0:
            return (diff, diff)

        # Z-score for confidence level
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        z = z_scores.get(confidence, 1.96)

        margin = z * std / math.sqrt(n)

        return (diff - margin, diff + margin)

    def _calculate_power(
        self,
        effect_size: float,
        n1: int,
        n2: int,
        alpha: float,
    ) -> float:
        """Calculate statistical power.

        Args:
            effect_size: Cohen's d
            n1, n2: Sample sizes
            alpha: Significance level

        Returns:
            Power (0-1)
        """
        if n1 == 0 or n2 == 0:
            return 0.0

        # Approximate power calculation
        n_harmonic = 2 * n1 * n2 / (n1 + n2)
        noncentrality = effect_size * math.sqrt(n_harmonic / 2)

        # Z-critical for alpha
        z_alpha = 1.96 if alpha == 0.05 else 1.645 if alpha == 0.10 else 2.576

        # Power approximation
        power = self._normal_cdf(abs(noncentrality) - z_alpha)

        return max(0, min(1, power))

    def _check_guardrails(self, result: ExperimentResult) -> List[str]:
        """Check for guardrail violations.

        Args:
            result: Experiment result

        Returns:
            List of violation descriptions
        """
        violations = []

        for metric_type, operator, threshold in self.config.guardrail_metrics:
            # Check control
            if metric_type in result.control_metrics:
                value = result.control_metrics[metric_type].value
                if not self._check_threshold(value, operator, threshold):
                    violations.append(
                        f"Control violates guardrail: {metric_type.value} {operator} {threshold}"
                    )

            # Check treatments
            for name, metrics in result.treatment_metrics.items():
                if metric_type in metrics:
                    value = metrics[metric_type].value
                    if not self._check_threshold(value, operator, threshold):
                        violations.append(
                            f"Treatment '{name}' violates guardrail: "
                            f"{metric_type.value} {operator} {threshold}"
                        )

        return violations

    def _check_threshold(self, value: float, operator: str, threshold: float) -> bool:
        """Check if value meets threshold condition."""
        if operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.0001
        return True


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_results(
    result: ExperimentResult,
    config: Optional[ExperimentConfig] = None,
) -> ExperimentResult:
    """Analyze experiment results.

    Args:
        result: Raw experiment result
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Result with statistical analysis
    """
    if config is None:
        from truthound.profiler.ab_testing.base import ExperimentConfig
        config = ExperimentConfig(name="analysis")

    analyzer = StatisticalAnalyzer(config)
    return analyzer.analyze(result)


def calculate_sample_size(
    baseline_rate: float,
    minimum_effect: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Calculate required sample size for an experiment.

    Args:
        baseline_rate: Expected baseline rate (e.g., 0.1 for 10%)
        minimum_effect: Minimum detectable effect (relative, e.g., 0.1 for 10% improvement)
        power: Statistical power (default 0.8)
        alpha: Significance level (default 0.05)

    Returns:
        Required sample size per group
    """
    # Z-scores
    z_alpha = 1.96 if alpha == 0.05 else 1.645 if alpha == 0.10 else 2.576
    z_beta = 0.84 if power == 0.8 else 1.28 if power == 0.9 else 0.52

    # Effect size
    p1 = baseline_rate
    p2 = baseline_rate * (1 - minimum_effect)  # Improved rate

    # Pooled proportion
    p_pooled = (p1 + p2) / 2

    # Sample size formula
    if p_pooled > 0 and p_pooled < 1:
        numerator = 2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta) ** 2
        denominator = (p1 - p2) ** 2
        n = numerator / denominator if denominator > 0 else 10000
    else:
        n = 10000

    return max(100, int(math.ceil(n)))


def is_significant(
    control_value: float,
    treatment_value: float,
    control_n: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> bool:
    """Quick check if difference is statistically significant.

    Args:
        control_value: Control group metric value
        treatment_value: Treatment group metric value
        control_n: Control sample size
        treatment_n: Treatment sample size
        alpha: Significance level

    Returns:
        True if difference is significant
    """
    from truthound.profiler.ab_testing.base import ExperimentConfig

    config = ExperimentConfig(name="quick_check", confidence_level=1 - alpha)
    analyzer = StatisticalAnalyzer(config)

    p_value = analyzer._calculate_p_value(
        control_value, treatment_value,
        0.1, 0.1,  # Assume some std
        control_n, treatment_n,
    )

    return p_value < alpha
