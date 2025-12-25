"""ML-based rule generator.

Generates validation rules using machine learning techniques:
- Outlier detection rules
- Anomaly detection rules
- Correlation-based rules
- Distribution-based rules
"""

from __future__ import annotations

from typing import Any

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    Strictness,
    TableProfile,
)
from truthound.profiler.generators.base import (
    DEFAULT_THRESHOLDS,
    GeneratedRule,
    RuleBuilder,
    RuleCategory,
    RuleConfidence,
    RuleGenerator,
    register_generator,
)


@register_generator("ml")
class MLRuleGenerator(RuleGenerator):
    """Generates ML-based validation rules.

    This generator creates rules that validate:
    - Outlier detection using statistical methods
    - Anomaly detection
    - Correlation constraints
    - Distribution shape validation
    """

    name = "ml"
    description = "Generates ML-based validation rules"
    categories = {RuleCategory.ANOMALY, RuleCategory.DISTRIBUTION, RuleCategory.RELATIONSHIP}
    priority = 50

    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        correlation_threshold: float = 0.7,
        **kwargs: Any,
    ):
        """Initialize ML rule generator.

        Args:
            z_score_threshold: Z-score threshold for outlier detection
            iqr_multiplier: IQR multiplier for outlier detection
            correlation_threshold: Minimum correlation for correlation rules
        """
        super().__init__(**kwargs)
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.correlation_threshold = correlation_threshold

    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        # Column-level rules
        for column in profile.columns:
            rules.extend(self.generate_for_column(column, strictness))

        # Correlation rules
        if profile.correlations:
            rules.extend(self._correlation_rules(profile, strictness))

        return rules

    def generate_for_column(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        # Outlier detection for numeric columns
        if column.distribution:
            rules.extend(self._outlier_rules(column, strictness))
            rules.extend(self._distribution_shape_rules(column, strictness))

        return rules

    def _outlier_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate outlier detection rules."""
        rules: list[GeneratedRule] = []
        dist = column.distribution

        if not dist:
            return rules

        # Z-score based outlier detection
        if dist.mean is not None and dist.std is not None and dist.std > 0:
            # Adjust threshold based on strictness
            z_threshold = {
                Strictness.LOOSE: 4.0,
                Strictness.MEDIUM: 3.0,
                Strictness.STRICT: 2.5,
            }[strictness]

            rules.append(
                RuleBuilder(f"zscore_outlier_{column.name}")
                .validator("ZScoreOutlierValidator")
                .category(RuleCategory.ANOMALY)
                .column(column.name)
                .params(threshold=z_threshold)
                .confidence(RuleConfidence.MEDIUM)
                .description(
                    f"Column '{column.name}' values should not have Z-score > {z_threshold}"
                )
                .rationale(
                    f"Outlier detection based on observed mean={dist.mean:.2f}, std={dist.std:.2f}"
                )
                .build()
            )

        # IQR-based outlier detection
        if dist.q1 is not None and dist.q3 is not None:
            iqr = dist.q3 - dist.q1
            if iqr > 0:
                # Adjust multiplier based on strictness
                multiplier = {
                    Strictness.LOOSE: 2.0,
                    Strictness.MEDIUM: 1.5,
                    Strictness.STRICT: 1.0,
                }[strictness]

                lower_bound = dist.q1 - multiplier * iqr
                upper_bound = dist.q3 + multiplier * iqr

                rules.append(
                    RuleBuilder(f"iqr_outlier_{column.name}")
                    .validator("IQRAnomalyValidator")
                    .category(RuleCategory.ANOMALY)
                    .column(column.name)
                    .params(iqr_multiplier=multiplier)
                    .confidence(RuleConfidence.MEDIUM)
                    .description(
                        f"Column '{column.name}' values should be within "
                        f"IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                    )
                    .rationale(
                        f"IQR-based detection with Q1={dist.q1:.2f}, Q3={dist.q3:.2f}"
                    )
                    .build()
                )

        return rules

    def _distribution_shape_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate distribution shape validation rules."""
        rules: list[GeneratedRule] = []
        dist = column.distribution

        if not dist:
            return rules

        # Only for strict mode - validate distribution shape
        if strictness != Strictness.STRICT:
            return rules

        # Skewness check
        if dist.skewness is not None:
            abs_skew = abs(dist.skewness)

            if abs_skew < 0.5:
                shape = "symmetric"
            elif abs_skew < 1.0:
                shape = "moderately_skewed"
            else:
                shape = "highly_skewed"

            # Only create rule if distribution is reasonably symmetric
            if abs_skew < 1.0:
                rules.append(
                    RuleBuilder(f"skewness_{column.name}")
                    .validator("DistributionShapeValidator")
                    .category(RuleCategory.DISTRIBUTION)
                    .column(column.name)
                    .params(
                        max_skewness=abs_skew * 1.5,  # Allow 50% increase
                    )
                    .confidence(RuleConfidence.LOW)
                    .description(
                        f"Column '{column.name}' should maintain {shape} distribution"
                    )
                    .rationale(f"Observed skewness: {dist.skewness:.2f}")
                    .build()
                )

        # Kurtosis check (for detecting unusual peakedness)
        if dist.kurtosis is not None:
            # Normal distribution has kurtosis of 3 (excess kurtosis of 0)
            excess_kurtosis = dist.kurtosis - 3

            if abs(excess_kurtosis) > 2:
                # Unusual distribution shape
                rules.append(
                    RuleBuilder(f"kurtosis_{column.name}")
                    .validator("DistributionShapeValidator")
                    .category(RuleCategory.DISTRIBUTION)
                    .column(column.name)
                    .params(
                        min_kurtosis=dist.kurtosis * 0.5,
                        max_kurtosis=dist.kurtosis * 1.5,
                    )
                    .confidence(RuleConfidence.LOW)
                    .description(
                        f"Column '{column.name}' kurtosis should be stable"
                    )
                    .rationale(f"Observed kurtosis: {dist.kurtosis:.2f}")
                    .build()
                )

        return rules

    def _correlation_rules(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate correlation-based validation rules."""
        rules: list[GeneratedRule] = []

        if not profile.correlations:
            return rules

        # Threshold based on strictness
        min_correlation = {
            Strictness.LOOSE: 0.8,
            Strictness.MEDIUM: 0.7,
            Strictness.STRICT: 0.6,
        }[strictness]

        for col1, col2, correlation in profile.correlations:
            if abs(correlation) >= min_correlation:
                rules.append(
                    RuleBuilder(f"correlation_{col1}_{col2}")
                    .validator("ColumnCorrelationValidator")
                    .category(RuleCategory.RELATIONSHIP)
                    .columns(col1, col2)
                    .params(
                        min_correlation=abs(correlation) * 0.8,  # Allow 20% decrease
                        max_correlation=min(1.0, abs(correlation) * 1.2),  # Allow 20% increase
                    )
                    .confidence(RuleConfidence.MEDIUM)
                    .description(
                        f"Correlation between '{col1}' and '{col2}' should be "
                        f"maintained around {correlation:.2f}"
                    )
                    .rationale(f"Observed correlation: {correlation:.2f}")
                    .build()
                )

        return rules


@register_generator("advanced_anomaly")
class AdvancedAnomalyRuleGenerator(RuleGenerator):
    """Generates advanced anomaly detection rules.

    This generator creates rules for more sophisticated anomaly detection
    that may require additional dependencies (sklearn, etc.).
    """

    name = "advanced_anomaly"
    description = "Generates advanced anomaly detection rules"
    categories = {RuleCategory.ANOMALY}
    priority = 40

    def __init__(
        self,
        enable_isolation_forest: bool = False,
        enable_lof: bool = False,
        contamination: float = 0.05,
        **kwargs: Any,
    ):
        """Initialize advanced anomaly generator.

        Args:
            enable_isolation_forest: Enable Isolation Forest rules
            enable_lof: Enable Local Outlier Factor rules
            contamination: Expected proportion of outliers
        """
        super().__init__(**kwargs)
        self.enable_isolation_forest = enable_isolation_forest
        self.enable_lof = enable_lof
        self.contamination = contamination

    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        # Only generate if enabled
        if not self.enable_isolation_forest and not self.enable_lof:
            return rules

        # Find numeric columns suitable for multivariate analysis
        numeric_columns = [
            col for col in profile.columns
            if col.distribution is not None
        ]

        if len(numeric_columns) < 2:
            return rules

        column_names = [col.name for col in numeric_columns]

        # Adjust contamination based on strictness
        contamination = {
            Strictness.LOOSE: 0.10,
            Strictness.MEDIUM: 0.05,
            Strictness.STRICT: 0.01,
        }[strictness]

        if self.enable_isolation_forest:
            rules.append(
                RuleBuilder("isolation_forest_anomaly")
                .validator("IsolationForestValidator")
                .category(RuleCategory.ANOMALY)
                .columns(*column_names)
                .params(contamination=contamination)
                .confidence(RuleConfidence.LOW)
                .description(
                    f"Multivariate anomaly detection on columns: {', '.join(column_names)}"
                )
                .rationale(
                    f"Isolation Forest with contamination={contamination}"
                )
                .build()
            )

        if self.enable_lof:
            rules.append(
                RuleBuilder("lof_anomaly")
                .validator("LOFValidator")
                .category(RuleCategory.ANOMALY)
                .columns(*column_names)
                .params(contamination=contamination, n_neighbors=20)
                .confidence(RuleConfidence.LOW)
                .description(
                    f"Local Outlier Factor detection on columns: {', '.join(column_names)}"
                )
                .rationale(
                    f"LOF with contamination={contamination}"
                )
                .build()
            )

        return rules
