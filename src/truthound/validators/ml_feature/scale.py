"""Feature scale validators.

This module provides validators for analyzing feature scale
consistency in ML pipelines.
"""

from enum import Enum
from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.ml_feature.base import MLFeatureValidator, FeatureStats


class ScaleType(str, Enum):
    """Feature scale types."""

    STANDARD = "standard"  # Mean ~0, std ~1
    MINMAX = "minmax"  # Range [0, 1]
    ROBUST = "robust"  # Median-centered, IQR-scaled
    RAW = "raw"  # Unscaled
    UNKNOWN = "unknown"


@register_validator
class FeatureScaleValidator(MLFeatureValidator):
    """Validates feature scale consistency.

    Checks for:
    - Scale inconsistency across features
    - Features with vastly different magnitudes
    - Features needing normalization
    - Expected scale violations

    Example:
        validator = FeatureScaleValidator(
            expected_scale=ScaleType.STANDARD,
            max_scale_ratio=100.0,
        )
    """

    name = "feature_scale"

    def __init__(
        self,
        columns: list[str] | None = None,
        expected_scale: ScaleType | str | None = None,
        max_scale_ratio: float = 100.0,
        max_magnitude: float = 1e6,
        check_standard_scale: bool = False,
        std_tolerance: float = 0.5,
        mean_tolerance: float = 0.5,
        **kwargs: Any,
    ):
        """Initialize scale validator.

        Args:
            columns: Specific columns to validate
            expected_scale: Expected scale type for all features
            max_scale_ratio: Maximum ratio between feature scales
            max_magnitude: Maximum allowed magnitude for any feature
            check_standard_scale: Whether to check for standard scaling
            std_tolerance: Tolerance for std deviation from 1
            mean_tolerance: Tolerance for mean deviation from 0
            **kwargs: Additional config
        """
        super().__init__(columns=columns, **kwargs)

        if isinstance(expected_scale, str):
            self.expected_scale = ScaleType(expected_scale)
        else:
            self.expected_scale = expected_scale

        self.max_scale_ratio = max_scale_ratio
        self.max_magnitude = max_magnitude
        self.check_standard_scale = check_standard_scale
        self.std_tolerance = std_tolerance
        self.mean_tolerance = mean_tolerance

    def _detect_scale_type(self, stats: FeatureStats) -> ScaleType:
        """Detect the scale type of a feature.

        Args:
            stats: Feature statistics

        Returns:
            Detected scale type
        """
        if stats.mean is None or stats.std is None:
            return ScaleType.UNKNOWN

        # Check for standard scaling (mean ~0, std ~1)
        if abs(stats.mean) < 0.5 and 0.5 <= stats.std <= 1.5:
            return ScaleType.STANDARD

        # Check for minmax scaling (range ~[0,1])
        if stats.min_value is not None and stats.max_value is not None:
            if (
                abs(stats.min_value) < 0.1
                and abs(stats.max_value - 1.0) < 0.1
            ):
                return ScaleType.MINMAX

        return ScaleType.RAW

    def validate_features(
        self, df: pl.DataFrame, columns: list[str]
    ) -> list[ValidationIssue]:
        """Validate feature scale consistency.

        Args:
            df: Input DataFrame
            columns: List of feature columns

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if len(columns) == 0:
            return issues

        # Compute stats for all features
        feature_stats: list[FeatureStats] = []
        for col in columns:
            stats = self._compute_feature_stats(df, col)
            feature_stats.append(stats)

        # Check for scale inconsistency
        scales = []
        for stats in feature_stats:
            if stats.scale is not None and stats.scale > 0:
                scales.append((stats.name, stats.scale))

        if len(scales) > 1:
            max_scale = max(s[1] for s in scales)
            min_scale = min(s[1] for s in scales)

            if min_scale > 0:
                scale_ratio = max_scale / min_scale
                if scale_ratio > self.max_scale_ratio:
                    largest = max(scales, key=lambda x: x[1])
                    smallest = min(scales, key=lambda x: x[1])
                    issues.append(
                        ValidationIssue(
                            column="multiple",
                            issue_type="scale_inconsistency",
                            count=len(columns),
                            severity=Severity.MEDIUM,
                            details=(
                                f"Feature scales vary by {scale_ratio:.1f}x. "
                                f"Largest: {largest[0]} (scale={largest[1]:.2f}), "
                                f"Smallest: {smallest[0]} (scale={smallest[1]:.2f}). "
                                f"Consider normalizing features."
                            ),
                            expected=f"Scale ratio <= {self.max_scale_ratio}",
                        )
                    )

        # Check for extreme magnitudes
        extreme_features = []
        for stats in feature_stats:
            if stats.max_value is not None and abs(stats.max_value) > self.max_magnitude:
                extreme_features.append((stats.name, stats.max_value))
            elif stats.min_value is not None and abs(stats.min_value) > self.max_magnitude:
                extreme_features.append((stats.name, stats.min_value))

        if extreme_features:
            details = [f"{name}: {val:.2e}" for name, val in extreme_features[:5]]
            issues.append(
                ValidationIssue(
                    column="multiple",
                    issue_type="extreme_magnitude",
                    count=len(extreme_features),
                    severity=Severity.MEDIUM,
                    details=(
                        f"Found {len(extreme_features)} features with extreme magnitudes. "
                        f"Features: {', '.join(details)}"
                    ),
                    expected=f"Magnitude <= {self.max_magnitude:.2e}",
                )
            )

        # Check expected scale violations
        if self.expected_scale:
            violations = []
            for stats in feature_stats:
                detected = self._detect_scale_type(stats)
                if detected != self.expected_scale and detected != ScaleType.UNKNOWN:
                    violations.append((stats.name, detected.value))

            if violations:
                details = [f"{name} ({scale})" for name, scale in violations[:5]]
                issues.append(
                    ValidationIssue(
                        column="multiple",
                        issue_type="unexpected_scale_type",
                        count=len(violations),
                        severity=Severity.LOW,
                        details=(
                            f"Found {len(violations)} features not matching expected "
                            f"scale type '{self.expected_scale.value}'. "
                            f"Features: {', '.join(details)}"
                        ),
                        expected=f"Scale type: {self.expected_scale.value}",
                    )
                )

        # Check for standard scaling if requested
        if self.check_standard_scale:
            not_standard = []
            for stats in feature_stats:
                if stats.mean is None or stats.std is None:
                    continue

                mean_off = abs(stats.mean) > self.mean_tolerance
                std_off = abs(stats.std - 1.0) > self.std_tolerance

                if mean_off or std_off:
                    not_standard.append(
                        (stats.name, stats.mean, stats.std)
                    )

            if not_standard:
                details = [
                    f"{name} (mean={mean:.2f}, std={std:.2f})"
                    for name, mean, std in not_standard[:5]
                ]
                issues.append(
                    ValidationIssue(
                        column="multiple",
                        issue_type="not_standard_scaled",
                        count=len(not_standard),
                        severity=Severity.LOW,
                        details=(
                            f"Found {len(not_standard)} features not standard scaled. "
                            f"Features: {', '.join(details)}"
                        ),
                        expected=f"Mean ~0 (±{self.mean_tolerance}), Std ~1 (±{self.std_tolerance})",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.collect()
        columns = self._get_feature_columns(df)

        if not columns:
            return []

        return self.validate_features(df, columns)
