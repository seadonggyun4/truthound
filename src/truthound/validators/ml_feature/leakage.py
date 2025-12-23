"""Target leakage detection validators.

This module provides validators for detecting target leakage
in ML training data.
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.ml_feature.base import MLFeatureValidator, LeakageResult


@register_validator
class TargetLeakageValidator(MLFeatureValidator):
    """Validates for target leakage in training features.

    Detects features that:
    - Have suspiciously high correlation with target
    - Contain future information (temporal leakage)
    - Are derived from the target (data leakage)

    Example:
        validator = TargetLeakageValidator(
            target_column="label",
            max_correlation=0.95,
        )
    """

    name = "target_leakage"

    def __init__(
        self,
        target_column: str,
        columns: list[str] | None = None,
        max_correlation: float = 0.95,
        warn_correlation: float = 0.8,
        check_mutual_info: bool = True,
        check_temporal: bool = False,
        timestamp_column: str | None = None,
        min_samples: int = 30,
        **kwargs: Any,
    ):
        """Initialize target leakage validator.

        Args:
            target_column: Name of the target column
            columns: Feature columns to check (None = all except target)
            max_correlation: Correlation threshold for leakage detection
            warn_correlation: Correlation threshold for warning
            check_mutual_info: Whether to compute mutual information
            check_temporal: Whether to check for temporal leakage
            timestamp_column: Column containing timestamps (for temporal check)
            min_samples: Minimum samples for computation
            **kwargs: Additional config
        """
        super().__init__(columns=columns, **kwargs)
        self.target_column = target_column
        self.max_correlation = max_correlation
        self.warn_correlation = warn_correlation
        self.check_mutual_info = check_mutual_info
        self.check_temporal = check_temporal
        self.timestamp_column = timestamp_column
        self.min_samples = min_samples

    def _compute_mutual_information(
        self, x: np.ndarray, y: np.ndarray, n_bins: int = 20
    ) -> float:
        """Compute mutual information between two arrays.

        Uses binned entropy estimation.

        Args:
            x: First array
            y: Second array
            n_bins: Number of bins for discretization

        Returns:
            Mutual information estimate
        """
        try:
            # Discretize continuous variables
            x_bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins))
            y_bins = np.digitize(y, np.linspace(y.min(), y.max(), n_bins))

            # Compute joint and marginal probabilities
            joint_hist = np.histogram2d(x_bins, y_bins, bins=n_bins)[0]
            joint_prob = joint_hist / joint_hist.sum()

            x_prob = joint_prob.sum(axis=1)
            y_prob = joint_prob.sum(axis=0)

            # Compute MI
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                        mi += joint_prob[i, j] * np.log2(
                            joint_prob[i, j] / (x_prob[i] * y_prob[j])
                        )

            return max(0.0, float(mi))
        except Exception:
            return 0.0

    def _check_temporal_leakage(
        self, df: pl.DataFrame, feature_col: str
    ) -> tuple[bool, str]:
        """Check if a feature contains temporal leakage.

        Args:
            df: Input DataFrame
            feature_col: Feature column to check

        Returns:
            Tuple of (is_leaky, reason)
        """
        if not self.timestamp_column or self.timestamp_column not in df.columns:
            return False, ""

        try:
            # Sort by timestamp
            sorted_df = df.sort(self.timestamp_column)

            # Check if feature values depend on future target values
            # Simple heuristic: check correlation of feature with shifted target
            target = sorted_df[self.target_column].to_numpy()
            feature = sorted_df[feature_col].to_numpy()

            # Create future target (shifted back)
            future_target = np.roll(target, -1)

            # Compute correlation with future
            valid_idx = ~(
                np.isnan(feature[:-1]) | np.isnan(future_target[:-1])
            )
            if valid_idx.sum() < self.min_samples:
                return False, ""

            future_corr = np.corrcoef(
                feature[:-1][valid_idx],
                future_target[:-1][valid_idx]
            )[0, 1]

            # Compute correlation with current target
            valid_idx = ~(np.isnan(feature) | np.isnan(target))
            if valid_idx.sum() < self.min_samples:
                return False, ""

            current_corr = np.corrcoef(feature[valid_idx], target[valid_idx])[0, 1]

            # If correlation with future target is higher, suspicious
            if not np.isnan(future_corr) and abs(future_corr) > abs(current_corr) * 1.2:
                return True, f"Higher correlation with future target ({future_corr:.3f} vs {current_corr:.3f})"

            return False, ""
        except Exception:
            return False, ""

    def _analyze_feature(
        self, df: pl.DataFrame, feature_col: str
    ) -> LeakageResult | None:
        """Analyze a single feature for leakage.

        Args:
            df: Input DataFrame
            feature_col: Feature column

        Returns:
            LeakageResult or None
        """
        try:
            # Get valid pairs
            valid_mask = (
                df[feature_col].is_not_null()
                & df[self.target_column].is_not_null()
            )
            valid_df = df.filter(valid_mask)

            if len(valid_df) < self.min_samples:
                return None

            feature = valid_df[feature_col].to_numpy().astype(float)
            target = valid_df[self.target_column].to_numpy().astype(float)

            # Compute correlation
            correlation = np.corrcoef(feature, target)[0, 1]
            if np.isnan(correlation):
                return None

            result = LeakageResult(
                feature=feature_col,
                correlation=float(correlation),
            )

            # Compute mutual information if requested
            if self.check_mutual_info:
                result.mutual_information = self._compute_mutual_information(
                    feature, target
                )

            # Check for temporal leakage
            if self.check_temporal:
                is_temporal_leak, reason = self._check_temporal_leakage(
                    df, feature_col
                )
                if is_temporal_leak:
                    result.is_suspicious = True
                    result.reason = f"Temporal leakage: {reason}"
                    return result

            # Compute leakage score
            result.leakage_score = abs(correlation)
            if result.mutual_information:
                # Normalize MI to [0, 1] range roughly
                result.leakage_score = max(
                    result.leakage_score,
                    min(1.0, result.mutual_information / 3.0)
                )

            # Check if suspicious
            if abs(correlation) >= self.max_correlation:
                result.is_suspicious = True
                result.reason = f"Very high correlation ({correlation:.3f})"
            elif abs(correlation) >= self.warn_correlation:
                result.is_suspicious = True
                result.reason = f"High correlation ({correlation:.3f})"

            return result
        except Exception:
            return None

    def validate_features(
        self, df: pl.DataFrame, columns: list[str]
    ) -> list[ValidationIssue]:
        """Validate features for target leakage.

        Args:
            df: Input DataFrame
            columns: List of feature columns

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if self.target_column not in df.columns:
            issues.append(
                ValidationIssue(
                    column=self.target_column,
                    issue_type="target_column_missing",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"Target column '{self.target_column}' not found in data",
                    expected="Target column present",
                )
            )
            return issues

        # Exclude target from features
        feature_cols = [c for c in columns if c != self.target_column]
        if self.timestamp_column:
            feature_cols = [c for c in feature_cols if c != self.timestamp_column]

        if not feature_cols:
            return issues

        # Analyze each feature
        leaky_features: list[LeakageResult] = []
        warning_features: list[LeakageResult] = []

        for col in feature_cols:
            result = self._analyze_feature(df, col)
            if result is None:
                continue

            if abs(result.correlation) >= self.max_correlation:
                leaky_features.append(result)
            elif abs(result.correlation) >= self.warn_correlation:
                warning_features.append(result)

        # Report definite leakage
        if leaky_features:
            leaky_features.sort(key=lambda x: abs(x.correlation), reverse=True)
            details = [
                f"{r.feature}: r={r.correlation:.3f}"
                for r in leaky_features[:5]
            ]
            issues.append(
                ValidationIssue(
                    column="multiple",
                    issue_type="target_leakage_detected",
                    count=len(leaky_features),
                    severity=Severity.CRITICAL,
                    details=(
                        f"Found {len(leaky_features)} features with potential target leakage "
                        f"(|r| >= {self.max_correlation}). These features may contain "
                        f"information derived from the target. Features: {', '.join(details)}"
                    ),
                    expected=f"|correlation| < {self.max_correlation}",
                )
            )

        # Report warnings
        if warning_features:
            warning_features.sort(key=lambda x: abs(x.correlation), reverse=True)
            details = [
                f"{r.feature}: r={r.correlation:.3f}"
                for r in warning_features[:5]
            ]
            issues.append(
                ValidationIssue(
                    column="multiple",
                    issue_type="potential_target_leakage",
                    count=len(warning_features),
                    severity=Severity.HIGH,
                    details=(
                        f"Found {len(warning_features)} features with suspicious correlation "
                        f"to target (|r| >= {self.warn_correlation}). Review these features "
                        f"for potential leakage. Features: {', '.join(details)}"
                    ),
                    expected=f"|correlation| < {self.warn_correlation}",
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

        return self.validate_features(df, columns)
