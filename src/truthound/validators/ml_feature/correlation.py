"""Feature correlation validators.

This module provides validators for detecting multicollinearity
and excessive correlation in ML features.
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.ml_feature.base import MLFeatureValidator, CorrelationResult


@register_validator
class FeatureCorrelationMatrixValidator(MLFeatureValidator):
    """Validates feature correlation matrix for multicollinearity.

    Detects:
    - Highly correlated feature pairs
    - Perfect multicollinearity
    - Features that can be dropped due to redundancy
    - Correlation clusters

    Example:
        validator = FeatureCorrelationMatrixValidator(
            max_correlation=0.9,
            max_vif=10.0,
        )
    """

    name = "feature_correlation_matrix"

    def __init__(
        self,
        columns: list[str] | None = None,
        max_correlation: float = 0.9,
        warn_correlation: float = 0.7,
        max_vif: float | None = 10.0,
        compute_vif: bool = False,
        min_samples: int = 30,
        **kwargs: Any,
    ):
        """Initialize correlation matrix validator.

        Args:
            columns: Specific columns to validate
            max_correlation: Maximum acceptable absolute correlation
            warn_correlation: Correlation threshold for warnings
            max_vif: Maximum variance inflation factor (for multicollinearity)
            compute_vif: Whether to compute VIF (expensive for many features)
            min_samples: Minimum samples for correlation computation
            **kwargs: Additional config
        """
        super().__init__(columns=columns, **kwargs)
        self.max_correlation = max_correlation
        self.warn_correlation = warn_correlation
        self.max_vif = max_vif
        self.compute_vif = compute_vif
        self.min_samples = min_samples

    def _compute_correlation_matrix(
        self, df: pl.DataFrame, columns: list[str]
    ) -> np.ndarray | None:
        """Compute correlation matrix.

        Args:
            df: Input DataFrame
            columns: Columns to include

        Returns:
            Correlation matrix as numpy array
        """
        if len(df) < self.min_samples:
            return None

        try:
            # Get data as numpy array
            data = df.select(columns).to_numpy()

            # Remove rows with any NaN
            valid_mask = ~np.any(np.isnan(data), axis=1)
            data = data[valid_mask]

            if len(data) < self.min_samples:
                return None

            # Compute correlation matrix
            return np.corrcoef(data, rowvar=False)
        except Exception:
            return None

    def _compute_vif(
        self, df: pl.DataFrame, columns: list[str], target_col: str
    ) -> float | None:
        """Compute Variance Inflation Factor for a feature.

        VIF = 1 / (1 - R²) where R² is from regressing target on others.

        Args:
            df: Input DataFrame
            columns: All feature columns
            target_col: Column to compute VIF for

        Returns:
            VIF value or None
        """
        if len(df) < self.min_samples:
            return None

        try:
            other_cols = [c for c in columns if c != target_col]
            if not other_cols:
                return 1.0

            # Get data
            y = df[target_col].drop_nulls().to_numpy()
            X = df.select(other_cols).drop_nulls().to_numpy()

            if len(y) != len(X) or len(y) < self.min_samples:
                return None

            # Simple OLS regression to get R²
            # Add intercept
            X = np.column_stack([np.ones(len(X)), X])

            # Solve normal equations: beta = (X'X)^(-1) X'y
            XtX = X.T @ X
            Xty = X.T @ y

            # Check for singularity
            try:
                beta = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                return float("inf")

            # Compute R²
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                return float("inf")

            r_squared = 1 - (ss_res / ss_tot)

            if r_squared >= 1.0:
                return float("inf")

            return 1.0 / (1.0 - r_squared)
        except Exception:
            return None

    def validate_features(
        self, df: pl.DataFrame, columns: list[str]
    ) -> list[ValidationIssue]:
        """Validate feature correlations.

        Args:
            df: Input DataFrame
            columns: List of feature columns

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if len(columns) < 2:
            return issues

        # Compute correlation matrix
        corr_matrix = self._compute_correlation_matrix(df, columns)

        if corr_matrix is None:
            return issues

        # Find highly correlated pairs
        high_corr_pairs: list[CorrelationResult] = []
        warn_corr_pairs: list[CorrelationResult] = []

        n_cols = len(columns)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                corr = corr_matrix[i, j]
                if np.isnan(corr):
                    continue

                abs_corr = abs(corr)

                if abs_corr >= self.max_correlation:
                    high_corr_pairs.append(
                        CorrelationResult(
                            feature1=columns[i],
                            feature2=columns[j],
                            correlation=float(corr),
                        )
                    )
                elif abs_corr >= self.warn_correlation:
                    warn_corr_pairs.append(
                        CorrelationResult(
                            feature1=columns[i],
                            feature2=columns[j],
                            correlation=float(corr),
                        )
                    )

        # Report high correlations
        if high_corr_pairs:
            # Sort by absolute correlation descending
            high_corr_pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
            details = [
                f"{r.feature1} <-> {r.feature2}: {r.correlation:.3f}"
                for r in high_corr_pairs[:5]
            ]
            issues.append(
                ValidationIssue(
                    column="multiple",
                    issue_type="high_feature_correlation",
                    count=len(high_corr_pairs),
                    severity=Severity.HIGH,
                    details=(
                        f"Found {len(high_corr_pairs)} highly correlated feature pairs "
                        f"(|r| >= {self.max_correlation}). Consider removing redundant features. "
                        f"Pairs: {'; '.join(details)}"
                    ),
                    expected=f"|correlation| < {self.max_correlation}",
                )
            )

        # Report warning-level correlations
        if warn_corr_pairs:
            warn_corr_pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
            details = [
                f"{r.feature1} <-> {r.feature2}: {r.correlation:.3f}"
                for r in warn_corr_pairs[:5]
            ]
            issues.append(
                ValidationIssue(
                    column="multiple",
                    issue_type="elevated_feature_correlation",
                    count=len(warn_corr_pairs),
                    severity=Severity.MEDIUM,
                    details=(
                        f"Found {len(warn_corr_pairs)} feature pairs with elevated correlation "
                        f"(|r| >= {self.warn_correlation}). Pairs: {'; '.join(details)}"
                    ),
                    expected=f"|correlation| < {self.warn_correlation}",
                )
            )

        # Compute VIF if requested
        if self.compute_vif and self.max_vif and len(columns) <= 20:
            high_vif_features = []
            for col in columns:
                vif = self._compute_vif(df, columns, col)
                if vif is not None and vif > self.max_vif:
                    high_vif_features.append((col, vif))

            if high_vif_features:
                high_vif_features.sort(key=lambda x: x[1], reverse=True)
                details = [
                    f"{name}: VIF={vif:.1f}"
                    for name, vif in high_vif_features[:5]
                ]
                issues.append(
                    ValidationIssue(
                        column="multiple",
                        issue_type="high_vif",
                        count=len(high_vif_features),
                        severity=Severity.MEDIUM,
                        details=(
                            f"Found {len(high_vif_features)} features with high VIF "
                            f"(multicollinearity). Features: {', '.join(details)}"
                        ),
                        expected=f"VIF < {self.max_vif}",
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

        if len(columns) < 2:
            return []

        return self.validate_features(df, columns)
