"""String casing validators."""

from typing import Any, Literal

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator, StringValidatorMixin
from truthound.validators.registry import register_validator


@register_validator
class ConsistentCasingValidator(Validator, StringValidatorMixin):
    """Validates that string values have consistent casing.

    Detects mixing of different casing conventions within a column.

    Example:
        validator = ConsistentCasingValidator(
            column="product_name",
            expected_casing="title",  # Title Case
        )
    """

    name = "consistent_casing"
    category = "string"

    CASING_CHECKS = {
        "upper": str.isupper,
        "lower": str.islower,
        "title": str.istitle,
    }

    def __init__(
        self,
        expected_casing: Literal["upper", "lower", "title"] | None = None,
        column: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.expected_casing = expected_casing
        self.target_column = column

    def _detect_casing(self, value: str) -> str | None:
        """Detect the casing style of a string."""
        if not value or not value.strip():
            return None

        # Only check alphabetic parts
        alpha_only = "".join(c for c in value if c.isalpha())
        if not alpha_only:
            return None

        if alpha_only.isupper():
            return "upper"
        elif alpha_only.islower():
            return "lower"
        elif value.istitle():
            return "title"
        else:
            return "mixed"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.target_column:
            columns = [self.target_column]
        else:
            columns = self._get_string_columns(lf)

        if not columns:
            return issues

        # Use streaming for large datasets
        df = lf.collect(engine="streaming")

        for col in columns:
            col_data = df.get_column(col).drop_nulls()

            if len(col_data) == 0:
                continue

            casing_counts: dict[str, int] = {}
            samples: dict[str, list[str]] = {}

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                casing = self._detect_casing(val)
                if casing:
                    casing_counts[casing] = casing_counts.get(casing, 0) + 1
                    if casing not in samples:
                        samples[casing] = []
                    if len(samples[casing]) < 3:
                        samples[casing].append(val[:30])

            if not casing_counts:
                continue

            if self.expected_casing:
                # Check against expected casing
                violations = sum(
                    count for casing, count in casing_counts.items()
                    if casing != self.expected_casing
                )
                total = sum(casing_counts.values())

                if violations > 0:
                    if self._passes_mostly(violations, total):
                        continue

                    ratio = violations / total
                    violation_samples = [
                        s for casing, sample_list in samples.items()
                        if casing != self.expected_casing
                        for s in sample_list
                    ]

                    issues.append(
                        ValidationIssue(
                            column=col,
                            issue_type="casing_violation",
                            count=violations,
                            severity=self._calculate_severity(ratio),
                            details=f"Expected {self.expected_casing} casing",
                            expected=self.expected_casing,
                            sample_values=violation_samples[: self.config.sample_size],
                        )
                    )
            else:
                # Check for inconsistent casing (multiple styles)
                if len(casing_counts) > 1 and "mixed" not in casing_counts:
                    total = sum(casing_counts.values())
                    dominant_casing = max(casing_counts, key=casing_counts.get)
                    violations = total - casing_counts[dominant_casing]

                    if violations > 0:
                        if self._passes_mostly(violations, total):
                            continue

                        ratio = violations / total
                        issues.append(
                            ValidationIssue(
                                column=col,
                                issue_type="inconsistent_casing",
                                count=violations,
                                severity=self._calculate_severity(ratio),
                                details=f"Mixed casing styles: {dict(casing_counts)}",
                                expected=f"Consistent {dominant_casing} casing",
                                sample_values=[
                                    s for casing, sample_list in samples.items()
                                    if casing != dominant_casing
                                    for s in sample_list
                                ][: self.config.sample_size],
                            )
                        )

        return issues
