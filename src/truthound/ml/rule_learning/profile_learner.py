"""Profile-based rule learning.

Learns validation rules from data profile statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl

from truthound.ml.base import (
    LearnedRule,
    RuleLearner,
    RuleLearningConfig,
    RuleLearningResult,
    ModelInfo,
    ModelState,
    ModelTrainingError,
    ModelType,
    register_model,
)


@dataclass
class ProfileLearnerConfig(RuleLearningConfig):
    """Configuration for profile-based rule learning.

    Attributes:
        strictness: Rule strictness ('loose', 'medium', 'strict')
        include_range_rules: Generate range/bounds rules
        include_uniqueness_rules: Generate uniqueness rules
        include_format_rules: Generate format/pattern rules
        include_null_rules: Generate null/completeness rules
        include_type_rules: Generate type rules
        null_threshold: Threshold for generating null rules
        uniqueness_threshold: Threshold for uniqueness rules
    """

    strictness: str = "medium"
    include_range_rules: bool = True
    include_uniqueness_rules: bool = True
    include_format_rules: bool = True
    include_null_rules: bool = True
    include_type_rules: bool = True
    null_threshold: float = 0.01  # Generate rule if < 1% nulls
    uniqueness_threshold: float = 0.99  # Generate rule if > 99% unique


@register_model("profile_learner")
class DataProfileRuleLearner(RuleLearner):
    """Learn validation rules from data profile.

    Analyzes data characteristics and generates appropriate
    validation rules based on observed patterns.

    Generates rules for:
    - Range constraints (numeric min/max)
    - Null/completeness checks
    - Uniqueness constraints
    - Type constraints
    - Format patterns (strings)

    Example:
        >>> learner = DataProfileRuleLearner(strictness="medium")
        >>> result = learner.learn_rules(data)
        >>> print(f"Learned {len(result.rules)} rules")
    """

    def __init__(self, config: ProfileLearnerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._profiles: dict[str, dict] = {}

    @property
    def config(self) -> ProfileLearnerConfig:
        return self._config  # type: ignore

    def _default_config(self) -> ProfileLearnerConfig:
        return ProfileLearnerConfig()

    def _get_model_name(self) -> str:
        return "profile_learner"

    def _get_description(self) -> str:
        return "Profile-based rule learning from data statistics"

    def learn_rules(self, data: pl.LazyFrame) -> RuleLearningResult:
        """Learn validation rules from data profile.

        Args:
            data: Data to analyze

        Returns:
            RuleLearningResult with learned rules
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            df = self._maybe_sample(data).collect()

            rules: list[LearnedRule] = []
            filtered_count = 0

            # Profile each column
            schema = df.schema
            for col in df.columns:
                dtype = schema[col]
                profile = self._profile_column(df, col, dtype)
                self._profiles[col] = profile

                # Generate rules for this column
                col_rules, col_filtered = self._generate_column_rules(
                    col, profile, row_count
                )
                rules.extend(col_rules)
                filtered_count += col_filtered

            # Filter by min_confidence
            final_rules = [
                r for r in rules
                if r.confidence >= self.config.min_confidence
            ]

            # Limit rules
            if len(final_rules) > self.config.max_rules:
                final_rules = sorted(
                    final_rules,
                    key=lambda r: (r.confidence, r.support),
                    reverse=True,
                )[:self.config.max_rules]

            elapsed = (time.perf_counter() - start) * 1000

            result = RuleLearningResult(
                rules=tuple(final_rules),
                total_rules=len(rules),
                filtered_rules=filtered_count + (len(rules) - len(final_rules)),
                learning_time_ms=elapsed,
                data_profile={"columns": len(df.columns), "rows": row_count},
            )

            self._learned_rules = result
            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED

            return result

        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelTrainingError(
                f"Failed to learn rules: {e}",
                model_name=self.info.name,
            ) from e

    def _profile_column(
        self, df: pl.DataFrame, column: str, dtype: pl.DataType
    ) -> dict[str, Any]:
        """Profile a single column."""
        total = len(df)
        null_count = df[column].null_count()

        profile = {
            "dtype": str(dtype),
            "dtype_class": type(dtype).__name__,
            "total": total,
            "null_count": null_count,
            "null_ratio": null_count / total if total > 0 else 0,
            "non_null_count": total - null_count,
        }

        # Distinct count
        distinct = df[column].n_unique()
        profile["distinct_count"] = distinct
        profile["distinct_ratio"] = distinct / total if total > 0 else 0

        # Type-specific profiling
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }
        string_types = {pl.String, pl.Utf8}

        if type(dtype) in numeric_types:
            profile.update(self._profile_numeric(df, column))
        elif type(dtype) in string_types:
            profile.update(self._profile_string(df, column))
        elif type(dtype) in {pl.Date, pl.Datetime}:
            profile.update(self._profile_temporal(df, column))
        elif type(dtype) == pl.Boolean:
            profile["is_boolean"] = True

        return profile

    def _profile_numeric(self, df: pl.DataFrame, column: str) -> dict[str, Any]:
        """Profile numeric column."""
        stats = df.select([
            pl.col(column).min().alias("min"),
            pl.col(column).max().alias("max"),
            pl.col(column).mean().alias("mean"),
            pl.col(column).std().alias("std"),
            pl.col(column).median().alias("median"),
            pl.col(column).quantile(0.01).alias("p01"),
            pl.col(column).quantile(0.05).alias("p05"),
            pl.col(column).quantile(0.95).alias("p95"),
            pl.col(column).quantile(0.99).alias("p99"),
        ]).row(0)

        return {
            "is_numeric": True,
            "min": stats[0],
            "max": stats[1],
            "mean": stats[2],
            "std": stats[3],
            "median": stats[4],
            "p01": stats[5],
            "p05": stats[6],
            "p95": stats[7],
            "p99": stats[8],
            "is_integer": type(df.schema[column]) in {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            },
            "is_positive": stats[0] is not None and stats[0] >= 0,
        }

    def _profile_string(self, df: pl.DataFrame, column: str) -> dict[str, Any]:
        """Profile string column."""
        # Length statistics
        length_stats = df.select([
            pl.col(column).str.len_chars().min().alias("min_len"),
            pl.col(column).str.len_chars().max().alias("max_len"),
            pl.col(column).str.len_chars().mean().alias("avg_len"),
        ]).row(0)

        profile = {
            "is_string": True,
            "min_length": length_stats[0],
            "max_length": length_stats[1],
            "avg_length": length_stats[2],
        }

        # Check for common patterns
        sample = df[column].drop_nulls().head(1000).to_list()
        if sample:
            profile["patterns"] = self._detect_patterns(sample)

        return profile

    def _profile_temporal(self, df: pl.DataFrame, column: str) -> dict[str, Any]:
        """Profile temporal column."""
        stats = df.select([
            pl.col(column).min().alias("min"),
            pl.col(column).max().alias("max"),
        ]).row(0)

        return {
            "is_temporal": True,
            "min_date": stats[0],
            "max_date": stats[1],
        }

    def _detect_patterns(self, sample: list[str]) -> list[dict]:
        """Detect common patterns in string data."""
        import re

        patterns = []
        pattern_tests = [
            ("email", r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
            ("url", r"^https?://"),
            ("phone", r"^\+?[\d\s\-().]{7,}$"),
            ("uuid", r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
            ("date_iso", r"^\d{4}-\d{2}-\d{2}$"),
            ("datetime_iso", r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),
            ("numeric_string", r"^-?\d+\.?\d*$"),
            ("alphanumeric", r"^[a-zA-Z0-9]+$"),
        ]

        for name, regex in pattern_tests:
            try:
                compiled = re.compile(regex, re.IGNORECASE)
                matches = sum(1 for s in sample if compiled.match(str(s)))
                ratio = matches / len(sample)
                if ratio >= 0.8:
                    patterns.append({"name": name, "regex": regex, "ratio": ratio})
            except Exception:
                pass

        return patterns

    def _generate_column_rules(
        self,
        column: str,
        profile: dict[str, Any],
        total_rows: int,
    ) -> tuple[list[LearnedRule], int]:
        """Generate rules for a single column."""
        rules = []
        filtered = 0
        strictness = self.config.strictness

        # Null rules
        if self.config.include_null_rules:
            null_ratio = profile.get("null_ratio", 0)
            if null_ratio < self.config.null_threshold:
                # Column is mostly non-null, generate not_null rule
                confidence = 1.0 - null_ratio
                rules.append(LearnedRule(
                    name=f"not_null_{column}",
                    rule_type="not_null",
                    column=column,
                    condition=f"{column} is not null",
                    support=1.0 - null_ratio,
                    confidence=confidence,
                    validator_config={"columns": [column]},
                    description=f"{column} should not be null",
                ))

        # Uniqueness rules
        if self.config.include_uniqueness_rules:
            distinct_ratio = profile.get("distinct_ratio", 0)
            if distinct_ratio >= self.config.uniqueness_threshold:
                rules.append(LearnedRule(
                    name=f"unique_{column}",
                    rule_type="unique",
                    column=column,
                    condition=f"{column} is unique",
                    support=distinct_ratio,
                    confidence=distinct_ratio,
                    validator_config={"columns": [column]},
                    description=f"{column} should be unique",
                ))

        # Range rules for numeric columns
        if self.config.include_range_rules and profile.get("is_numeric"):
            min_val = profile.get("min")
            max_val = profile.get("max")

            if min_val is not None and max_val is not None:
                # Use percentiles for stricter/looser bounds
                if strictness == "strict":
                    lower = profile.get("p01", min_val)
                    upper = profile.get("p99", max_val)
                elif strictness == "loose":
                    # Add 10% margin
                    range_val = max_val - min_val if max_val != min_val else abs(min_val) * 0.1
                    lower = min_val - range_val * 0.1
                    upper = max_val + range_val * 0.1
                else:  # medium
                    lower = profile.get("p05", min_val)
                    upper = profile.get("p95", max_val)

                rules.append(LearnedRule(
                    name=f"range_{column}",
                    rule_type="range",
                    column=column,
                    condition=f"{lower:.2f} <= {column} <= {upper:.2f}",
                    support=0.9 if strictness == "medium" else 0.95,
                    confidence=0.95,
                    validator_config={
                        "columns": [column],
                        "min_value": lower,
                        "max_value": upper,
                    },
                    description=f"{column} should be between {lower:.2f} and {upper:.2f}",
                ))

            # Non-negative rule
            if profile.get("is_positive") and min_val is not None and min_val >= 0:
                rules.append(LearnedRule(
                    name=f"non_negative_{column}",
                    rule_type="min_value",
                    column=column,
                    condition=f"{column} >= 0",
                    support=1.0,
                    confidence=1.0,
                    validator_config={"columns": [column], "min_value": 0},
                    description=f"{column} should be non-negative",
                ))

        # Type rules
        if self.config.include_type_rules:
            if profile.get("is_integer"):
                rules.append(LearnedRule(
                    name=f"integer_{column}",
                    rule_type="dtype",
                    column=column,
                    condition=f"{column} is integer",
                    support=1.0,
                    confidence=1.0,
                    validator_config={"columns": [column], "dtype": "integer"},
                    description=f"{column} should be integer",
                ))

        # String format rules
        if self.config.include_format_rules and profile.get("is_string"):
            patterns = profile.get("patterns", [])
            for pattern in patterns:
                if pattern["ratio"] >= 0.9:
                    rules.append(LearnedRule(
                        name=f"{pattern['name']}_{column}",
                        rule_type="regex",
                        column=column,
                        condition=f"{column} matches {pattern['name']} pattern",
                        support=pattern["ratio"],
                        confidence=pattern["ratio"],
                        validator_config={
                            "columns": [column],
                            "pattern": pattern["regex"],
                        },
                        description=f"{column} should match {pattern['name']} format",
                    ))

            # Length rules
            min_len = profile.get("min_length")
            max_len = profile.get("max_length")
            if min_len is not None and max_len is not None:
                if min_len == max_len:
                    # Fixed length
                    rules.append(LearnedRule(
                        name=f"fixed_length_{column}",
                        rule_type="length",
                        column=column,
                        condition=f"len({column}) == {min_len}",
                        support=1.0,
                        confidence=1.0,
                        validator_config={
                            "columns": [column],
                            "min_length": min_len,
                            "max_length": min_len,
                        },
                        description=f"{column} should have length {min_len}",
                    ))
                else:
                    rules.append(LearnedRule(
                        name=f"length_{column}",
                        rule_type="length",
                        column=column,
                        condition=f"{min_len} <= len({column}) <= {max_len}",
                        support=1.0,
                        confidence=0.95,
                        validator_config={
                            "columns": [column],
                            "min_length": min_len,
                            "max_length": max_len,
                        },
                        description=f"{column} length should be between {min_len} and {max_len}",
                    ))

        return rules, filtered

    def get_column_profiles(self) -> dict[str, dict]:
        """Get computed column profiles."""
        return dict(self._profiles)
