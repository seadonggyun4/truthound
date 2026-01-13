"""Constraint mining for rule learning.

Discovers functional dependencies and value constraints
from data using association rule mining techniques.
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
class ConstraintMinerConfig(RuleLearningConfig):
    """Configuration for constraint mining.

    Attributes:
        discover_functional_deps: Discover functional dependencies
        discover_value_constraints: Discover value-based constraints
        discover_conditional_rules: Discover conditional rules
        max_antecedent_size: Maximum number of columns in antecedent
        sample_for_discovery: Sample size for constraint discovery
    """

    discover_functional_deps: bool = True
    discover_value_constraints: bool = True
    discover_conditional_rules: bool = True
    max_antecedent_size: int = 2
    sample_for_discovery: int = 10000


@register_model("constraint_miner")
class ConstraintMiner(RuleLearner):
    """Mine data constraints and dependencies.

    Discovers:
    - Functional dependencies (X -> Y)
    - Value constraints (if X=a then Y=b)
    - Conditional rules (if condition then constraint)

    Uses efficient algorithms for constraint discovery
    without requiring external ML libraries.

    Example:
        >>> miner = ConstraintMiner()
        >>> result = miner.learn_rules(data)
        >>> for rule in result.get_rules_by_type("functional_dependency"):
        ...     print(f"{rule.condition}")
    """

    def __init__(self, config: ConstraintMinerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._functional_deps: list[tuple] = []
        self._value_constraints: list[dict] = []

    @property
    def config(self) -> ConstraintMinerConfig:
        return self._config  # type: ignore

    def _default_config(self) -> ConstraintMinerConfig:
        return ConstraintMinerConfig()

    def _get_model_name(self) -> str:
        return "constraint_miner"

    def _get_description(self) -> str:
        return "Constraint mining for functional dependencies and value rules"

    def learn_rules(self, data: pl.LazyFrame) -> RuleLearningResult:
        """Discover constraints from data.

        Args:
            data: Data to analyze

        Returns:
            RuleLearningResult with discovered constraints
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            df = self._maybe_sample(data).collect()

            # Sample if needed
            if len(df) > self.config.sample_for_discovery:
                df = df.sample(self.config.sample_for_discovery, seed=self.config.random_seed)

            rules: list[LearnedRule] = []

            # Discover functional dependencies
            if self.config.discover_functional_deps:
                fd_rules = self._discover_functional_dependencies(df)
                rules.extend(fd_rules)

            # Discover value constraints
            if self.config.discover_value_constraints:
                value_rules = self._discover_value_constraints(df)
                rules.extend(value_rules)

            # Discover conditional rules
            if self.config.discover_conditional_rules:
                cond_rules = self._discover_conditional_rules(df)
                rules.extend(cond_rules)

            # Filter by min_confidence and min_support
            filtered_rules = [
                r for r in rules
                if r.confidence >= self.config.min_confidence
                and r.support >= self.config.min_support
            ]

            # Limit rules
            if len(filtered_rules) > self.config.max_rules:
                filtered_rules = sorted(
                    filtered_rules,
                    key=lambda r: (r.confidence, r.support),
                    reverse=True,
                )[:self.config.max_rules]

            elapsed = (time.perf_counter() - start) * 1000

            result = RuleLearningResult(
                rules=tuple(filtered_rules),
                total_rules=len(rules),
                filtered_rules=len(rules) - len(filtered_rules),
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
                f"Failed to mine constraints: {e}",
                model_name=self.info.name,
            ) from e

    def _discover_functional_dependencies(
        self, df: pl.DataFrame
    ) -> list[LearnedRule]:
        """Discover functional dependencies (X -> Y).

        A functional dependency X -> Y holds if for each unique
        value of X, there is exactly one value of Y.
        """
        rules = []
        columns = df.columns
        n_rows = len(df)

        # Check single column dependencies
        for i, col_x in enumerate(columns):
            for j, col_y in enumerate(columns):
                if i == j:
                    continue

                # Check if X -> Y holds
                x_unique = df[col_x].n_unique()

                # Group by X and count unique Y values
                grouped = (
                    df.group_by(col_x)
                    .agg(pl.col(col_y).n_unique().alias("y_unique"))
                )

                # FD holds if all groups have exactly 1 unique Y
                max_y_per_x = grouped["y_unique"].max()
                groups_with_single_y = (grouped["y_unique"] == 1).sum()
                total_groups = len(grouped)

                if max_y_per_x == 1:
                    # Perfect FD
                    confidence = 1.0
                    support = 1.0
                elif total_groups > 0:
                    confidence = groups_with_single_y / total_groups
                    support = confidence
                else:
                    continue

                if confidence >= self.config.min_confidence:
                    rules.append(LearnedRule(
                        name=f"fd_{col_x}_to_{col_y}",
                        rule_type="functional_dependency",
                        column=(col_x, col_y),
                        condition=f"{col_x} -> {col_y}",
                        support=support,
                        confidence=confidence,
                        validator_config={
                            "determinant": col_x,
                            "dependent": col_y,
                        },
                        description=f"{col_x} functionally determines {col_y}",
                    ))

        # Check composite dependencies (X1, X2 -> Y) if configured
        if self.config.max_antecedent_size >= 2 and len(columns) >= 3:
            for i, col_x1 in enumerate(columns):
                for j, col_x2 in enumerate(columns[i+1:], i+1):
                    for k, col_y in enumerate(columns):
                        if k == i or k == j:
                            continue

                        # Check if (X1, X2) -> Y
                        grouped = (
                            df.group_by([col_x1, col_x2])
                            .agg(pl.col(col_y).n_unique().alias("y_unique"))
                        )

                        max_y = grouped["y_unique"].max()
                        groups_single = (grouped["y_unique"] == 1).sum()
                        total_groups = len(grouped)

                        if max_y == 1:
                            confidence = 1.0
                        elif total_groups > 0:
                            confidence = groups_single / total_groups
                        else:
                            continue

                        if confidence >= self.config.min_confidence:
                            rules.append(LearnedRule(
                                name=f"fd_{col_x1}_{col_x2}_to_{col_y}",
                                rule_type="functional_dependency",
                                column=(col_x1, col_x2, col_y),
                                condition=f"({col_x1}, {col_x2}) -> {col_y}",
                                support=confidence,
                                confidence=confidence,
                                validator_config={
                                    "determinant": [col_x1, col_x2],
                                    "dependent": col_y,
                                },
                                description=f"({col_x1}, {col_x2}) functionally determines {col_y}",
                            ))

        return rules

    def _discover_value_constraints(
        self, df: pl.DataFrame
    ) -> list[LearnedRule]:
        """Discover value-based constraints.

        Examples:
        - If status='active' then deleted_at is null
        - If type='A' then category in ['X', 'Y']
        """
        rules = []
        columns = df.columns
        n_rows = len(df)

        # Find categorical columns (low cardinality)
        categorical_cols = []
        for col in columns:
            n_unique = df[col].n_unique()
            if 2 <= n_unique <= 20:  # Low cardinality
                categorical_cols.append(col)

        # Check value implications
        for col_x in categorical_cols:
            for col_y in columns:
                if col_x == col_y:
                    continue

                # Get value distribution
                value_counts = (
                    df.group_by([col_x, col_y])
                    .agg(pl.len().alias("count"))
                )

                x_totals = (
                    df.group_by(col_x)
                    .agg(pl.len().alias("total"))
                )

                # Find strong implications
                for row in value_counts.iter_rows():
                    x_val, y_val, count = row

                    # Get total for this x value
                    x_total = x_totals.filter(pl.col(col_x) == x_val)["total"][0]

                    confidence = count / x_total if x_total > 0 else 0
                    support = count / n_rows

                    if confidence >= 0.95 and support >= 0.01:
                        rules.append(LearnedRule(
                            name=f"value_impl_{col_x}_{x_val}_to_{col_y}",
                            rule_type="value_implication",
                            column=(col_x, col_y),
                            condition=f"if {col_x}='{x_val}' then {col_y}='{y_val}'",
                            support=support,
                            confidence=confidence,
                            validator_config={
                                "condition_column": col_x,
                                "condition_value": x_val,
                                "expected_column": col_y,
                                "expected_value": y_val,
                            },
                            description=f"When {col_x} is '{x_val}', {col_y} should be '{y_val}'",
                        ))

        return rules

    def _discover_conditional_rules(
        self, df: pl.DataFrame
    ) -> list[LearnedRule]:
        """Discover conditional constraint rules.

        Examples:
        - If end_date is not null then start_date is not null
        - If quantity > 0 then price > 0
        """
        rules = []
        columns = df.columns
        schema = df.schema
        n_rows = len(df)

        # Check null implications
        for col_x in columns:
            for col_y in columns:
                if col_x == col_y:
                    continue

                # If X is not null, Y should also be not null
                x_not_null = df.filter(pl.col(col_x).is_not_null())
                if len(x_not_null) == 0:
                    continue

                y_not_null_given_x = x_not_null.filter(
                    pl.col(col_y).is_not_null()
                )

                confidence = len(y_not_null_given_x) / len(x_not_null)
                support = len(y_not_null_given_x) / n_rows

                if confidence >= 0.99 and len(x_not_null) > n_rows * 0.1:
                    rules.append(LearnedRule(
                        name=f"null_impl_{col_x}_to_{col_y}",
                        rule_type="null_implication",
                        column=(col_x, col_y),
                        condition=f"if {col_x} is not null then {col_y} is not null",
                        support=support,
                        confidence=confidence,
                        validator_config={
                            "condition": f"{col_x} is not null",
                            "constraint": f"{col_y} is not null",
                        },
                        description=f"When {col_x} has a value, {col_y} should also have a value",
                    ))

        # Check numeric comparisons
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        numeric_cols = [
            c for c in columns
            if type(schema[c]) in numeric_types
        ]

        for col_x in numeric_cols:
            for col_y in numeric_cols:
                if col_x == col_y:
                    continue

                # Check if X > 0 implies Y > 0
                x_positive = df.filter(pl.col(col_x) > 0)
                if len(x_positive) < n_rows * 0.1:
                    continue

                y_positive_given_x = x_positive.filter(pl.col(col_y) > 0)

                confidence = len(y_positive_given_x) / len(x_positive)
                support = len(y_positive_given_x) / n_rows

                if confidence >= 0.95:
                    rules.append(LearnedRule(
                        name=f"positive_impl_{col_x}_to_{col_y}",
                        rule_type="comparison_implication",
                        column=(col_x, col_y),
                        condition=f"if {col_x} > 0 then {col_y} > 0",
                        support=support,
                        confidence=confidence,
                        validator_config={
                            "condition": f"{col_x} > 0",
                            "constraint": f"{col_y} > 0",
                        },
                        description=f"When {col_x} is positive, {col_y} should also be positive",
                    ))

                # Check if X <= Y (ordering)
                valid_rows = df.filter(
                    pl.col(col_x).is_not_null() & pl.col(col_y).is_not_null()
                )
                if len(valid_rows) < n_rows * 0.5:
                    continue

                ordered = valid_rows.filter(pl.col(col_x) <= pl.col(col_y))
                confidence = len(ordered) / len(valid_rows)

                if confidence >= 0.99:
                    rules.append(LearnedRule(
                        name=f"order_{col_x}_leq_{col_y}",
                        rule_type="ordering",
                        column=(col_x, col_y),
                        condition=f"{col_x} <= {col_y}",
                        support=len(ordered) / n_rows,
                        confidence=confidence,
                        validator_config={
                            "column1": col_x,
                            "column2": col_y,
                            "operator": "<=",
                        },
                        description=f"{col_x} should be less than or equal to {col_y}",
                    ))

        return rules
