"""Pattern-based rule learning.

Learns validation rules based on detected patterns in string data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections import Counter

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
class PatternLearnerConfig(RuleLearningConfig):
    """Configuration for pattern-based rule learning.

    Attributes:
        min_pattern_ratio: Minimum ratio of values matching pattern
        learn_custom_patterns: Learn custom regex patterns from data
        pattern_sample_size: Sample size for pattern learning
        max_pattern_length: Maximum length for learned patterns
        generalization_level: How much to generalize patterns (1-3)
    """

    min_pattern_ratio: float = 0.9
    learn_custom_patterns: bool = True
    pattern_sample_size: int = 1000
    max_pattern_length: int = 100
    generalization_level: int = 2  # 1=specific, 2=medium, 3=general


# Common pattern templates
COMMON_PATTERNS = [
    ("email", r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
    ("url", r"^https?://[^\s]+$"),
    ("phone_us", r"^\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$"),
    ("phone_intl", r"^\+[0-9]{1,3}[-.\s]?[0-9]{1,14}$"),
    ("phone_kr", r"^0[0-9]{1,2}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{4}$"),
    ("uuid", r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
    ("uuid_no_dash", r"^[0-9a-f]{32}$"),
    ("date_iso", r"^\d{4}-\d{2}-\d{2}$"),
    ("date_us", r"^\d{2}/\d{2}/\d{4}$"),
    ("datetime_iso", r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"),
    ("time_24h", r"^\d{2}:\d{2}(:\d{2})?$"),
    ("ipv4", r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"),
    ("ipv6", r"^([0-9a-f]{1,4}:){7}[0-9a-f]{1,4}$"),
    ("mac_address", r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$"),
    ("credit_card", r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$"),
    ("ssn_us", r"^\d{3}-\d{2}-\d{4}$"),
    ("rrn_kr", r"^\d{6}-[1-4]\d{6}$"),  # Korean RRN
    ("zip_us", r"^\d{5}(-\d{4})?$"),
    ("zip_kr", r"^\d{5}$"),
    ("currency_usd", r"^\$[\d,]+(\.\d{2})?$"),
    ("currency_krw", r"^[\d,]+원?$"),
    ("percentage", r"^-?\d+(\.\d+)?%$"),
    ("hex_color", r"^#[0-9a-f]{6}$"),
    ("slug", r"^[a-z0-9]+(-[a-z0-9]+)*$"),
    ("alphanumeric", r"^[a-zA-Z0-9]+$"),
    ("alpha_only", r"^[a-zA-Z]+$"),
    ("numeric_only", r"^\d+$"),
    ("uppercase", r"^[A-Z]+$"),
    ("lowercase", r"^[a-z]+$"),
]


@register_model("pattern_learner")
class PatternRuleLearner(RuleLearner):
    """Pattern-based rule learning.

    Analyzes string columns to detect consistent patterns and
    generates regex-based validation rules.

    Features:
    - Detects common patterns (email, phone, UUID, etc.)
    - Learns custom patterns from data
    - Generalizes patterns to appropriate level
    - Handles mixed-format columns

    Example:
        >>> learner = PatternRuleLearner()
        >>> result = learner.learn_rules(data)
        >>> for rule in result.get_rules_by_type("regex"):
        ...     print(f"{rule.column}: {rule.condition}")
    """

    def __init__(self, config: PatternLearnerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._detected_patterns: dict[str, list[dict]] = {}

    @property
    def config(self) -> PatternLearnerConfig:
        return self._config  # type: ignore

    def _default_config(self) -> PatternLearnerConfig:
        return PatternLearnerConfig()

    def _get_model_name(self) -> str:
        return "pattern_learner"

    def _get_description(self) -> str:
        return "Pattern-based rule learning for string validation"

    def learn_rules(self, data: pl.LazyFrame) -> RuleLearningResult:
        """Learn pattern-based validation rules.

        Args:
            data: Data to analyze

        Returns:
            RuleLearningResult with pattern rules
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            df = self._maybe_sample(data).collect()

            rules: list[LearnedRule] = []

            # Find string columns
            schema = df.schema
            string_cols = [
                c for c in df.columns
                if type(schema[c]) in {pl.String, pl.Utf8}
            ]

            for col in string_cols:
                col_rules = self._learn_column_patterns(df, col)
                rules.extend(col_rules)

            # Filter by min_confidence
            filtered_rules = [
                r for r in rules
                if r.confidence >= self.config.min_confidence
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
                data_profile={"columns": len(string_cols), "rows": row_count},
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
                f"Failed to learn patterns: {e}",
                model_name=self.info.name,
            ) from e

    def _learn_column_patterns(
        self, df: pl.DataFrame, column: str
    ) -> list[LearnedRule]:
        """Learn patterns for a single column."""
        rules = []

        # Get sample values
        sample = (
            df[column]
            .drop_nulls()
            .head(self.config.pattern_sample_size)
            .to_list()
        )

        if not sample:
            return rules

        total = len(sample)
        self._detected_patterns[column] = []

        # Test common patterns
        for pattern_name, regex in COMMON_PATTERNS:
            try:
                compiled = re.compile(regex, re.IGNORECASE)
                matches = sum(1 for s in sample if compiled.match(str(s)))
                ratio = matches / total

                if ratio >= self.config.min_pattern_ratio:
                    self._detected_patterns[column].append({
                        "name": pattern_name,
                        "regex": regex,
                        "ratio": ratio,
                        "type": "common",
                    })

                    rules.append(LearnedRule(
                        name=f"pattern_{pattern_name}_{column}",
                        rule_type="regex",
                        column=column,
                        condition=f"{column} matches {pattern_name} pattern",
                        support=ratio,
                        confidence=ratio,
                        validator_config={
                            "columns": [column],
                            "pattern": regex,
                            "pattern_name": pattern_name,
                        },
                        description=f"{column} should match {pattern_name} format",
                    ))
            except re.error:
                continue

        # Learn custom patterns if no common pattern found
        if self.config.learn_custom_patterns and not rules:
            custom_rules = self._learn_custom_pattern(sample, column)
            rules.extend(custom_rules)

        # Length-based rules
        lengths = [len(str(s)) for s in sample]
        min_len = min(lengths)
        max_len = max(lengths)
        length_counter = Counter(lengths)
        most_common_len, most_common_count = length_counter.most_common(1)[0]

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
                description=f"{column} should have exactly {min_len} characters",
            ))
        elif most_common_count / total >= 0.9:
            # Dominant length
            rules.append(LearnedRule(
                name=f"typical_length_{column}",
                rule_type="length",
                column=column,
                condition=f"len({column}) == {most_common_len}",
                support=most_common_count / total,
                confidence=most_common_count / total,
                validator_config={
                    "columns": [column],
                    "expected_length": most_common_len,
                },
                description=f"{column} typically has {most_common_len} characters",
            ))

        return rules

    def _learn_custom_pattern(
        self, sample: list[str], column: str
    ) -> list[LearnedRule]:
        """Learn custom regex patterns from data."""
        rules = []

        # Analyze character classes
        patterns = []
        for s in sample[:100]:  # Analyze subset
            pattern = self._generalize_string(str(s))
            patterns.append(pattern)

        # Find most common pattern
        pattern_counter = Counter(patterns)
        most_common_patterns = pattern_counter.most_common(3)

        for pattern, count in most_common_patterns:
            ratio = count / len(patterns)
            if ratio >= 0.5 and len(pattern) <= self.config.max_pattern_length:
                # Convert to regex and validate
                regex = self._pattern_to_regex(pattern)
                try:
                    compiled = re.compile(regex)
                    # Verify against full sample
                    matches = sum(1 for s in sample if compiled.match(str(s)))
                    full_ratio = matches / len(sample)

                    if full_ratio >= self.config.min_pattern_ratio:
                        self._detected_patterns[column].append({
                            "name": "custom",
                            "regex": regex,
                            "ratio": full_ratio,
                            "type": "learned",
                            "pattern": pattern,
                        })

                        rules.append(LearnedRule(
                            name=f"custom_pattern_{column}",
                            rule_type="regex",
                            column=column,
                            condition=f"{column} matches learned pattern",
                            support=full_ratio,
                            confidence=full_ratio,
                            validator_config={
                                "columns": [column],
                                "pattern": regex,
                                "pattern_name": "custom",
                            },
                            description=f"{column} should match pattern: {pattern}",
                        ))
                        break
                except re.error:
                    continue

        return rules

    def _generalize_string(self, s: str) -> str:
        """Convert string to generalized pattern.

        Level 1: Keep exact characters
        Level 2: Group consecutive same-class characters
        Level 3: Maximum generalization
        """
        level = self.config.generalization_level

        if level == 1:
            # Minimal generalization: just character classes
            result = []
            for c in s:
                if c.isdigit():
                    result.append("D")
                elif c.isupper():
                    result.append("U")
                elif c.islower():
                    result.append("L")
                elif c.isspace():
                    result.append("S")
                else:
                    result.append(c)  # Keep special chars
            return "".join(result)

        elif level == 2:
            # Group consecutive same-class characters
            result = []
            prev_class = None
            count = 0

            for c in s:
                if c.isdigit():
                    curr_class = "D"
                elif c.isalpha():
                    curr_class = "A"
                elif c.isspace():
                    curr_class = "S"
                else:
                    curr_class = c  # Keep special chars

                if curr_class == prev_class and curr_class in "DAS":
                    count += 1
                else:
                    if prev_class and prev_class in "DAS":
                        result.append(f"{prev_class}{count}")
                    elif prev_class:
                        result.append(prev_class)
                    prev_class = curr_class
                    count = 1

            if prev_class and prev_class in "DAS":
                result.append(f"{prev_class}{count}")
            elif prev_class:
                result.append(prev_class)

            return "".join(result)

        else:  # level 3
            # Maximum generalization
            result = []
            prev_class = None

            for c in s:
                if c.isdigit():
                    curr_class = "D"
                elif c.isalpha():
                    curr_class = "A"
                else:
                    curr_class = c

                if curr_class != prev_class:
                    if prev_class in ("D", "A"):
                        result.append(f"{prev_class}+")
                    elif prev_class:
                        result.append(prev_class)
                    prev_class = curr_class

            if prev_class in ("D", "A"):
                result.append(f"{prev_class}+")
            elif prev_class:
                result.append(prev_class)

            return "".join(result)

    def _pattern_to_regex(self, pattern: str) -> str:
        """Convert generalized pattern to regex."""
        result = "^"
        i = 0

        while i < len(pattern):
            c = pattern[i]

            if c == "D":
                if i + 1 < len(pattern) and pattern[i + 1].isdigit():
                    # Specific count
                    count = ""
                    j = i + 1
                    while j < len(pattern) and pattern[j].isdigit():
                        count += pattern[j]
                        j += 1
                    result += f"\\d{{{count}}}"
                    i = j
                    continue
                elif i + 1 < len(pattern) and pattern[i + 1] == "+":
                    result += "\\d+"
                    i += 2
                    continue
                else:
                    result += "\\d"

            elif c == "A":
                if i + 1 < len(pattern) and pattern[i + 1].isdigit():
                    count = ""
                    j = i + 1
                    while j < len(pattern) and pattern[j].isdigit():
                        count += pattern[j]
                        j += 1
                    result += f"[a-zA-Z]{{{count}}}"
                    i = j
                    continue
                elif i + 1 < len(pattern) and pattern[i + 1] == "+":
                    result += "[a-zA-Z]+"
                    i += 2
                    continue
                else:
                    result += "[a-zA-Z]"

            elif c == "U":
                result += "[A-Z]"

            elif c == "L":
                result += "[a-z]"

            elif c == "S":
                if i + 1 < len(pattern) and pattern[i + 1].isdigit():
                    count = pattern[i + 1]
                    result += f"\\s{{{count}}}"
                    i += 2
                    continue
                else:
                    result += "\\s"

            elif c in ".*+?[](){}^$|\\":
                result += "\\" + c

            else:
                result += re.escape(c)

            i += 1

        result += "$"
        return result

    def get_detected_patterns(self) -> dict[str, list[dict]]:
        """Get all detected patterns per column."""
        return dict(self._detected_patterns)
