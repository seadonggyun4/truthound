"""Feature extraction for ML-based type inference.

Provides comprehensive feature extraction from column data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import polars as pl


# =============================================================================
# Feature Set Definition
# =============================================================================


@dataclass
class FeatureSet:
    """Container for extracted features."""

    values: List[float]
    names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.values)

    def to_dict(self) -> Dict[str, float]:
        return dict(zip(self.names, self.values))

    def get(self, name: str, default: float = 0.0) -> float:
        try:
            idx = self.names.index(name)
            return self.values[idx]
        except ValueError:
            return default


# =============================================================================
# Feature Extractors
# =============================================================================


class FeatureExtractor:
    """Extract features for ML model from column data.

    Extracts comprehensive features including:
    - Statistical features (mean, std, min, max, etc.)
    - String pattern features (length, character types)
    - Distribution features (entropy, cardinality)
    - Domain-specific features (email, phone, date patterns)

    Example:
        extractor = FeatureExtractor()
        features = extractor.extract(column, context)
    """

    # Feature name prefixes
    STAT_PREFIX = "stat_"
    STR_PREFIX = "str_"
    PATTERN_PREFIX = "pat_"
    DIST_PREFIX = "dist_"
    NAME_PREFIX = "name_"

    # Type-specific keywords for name matching
    TYPE_KEYWORDS = {
        "email": ["email", "mail", "e_mail", "correo", "email_address"],
        "phone": ["phone", "tel", "mobile", "cell", "fax", "telephone", "hp"],
        "url": ["url", "link", "href", "website", "uri", "endpoint"],
        "uuid": ["uuid", "guid", "id", "identifier", "uid"],
        "date": ["date", "day", "birth", "created", "updated", "modified"],
        "datetime": ["datetime", "timestamp", "time", "at", "when"],
        "integer": ["count", "num", "qty", "quantity", "amount", "total"],
        "float": ["price", "rate", "ratio", "percent", "score", "value"],
        "boolean": ["is_", "has_", "flag", "active", "enabled", "valid"],
        "currency": ["price", "cost", "amount", "fee", "payment", "salary"],
        "percentage": ["percent", "pct", "ratio", "rate"],
        "categorical": ["type", "status", "category", "class", "kind", "level"],
        "identifier": ["id", "key", "code", "no", "number"],
    }

    # Regex patterns for detection
    PATTERNS = {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        "url": r"^https?://[^\s/$.?#].[^\s]*$",
        "ip_v4": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
        "date_iso": r"^\d{4}-\d{2}-\d{2}$",
        "datetime_iso": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",
        "phone_intl": r"^\+?[1-9]\d{6,14}$",
        "korean_phone": r"^01[0-9]-?\d{3,4}-?\d{4}$",
        "korean_rrn": r"^\d{6}-?[1-4]\d{6}$",
        "credit_card": r"^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$",
        "currency": r"^[$\u20ac\u00a3\u00a5]?\s*\d+([,.]?\d{3})*([,.]\d{2})?$",
        "percentage": r"^\d+\.?\d*%$",
        "hex_color": r"^#[0-9a-fA-F]{6}$",
    }

    def __init__(self, sample_size: int = 1000):
        """Initialize extractor.

        Args:
            sample_size: Maximum sample size for pattern analysis
        """
        self.sample_size = sample_size
        self._compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.PATTERNS.items()
        }

    def extract(
        self,
        column: pl.Series,
        context: Dict[str, Any] | None = None,
    ) -> FeatureSet:
        """Extract features from a column.

        Args:
            column: Column data
            context: Additional context (column name, table name, etc.)

        Returns:
            FeatureSet with all extracted features
        """
        context = context or {}
        values: List[float] = []
        names: List[str] = []

        # Basic stats
        basic = self._extract_basic_stats(column)
        values.extend(basic[0])
        names.extend(basic[1])

        # Type-specific features
        if column.dtype == pl.Utf8:
            string_features = self._extract_string_features(column)
            values.extend(string_features[0])
            names.extend(string_features[1])

            pattern_features = self._extract_pattern_features(column)
            values.extend(pattern_features[0])
            names.extend(pattern_features[1])

        elif column.dtype.is_numeric():
            numeric_features = self._extract_numeric_features(column)
            values.extend(numeric_features[0])
            names.extend(numeric_features[1])

        # Distribution features
        dist_features = self._extract_distribution_features(column)
        values.extend(dist_features[0])
        names.extend(dist_features[1])

        # Name-based features
        col_name = context.get("column_name", column.name or "")
        name_features = self._extract_name_features(col_name)
        values.extend(name_features[0])
        names.extend(name_features[1])

        # Context features
        ctx_features = self._extract_context_features(column, context)
        values.extend(ctx_features[0])
        names.extend(ctx_features[1])

        return FeatureSet(
            values=values,
            names=names,
            metadata={"column_name": col_name, "dtype": str(column.dtype)},
        )

    def _extract_basic_stats(self, column: pl.Series) -> tuple[List[float], List[str]]:
        """Extract basic statistics."""
        values = []
        names = []

        row_count = len(column)
        null_count = column.null_count()

        # Null ratio
        values.append(null_count / row_count if row_count > 0 else 0)
        names.append(f"{self.STAT_PREFIX}null_ratio")

        # Distinct ratio
        distinct = column.n_unique()
        values.append(distinct / row_count if row_count > 0 else 0)
        names.append(f"{self.STAT_PREFIX}distinct_ratio")

        # Is all null
        values.append(1.0 if null_count == row_count else 0.0)
        names.append(f"{self.STAT_PREFIX}is_all_null")

        # Is unique
        values.append(1.0 if distinct == row_count else 0.0)
        names.append(f"{self.STAT_PREFIX}is_unique")

        # Is constant
        values.append(1.0 if distinct <= 1 else 0.0)
        names.append(f"{self.STAT_PREFIX}is_constant")

        # Physical type indicators
        dtype = str(column.dtype).lower()
        values.append(1.0 if "int" in dtype else 0.0)
        names.append(f"{self.STAT_PREFIX}is_integer_type")

        values.append(1.0 if "float" in dtype else 0.0)
        names.append(f"{self.STAT_PREFIX}is_float_type")

        values.append(1.0 if dtype == "utf8" or dtype == "str" else 0.0)
        names.append(f"{self.STAT_PREFIX}is_string_type")

        values.append(1.0 if "bool" in dtype else 0.0)
        names.append(f"{self.STAT_PREFIX}is_boolean_type")

        values.append(1.0 if "date" in dtype else 0.0)
        names.append(f"{self.STAT_PREFIX}is_date_type")

        return values, names

    def _extract_string_features(self, column: pl.Series) -> tuple[List[float], List[str]]:
        """Extract string-specific features."""
        values = []
        names = []

        non_null = column.drop_nulls()
        if len(non_null) == 0:
            # Return zeros for all string features
            return [0.0] * 15, [
                f"{self.STR_PREFIX}avg_length",
                f"{self.STR_PREFIX}min_length",
                f"{self.STR_PREFIX}max_length",
                f"{self.STR_PREFIX}length_std",
                f"{self.STR_PREFIX}has_at_sign",
                f"{self.STR_PREFIX}has_dot",
                f"{self.STR_PREFIX}has_slash",
                f"{self.STR_PREFIX}has_dash",
                f"{self.STR_PREFIX}has_colon",
                f"{self.STR_PREFIX}has_space",
                f"{self.STR_PREFIX}digit_ratio",
                f"{self.STR_PREFIX}alpha_ratio",
                f"{self.STR_PREFIX}upper_ratio",
                f"{self.STR_PREFIX}special_ratio",
                f"{self.STR_PREFIX}uniform_length",
            ]

        sample = non_null.head(self.sample_size)
        sample_list = sample.to_list()

        # Length statistics
        lengths = sample.str.len_chars()
        avg_len = float(lengths.mean() or 0)
        min_len = int(lengths.min() or 0)
        max_len = int(lengths.max() or 0)
        std_len = float(lengths.std() or 0)

        values.extend([
            min(avg_len / 100, 1.0),
            min(min_len / 50, 1.0),
            min(max_len / 200, 1.0),
            min(std_len / 50, 1.0) if avg_len > 0 else 0,
        ])
        names.extend([
            f"{self.STR_PREFIX}avg_length",
            f"{self.STR_PREFIX}min_length",
            f"{self.STR_PREFIX}max_length",
            f"{self.STR_PREFIX}length_std",
        ])

        # Character analysis
        def char_stats(strings: List[str]) -> Dict[str, float]:
            n = len(strings)
            if n == 0:
                return {}

            has_at = sum(1 for s in strings if "@" in str(s)) / n
            has_dot = sum(1 for s in strings if "." in str(s)) / n
            has_slash = sum(1 for s in strings if "/" in str(s)) / n
            has_dash = sum(1 for s in strings if "-" in str(s)) / n
            has_colon = sum(1 for s in strings if ":" in str(s)) / n
            has_space = sum(1 for s in strings if " " in str(s)) / n

            # Character type ratios
            digit_ratios = []
            alpha_ratios = []
            upper_ratios = []
            special_ratios = []

            for s in strings[:100]:
                s = str(s)
                if len(s) > 0:
                    digit_ratios.append(sum(c.isdigit() for c in s) / len(s))
                    alpha_ratios.append(sum(c.isalpha() for c in s) / len(s))
                    upper_ratios.append(sum(c.isupper() for c in s) / len(s))
                    special_ratios.append(sum(not c.isalnum() for c in s) / len(s))

            return {
                "has_at": has_at,
                "has_dot": has_dot,
                "has_slash": has_slash,
                "has_dash": has_dash,
                "has_colon": has_colon,
                "has_space": has_space,
                "digit_ratio": sum(digit_ratios) / len(digit_ratios) if digit_ratios else 0,
                "alpha_ratio": sum(alpha_ratios) / len(alpha_ratios) if alpha_ratios else 0,
                "upper_ratio": sum(upper_ratios) / len(upper_ratios) if upper_ratios else 0,
                "special_ratio": sum(special_ratios) / len(special_ratios) if special_ratios else 0,
            }

        stats = char_stats(sample_list)
        values.extend([
            stats.get("has_at", 0),
            stats.get("has_dot", 0),
            stats.get("has_slash", 0),
            stats.get("has_dash", 0),
            stats.get("has_colon", 0),
            stats.get("has_space", 0),
            stats.get("digit_ratio", 0),
            stats.get("alpha_ratio", 0),
            stats.get("upper_ratio", 0),
            stats.get("special_ratio", 0),
        ])
        names.extend([
            f"{self.STR_PREFIX}has_at_sign",
            f"{self.STR_PREFIX}has_dot",
            f"{self.STR_PREFIX}has_slash",
            f"{self.STR_PREFIX}has_dash",
            f"{self.STR_PREFIX}has_colon",
            f"{self.STR_PREFIX}has_space",
            f"{self.STR_PREFIX}digit_ratio",
            f"{self.STR_PREFIX}alpha_ratio",
            f"{self.STR_PREFIX}upper_ratio",
            f"{self.STR_PREFIX}special_ratio",
        ])

        # Uniform length indicator
        uniform = 1.0 if min_len == max_len and min_len > 0 else 0.0
        values.append(uniform)
        names.append(f"{self.STR_PREFIX}uniform_length")

        return values, names

    def _extract_pattern_features(self, column: pl.Series) -> tuple[List[float], List[str]]:
        """Extract pattern matching features."""
        values = []
        names = []

        non_null = column.drop_nulls()
        if len(non_null) == 0:
            return [0.0] * len(self._compiled_patterns), [
                f"{self.PATTERN_PREFIX}{name}" for name in self._compiled_patterns
            ]

        sample = non_null.head(self.sample_size)
        sample_list = [str(s) for s in sample.to_list()]
        n = len(sample_list)

        for pattern_name, regex in self._compiled_patterns.items():
            matches = sum(1 for s in sample_list if regex.match(s))
            ratio = matches / n if n > 0 else 0
            values.append(ratio)
            names.append(f"{self.PATTERN_PREFIX}{pattern_name}")

        return values, names

    def _extract_numeric_features(self, column: pl.Series) -> tuple[List[float], List[str]]:
        """Extract numeric-specific features."""
        values = []
        names = []

        non_null = column.drop_nulls()
        if len(non_null) == 0:
            return [0.0] * 12, [
                f"{self.STAT_PREFIX}mean_normalized",
                f"{self.STAT_PREFIX}std_normalized",
                f"{self.STAT_PREFIX}min_normalized",
                f"{self.STAT_PREFIX}max_normalized",
                f"{self.STAT_PREFIX}range_normalized",
                f"{self.STAT_PREFIX}is_sequential",
                f"{self.STAT_PREFIX}is_0_1_range",
                f"{self.STAT_PREFIX}is_0_100_range",
                f"{self.STAT_PREFIX}is_positive",
                f"{self.STAT_PREFIX}is_integer_values",
                f"{self.STAT_PREFIX}has_2_decimals",
                f"{self.STAT_PREFIX}skewness",
            ]

        # Basic stats
        min_val = float(non_null.min() or 0)
        max_val = float(non_null.max() or 0)
        mean_val = float(non_null.mean() or 0)
        std_val = float(non_null.std() or 0)

        # Normalize to 0-1 range
        range_val = max_val - min_val
        range_log = math.log10(abs(range_val) + 1) / 10 if range_val != 0 else 0

        values.extend([
            math.log10(abs(mean_val) + 1) / 10,
            math.log10(abs(std_val) + 1) / 10,
            math.log10(abs(min_val) + 1) / 10,
            math.log10(abs(max_val) + 1) / 10,
            range_log,
        ])
        names.extend([
            f"{self.STAT_PREFIX}mean_normalized",
            f"{self.STAT_PREFIX}std_normalized",
            f"{self.STAT_PREFIX}min_normalized",
            f"{self.STAT_PREFIX}max_normalized",
            f"{self.STAT_PREFIX}range_normalized",
        ])

        # Is sequential (like IDs)
        if column.dtype in [pl.Int32, pl.Int64]:
            sorted_vals = non_null.sort()
            diffs = sorted_vals.diff().drop_nulls()
            is_sequential = float((diffs == 1).mean()) if len(diffs) > 0 else 0
        else:
            is_sequential = 0.0
        values.append(is_sequential)
        names.append(f"{self.STAT_PREFIX}is_sequential")

        # Range checks
        in_0_1 = float(((non_null >= 0) & (non_null <= 1)).mean())
        in_0_100 = float(((non_null >= 0) & (non_null <= 100)).mean())
        is_positive = float((non_null >= 0).mean())

        values.extend([in_0_1, in_0_100, is_positive])
        names.extend([
            f"{self.STAT_PREFIX}is_0_1_range",
            f"{self.STAT_PREFIX}is_0_100_range",
            f"{self.STAT_PREFIX}is_positive",
        ])

        # Integer check for floats
        if column.dtype in [pl.Float32, pl.Float64]:
            is_int_vals = float((non_null == non_null.floor()).mean())
            # Check for 2 decimal places (currency)
            sample = non_null.head(100).to_list()
            two_decimals = sum(
                1 for v in sample
                if v is not None and round(v, 2) == v
            ) / len(sample) if sample else 0
        else:
            is_int_vals = 1.0
            two_decimals = 1.0

        values.extend([is_int_vals, two_decimals])
        names.extend([
            f"{self.STAT_PREFIX}is_integer_values",
            f"{self.STAT_PREFIX}has_2_decimals",
        ])

        # Skewness approximation
        if std_val > 0:
            skewness = min(1, max(-1, (mean_val - float(non_null.median() or 0)) / std_val))
        else:
            skewness = 0
        values.append(skewness)
        names.append(f"{self.STAT_PREFIX}skewness")

        return values, names

    def _extract_distribution_features(self, column: pl.Series) -> tuple[List[float], List[str]]:
        """Extract distribution features."""
        values = []
        names = []

        non_null = column.drop_nulls()
        n = len(non_null)

        if n == 0:
            return [0.0] * 4, [
                f"{self.DIST_PREFIX}entropy",
                f"{self.DIST_PREFIX}max_frequency",
                f"{self.DIST_PREFIX}top10_coverage",
                f"{self.DIST_PREFIX}is_categorical",
            ]

        # Value counts
        value_counts = non_null.value_counts()
        counts = value_counts.get_column("count").to_list()

        # Entropy
        probs = [c / n for c in counts]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        n_unique = len(counts)
        max_entropy = math.log2(n_unique) if n_unique > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        values.append(normalized_entropy)
        names.append(f"{self.DIST_PREFIX}entropy")

        # Max frequency
        max_freq = max(counts) / n if counts else 0
        values.append(max_freq)
        names.append(f"{self.DIST_PREFIX}max_frequency")

        # Top 10 coverage
        top10_sum = sum(sorted(counts, reverse=True)[:10])
        top10_coverage = top10_sum / n if n > 0 else 0
        values.append(top10_coverage)
        names.append(f"{self.DIST_PREFIX}top10_coverage")

        # Is categorical (low cardinality)
        cardinality = n_unique / n if n > 0 else 0
        is_categorical = 1.0 if n_unique < 50 and cardinality < 0.1 else 0.0
        values.append(is_categorical)
        names.append(f"{self.DIST_PREFIX}is_categorical")

        return values, names

    def _extract_name_features(self, column_name: str) -> tuple[List[float], List[str]]:
        """Extract features from column name."""
        values = []
        names = []

        clean_name = column_name.lower().strip()
        tokens = re.split(r"[_\s\-\.]+", clean_name)

        for type_name, keywords in self.TYPE_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in clean_name:
                    score += 1.0
                elif any(keyword in token for token in tokens):
                    score += 0.5
            values.append(min(1.0, score / (len(keywords) * 0.5)))
            names.append(f"{self.NAME_PREFIX}{type_name}")

        # Name length
        values.append(min(1.0, len(column_name) / 50))
        names.append(f"{self.NAME_PREFIX}length")

        # Has underscore
        values.append(1.0 if "_" in column_name else 0.0)
        names.append(f"{self.NAME_PREFIX}has_underscore")

        # Has number
        values.append(1.0 if any(c.isdigit() for c in column_name) else 0.0)
        names.append(f"{self.NAME_PREFIX}has_number")

        return values, names

    def _extract_context_features(
        self,
        column: pl.Series,
        context: Dict[str, Any],
    ) -> tuple[List[float], List[str]]:
        """Extract context features."""
        values = []
        names = []

        # Column position
        col_index = context.get("column_index", 0)
        total_cols = context.get("total_columns", 1)
        position = col_index / total_cols if total_cols > 0 else 0

        values.append(position)
        names.append("ctx_position")

        # Is first column
        values.append(1.0 if col_index == 0 else 0.0)
        names.append("ctx_is_first")

        # Is last column
        values.append(1.0 if col_index == total_cols - 1 else 0.0)
        names.append("ctx_is_last")

        # Table context hints
        table_name = context.get("table_name", "").lower()
        values.append(1.0 if any(kw in table_name for kw in ["user", "customer", "member"]) else 0.0)
        names.append("ctx_user_table")

        values.append(1.0 if any(kw in table_name for kw in ["order", "transaction", "payment"]) else 0.0)
        names.append("ctx_transaction_table")

        return values, names


# =============================================================================
# Convenience Function
# =============================================================================


def extract_features(
    column: pl.Series,
    context: Dict[str, Any] | None = None,
    sample_size: int = 1000,
) -> FeatureSet:
    """Extract features from a column.

    Args:
        column: Column data
        context: Additional context
        sample_size: Sample size for pattern analysis

    Returns:
        FeatureSet with extracted features
    """
    extractor = FeatureExtractor(sample_size=sample_size)
    return extractor.extract(column, context)
