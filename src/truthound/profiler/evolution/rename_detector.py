"""Column rename detection with advanced algorithms.

This module provides sophisticated column rename detection using
multiple similarity algorithms and configurable strategies.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class SimilarityCalculator(Protocol):
    """Protocol for string similarity calculation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get calculator name."""
        ...

    @abstractmethod
    def calculate(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Similarity score between 0.0 (no similarity) and 1.0 (identical).
        """
        ...


# =============================================================================
# Data Structures
# =============================================================================


class RenameConfidence(str, Enum):
    """Confidence levels for rename detection."""

    HIGH = "high"  # >= 0.9 similarity + type match
    MEDIUM = "medium"  # >= 0.8 similarity + type match
    LOW = "low"  # >= 0.7 similarity + type match
    UNCERTAIN = "uncertain"  # < 0.7 or type mismatch


@dataclass
class RenameCandidate:
    """A potential column rename candidate.

    Attributes:
        old_name: Original column name.
        new_name: New column name.
        similarity: Name similarity score (0.0 to 1.0).
        confidence: Confidence level of the rename.
        old_type: Type of the old column.
        new_type: Type of the new column.
        type_match: Whether types match or are compatible.
        reasons: List of reasons supporting this candidate.
        metadata: Additional metadata.
    """

    old_name: str
    new_name: str
    similarity: float
    confidence: RenameConfidence
    old_type: Any = None
    new_type: Any = None
    type_match: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "old_name": self.old_name,
            "new_name": self.new_name,
            "similarity": self.similarity,
            "confidence": self.confidence.value,
            "old_type": str(self.old_type) if self.old_type else None,
            "new_type": str(self.new_type) if self.new_type else None,
            "type_match": self.type_match,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


@dataclass
class RenameDetectionResult:
    """Result of column rename detection.

    Attributes:
        confirmed_renames: High-confidence renames.
        possible_renames: Medium/low-confidence renames needing review.
        unmatched_removed: Columns removed without a rename match.
        unmatched_added: Columns added without a rename match.
    """

    confirmed_renames: list[RenameCandidate] = field(default_factory=list)
    possible_renames: list[RenameCandidate] = field(default_factory=list)
    unmatched_removed: list[str] = field(default_factory=list)
    unmatched_added: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confirmed_renames": [r.to_dict() for r in self.confirmed_renames],
            "possible_renames": [r.to_dict() for r in self.possible_renames],
            "unmatched_removed": self.unmatched_removed,
            "unmatched_added": self.unmatched_added,
        }

    @property
    def all_renames(self) -> list[RenameCandidate]:
        """Get all detected renames."""
        return self.confirmed_renames + self.possible_renames


# =============================================================================
# Similarity Calculators
# =============================================================================


class LevenshteinSimilarity(SimilarityCalculator):
    """Levenshtein distance-based similarity.

    Computes edit distance normalized by string length.
    """

    @property
    def name(self) -> str:
        return "levenshtein"

    def calculate(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity."""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Create distance matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1].lower() == s2[j - 1].lower() else 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # deletion
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j - 1] + cost,  # substitution
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)


class JaroWinklerSimilarity(SimilarityCalculator):
    """Jaro-Winkler similarity algorithm.

    Particularly good for short strings and common prefixes.
    """

    def __init__(self, prefix_scale: float = 0.1):
        """Initialize with prefix scale.

        Args:
            prefix_scale: Weight for common prefix (0.0 to 0.25).
        """
        self._prefix_scale = min(0.25, max(0.0, prefix_scale))

    @property
    def name(self) -> str:
        return "jaro_winkler"

    def calculate(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity."""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Calculate match window
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        # Find matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i].lower() != s2[j].lower():
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i].lower() != s2[k].lower():
                transpositions += 1
            k += 1

        jaro = (
            matches / len1
            + matches / len2
            + (matches - transpositions / 2) / matches
        ) / 3

        # Jaro-Winkler: boost for common prefix
        prefix = 0
        for i in range(min(4, len1, len2)):
            if s1[i].lower() == s2[i].lower():
                prefix += 1
            else:
                break

        return jaro + prefix * self._prefix_scale * (1 - jaro)


class NgramSimilarity(SimilarityCalculator):
    """N-gram based similarity using Jaccard coefficient.

    Good for catching partial matches and abbreviations.
    """

    def __init__(self, n: int = 2):
        """Initialize with n-gram size.

        Args:
            n: Size of n-grams (default 2 for bigrams).
        """
        self._n = n

    @property
    def name(self) -> str:
        return f"ngram_{self._n}"

    def calculate(self, s1: str, s2: str) -> float:
        """Calculate n-gram Jaccard similarity."""
        if s1 == s2:
            return 1.0

        s1_lower = s1.lower()
        s2_lower = s2.lower()

        if len(s1_lower) < self._n or len(s2_lower) < self._n:
            # Fall back to character comparison for short strings
            return LevenshteinSimilarity().calculate(s1, s2)

        ngrams1 = set(s1_lower[i : i + self._n] for i in range(len(s1_lower) - self._n + 1))
        ngrams2 = set(s2_lower[i : i + self._n] for i in range(len(s2_lower) - self._n + 1))

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0


class TokenSimilarity(SimilarityCalculator):
    """Token-based similarity for snake_case/camelCase names.

    Splits names into tokens and compares token sets.
    """

    @property
    def name(self) -> str:
        return "token"

    def calculate(self, s1: str, s2: str) -> float:
        """Calculate token-based similarity."""
        if s1 == s2:
            return 1.0

        tokens1 = self._tokenize(s1)
        tokens2 = self._tokenize(s2)

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity of token sets
        set1 = set(tokens1)
        set2 = set(tokens2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        jaccard = intersection / union if union > 0 else 0.0

        # Boost for same token order
        order_score = 0.0
        if tokens1 and tokens2:
            # Check for prefix/suffix matches
            if tokens1[0] == tokens2[0]:
                order_score += 0.1
            if tokens1[-1] == tokens2[-1]:
                order_score += 0.1

        return min(1.0, jaccard + order_score)

    def _tokenize(self, s: str) -> list[str]:
        """Split string into tokens.

        Handles snake_case, camelCase, PascalCase, and kebab-case.
        """
        # Handle snake_case and kebab-case
        s = s.replace("-", "_")

        # Handle camelCase and PascalCase
        s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)

        # Split and normalize
        tokens = [t.lower() for t in s.split("_") if t]

        return tokens


class CompositeSimilarity(SimilarityCalculator):
    """Composite calculator using weighted average of multiple calculators."""

    def __init__(
        self,
        calculators: list[tuple[SimilarityCalculator, float]] | None = None,
    ):
        """Initialize with calculators and weights.

        Args:
            calculators: List of (calculator, weight) tuples.
        """
        if calculators is None:
            calculators = [
                (LevenshteinSimilarity(), 0.4),
                (JaroWinklerSimilarity(), 0.3),
                (TokenSimilarity(), 0.3),
            ]
        self._calculators = calculators

    @property
    def name(self) -> str:
        return "composite"

    def calculate(self, s1: str, s2: str) -> float:
        """Calculate weighted average similarity."""
        if s1 == s2:
            return 1.0

        total_weight = sum(w for _, w in self._calculators)
        weighted_sum = sum(
            calc.calculate(s1, s2) * weight for calc, weight in self._calculators
        )

        return weighted_sum / total_weight if total_weight > 0 else 0.0


# =============================================================================
# Column Rename Detector
# =============================================================================


class ColumnRenameDetector:
    """Advanced column rename detector.

    Detects potential column renames using multiple similarity algorithms
    and configurable thresholds.

    Example:
        detector = ColumnRenameDetector(
            similarity_threshold=0.8,
            require_type_match=True,
        )

        result = detector.detect(
            added_columns={"user_email": "Utf8"},
            removed_columns={"email": "Utf8"},
        )

        for rename in result.confirmed_renames:
            print(f"{rename.old_name} -> {rename.new_name} ({rename.confidence})")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        require_type_match: bool = True,
        allow_compatible_types: bool = True,
        calculator: SimilarityCalculator | None = None,
        high_confidence_threshold: float = 0.9,
        low_confidence_threshold: float = 0.7,
    ):
        """Initialize the detector.

        Args:
            similarity_threshold: Minimum similarity for confirmed renames.
            require_type_match: Whether types must match/be compatible.
            allow_compatible_types: Allow type widening (Int32 -> Int64).
            calculator: Similarity calculator to use.
            high_confidence_threshold: Threshold for HIGH confidence.
            low_confidence_threshold: Threshold for LOW confidence.
        """
        self._threshold = similarity_threshold
        self._require_type_match = require_type_match
        self._allow_compatible_types = allow_compatible_types
        self._calculator = calculator or CompositeSimilarity()
        self._high_threshold = high_confidence_threshold
        self._low_threshold = low_confidence_threshold

        # Type compatibility (from detector.py)
        self._compatible_types = self._load_compatible_types()

    def _load_compatible_types(self) -> set[tuple[str, str]]:
        """Load compatible type pairs."""
        return {
            ("Int8", "Int16"),
            ("Int8", "Int32"),
            ("Int8", "Int64"),
            ("Int16", "Int32"),
            ("Int16", "Int64"),
            ("Int32", "Int64"),
            ("UInt8", "UInt16"),
            ("UInt8", "UInt32"),
            ("UInt8", "UInt64"),
            ("UInt16", "UInt32"),
            ("UInt16", "UInt64"),
            ("UInt32", "UInt64"),
            ("Float32", "Float64"),
        }

    def detect(
        self,
        added_columns: dict[str, Any],
        removed_columns: dict[str, Any],
    ) -> RenameDetectionResult:
        """Detect column renames.

        Args:
            added_columns: Dict mapping new column names to types.
            removed_columns: Dict mapping removed column names to types.

        Returns:
            RenameDetectionResult with detected renames.
        """
        result = RenameDetectionResult()
        matched_added: set[str] = set()
        matched_removed: set[str] = set()

        # Find all potential rename candidates
        candidates: list[RenameCandidate] = []

        for old_name, old_type in removed_columns.items():
            best_match: RenameCandidate | None = None
            best_score = 0.0

            for new_name, new_type in added_columns.items():
                if new_name in matched_added:
                    continue

                # Check type compatibility
                type_match = self._check_type_compatibility(old_type, new_type)
                if self._require_type_match and not type_match:
                    continue

                # Calculate similarity
                similarity = self._calculator.calculate(old_name, new_name)

                # Skip if below low threshold
                if similarity < self._low_threshold:
                    continue

                # Create candidate
                confidence = self._determine_confidence(similarity, type_match)
                reasons = self._generate_reasons(old_name, new_name, similarity, type_match)

                candidate = RenameCandidate(
                    old_name=old_name,
                    new_name=new_name,
                    similarity=similarity,
                    confidence=confidence,
                    old_type=old_type,
                    new_type=new_type,
                    type_match=type_match,
                    reasons=reasons,
                    metadata={
                        "algorithm": self._calculator.name,
                        "threshold": self._threshold,
                    },
                )

                if similarity > best_score:
                    best_score = similarity
                    best_match = candidate

            if best_match:
                candidates.append(best_match)
                matched_added.add(best_match.new_name)
                matched_removed.add(best_match.old_name)

        # Categorize candidates
        for candidate in candidates:
            if candidate.similarity >= self._threshold and candidate.type_match:
                result.confirmed_renames.append(candidate)
            else:
                result.possible_renames.append(candidate)

        # Collect unmatched columns
        result.unmatched_removed = [
            name for name in removed_columns if name not in matched_removed
        ]
        result.unmatched_added = [
            name for name in added_columns if name not in matched_added
        ]

        return result

    def _check_type_compatibility(self, old_type: Any, new_type: Any) -> bool:
        """Check if types are compatible."""
        if old_type == new_type:
            return True

        if not self._allow_compatible_types:
            return False

        # Normalize types to strings for comparison
        old_str = self._type_to_string(old_type)
        new_str = self._type_to_string(new_type)

        return (old_str, new_str) in self._compatible_types

    def _type_to_string(self, dtype: Any) -> str:
        """Convert type to string for comparison."""
        if isinstance(dtype, str):
            return dtype

        type_name = type(dtype).__name__
        # Handle Polars types
        if hasattr(dtype, "__class__"):
            type_name = dtype.__class__.__name__

        return type_name

    def _determine_confidence(
        self,
        similarity: float,
        type_match: bool,
    ) -> RenameConfidence:
        """Determine confidence level."""
        if not type_match:
            return RenameConfidence.UNCERTAIN

        if similarity >= self._high_threshold:
            return RenameConfidence.HIGH
        elif similarity >= self._threshold:
            return RenameConfidence.MEDIUM
        elif similarity >= self._low_threshold:
            return RenameConfidence.LOW
        else:
            return RenameConfidence.UNCERTAIN

    def _generate_reasons(
        self,
        old_name: str,
        new_name: str,
        similarity: float,
        type_match: bool,
    ) -> list[str]:
        """Generate reasons supporting this rename candidate."""
        reasons = []

        # Similarity reasons
        if similarity >= 0.9:
            reasons.append(f"High name similarity ({similarity:.0%})")
        elif similarity >= 0.8:
            reasons.append(f"Good name similarity ({similarity:.0%})")
        else:
            reasons.append(f"Moderate name similarity ({similarity:.0%})")

        # Type match
        if type_match:
            reasons.append("Types match or are compatible")

        # Specific patterns
        if old_name.lower() in new_name.lower() or new_name.lower() in old_name.lower():
            reasons.append("Name is a substring of the other")

        # Common prefixes
        common_prefix_len = 0
        for c1, c2 in zip(old_name.lower(), new_name.lower()):
            if c1 == c2:
                common_prefix_len += 1
            else:
                break
        if common_prefix_len >= 3:
            reasons.append(f"Common prefix of {common_prefix_len} characters")

        # Token overlap
        old_tokens = set(self._tokenize(old_name))
        new_tokens = set(self._tokenize(new_name))
        overlap = old_tokens & new_tokens
        if overlap:
            reasons.append(f"Shared tokens: {', '.join(overlap)}")

        return reasons

    def _tokenize(self, s: str) -> list[str]:
        """Split string into tokens."""
        s = s.replace("-", "_")
        s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
        return [t.lower() for t in s.split("_") if t]


# =============================================================================
# Factory Function
# =============================================================================


def create_rename_detector(
    algorithm: str = "composite",
    threshold: float = 0.8,
    require_type_match: bool = True,
) -> ColumnRenameDetector:
    """Factory function to create a rename detector.

    Args:
        algorithm: Similarity algorithm ("levenshtein", "jaro_winkler", "ngram", "token", "composite").
        threshold: Similarity threshold.
        require_type_match: Whether to require type match.

    Returns:
        Configured ColumnRenameDetector.
    """
    calculators: dict[str, SimilarityCalculator] = {
        "levenshtein": LevenshteinSimilarity(),
        "jaro_winkler": JaroWinklerSimilarity(),
        "ngram": NgramSimilarity(),
        "token": TokenSimilarity(),
        "composite": CompositeSimilarity(),
    }

    calculator = calculators.get(algorithm, CompositeSimilarity())

    return ColumnRenameDetector(
        similarity_threshold=threshold,
        require_type_match=require_type_match,
        calculator=calculator,
    )
