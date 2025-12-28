"""Training dataset generation for ReDoS ML models.

This module provides utilities for generating, managing, and augmenting
training datasets for ReDoS vulnerability detection.

Features:
    - Built-in collection of known vulnerable patterns
    - Pattern mutation for data augmentation
    - Balanced dataset generation
    - Import from external sources (CSV, JSON)
    - Export for sharing and reproducibility

Example:
    >>> from truthound.validators.security.redos.ml.datasets import (
    ...     ReDoSDatasetGenerator,
    ...     generate_training_dataset,
    ... )
    >>> dataset = generate_training_dataset(n_samples=1000)
    >>> print(f"Generated {len(dataset)} samples")
    >>> print(f"Vulnerable: {dataset.num_vulnerable}, Safe: {dataset.num_safe}")
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from truthound.validators.security.redos.ml.base import ReDoSTrainingData


class PatternLabel(int, Enum):
    """Labels for regex patterns."""

    SAFE = 0
    VULNERABLE = 1


# =============================================================================
# Built-in Pattern Collections
# =============================================================================

# Known vulnerable patterns from security research and CVE databases
VULNERABLE_PATTERNS: List[str] = [
    # Nested quantifiers (exponential backtracking)
    r"(a+)+",
    r"(a+)+b",
    r"(a*)*",
    r"(a*)*b",
    r"(a+)*",
    r"([a-z]+)+",
    r"([a-zA-Z]+)+",
    r"(.*)+",
    r"(.+)+",
    r"(\w+)+",
    r"(\d+)+",
    r"([0-9]+)+",
    r"(x+x+)+y",
    r"((a+)+)+",
    r"(a|aa)+",
    r"(a|a)+",
    # Quantified alternation
    r"(a|b)+c",
    r"(ab|cd)+e",
    r"(foo|foobar)+",
    r"([a-z]|[0-9])+",
    r"(cat|car|card)+",
    r"(a|ab)*c",
    r"(aaa|aab|aba|abb|baa|bab|bba|bbb)+",
    # Overlapping patterns
    r".*a.*b.*c",
    r".*.*.*",
    r".*.+.+",
    r"a.*b.*c.*d.*e",
    r"[\s\S]*[\s\S]*",
    # Email-like ReDoS
    r"^([a-zA-Z0-9])(([\-.]|[_]+)?([a-zA-Z0-9]+))*(@){1}[a-z0-9]+[.]{1}(([a-z]{2,3})|([a-z]{2,3}[.]{1}[a-z]{2,3}))$",
    r"^([0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*@([0-9a-zA-Z][-\w]*[0-9a-zA-Z]\.)+[a-zA-Z]{2,9})$",
    # URL-like ReDoS
    r"^(([a-z]+://|)([a-z0-9_-]+(\.[a-z0-9_-]+)+))([a-z0-9/\._-]+)*$",
    # Additional patterns from CVE database
    r"([a-z]+)*$",
    r"(a{1,})+$",
    r"^(a+)+$",
    r"^(0|[1-9][0-9]*)(.[0-9]+)?$",
    r"^(.+\.)+.+$",
    r"^(\w+\.)+\w+$",
    r"(?:a+)+b",
    r"(?:(?:a+)+)+b",
    r"(([a-z])+.)+[A-Z]([a-z])+$",
    # Real-world vulnerable patterns
    r"^\d+1?\d*$",
    r"^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){2}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
]

# Known safe patterns
SAFE_PATTERNS: List[str] = [
    # Simple patterns
    r"^[a-z]+$",
    r"^[A-Z]+$",
    r"^[0-9]+$",
    r"^\d+$",
    r"^\w+$",
    r"^[a-zA-Z0-9]+$",
    r"^.+$",
    r"^.*$",
    # Anchored patterns with bounded quantifiers
    r"^[a-z]{1,10}$",
    r"^[0-9]{1,20}$",
    r"^\w{3,50}$",
    r"^.{1,100}$",
    # Character classes only
    r"[a-z]",
    r"[A-Z]",
    r"[0-9]",
    r"[a-zA-Z]",
    r"[a-zA-Z0-9]",
    r"[\w\s]",
    # Literals
    r"hello",
    r"world",
    r"test",
    r"^hello$",
    r"^test$",
    # Simple optional patterns
    r"^https?://",
    r"^www\.",
    r"^[a-z]+@[a-z]+\.[a-z]+$",
    # Well-structured patterns
    r"^\d{4}-\d{2}-\d{2}$",
    r"^\d{2}/\d{2}/\d{4}$",
    r"^[A-Z]{2}\d{4}$",
    r"^#[0-9a-fA-F]{6}$",
    r"^rgb\(\d{1,3},\d{1,3},\d{1,3}\)$",
    # Common safe regex patterns
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",  # Anchored email
    r"^(https?|ftp)://[^\s/$.?#].[^\s]*$",  # URL with proper structure
    r"^\+?[1-9]\d{1,14}$",  # E.164 phone format
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",  # UUID
    # Simple patterns with single quantifiers
    r"a+",
    r"a*",
    r"a?",
    r"[a-z]+",
    r"\d+",
    r"\w+",
    # Alternation without quantifiers
    r"cat|dog",
    r"yes|no",
    r"true|false",
    r"^(true|false)$",
    # Lookahead/lookbehind (generally safe)
    r"(?=.*[A-Z])",
    r"(?=.*\d)",
    r"(?<=@)[a-z]+",
    r"(?<!\\)\"",
]


@dataclass
class PatternMutation:
    """A mutation strategy for pattern augmentation."""

    name: str
    description: str
    apply_to: str  # "vulnerable" or "safe" or "both"

    def mutate(self, pattern: str) -> Optional[str]:
        """Apply the mutation to a pattern.

        Returns None if mutation is not applicable.
        """
        raise NotImplementedError


class AddNestedQuantifier(PatternMutation):
    """Add nested quantifiers to make pattern vulnerable."""

    def __init__(self):
        super().__init__(
            name="add_nested_quantifier",
            description="Wrap quantified groups in another quantifier",
            apply_to="safe",
        )

    def mutate(self, pattern: str) -> Optional[str]:
        # Find groups with quantifiers and nest them
        group_match = re.search(r"\(([^)]+)\)([+*])", pattern)
        if group_match:
            inner = group_match.group(1)
            quant = group_match.group(2)
            replacement = f"(({inner}){quant}){quant}"
            return pattern[: group_match.start()] + replacement + pattern[group_match.end() :]
        return None


class RemoveAnchors(PatternMutation):
    """Remove anchors from a pattern."""

    def __init__(self):
        super().__init__(
            name="remove_anchors",
            description="Remove ^ and $ anchors",
            apply_to="safe",
        )

    def mutate(self, pattern: str) -> Optional[str]:
        result = pattern
        if result.startswith("^"):
            result = result[1:]
        if result.endswith("$"):
            result = result[:-1]
        return result if result != pattern else None


class AddAnchors(PatternMutation):
    """Add anchors to a pattern."""

    def __init__(self):
        super().__init__(
            name="add_anchors",
            description="Add ^ and $ anchors",
            apply_to="vulnerable",
        )

    def mutate(self, pattern: str) -> Optional[str]:
        result = pattern
        if not result.startswith("^"):
            result = "^" + result
        if not result.endswith("$"):
            result = result + "$"
        return result if result != pattern else None


class CharacterClassVariation(PatternMutation):
    """Create variations of character classes."""

    def __init__(self):
        super().__init__(
            name="char_class_variation",
            description="Create variations of character classes",
            apply_to="both",
        )
        self.replacements = [
            (r"\[a-z\]", "[a-zA-Z]"),
            (r"\[A-Z\]", "[a-zA-Z0-9]"),
            (r"\[0-9\]", "[0-9a-f]"),
            (r"\\d", "[0-9]"),
            (r"\\w", "[a-zA-Z0-9_]"),
            (r"a", "[aA]"),
        ]

    def mutate(self, pattern: str) -> Optional[str]:
        for old, new in self.replacements:
            if re.search(old, pattern):
                return re.sub(old, new, pattern, count=1)
        return None


class QuantifierVariation(PatternMutation):
    """Create variations of quantifiers."""

    def __init__(self):
        super().__init__(
            name="quantifier_variation",
            description="Create variations of quantifiers",
            apply_to="both",
        )

    def mutate(self, pattern: str) -> Optional[str]:
        variations = [
            (r"\+", "*"),
            (r"\*", "+"),
            (r"\+", "{1,}"),
            (r"\*", "{0,}"),
            (r"\{(\d+)\}", lambda m: f"{{{m.group(1)},{int(m.group(1))+10}}}"),
        ]
        for old, new in variations:
            match = re.search(old, pattern)
            if match:
                if callable(new):
                    replacement = new(match)
                else:
                    replacement = new
                return pattern[: match.start()] + replacement + pattern[match.end() :]
        return None


# =============================================================================
# Dataset Generator
# =============================================================================


class ReDoSDatasetGenerator:
    """Generator for ReDoS training datasets.

    This class provides comprehensive dataset generation capabilities:
    - Built-in vulnerable and safe pattern collections
    - Pattern mutation for data augmentation
    - Balanced dataset generation
    - Custom pattern sources
    - Export/import functionality

    Example:
        >>> generator = ReDoSDatasetGenerator(random_state=42)
        >>> dataset = generator.generate(n_samples=500, balance=0.5)
        >>> print(f"Generated {len(dataset)} samples")
    """

    def __init__(
        self,
        vulnerable_patterns: Optional[List[str]] = None,
        safe_patterns: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        """Initialize the dataset generator.

        Args:
            vulnerable_patterns: Custom vulnerable patterns (uses built-in if None)
            safe_patterns: Custom safe patterns (uses built-in if None)
            random_state: Random seed for reproducibility
        """
        self.vulnerable_patterns = vulnerable_patterns or VULNERABLE_PATTERNS.copy()
        self.safe_patterns = safe_patterns or SAFE_PATTERNS.copy()
        self.random_state = random_state
        self._rng = random.Random(random_state)

        # Mutation strategies
        self._mutations = [
            AddNestedQuantifier(),
            RemoveAnchors(),
            AddAnchors(),
            CharacterClassVariation(),
            QuantifierVariation(),
        ]

    def generate(
        self,
        n_samples: Optional[int] = None,
        balance: float = 0.5,
        augment: bool = True,
        augment_factor: float = 2.0,
    ) -> ReDoSTrainingData:
        """Generate a training dataset.

        Args:
            n_samples: Number of samples to generate (None for all available)
            balance: Ratio of vulnerable samples (0.0 to 1.0)
            augment: Whether to augment with mutations
            augment_factor: How many times to augment each pattern

        Returns:
            ReDoSTrainingData with patterns and labels
        """
        patterns = []
        labels = []

        # Start with base patterns
        base_vulnerable = self.vulnerable_patterns.copy()
        base_safe = self.safe_patterns.copy()

        # Augment if requested
        if augment:
            augmented_vulnerable = self._augment_patterns(
                base_vulnerable, PatternLabel.VULNERABLE, augment_factor
            )
            augmented_safe = self._augment_patterns(
                base_safe, PatternLabel.SAFE, augment_factor
            )
            base_vulnerable.extend(augmented_vulnerable)
            base_safe.extend(augmented_safe)

        # Determine sample counts
        if n_samples is None:
            n_vulnerable = len(base_vulnerable)
            n_safe = len(base_safe)
        else:
            n_vulnerable = int(n_samples * balance)
            n_safe = n_samples - n_vulnerable

        # Sample patterns
        vulnerable_samples = self._sample_patterns(base_vulnerable, n_vulnerable)
        safe_samples = self._sample_patterns(base_safe, n_safe)

        # Combine
        for pattern in vulnerable_samples:
            patterns.append(pattern)
            labels.append(PatternLabel.VULNERABLE.value)

        for pattern in safe_samples:
            patterns.append(pattern)
            labels.append(PatternLabel.SAFE.value)

        # Shuffle
        combined = list(zip(patterns, labels))
        self._rng.shuffle(combined)
        patterns, labels = zip(*combined) if combined else ([], [])

        return ReDoSTrainingData(
            patterns=list(patterns),
            labels=list(labels),
            metadata={
                "generator": "ReDoSDatasetGenerator",
                "n_samples": len(patterns),
                "balance": sum(labels) / len(labels) if labels else 0,
                "augmented": augment,
                "random_state": self.random_state,
            },
        )

    def _sample_patterns(self, patterns: List[str], n: int) -> List[str]:
        """Sample n patterns from a list, with replacement if needed."""
        if len(patterns) >= n:
            return self._rng.sample(patterns, n)
        else:
            # Need to sample with replacement
            result = patterns.copy()
            while len(result) < n:
                result.extend(self._rng.choices(patterns, k=min(n - len(result), len(patterns))))
            return result[:n]

    def _augment_patterns(
        self,
        patterns: List[str],
        label: PatternLabel,
        factor: float,
    ) -> List[str]:
        """Augment patterns using mutations."""
        augmented = []
        target_count = int(len(patterns) * factor)

        applicable_mutations = [
            m
            for m in self._mutations
            if m.apply_to == "both"
            or (m.apply_to == "vulnerable" and label == PatternLabel.VULNERABLE)
            or (m.apply_to == "safe" and label == PatternLabel.SAFE)
        ]

        if not applicable_mutations:
            return augmented

        attempts = 0
        max_attempts = target_count * 10

        while len(augmented) < target_count and attempts < max_attempts:
            pattern = self._rng.choice(patterns)
            mutation = self._rng.choice(applicable_mutations)

            try:
                mutated = mutation.mutate(pattern)
                if mutated and mutated not in patterns and mutated not in augmented:
                    # Validate the pattern compiles
                    re.compile(mutated)
                    augmented.append(mutated)
            except re.error:
                pass

            attempts += 1

        return augmented

    def add_patterns(
        self,
        patterns: List[str],
        labels: List[int],
    ) -> None:
        """Add custom patterns to the generator.

        Args:
            patterns: List of regex patterns
            labels: List of labels (0=safe, 1=vulnerable)
        """
        for pattern, label in zip(patterns, labels):
            if label == PatternLabel.VULNERABLE.value:
                if pattern not in self.vulnerable_patterns:
                    self.vulnerable_patterns.append(pattern)
            else:
                if pattern not in self.safe_patterns:
                    self.safe_patterns.append(pattern)

    def load_from_json(self, path: str | Path) -> None:
        """Load patterns from a JSON file.

        Expected format:
            {
                "vulnerable": ["pattern1", "pattern2", ...],
                "safe": ["pattern1", "pattern2", ...]
            }

        Args:
            path: Path to JSON file
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        if "vulnerable" in data:
            for pattern in data["vulnerable"]:
                if pattern not in self.vulnerable_patterns:
                    self.vulnerable_patterns.append(pattern)

        if "safe" in data:
            for pattern in data["safe"]:
                if pattern not in self.safe_patterns:
                    self.safe_patterns.append(pattern)

    def save_to_json(self, path: str | Path) -> None:
        """Save current patterns to a JSON file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        data = {
            "vulnerable": self.vulnerable_patterns,
            "safe": self.safe_patterns,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the available patterns.

        Returns:
            Dictionary with pattern statistics
        """
        return {
            "vulnerable_count": len(self.vulnerable_patterns),
            "safe_count": len(self.safe_patterns),
            "total_count": len(self.vulnerable_patterns) + len(self.safe_patterns),
            "balance": len(self.vulnerable_patterns)
            / (len(self.vulnerable_patterns) + len(self.safe_patterns)),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_training_dataset(
    n_samples: Optional[int] = None,
    balance: float = 0.5,
    augment: bool = True,
    random_state: int = 42,
) -> ReDoSTrainingData:
    """Generate a training dataset using default settings.

    This is a convenience function for quick dataset generation.

    Args:
        n_samples: Number of samples (None for all available)
        balance: Ratio of vulnerable samples
        augment: Whether to augment with mutations
        random_state: Random seed

    Returns:
        ReDoSTrainingData
    """
    generator = ReDoSDatasetGenerator(random_state=random_state)
    return generator.generate(n_samples=n_samples, balance=balance, augment=augment)


def load_dataset(path: str | Path) -> ReDoSTrainingData:
    """Load a dataset from a JSON file.

    Expected format:
        {
            "patterns": ["pattern1", "pattern2", ...],
            "labels": [0, 1, 0, ...]
        }

    Args:
        path: Path to JSON file

    Returns:
        ReDoSTrainingData
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    return ReDoSTrainingData(
        patterns=data["patterns"],
        labels=data["labels"],
        metadata=data.get("metadata", {}),
    )


def save_dataset(dataset: ReDoSTrainingData, path: str | Path) -> None:
    """Save a dataset to a JSON file.

    Args:
        dataset: Dataset to save
        path: Path to save to
    """
    path = Path(path)
    data = {
        "patterns": dataset.patterns,
        "labels": dataset.labels,
        "metadata": dataset.metadata,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_builtin_patterns() -> Tuple[List[str], List[str]]:
    """Get the built-in vulnerable and safe patterns.

    Returns:
        Tuple of (vulnerable_patterns, safe_patterns)
    """
    return VULNERABLE_PATTERNS.copy(), SAFE_PATTERNS.copy()
