"""Feature extraction for ReDoS ML analysis.

This module provides feature extraction capabilities for regex patterns,
converting raw pattern strings into numeric feature vectors suitable for
machine learning models.

The feature extractor analyzes patterns for:
- Structural characteristics (length, groups, nesting depth)
- Quantifier usage and patterns
- Dangerous constructs (nested quantifiers, quantified alternation)
- Character class and lookaround features
- Complexity metrics

Example:
    >>> extractor = PatternFeatureExtractor()
    >>> features = extractor.extract(r"(a+)+b")
    >>> print(features.nested_quantifier_count)  # 1
    >>> print(features.backtracking_potential)   # High value
"""

from __future__ import annotations

import re
from typing import List, Sequence

from truthound.validators.security.redos.ml.base import PatternFeatures


class PatternFeatureExtractor:
    """Extracts ML-relevant features from regex patterns.

    This extractor analyzes regex patterns and produces feature vectors
    suitable for machine learning models. It uses static analysis techniques
    to identify pattern characteristics without executing the regex.

    The extractor is stateless and thread-safe for concurrent use.

    Example:
        >>> extractor = PatternFeatureExtractor()
        >>> features = extractor.extract(r"(a+)+b")
        >>> vector = features.to_vector()
        >>> print(len(vector))  # 32 features
    """

    # Compiled patterns for feature extraction (class-level for efficiency)
    _PLUS_PATTERN = re.compile(r"(?<!\\)\+")
    _STAR_PATTERN = re.compile(r"(?<!\\)\*")
    _QUESTION_PATTERN = re.compile(r"(?<!\\)\?(?![=!<:])")
    _BOUNDED_QUANT_PATTERN = re.compile(r"\{(\d+)(?:,(\d*))?\}")
    _LAZY_QUANT_PATTERN = re.compile(r"[+*?]\?|\{[^}]+\}\?")
    _CHAR_CLASS_PATTERN = re.compile(r"\[[^\]]+\]")
    _NEGATED_CLASS_PATTERN = re.compile(r"\[\^[^\]]+\]")
    _LOOKAHEAD_PATTERN = re.compile(r"\(\?[=!]")
    _LOOKBEHIND_PATTERN = re.compile(r"\(\?<[=!]")
    _BACKREFERENCE_PATTERN = re.compile(r"\\([1-9]\d*)")
    _NESTED_QUANT_PATTERN = re.compile(r"\([^)]*[+*][^)]*\)[+*]")
    _ADJACENT_QUANT_PATTERN = re.compile(r"[+*][+*]")
    _QUANTIFIED_ALT_PATTERN = re.compile(r"\([^)]*\|[^)]*\)[+*?]")
    _QUANTIFIED_BACKREF_PATTERN = re.compile(r"\\[1-9][+*]|\{[^}]+\}")
    _DOT_PATTERN = re.compile(r"(?<!\\)\.")
    _WORD_BOUNDARY_PATTERN = re.compile(r"\\b")
    _NON_CAPTURE_GROUP_PATTERN = re.compile(r"\(\?(?:[imsxLu]|:)")
    _CAPTURE_GROUP_PATTERN = re.compile(r"\((?!\?)")

    def extract(self, pattern: str) -> PatternFeatures:
        """Extract features from a regex pattern.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            PatternFeatures with all extracted features
        """
        if not pattern:
            return PatternFeatures()

        # Structural features
        length = len(pattern)
        max_nesting_depth = self._calculate_nesting_depth(pattern)
        alternation_count = pattern.count("|")

        # Group counts
        capture_group_count = len(self._CAPTURE_GROUP_PATTERN.findall(pattern))
        non_capture_group_count = len(self._NON_CAPTURE_GROUP_PATTERN.findall(pattern))
        group_count = capture_group_count + non_capture_group_count

        # Quantifier features
        plus_count = len(self._PLUS_PATTERN.findall(pattern))
        star_count = len(self._STAR_PATTERN.findall(pattern))
        question_count = len(self._QUESTION_PATTERN.findall(pattern))
        lazy_quantifier_count = len(self._LAZY_QUANT_PATTERN.findall(pattern))

        bounded_matches = self._BOUNDED_QUANT_PATTERN.findall(pattern)
        bounded_quantifier_count = len(bounded_matches)

        # Unbounded quantifiers
        unbounded_count = 0
        for min_val, max_val in bounded_matches:
            if max_val == "":  # {n,} form
                unbounded_count += 1
        unbounded_quantifier_count = plus_count + star_count + unbounded_count

        # Quantifier density
        total_quantifiers = (
            plus_count + star_count + question_count + bounded_quantifier_count
        )
        quantifier_density = total_quantifiers / max(length, 1)

        # Dangerous patterns
        nested_quantifier_count = len(self._NESTED_QUANT_PATTERN.findall(pattern))
        adjacent_quantifier_count = len(self._ADJACENT_QUANT_PATTERN.findall(pattern))
        quantified_alternation_count = len(self._QUANTIFIED_ALT_PATTERN.findall(pattern))
        quantified_backreference_count = len(
            self._QUANTIFIED_BACKREF_PATTERN.findall(pattern)
        )

        # Character class features
        char_class_count = len(self._CHAR_CLASS_PATTERN.findall(pattern))
        negated_char_class_count = len(self._NEGATED_CLASS_PATTERN.findall(pattern))
        dot_count = len(self._DOT_PATTERN.findall(pattern))
        word_boundary_count = len(self._WORD_BOUNDARY_PATTERN.findall(pattern))

        # Lookaround features
        lookahead_matches = self._LOOKAHEAD_PATTERN.findall(pattern)
        lookbehind_matches = self._LOOKBEHIND_PATTERN.findall(pattern)
        lookahead_count = len(lookahead_matches)
        lookbehind_count = len(lookbehind_matches)
        negative_lookaround_count = pattern.count("(?!") + pattern.count("(?<!")

        # Backreference features
        backref_matches = self._BACKREFERENCE_PATTERN.findall(pattern)
        backreference_count = len(backref_matches)
        max_backreference_index = (
            max(int(m) for m in backref_matches) if backref_matches else 0
        )

        # Anchor features
        start_anchor = pattern.startswith("^") or "\\A" in pattern
        end_anchor = pattern.endswith("$") or "\\Z" in pattern or "\\z" in pattern
        anchored = start_anchor and end_anchor

        # Calculate complexity metrics
        features_partial = PatternFeatures(
            length=length,
            group_count=group_count,
            capture_group_count=capture_group_count,
            non_capture_group_count=non_capture_group_count,
            max_nesting_depth=max_nesting_depth,
            alternation_count=alternation_count,
            plus_count=plus_count,
            star_count=star_count,
            question_count=question_count,
            bounded_quantifier_count=bounded_quantifier_count,
            unbounded_quantifier_count=unbounded_quantifier_count,
            lazy_quantifier_count=lazy_quantifier_count,
            possessive_quantifier_count=0,  # Python doesn't support possessive
            quantifier_density=quantifier_density,
            nested_quantifier_count=nested_quantifier_count,
            adjacent_quantifier_count=adjacent_quantifier_count,
            quantified_alternation_count=quantified_alternation_count,
            quantified_backreference_count=quantified_backreference_count,
            char_class_count=char_class_count,
            negated_char_class_count=negated_char_class_count,
            dot_count=dot_count,
            word_boundary_count=word_boundary_count,
            lookahead_count=lookahead_count,
            lookbehind_count=lookbehind_count,
            negative_lookaround_count=negative_lookaround_count,
            backreference_count=backreference_count,
            max_backreference_index=max_backreference_index,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            anchored=anchored,
        )

        backtracking_potential = self._calculate_backtracking_potential(features_partial)
        estimated_states = self._estimate_nfa_states(features_partial)

        # Return complete features with complexity metrics
        return PatternFeatures(
            length=length,
            group_count=group_count,
            capture_group_count=capture_group_count,
            non_capture_group_count=non_capture_group_count,
            max_nesting_depth=max_nesting_depth,
            alternation_count=alternation_count,
            plus_count=plus_count,
            star_count=star_count,
            question_count=question_count,
            bounded_quantifier_count=bounded_quantifier_count,
            unbounded_quantifier_count=unbounded_quantifier_count,
            lazy_quantifier_count=lazy_quantifier_count,
            possessive_quantifier_count=0,
            quantifier_density=quantifier_density,
            nested_quantifier_count=nested_quantifier_count,
            adjacent_quantifier_count=adjacent_quantifier_count,
            quantified_alternation_count=quantified_alternation_count,
            quantified_backreference_count=quantified_backreference_count,
            char_class_count=char_class_count,
            negated_char_class_count=negated_char_class_count,
            dot_count=dot_count,
            word_boundary_count=word_boundary_count,
            lookahead_count=lookahead_count,
            lookbehind_count=lookbehind_count,
            negative_lookaround_count=negative_lookaround_count,
            backreference_count=backreference_count,
            max_backreference_index=max_backreference_index,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            anchored=anchored,
            backtracking_potential=backtracking_potential,
            estimated_states=estimated_states,
        )

    def extract_batch(self, patterns: Sequence[str]) -> List[PatternFeatures]:
        """Extract features from multiple patterns.

        Args:
            patterns: Sequence of regex patterns

        Returns:
            List of PatternFeatures for each pattern
        """
        return [self.extract(pattern) for pattern in patterns]

    def extract_vectors(self, patterns: Sequence[str]) -> List[List[float]]:
        """Extract feature vectors from multiple patterns.

        Convenience method that returns vectors directly instead of
        PatternFeatures objects.

        Args:
            patterns: Sequence of regex patterns

        Returns:
            List of feature vectors
        """
        return [self.extract(pattern).to_vector() for pattern in patterns]

    def _calculate_nesting_depth(self, pattern: str) -> int:
        """Calculate maximum nesting depth of groups.

        Args:
            pattern: Regex pattern

        Returns:
            Maximum depth of nested parentheses
        """
        depth = 0
        max_depth = 0
        for char in pattern:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth = max(0, depth - 1)
        return max_depth

    def _calculate_backtracking_potential(self, features: PatternFeatures) -> float:
        """Estimate backtracking potential based on features.

        Higher values indicate higher risk of catastrophic backtracking.
        The score is normalized to 0-100 range.

        Args:
            features: Extracted pattern features

        Returns:
            Backtracking potential score (0-100)
        """
        potential = 0.0

        # Nested quantifiers are the biggest risk (exponential backtracking)
        potential += features.nested_quantifier_count * 50.0

        # Quantified alternation can also cause exponential backtracking
        potential += features.quantified_alternation_count * 30.0

        # Adjacent quantifiers indicate overlapping patterns
        potential += features.adjacent_quantifier_count * 20.0

        # Unbounded quantifiers increase potential
        potential += features.unbounded_quantifier_count * 5.0

        # Deep nesting increases search space
        potential += features.max_nesting_depth * 3.0

        # Backreferences with quantifiers are very risky
        potential += features.quantified_backreference_count * 40.0

        # Lack of anchoring increases potential matches
        if not features.anchored:
            potential *= 1.2

        return min(potential, 100.0)

    def _estimate_nfa_states(self, features: PatternFeatures) -> int:
        """Estimate number of NFA states.

        This is a rough approximation based on pattern features,
        useful for comparing relative pattern complexity.

        Args:
            features: Extracted pattern features

        Returns:
            Estimated number of NFA states
        """
        # Base states from length
        states = features.length

        # Groups add entry/exit states
        states += features.group_count * 2

        # Quantifiers add loop states
        states += features.plus_count * 2
        states += features.star_count * 2
        states += features.question_count

        # Bounded quantifiers can add many states
        states += features.bounded_quantifier_count * 5

        # Alternations add branch states
        states += features.alternation_count * 2

        # Character classes add states per character
        states += features.char_class_count * 5

        return states


# Module-level singleton for convenience
_default_extractor: PatternFeatureExtractor | None = None


def get_default_extractor() -> PatternFeatureExtractor:
    """Get the default feature extractor singleton.

    Returns:
        Shared PatternFeatureExtractor instance
    """
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = PatternFeatureExtractor()
    return _default_extractor


def extract_features(pattern: str) -> PatternFeatures:
    """Extract features from a pattern using the default extractor.

    Convenience function for quick feature extraction.

    Args:
        pattern: Regex pattern string

    Returns:
        PatternFeatures instance
    """
    return get_default_extractor().extract(pattern)
