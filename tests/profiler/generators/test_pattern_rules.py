"""Tests for PatternRuleGenerator empty regex handling."""

from __future__ import annotations

import pytest

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    PatternMatch,
    Strictness,
    TableProfile,
)
from truthound.profiler.generators.pattern_rules import PatternRuleGenerator


class TestPatternRuleGeneratorEmptyRegex:
    """Test PatternRuleGenerator handles empty regex patterns correctly."""

    def test_skips_empty_regex_pattern(self):
        """Test that patterns with empty regex are skipped."""
        column = ColumnProfile(
            name="test_col",
            physical_type="String",
            inferred_type=DataType.STRING,
            row_count=100,
            detected_patterns=(
                PatternMatch(
                    pattern="empty_pattern",
                    regex="",  # Empty regex
                    match_ratio=0.95,
                    sample_matches=(),
                ),
            ),
        )
        profile = TableProfile(
            name="test_table",
            row_count=100,
            column_count=1,
            columns=(column,),
        )

        generator = PatternRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should not generate any pattern rules for empty regex
        pattern_rules = [r for r in rules if "RegexValidator" in r.validator_class]
        assert len(pattern_rules) == 0

    def test_generates_rules_for_valid_regex_pattern(self):
        """Test that patterns with valid regex generate rules."""
        column = ColumnProfile(
            name="email_col",
            physical_type="String",
            inferred_type=DataType.STRING,
            row_count=100,
            detected_patterns=(
                PatternMatch(
                    pattern="email",
                    regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                    match_ratio=0.95,
                    sample_matches=("test@example.com",),
                ),
            ),
        )
        profile = TableProfile(
            name="test_table",
            row_count=100,
            column_count=1,
            columns=(column,),
        )

        generator = PatternRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should generate a pattern rule for valid regex
        pattern_rules = [r for r in rules if "RegexValidator" in r.validator_class]
        assert len(pattern_rules) == 1
        assert pattern_rules[0].parameters.get("pattern") is not None

    def test_skips_none_regex_pattern(self):
        """Test that patterns with None regex are skipped (edge case)."""
        # Create a pattern match with a valid regex, then manually set it to None
        # This simulates a deserialization issue
        column = ColumnProfile(
            name="test_col",
            physical_type="String",
            inferred_type=DataType.STRING,
            row_count=100,
            detected_patterns=(
                PatternMatch(
                    pattern="broken_pattern",
                    regex="",  # Empty string (closest to None in type-safe way)
                    match_ratio=0.95,
                    sample_matches=(),
                ),
            ),
        )
        profile = TableProfile(
            name="test_table",
            row_count=100,
            column_count=1,
            columns=(column,),
        )

        generator = PatternRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should not generate any pattern rules
        pattern_rules = [r for r in rules if "RegexValidator" in r.validator_class]
        assert len(pattern_rules) == 0

    def test_mixed_valid_and_empty_patterns(self):
        """Test handling mixed valid and empty regex patterns."""
        column = ColumnProfile(
            name="mixed_col",
            physical_type="String",
            inferred_type=DataType.STRING,
            row_count=100,
            detected_patterns=(
                PatternMatch(
                    pattern="empty_pattern",
                    regex="",  # Empty - should be skipped
                    match_ratio=0.95,
                    sample_matches=(),
                ),
                PatternMatch(
                    pattern="valid_pattern",
                    regex=r"^\d{4}-\d{2}-\d{2}$",  # Valid date pattern
                    match_ratio=0.96,  # Above MEDIUM threshold (0.95)
                    sample_matches=("2024-01-15",),
                ),
            ),
        )
        profile = TableProfile(
            name="test_table",
            row_count=100,
            column_count=1,
            columns=(column,),
        )

        generator = PatternRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should only generate rule for valid regex
        pattern_rules = [r for r in rules if "RegexValidator" in r.validator_class]
        assert len(pattern_rules) == 1
        assert pattern_rules[0].parameters["pattern"] == r"^\d{4}-\d{2}-\d{2}$"

    def test_low_match_ratio_skipped(self):
        """Test that patterns below threshold are skipped regardless of regex."""
        column = ColumnProfile(
            name="test_col",
            physical_type="String",
            inferred_type=DataType.STRING,
            row_count=100,
            detected_patterns=(
                PatternMatch(
                    pattern="low_match",
                    regex=r"^\d+$",
                    match_ratio=0.5,  # Below threshold
                    sample_matches=("123",),
                ),
            ),
        )
        profile = TableProfile(
            name="test_table",
            row_count=100,
            column_count=1,
            columns=(column,),
        )

        generator = PatternRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should not generate rules for low match ratio
        pattern_rules = [r for r in rules if "RegexValidator" in r.validator_class]
        assert len(pattern_rules) == 0
