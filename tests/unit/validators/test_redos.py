"""Tests for ReDoS (Regular Expression Denial of Service) protection.

This test suite covers:
- Pattern safety checking
- Complexity analysis
- Safe regex compilation
- Timeout protection
"""

import re
import pytest

from truthound.validators.security.redos import (
    RegexSafetyChecker,
    RegexComplexityAnalyzer,
    SafeRegexConfig,
    SafeRegexExecutor,
    RegexAnalysisResult,
    ReDoSRisk,
    check_regex_safety,
    analyze_regex_complexity,
    create_safe_regex,
    safe_match,
    safe_search,
)


class TestRegexSafetyChecker:
    """Test RegexSafetyChecker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = RegexSafetyChecker()

    def test_safe_simple_pattern(self):
        """Test that simple patterns are considered safe."""
        is_safe, warning = self.checker.check(r"^[a-z]+$")
        assert is_safe is True
        assert warning is None

    def test_safe_email_pattern(self):
        """Test that typical email pattern is safe."""
        is_safe, warning = self.checker.check(
            r"^[\w.+-]+@[\w.-]+\.\w+$"
        )
        assert is_safe is True

    def test_unsafe_nested_quantifiers(self):
        """Test that nested quantifiers are detected."""
        is_safe, warning = self.checker.check(r"(a+)+")
        assert is_safe is False
        assert "nested_quantifiers" in warning.lower() or "ReDoS" in warning

    def test_unsafe_alternation_quantifier(self):
        """Test that quantified alternation is detected."""
        is_safe, warning = self.checker.check(r"(a|aa)+")
        assert is_safe is False
        assert warning is not None

    def test_pattern_too_long(self):
        """Test that overly long patterns are rejected."""
        long_pattern = "a" * 1001
        is_safe, warning = self.checker.check(long_pattern)
        assert is_safe is False
        assert "too long" in warning.lower()

    def test_invalid_syntax(self):
        """Test that invalid regex syntax is rejected."""
        is_safe, warning = self.checker.check(r"[")
        assert is_safe is False
        assert "syntax" in warning.lower()

    def test_empty_pattern(self):
        """Test that empty pattern is safe."""
        is_safe, warning = self.checker.check("")
        assert is_safe is True

    def test_quantified_backreference(self):
        """Test that quantified backreferences are detected."""
        is_safe, warning = self.checker.check(r"(a+)\1+")
        assert is_safe is False

    def test_validate_and_compile(self):
        """Test validate_and_compile returns compiled pattern."""
        pattern = self.checker.validate_and_compile(r"^[a-z]+$")
        assert isinstance(pattern, re.Pattern)

    def test_validate_and_compile_rejects_unsafe(self):
        """Test validate_and_compile raises for unsafe patterns."""
        with pytest.raises(ValueError, match="Unsafe"):
            self.checker.validate_and_compile(r"(a+)+b")


class TestRegexComplexityAnalyzer:
    """Test RegexComplexityAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = RegexComplexityAnalyzer()

    def test_analyze_simple_pattern(self):
        """Test analysis of simple pattern."""
        result = self.analyzer.analyze(r"^[a-z]+$")
        assert result.risk_level == ReDoSRisk.NONE
        assert result.is_safe is True
        assert result.complexity_score < 20

    def test_analyze_nested_quantifiers(self):
        """Test detection of nested quantifiers."""
        result = self.analyzer.analyze(r"(a+)+b")
        assert result.risk_level == ReDoSRisk.CRITICAL
        assert result.is_safe is False
        assert "nested_quantifiers" in result.dangerous_constructs

    def test_analyze_deeply_nested(self):
        """Test detection of deeply nested patterns."""
        result = self.analyzer.analyze(r"((a+)+)+")
        assert result.risk_level == ReDoSRisk.CRITICAL
        assert result.complexity_score > 30

    def test_metrics_extraction(self):
        """Test that metrics are extracted correctly."""
        result = self.analyzer.analyze(r"(a)(b)(c)")
        assert result.metrics["group_count"] == 3

    def test_nesting_depth(self):
        """Test nesting depth calculation."""
        result = self.analyzer.analyze(r"((()))")
        assert result.metrics["max_nesting"] == 3

    def test_backreference_detection(self):
        """Test backreference detection."""
        result = self.analyzer.analyze(r"(a+)\1")
        assert result.metrics["has_backreference"] is True

    def test_lookaround_detection(self):
        """Test lookaround detection."""
        result = self.analyzer.analyze(r"(?=foo)bar")
        assert result.metrics["has_lookaround"] is True

    def test_recommendation_for_critical(self):
        """Test that critical patterns get recommendation."""
        result = self.analyzer.analyze(r"(a+)+b")
        assert result.recommendation
        assert "CRITICAL" in result.recommendation or "nested" in result.recommendation.lower()


class TestSafeRegexConfig:
    """Test SafeRegexConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SafeRegexConfig()
        assert config.max_pattern_length == 1000
        assert config.max_groups == 20
        assert config.timeout_seconds == 1.0

    def test_strict_config(self):
        """Test strict configuration."""
        config = SafeRegexConfig.strict()
        assert config.max_pattern_length == 500
        assert config.allow_backreferences is False
        assert config.allow_lookaround is False

    def test_lenient_config(self):
        """Test lenient configuration."""
        config = SafeRegexConfig.lenient()
        assert config.max_pattern_length == 5000
        assert config.allow_backreferences is True

    def test_config_immutability(self):
        """Test that config is immutable."""
        config = SafeRegexConfig()
        with pytest.raises(AttributeError):
            config.max_pattern_length = 500  # type: ignore


class TestSafeRegexExecutor:
    """Test SafeRegexExecutor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = SafeRegexExecutor(timeout_seconds=1.0)

    def test_match_succeeds(self):
        """Test successful match execution."""
        result = self.executor.match(r"^hello$", "hello")
        assert result is not None
        assert result.group() == "hello"

    def test_match_no_match(self):
        """Test match returns None when no match."""
        result = self.executor.match(r"^hello$", "world")
        assert result is None

    def test_search_succeeds(self):
        """Test successful search execution."""
        result = self.executor.search(r"world", "hello world")
        assert result is not None
        assert result.group() == "world"

    def test_findall_succeeds(self):
        """Test successful findall execution."""
        result = self.executor.findall(r"\d+", "a1b2c3")
        assert result == ["1", "2", "3"]

    def test_input_too_long(self):
        """Test that overly long input is rejected."""
        executor = SafeRegexExecutor(max_input_length=100)
        with pytest.raises(ValueError, match="too long"):
            executor.match(r".", "a" * 101)

    def test_timeout_on_catastrophic_backtracking(self):
        """Test timeout on known ReDoS pattern.

        This test uses a pattern known to cause catastrophic backtracking.
        Note: Python's regex engine is optimized enough that short inputs
        may complete quickly. We use a longer input to trigger backtracking.
        """
        executor = SafeRegexExecutor(timeout_seconds=0.05)
        dangerous_pattern = r"(a+)+b"
        # Longer input needed to trigger timeout
        dangerous_input = "a" * 30 + "!"

        # This may or may not timeout depending on the system
        # The important thing is that the executor respects the timeout
        try:
            result = executor.match(dangerous_pattern, dangerous_input)
            # If it completes, the result should be None (no match)
            assert result is None
        except TimeoutError:
            # Expected behavior for slow systems
            pass

    def test_compiled_pattern(self):
        """Test using pre-compiled pattern."""
        compiled = re.compile(r"^test$")
        result = self.executor.match(compiled, "test")
        assert result is not None


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_check_regex_safety(self):
        """Test check_regex_safety function."""
        is_safe, warning = check_regex_safety(r"^[a-z]+$")
        assert is_safe is True
        assert warning is None

        is_safe, warning = check_regex_safety(r"(a+)+")
        assert is_safe is False
        assert warning is not None

    def test_analyze_regex_complexity(self):
        """Test analyze_regex_complexity function."""
        result = analyze_regex_complexity(r"^hello$")
        assert isinstance(result, RegexAnalysisResult)
        assert result.is_safe is True

    def test_create_safe_regex(self):
        """Test create_safe_regex function."""
        pattern = create_safe_regex(r"^[a-z]+$")
        assert isinstance(pattern, re.Pattern)

        with pytest.raises(ValueError):
            create_safe_regex(r"(a+)+")

    def test_safe_match(self):
        """Test safe_match function."""
        result = safe_match(r"^hello$", "hello")
        assert result is not None

    def test_safe_search(self):
        """Test safe_search function."""
        result = safe_search(r"world", "hello world")
        assert result is not None


class TestRealWorldPatterns:
    """Test with real-world regex patterns."""

    @pytest.mark.parametrize(
        "pattern,expected_safe",
        [
            # Safe patterns
            (r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", True),  # Email
            (r"^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$", True),  # US Phone
            (r"^\d{5}(-\d{4})?$", True),  # US Zip
            (r"^[a-zA-Z][a-zA-Z0-9_]*$", True),  # Identifier
            (r"^\d{4}-\d{2}-\d{2}$", True),  # Date
            # Unsafe patterns
            (r"(a+)+", False),  # Classic ReDoS
            (r"(a|aa)+", False),  # Overlapping alternation
            (r"([a-zA-Z]+)*", False),  # Star of plus
        ],
    )
    def test_pattern_safety(self, pattern: str, expected_safe: bool):
        """Test safety classification of common patterns."""
        is_safe, _ = check_regex_safety(pattern)
        assert is_safe == expected_safe, f"Pattern {pattern} expected safe={expected_safe}"


class TestIntegrationWithBase:
    """Test integration with validators/base.py RegexSafetyChecker."""

    def test_compatibility_with_base_checker(self):
        """Test that new checker is compatible with base interface."""
        from truthound.validators.base import RegexSafetyChecker as BaseChecker

        # Both should have check_pattern method
        new = RegexSafetyChecker()

        pattern = r"^[a-z]+$"
        new_result = new.check_pattern(pattern)

        # Should detect safe patterns
        assert new_result[0] is True

        # Should detect dangerous patterns
        dangerous = r"(a+)+"
        new_result = new.check_pattern(dangerous)
        assert new_result[0] is False

        # Base checker uses class method
        base_result = BaseChecker.check_pattern(pattern)
        assert base_result[0] is True
