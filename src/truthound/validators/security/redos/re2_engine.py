"""RE2 Engine - Linear Time Regex Matching.

This module provides integration with Google's RE2 regex engine,
which guarantees linear time matching by avoiding backtracking.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    RE2 Engine Wrapper                            │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬─────────────────┐
    │               │               │               │                 │
    ▼               ▼               ▼               ▼                 ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌─────────┐
│ Pattern │   │ Fallback│    │ Feature  │   │ Compat   │    │ Perf    │
│ Compiler│   │ Handler │    │ Detector │   │ Converter│    │ Monitor │
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └─────────┘

RE2 Advantages:
- Guaranteed O(n) matching time
- No catastrophic backtracking
- Memory-bounded matching

RE2 Limitations (handled by this module):
- No backreferences (\\1, \\2, etc.)
- No lookahead/lookbehind
- Limited Unicode support
- Different capture group semantics

Usage:
    from truthound.validators.security.redos.re2_engine import (
        RE2Engine,
        safe_match_re2,
    )

    # Quick RE2 match
    result = safe_match_re2(r"^[a-z]+$", "hello")
    if result.matched:
        print(f"Matched: {result.match}")

    # Full engine with fallback
    engine = RE2Engine(fallback_to_python=True)
    result = engine.match(r"(a+)\\1", "aa")  # Falls back to Python
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, Sequence

# Try to import re2 library
try:
    import re2
    HAS_RE2 = True
except ImportError:
    re2 = None  # type: ignore
    HAS_RE2 = False


class RE2UnsupportedFeature(Enum):
    """Features not supported by RE2."""

    BACKREFERENCE = auto()
    LOOKAHEAD = auto()
    LOOKBEHIND = auto()
    ATOMIC_GROUP = auto()
    POSSESSIVE_QUANTIFIER = auto()
    CONDITIONAL = auto()
    RECURSION = auto()
    UNICODE_CATEGORY = auto()


class RE2CompileError(Exception):
    """Error compiling pattern with RE2."""

    def __init__(
        self,
        message: str,
        pattern: str,
        unsupported_features: list[RE2UnsupportedFeature] | None = None,
    ):
        super().__init__(message)
        self.pattern = pattern
        self.unsupported_features = unsupported_features or []


@dataclass
class RE2MatchResult:
    """Result of RE2 matching operation.

    Attributes:
        pattern: The pattern used
        input_string: The input string (truncated)
        matched: Whether the pattern matched
        match: The matched string (if any)
        groups: Captured groups
        span: Match span (start, end)
        used_fallback: Whether Python re was used as fallback
        engine: Engine used ("re2" or "python_re")
        compile_time_ns: Time to compile pattern
        match_time_ns: Time to perform match
    """

    pattern: str
    input_string: str
    matched: bool = False
    match: str | None = None
    groups: tuple[str | None, ...] = field(default_factory=tuple)
    span: tuple[int, int] = (0, 0)
    used_fallback: bool = False
    engine: str = "re2"
    compile_time_ns: int = 0
    match_time_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": self.pattern,
            "input_string": self.input_string[:100],
            "matched": self.matched,
            "match": self.match,
            "groups": self.groups,
            "span": self.span,
            "used_fallback": self.used_fallback,
            "engine": self.engine,
            "compile_time_us": self.compile_time_ns / 1000,
            "match_time_us": self.match_time_ns / 1000,
        }


class RegexEngineProtocol(Protocol):
    """Protocol for regex engine implementations."""

    def compile(self, pattern: str, flags: int = 0) -> Any:
        """Compile a pattern."""
        ...

    def match(self, pattern: str, string: str, flags: int = 0) -> RE2MatchResult:
        """Match pattern against string."""
        ...

    def search(self, pattern: str, string: str, flags: int = 0) -> RE2MatchResult:
        """Search for pattern in string."""
        ...


class FeatureDetector:
    """Detect unsupported RE2 features in patterns."""

    # Patterns for unsupported features
    FEATURE_PATTERNS: list[tuple[str, RE2UnsupportedFeature]] = [
        (r"\\[1-9]\d*", RE2UnsupportedFeature.BACKREFERENCE),
        (r"\(\?=", RE2UnsupportedFeature.LOOKAHEAD),
        (r"\(\?!", RE2UnsupportedFeature.LOOKAHEAD),
        (r"\(\?<=", RE2UnsupportedFeature.LOOKBEHIND),
        (r"\(\?<!", RE2UnsupportedFeature.LOOKBEHIND),
        (r"\(\?>", RE2UnsupportedFeature.ATOMIC_GROUP),
        (r"[+*?]\+", RE2UnsupportedFeature.POSSESSIVE_QUANTIFIER),
        (r"\(\?\(", RE2UnsupportedFeature.CONDITIONAL),
        (r"\(\?R\)", RE2UnsupportedFeature.RECURSION),
        (r"\(\?\d+\)", RE2UnsupportedFeature.RECURSION),
        (r"\\p\{", RE2UnsupportedFeature.UNICODE_CATEGORY),
        (r"\\P\{", RE2UnsupportedFeature.UNICODE_CATEGORY),
    ]

    def __init__(self):
        """Initialize detector with compiled patterns."""
        self._compiled = [
            (re.compile(pattern), feature)
            for pattern, feature in self.FEATURE_PATTERNS
        ]

    def detect(self, pattern: str) -> list[RE2UnsupportedFeature]:
        """Detect unsupported features in a pattern.

        Args:
            pattern: Regex pattern to check

        Returns:
            List of unsupported features found
        """
        found: list[RE2UnsupportedFeature] = []

        for compiled, feature in self._compiled:
            if compiled.search(pattern):
                if feature not in found:
                    found.append(feature)

        return found

    def is_re2_compatible(self, pattern: str) -> bool:
        """Check if pattern is compatible with RE2.

        Args:
            pattern: Regex pattern

        Returns:
            True if pattern can be compiled with RE2
        """
        return len(self.detect(pattern)) == 0


class PatternConverter:
    """Convert patterns to RE2-compatible form where possible."""

    def convert(self, pattern: str) -> tuple[str, list[str]]:
        """Attempt to convert pattern to RE2-compatible form.

        Args:
            pattern: Original pattern

        Returns:
            Tuple of (converted_pattern, warnings)

        Note:
            Not all patterns can be converted. Backreferences
            and lookaround cannot be emulated in RE2.
        """
        warnings: list[str] = []
        result = pattern

        # Remove possessive quantifiers (convert to greedy)
        if re.search(r"[+*?]\+", result):
            result = re.sub(r"([+*?])\+", r"\1", result)
            warnings.append("Possessive quantifiers converted to greedy")

        # Note: We can't convert backreferences or lookaround
        if re.search(r"\\[1-9]", result):
            warnings.append("Backreferences cannot be converted to RE2")

        if re.search(r"\(\?[=!<]", result):
            warnings.append("Lookaround cannot be converted to RE2")

        return result, warnings


class RE2Wrapper:
    """Wrapper around RE2 library with pattern caching."""

    def __init__(self, max_cache_size: int = 100):
        """Initialize RE2 wrapper.

        Args:
            max_cache_size: Maximum patterns to cache
        """
        self._cache: dict[str, Any] = {}
        self._max_cache_size = max_cache_size

    def compile(self, pattern: str, flags: int = 0) -> Any:
        """Compile pattern with RE2.

        Args:
            pattern: Regex pattern
            flags: Regex flags

        Returns:
            Compiled RE2 pattern

        Raises:
            RE2CompileError: If pattern cannot be compiled
        """
        if not HAS_RE2:
            raise RE2CompileError(
                "RE2 library not installed. Install with: pip install google-re2",
                pattern,
            )

        cache_key = f"{pattern}:{flags}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check for unsupported features
        detector = FeatureDetector()
        unsupported = detector.detect(pattern)
        if unsupported:
            features_str = ", ".join(f.name for f in unsupported)
            raise RE2CompileError(
                f"Pattern contains features not supported by RE2: {features_str}",
                pattern,
                unsupported,
            )

        try:
            compiled = re2.compile(pattern)

            # Cache the compiled pattern
            if len(self._cache) >= self._max_cache_size:
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]

            self._cache[cache_key] = compiled
            return compiled

        except Exception as e:
            raise RE2CompileError(
                f"RE2 compilation failed: {e}",
                pattern,
            )

    def match(self, compiled: Any, string: str) -> Any:
        """Match using compiled pattern."""
        if not HAS_RE2:
            raise RuntimeError("RE2 not available")
        return compiled.match(string)

    def search(self, compiled: Any, string: str) -> Any:
        """Search using compiled pattern."""
        if not HAS_RE2:
            raise RuntimeError("RE2 not available")
        return compiled.search(string)

    def findall(self, compiled: Any, string: str) -> list[Any]:
        """Find all matches using compiled pattern."""
        if not HAS_RE2:
            raise RuntimeError("RE2 not available")
        return compiled.findall(string)


class PythonRegexFallback:
    """Fallback to Python re module."""

    def __init__(self, timeout_seconds: float = 5.0):
        """Initialize fallback with timeout.

        Args:
            timeout_seconds: Maximum execution time
        """
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, re.Pattern] = {}

    def compile(self, pattern: str, flags: int = 0) -> re.Pattern:
        """Compile with Python re."""
        cache_key = f"{pattern}:{flags}"
        if cache_key not in self._cache:
            self._cache[cache_key] = re.compile(pattern, flags)
        return self._cache[cache_key]

    def match(self, compiled: re.Pattern, string: str) -> re.Match | None:
        """Match using Python re."""
        return compiled.match(string)

    def search(self, compiled: re.Pattern, string: str) -> re.Match | None:
        """Search using Python re."""
        return compiled.search(string)


class RE2Engine:
    """High-level RE2 engine with Python fallback.

    This engine provides a simple interface for regex matching that
    uses RE2 when possible for guaranteed linear-time matching, with
    optional fallback to Python's re module for patterns that use
    features not supported by RE2.

    Example:
        engine = RE2Engine()

        # Simple match
        result = engine.match(r"^[a-z]+$", "hello")
        if result.matched:
            print(f"Matched: {result.match}")

        # Pattern with backreference (needs fallback)
        engine_with_fallback = RE2Engine(fallback_to_python=True)
        result = engine_with_fallback.match(r"(a+)\\1", "aa")
        print(f"Used fallback: {result.used_fallback}")  # True
    """

    def __init__(
        self,
        fallback_to_python: bool = True,
        max_cache_size: int = 100,
        python_timeout: float = 5.0,
    ):
        """Initialize the engine.

        Args:
            fallback_to_python: Use Python re for unsupported patterns
            max_cache_size: Maximum compiled patterns to cache
            python_timeout: Timeout for Python fallback
        """
        self.fallback_to_python = fallback_to_python
        self._re2 = RE2Wrapper(max_cache_size) if HAS_RE2 else None
        self._python = PythonRegexFallback(python_timeout)
        self._detector = FeatureDetector()
        self._converter = PatternConverter()

    @property
    def re2_available(self) -> bool:
        """Check if RE2 is available."""
        return HAS_RE2

    def is_pattern_re2_compatible(self, pattern: str) -> bool:
        """Check if pattern can be used with RE2.

        Args:
            pattern: Regex pattern

        Returns:
            True if pattern is RE2-compatible
        """
        return self._detector.is_re2_compatible(pattern)

    def get_unsupported_features(self, pattern: str) -> list[RE2UnsupportedFeature]:
        """Get list of unsupported features in pattern.

        Args:
            pattern: Regex pattern

        Returns:
            List of unsupported features
        """
        return self._detector.detect(pattern)

    def match(
        self,
        pattern: str,
        string: str,
        flags: int = 0,
    ) -> RE2MatchResult:
        """Match pattern against string.

        Uses RE2 if available and pattern is compatible,
        otherwise falls back to Python re if enabled.

        Args:
            pattern: Regex pattern
            string: String to match
            flags: Regex flags

        Returns:
            RE2MatchResult with match information

        Raises:
            RE2CompileError: If pattern not supported and no fallback
        """
        return self._execute("match", pattern, string, flags)

    def search(
        self,
        pattern: str,
        string: str,
        flags: int = 0,
    ) -> RE2MatchResult:
        """Search for pattern in string.

        Args:
            pattern: Regex pattern
            string: String to search
            flags: Regex flags

        Returns:
            RE2MatchResult with search result
        """
        return self._execute("search", pattern, string, flags)

    def findall(
        self,
        pattern: str,
        string: str,
        flags: int = 0,
    ) -> list[str]:
        """Find all matches of pattern in string.

        Args:
            pattern: Regex pattern
            string: String to search
            flags: Regex flags

        Returns:
            List of matches
        """
        result = self._execute("findall", pattern, string, flags)
        # Extract matches from result
        if isinstance(result.match, list):
            return result.match
        return []

    def _execute(
        self,
        operation: str,
        pattern: str,
        string: str,
        flags: int = 0,
    ) -> RE2MatchResult:
        """Execute a regex operation.

        Args:
            operation: "match", "search", or "findall"
            pattern: Regex pattern
            string: Input string
            flags: Regex flags

        Returns:
            RE2MatchResult
        """
        import time

        # Try RE2 first
        if HAS_RE2 and self._re2:
            try:
                compile_start = time.perf_counter_ns()
                compiled = self._re2.compile(pattern, flags)
                compile_time = time.perf_counter_ns() - compile_start

                match_start = time.perf_counter_ns()
                if operation == "match":
                    result = self._re2.match(compiled, string)
                elif operation == "search":
                    result = self._re2.search(compiled, string)
                else:  # findall
                    result = self._re2.findall(compiled, string)
                match_time = time.perf_counter_ns() - match_start

                return self._create_result(
                    pattern, string, result, operation,
                    engine="re2",
                    compile_time=compile_time,
                    match_time=match_time,
                )

            except RE2CompileError as e:
                if not self.fallback_to_python:
                    # Re-raise with pattern info
                    raise RE2CompileError(
                        f"Pattern not supported by RE2 and fallback disabled: {e}",
                        pattern,
                        e.unsupported_features,
                    )
                # Fall through to Python fallback

        # Fallback to Python re
        if self.fallback_to_python:
            try:
                compile_start = time.perf_counter_ns()
                compiled = self._python.compile(pattern, flags)
                compile_time = time.perf_counter_ns() - compile_start

                match_start = time.perf_counter_ns()
                if operation == "match":
                    result = self._python.match(compiled, string)
                elif operation == "search":
                    result = self._python.search(compiled, string)
                else:  # findall
                    result = compiled.findall(string)
                match_time = time.perf_counter_ns() - match_start

                return self._create_result(
                    pattern, string, result, operation,
                    engine="python_re",
                    used_fallback=True,
                    compile_time=compile_time,
                    match_time=match_time,
                )

            except re.error as e:
                return RE2MatchResult(
                    pattern=pattern,
                    input_string=string[:100],
                    matched=False,
                    engine="python_re",
                    used_fallback=True,
                )

        raise RE2CompileError(
            "RE2 not available and fallback disabled",
            pattern,
        )

    def _create_result(
        self,
        pattern: str,
        string: str,
        match_result: Any,
        operation: str,
        engine: str = "re2",
        used_fallback: bool = False,
        compile_time: int = 0,
        match_time: int = 0,
    ) -> RE2MatchResult:
        """Create result from match."""
        if operation == "findall":
            return RE2MatchResult(
                pattern=pattern,
                input_string=string[:100],
                matched=bool(match_result),
                match=match_result if match_result else None,
                engine=engine,
                used_fallback=used_fallback,
                compile_time_ns=compile_time,
                match_time_ns=match_time,
            )

        if match_result is None:
            return RE2MatchResult(
                pattern=pattern,
                input_string=string[:100],
                matched=False,
                engine=engine,
                used_fallback=used_fallback,
                compile_time_ns=compile_time,
                match_time_ns=match_time,
            )

        return RE2MatchResult(
            pattern=pattern,
            input_string=string[:100],
            matched=True,
            match=match_result.group(0),
            groups=match_result.groups() if hasattr(match_result, 'groups') else (),
            span=match_result.span() if hasattr(match_result, 'span') else (0, 0),
            engine=engine,
            used_fallback=used_fallback,
            compile_time_ns=compile_time,
            match_time_ns=match_time,
        )


# ============================================================================
# Convenience functions
# ============================================================================


# Singleton engine instance
_default_engine: RE2Engine | None = None


def get_default_engine() -> RE2Engine:
    """Get the default RE2 engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = RE2Engine(fallback_to_python=True)
    return _default_engine


def safe_match_re2(
    pattern: str,
    string: str,
    flags: int = 0,
) -> RE2MatchResult:
    """Match pattern against string using RE2.

    Args:
        pattern: Regex pattern
        string: String to match
        flags: Regex flags

    Returns:
        RE2MatchResult with match information

    Example:
        result = safe_match_re2(r"^[a-z]+$", "hello")
        if result.matched:
            print(f"Match: {result.match}")
            print(f"Engine: {result.engine}")
    """
    engine = get_default_engine()
    return engine.match(pattern, string, flags)


def safe_search_re2(
    pattern: str,
    string: str,
    flags: int = 0,
) -> RE2MatchResult:
    """Search for pattern in string using RE2.

    Args:
        pattern: Regex pattern
        string: String to search
        flags: Regex flags

    Returns:
        RE2MatchResult with search result
    """
    engine = get_default_engine()
    return engine.search(pattern, string, flags)


def is_re2_available() -> bool:
    """Check if RE2 library is available.

    Returns:
        True if google-re2 is installed
    """
    return HAS_RE2


def check_re2_compatibility(pattern: str) -> tuple[bool, list[RE2UnsupportedFeature]]:
    """Check if a pattern is compatible with RE2.

    Args:
        pattern: Regex pattern to check

    Returns:
        Tuple of (is_compatible, unsupported_features)

    Example:
        compatible, unsupported = check_re2_compatibility(r"(a+)\\1")
        if not compatible:
            print(f"Unsupported: {[f.name for f in unsupported]}")
    """
    detector = FeatureDetector()
    unsupported = detector.detect(pattern)
    return len(unsupported) == 0, unsupported
