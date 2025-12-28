"""Advanced ReDoS (Regular Expression Denial of Service) Protection.

This module provides comprehensive protection against ReDoS attacks:
- Static analysis of regex patterns for dangerous constructs
- Complexity estimation for potential exponential backtracking
- Safe regex compilation with configurable limits
- Runtime execution monitoring

ReDoS attacks exploit the exponential time complexity of certain regex
patterns, causing validation to hang or consume excessive CPU.

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                    ReDoS Protection Pipeline                    │
    └────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬────────────────┐
    │               │               │               │                │
    ▼               ▼               ▼               ▼                ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌────────┐
│ Static  │   │Complexity│    │ Quantifier│   │Alternation│    │ Safe   │
│ Analysis│   │ Estimator│    │ Analysis  │   │ Analysis  │    │ Compile│
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └────────┘

Usage:
    from truthound.validators.security.redos import (
        check_regex_safety,
        analyze_regex_complexity,
        create_safe_regex,
    )

    # Quick safety check
    is_safe, warning = check_regex_safety(r"(a+)+")
    # is_safe = False, warning = "Nested quantifiers detected"

    # Detailed analysis
    result = analyze_regex_complexity(r"^[a-z]+@[a-z]+\\.com$")
    print(result.risk_level)  # ReDoSRisk.LOW
    print(result.complexity_score)  # 2.5

    # Create safe regex with limits
    pattern = create_safe_regex(r"^\\w+$", max_length=1000)
"""

from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class ReDoSRisk(Enum):
    """Risk level for ReDoS vulnerability."""

    NONE = auto()      # No known vulnerability patterns
    LOW = auto()       # Minor concerns, likely safe
    MEDIUM = auto()    # Some concerning patterns, use with caution
    HIGH = auto()      # Dangerous patterns detected, avoid
    CRITICAL = auto()  # Known ReDoS pattern, reject


@dataclass(frozen=True)
class SafeRegexConfig:
    """Configuration for safe regex operations.

    Attributes:
        max_pattern_length: Maximum pattern length (chars)
        max_groups: Maximum capture groups allowed
        max_quantifier_range: Maximum {n,m} range (m-n)
        max_alternations: Maximum alternation branches
        max_nested_depth: Maximum nesting depth
        allow_backreferences: Whether to allow backreferences
        allow_lookaround: Whether to allow lookahead/lookbehind
        timeout_seconds: Max execution time for matching
        max_input_length: Maximum input string length to match
    """

    max_pattern_length: int = 1000
    max_groups: int = 20
    max_quantifier_range: int = 100
    max_alternations: int = 50
    max_nested_depth: int = 10
    allow_backreferences: bool = False
    allow_lookaround: bool = True
    timeout_seconds: float = 1.0
    max_input_length: int = 100_000

    @classmethod
    def strict(cls) -> "SafeRegexConfig":
        """Create strict configuration for untrusted patterns."""
        return cls(
            max_pattern_length=500,
            max_groups=10,
            max_quantifier_range=50,
            max_alternations=20,
            max_nested_depth=5,
            allow_backreferences=False,
            allow_lookaround=False,
            timeout_seconds=0.5,
            max_input_length=10_000,
        )

    @classmethod
    def lenient(cls) -> "SafeRegexConfig":
        """Create lenient configuration for trusted patterns."""
        return cls(
            max_pattern_length=5000,
            max_groups=50,
            max_quantifier_range=1000,
            max_alternations=100,
            max_nested_depth=20,
            allow_backreferences=True,
            allow_lookaround=True,
            timeout_seconds=5.0,
            max_input_length=1_000_000,
        )


@dataclass
class RegexAnalysisResult:
    """Result of regex pattern analysis.

    Attributes:
        pattern: The analyzed pattern
        risk_level: Overall ReDoS risk level
        complexity_score: Numeric complexity estimate (0-100)
        warnings: List of warning messages
        dangerous_constructs: List of detected dangerous constructs
        metrics: Detailed pattern metrics
        is_safe: Whether the pattern is considered safe
        recommendation: Suggested action or alternative
    """

    pattern: str
    risk_level: ReDoSRisk
    complexity_score: float
    warnings: list[str] = field(default_factory=list)
    dangerous_constructs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    is_safe: bool = True
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": self.pattern,
            "risk_level": self.risk_level.name,
            "complexity_score": round(self.complexity_score, 2),
            "warnings": self.warnings,
            "dangerous_constructs": self.dangerous_constructs,
            "metrics": self.metrics,
            "is_safe": self.is_safe,
            "recommendation": self.recommendation,
        }


class RegexComplexityAnalyzer:
    """Analyzes regex patterns for complexity and ReDoS vulnerability.

    This analyzer performs static analysis on regex patterns to detect
    potentially dangerous constructs that could lead to exponential
    backtracking (ReDoS attacks).

    Detection Categories:
    1. Nested Quantifiers: (a+)+ - exponential backtracking
    2. Overlapping Alternation: (a|a)+ - ambiguous matching
    3. Polynomial Backtracking: a*b*c*d* on non-matching input
    4. Atomic Group Absence: Patterns that would benefit from atomic groups
    5. Catastrophic Backreference: (a+)\\1+ with long inputs

    Example:
        analyzer = RegexComplexityAnalyzer()
        result = analyzer.analyze(r"(a+)+b")
        print(result.risk_level)  # ReDoSRisk.CRITICAL
        print(result.dangerous_constructs)  # ["nested_quantifiers"]
    """

    # Dangerous pattern signatures
    DANGEROUS_PATTERNS: list[tuple[str, str, ReDoSRisk]] = [
        # Nested quantifiers - exponential
        (r"\([^)]*[+*][^)]*\)[+*]", "nested_quantifiers", ReDoSRisk.CRITICAL),
        (r"\([^)]*[+*][^)]*\)\{[0-9]+,\}", "nested_quantifiers_bounded", ReDoSRisk.CRITICAL),

        # Nested groups with quantifiers
        (r"\(\([^)]*\)[+*]\)[+*]", "deeply_nested_quantifiers", ReDoSRisk.CRITICAL),

        # Overlapping character classes in alternation
        (r"\([^)]*\|[^)]*\)[+*]", "alternation_with_quantifier", ReDoSRisk.HIGH),

        # Backreference with quantifier
        (r"\\[0-9]+[+*]", "quantified_backreference", ReDoSRisk.HIGH),
        (r"\\[0-9]+\{[0-9]+,\}", "bounded_quantified_backreference", ReDoSRisk.HIGH),

        # Multiple adjacent quantifiers (greedy conflict)
        (r"[+*][+*]", "adjacent_quantifiers", ReDoSRisk.MEDIUM),

        # Long alternation chains
        (r"(?:\|[^|)]+){10,}", "long_alternation_chain", ReDoSRisk.MEDIUM),

        # Greedy quantifier followed by same pattern
        (r"\.+\.", "greedy_dot_conflict", ReDoSRisk.MEDIUM),
        (r"\.\*\.", "greedy_dotstar_conflict", ReDoSRisk.MEDIUM),

        # Unbounded repetition at start
        (r"^[+*]", "start_with_quantifier", ReDoSRisk.LOW),

        # Possessive/atomic group simulation (not actually supported in Python)
        (r"\(\?\>", "atomic_group_attempt", ReDoSRisk.LOW),
    ]

    # Quantifier patterns for extraction
    QUANTIFIER_PATTERN = re.compile(
        r"""
        (?:
            \+\??           |   # + or +?
            \*\??           |   # * or *?
            \?\??           |   # ? or ??
            \{(\d+)\}       |   # {n}
            \{(\d+),\}      |   # {n,}
            \{(\d+),(\d+)\}     # {n,m}
        )
        """,
        re.VERBOSE,
    )

    def __init__(self, config: SafeRegexConfig | None = None):
        """Initialize the analyzer.

        Args:
            config: Safety configuration
        """
        self.config = config or SafeRegexConfig()
        self._compile_dangerous_patterns()

    def _compile_dangerous_patterns(self) -> None:
        """Pre-compile dangerous pattern detectors."""
        self._compiled_patterns: list[tuple[re.Pattern, str, ReDoSRisk]] = []
        for pattern_str, name, risk in self.DANGEROUS_PATTERNS:
            try:
                compiled = re.compile(pattern_str)
                self._compiled_patterns.append((compiled, name, risk))
            except re.error:
                # Skip invalid patterns
                pass

    def analyze(self, pattern: str) -> RegexAnalysisResult:
        """Analyze a regex pattern for ReDoS vulnerability.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            RegexAnalysisResult with risk assessment
        """
        warnings: list[str] = []
        dangerous_constructs: list[str] = []
        max_risk = ReDoSRisk.NONE
        complexity_score = 0.0

        # Basic validation
        if not pattern:
            return RegexAnalysisResult(
                pattern=pattern,
                risk_level=ReDoSRisk.NONE,
                complexity_score=0.0,
                is_safe=True,
            )

        # Check pattern length
        if len(pattern) > self.config.max_pattern_length:
            warnings.append(
                f"Pattern length ({len(pattern)}) exceeds limit "
                f"({self.config.max_pattern_length})"
            )
            max_risk = max(max_risk, ReDoSRisk.MEDIUM, key=lambda r: r.value)
            complexity_score += 10

        # Extract metrics
        metrics = self._extract_metrics(pattern)

        # Check group count
        if metrics["group_count"] > self.config.max_groups:
            warnings.append(
                f"Too many groups ({metrics['group_count']} > {self.config.max_groups})"
            )
            complexity_score += 5

        # Check nesting depth
        if metrics["max_nesting"] > self.config.max_nested_depth:
            warnings.append(
                f"Nesting too deep ({metrics['max_nesting']} > {self.config.max_nested_depth})"
            )
            complexity_score += 15
            max_risk = max(max_risk, ReDoSRisk.MEDIUM, key=lambda r: r.value)

        # Check for backreferences
        if metrics["has_backreference"] and not self.config.allow_backreferences:
            warnings.append("Backreferences not allowed")
            dangerous_constructs.append("backreference")
            complexity_score += 20
            max_risk = max(max_risk, ReDoSRisk.HIGH, key=lambda r: r.value)

        # Check for lookaround
        if metrics["has_lookaround"] and not self.config.allow_lookaround:
            warnings.append("Lookaround assertions not allowed")
            complexity_score += 5

        # Check quantifier ranges
        for qmin, qmax in metrics.get("quantifier_ranges", []):
            if qmax is not None and qmax - qmin > self.config.max_quantifier_range:
                warnings.append(
                    f"Quantifier range too large: {{{qmin},{qmax}}}"
                )
                complexity_score += 10

        # Check for dangerous patterns
        for compiled, name, risk in self._compiled_patterns:
            if compiled.search(pattern):
                dangerous_constructs.append(name)
                max_risk = max(max_risk, risk, key=lambda r: r.value)
                complexity_score += self._risk_to_score(risk)

        # Additional heuristic checks
        complexity_score += self._analyze_quantifier_density(pattern)
        complexity_score += self._analyze_alternation_complexity(pattern)

        # Determine if safe
        is_safe = max_risk.value <= ReDoSRisk.LOW.value

        # Generate recommendation
        recommendation = self._generate_recommendation(
            max_risk, dangerous_constructs, warnings
        )

        return RegexAnalysisResult(
            pattern=pattern,
            risk_level=max_risk,
            complexity_score=min(complexity_score, 100),
            warnings=warnings,
            dangerous_constructs=dangerous_constructs,
            metrics=metrics,
            is_safe=is_safe,
            recommendation=recommendation,
        )

    def _extract_metrics(self, pattern: str) -> dict[str, Any]:
        """Extract metrics from pattern.

        Args:
            pattern: Regex pattern

        Returns:
            Dictionary of metrics
        """
        metrics: dict[str, Any] = {
            "length": len(pattern),
            "group_count": 0,
            "max_nesting": 0,
            "quantifier_count": 0,
            "alternation_count": pattern.count("|"),
            "has_backreference": bool(re.search(r"\\[1-9]", pattern)),
            "has_lookaround": bool(re.search(r"\(\?[=!<]", pattern)),
            "has_atomic": bool(re.search(r"\(\?>", pattern)),
            "quantifier_ranges": [],
        }

        # Count groups and nesting
        depth = 0
        max_depth = 0
        for char in pattern:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth = max(0, depth - 1)

        metrics["group_count"] = pattern.count("(") - pattern.count("(?")
        metrics["max_nesting"] = max_depth

        # Extract quantifier information
        for match in self.QUANTIFIER_PATTERN.finditer(pattern):
            metrics["quantifier_count"] += 1
            groups = match.groups()
            if groups[0]:  # {n}
                n = int(groups[0])
                metrics["quantifier_ranges"].append((n, n))
            elif groups[1]:  # {n,}
                n = int(groups[1])
                metrics["quantifier_ranges"].append((n, None))
            elif groups[2] and groups[3]:  # {n,m}
                n, m = int(groups[2]), int(groups[3])
                metrics["quantifier_ranges"].append((n, m))

        return metrics

    def _analyze_quantifier_density(self, pattern: str) -> float:
        """Analyze quantifier density for complexity.

        High density of quantifiers increases backtracking potential.
        """
        quantifier_chars = sum(1 for c in pattern if c in "+*?{}")
        if len(pattern) == 0:
            return 0
        density = quantifier_chars / len(pattern)
        return density * 20  # Scale to 0-20

    def _analyze_alternation_complexity(self, pattern: str) -> float:
        """Analyze alternation complexity.

        Overlapping alternatives can cause exponential matching.
        """
        if "|" not in pattern:
            return 0

        # Count alternations in quantified groups
        quantified_alt_pattern = r"\([^)]*\|[^)]*\)[+*?]"
        matches = re.findall(quantified_alt_pattern, pattern)

        return len(matches) * 15  # Each quantified alternation adds risk

    def _risk_to_score(self, risk: ReDoSRisk) -> float:
        """Convert risk level to complexity score contribution."""
        scores = {
            ReDoSRisk.NONE: 0,
            ReDoSRisk.LOW: 5,
            ReDoSRisk.MEDIUM: 15,
            ReDoSRisk.HIGH: 30,
            ReDoSRisk.CRITICAL: 50,
        }
        return scores.get(risk, 0)

    def _generate_recommendation(
        self,
        risk: ReDoSRisk,
        constructs: list[str],
        warnings: list[str],
    ) -> str:
        """Generate recommendation based on analysis.

        Args:
            risk: Overall risk level
            constructs: Dangerous constructs found
            warnings: Warning messages

        Returns:
            Recommendation string
        """
        if risk == ReDoSRisk.NONE:
            return "Pattern appears safe."

        if risk == ReDoSRisk.LOW:
            return "Pattern has minor concerns but is likely safe for typical inputs."

        if risk == ReDoSRisk.MEDIUM:
            return (
                "Pattern has moderate risk. Consider simplifying or adding input "
                "length limits."
            )

        if risk == ReDoSRisk.HIGH:
            recommendations = ["Pattern has high ReDoS risk. Consider:"]
            if "nested_quantifiers" in constructs:
                recommendations.append("- Avoid nested quantifiers like (a+)+")
            if "alternation_with_quantifier" in constructs:
                recommendations.append("- Avoid quantified alternation like (a|b)+")
            if "quantified_backreference" in constructs:
                recommendations.append("- Avoid quantified backreferences like (a+)\\1+")
            recommendations.append("- Use possessive quantifiers if available")
            recommendations.append("- Limit input length strictly")
            return "\n".join(recommendations)

        # CRITICAL
        return (
            "CRITICAL: Pattern contains known ReDoS vulnerability. "
            "Do NOT use with untrusted input. Rewrite the pattern to avoid "
            "nested quantifiers and overlapping alternatives."
        )


class RegexSafetyChecker:
    """High-level API for checking regex pattern safety.

    This class provides a simple interface for validating regex patterns
    before use. It combines static analysis with optional runtime testing.

    Example:
        checker = RegexSafetyChecker()

        # Quick check
        is_safe, warning = checker.check(r"^[a-z]+$")
        # is_safe = True, warning = None

        # Check dangerous pattern
        is_safe, warning = checker.check(r"(a+)+b")
        # is_safe = False, warning = "Nested quantifiers detected..."

        # Check with custom config
        config = SafeRegexConfig.strict()
        checker = RegexSafetyChecker(config)
    """

    def __init__(self, config: SafeRegexConfig | None = None):
        """Initialize the checker.

        Args:
            config: Safety configuration
        """
        self.config = config or SafeRegexConfig()
        self.analyzer = RegexComplexityAnalyzer(self.config)

    def check(self, pattern: str) -> tuple[bool, str | None]:
        """Check if a regex pattern is safe to use.

        Args:
            pattern: Regex pattern to check

        Returns:
            Tuple of (is_safe, warning_message)
        """
        # Length check
        if len(pattern) > self.config.max_pattern_length:
            return False, f"Pattern too long ({len(pattern)} > {self.config.max_pattern_length})"

        # Syntax validation
        try:
            re.compile(pattern)
        except re.error as e:
            return False, f"Invalid regex syntax: {e}"

        # Analyze for ReDoS
        result = self.analyzer.analyze(pattern)

        if not result.is_safe:
            warnings = "; ".join(result.warnings) if result.warnings else ""
            constructs = ", ".join(result.dangerous_constructs)
            message = f"ReDoS risk ({result.risk_level.name})"
            if constructs:
                message += f": {constructs}"
            if warnings:
                message += f". {warnings}"
            return False, message

        return True, None

    def check_pattern(self, pattern: str) -> tuple[bool, str | None]:
        """Alias for check() for backward compatibility."""
        return self.check(pattern)

    def analyze(self, pattern: str) -> RegexAnalysisResult:
        """Get detailed analysis of a pattern.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            RegexAnalysisResult with full details
        """
        return self.analyzer.analyze(pattern)

    def validate_and_compile(
        self,
        pattern: str,
        flags: int = 0,
    ) -> re.Pattern:
        """Validate pattern and compile if safe.

        Args:
            pattern: Regex pattern
            flags: Regex flags

        Returns:
            Compiled pattern

        Raises:
            ValueError: If pattern is unsafe or invalid
        """
        is_safe, warning = self.check(pattern)
        if not is_safe:
            raise ValueError(f"Unsafe regex pattern: {warning}")

        return re.compile(pattern, flags)


class SafeRegexExecutor:
    """Execute regex matching with timeout protection.

    This class wraps regex operations to prevent ReDoS by enforcing
    timeouts on matching operations.

    Example:
        executor = SafeRegexExecutor(timeout_seconds=1.0)

        # Safe execution
        result = executor.match(r"^[a-z]+$", "hello")
        # result = <Match object>

        # Timeout on dangerous pattern
        result = executor.match(r"(a+)+b", "a" * 30)
        # Raises TimeoutError after 1 second
    """

    def __init__(
        self,
        timeout_seconds: float = 1.0,
        max_input_length: int = 100_000,
    ):
        """Initialize the executor.

        Args:
            timeout_seconds: Maximum execution time
            max_input_length: Maximum input string length
        """
        self.timeout_seconds = timeout_seconds
        self.max_input_length = max_input_length

    def match(
        self,
        pattern: str | re.Pattern,
        string: str,
        flags: int = 0,
    ) -> re.Match | None:
        """Execute regex match with timeout.

        Args:
            pattern: Regex pattern or compiled pattern
            string: String to match
            flags: Regex flags (if pattern is string)

        Returns:
            Match object or None

        Raises:
            TimeoutError: If matching exceeds timeout
            ValueError: If input exceeds max length
        """
        if len(string) > self.max_input_length:
            raise ValueError(
                f"Input too long ({len(string)} > {self.max_input_length})"
            )

        if isinstance(pattern, str):
            compiled = re.compile(pattern, flags)
        else:
            compiled = pattern

        return self._execute_with_timeout(compiled.match, string)

    def search(
        self,
        pattern: str | re.Pattern,
        string: str,
        flags: int = 0,
    ) -> re.Match | None:
        """Execute regex search with timeout.

        Args:
            pattern: Regex pattern or compiled pattern
            string: String to search
            flags: Regex flags

        Returns:
            Match object or None

        Raises:
            TimeoutError: If search exceeds timeout
        """
        if len(string) > self.max_input_length:
            raise ValueError(
                f"Input too long ({len(string)} > {self.max_input_length})"
            )

        if isinstance(pattern, str):
            compiled = re.compile(pattern, flags)
        else:
            compiled = pattern

        return self._execute_with_timeout(compiled.search, string)

    def findall(
        self,
        pattern: str | re.Pattern,
        string: str,
        flags: int = 0,
    ) -> list[Any]:
        """Execute regex findall with timeout.

        Args:
            pattern: Regex pattern or compiled pattern
            string: String to search
            flags: Regex flags

        Returns:
            List of matches

        Raises:
            TimeoutError: If operation exceeds timeout
        """
        if len(string) > self.max_input_length:
            raise ValueError(
                f"Input too long ({len(string)} > {self.max_input_length})"
            )

        if isinstance(pattern, str):
            compiled = re.compile(pattern, flags)
        else:
            compiled = pattern

        return self._execute_with_timeout(compiled.findall, string)

    def _execute_with_timeout(
        self,
        func: Callable,
        *args: Any,
    ) -> Any:
        """Execute function with timeout.

        Uses threading for cross-platform timeout support.

        Args:
            func: Function to execute
            *args: Function arguments

        Returns:
            Function result

        Raises:
            TimeoutError: If execution exceeds timeout
        """
        result: list[Any] = [None]
        exception: list[Exception | None] = [None]
        completed = threading.Event()

        def target() -> None:
            try:
                result[0] = func(*args)
            except Exception as e:
                exception[0] = e
            finally:
                completed.set()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        if not completed.wait(timeout=self.timeout_seconds):
            raise TimeoutError(
                f"Regex operation timed out after {self.timeout_seconds}s"
            )

        if exception[0]:
            raise exception[0]

        return result[0]


# ============================================================================
# Module-level convenience functions
# ============================================================================


def check_regex_safety(
    pattern: str,
    config: SafeRegexConfig | None = None,
) -> tuple[bool, str | None]:
    """Check if a regex pattern is safe to use.

    Args:
        pattern: Regex pattern to check
        config: Optional safety configuration

    Returns:
        Tuple of (is_safe, warning_message)

    Example:
        is_safe, warning = check_regex_safety(r"(a+)+b")
        # is_safe = False, warning = "ReDoS risk (CRITICAL): nested_quantifiers"
    """
    checker = RegexSafetyChecker(config)
    return checker.check(pattern)


def analyze_regex_complexity(
    pattern: str,
    config: SafeRegexConfig | None = None,
) -> RegexAnalysisResult:
    """Get detailed complexity analysis of a regex pattern.

    Args:
        pattern: Regex pattern to analyze
        config: Optional safety configuration

    Returns:
        RegexAnalysisResult with full analysis

    Example:
        result = analyze_regex_complexity(r"^[a-z]+@[a-z]+\\.com$")
        print(result.risk_level)  # ReDoSRisk.LOW
        print(result.complexity_score)  # 2.5
    """
    analyzer = RegexComplexityAnalyzer(config)
    return analyzer.analyze(pattern)


def create_safe_regex(
    pattern: str,
    flags: int = 0,
    config: SafeRegexConfig | None = None,
) -> re.Pattern:
    """Create a compiled regex pattern after safety validation.

    Args:
        pattern: Regex pattern to compile
        flags: Regex flags
        config: Optional safety configuration

    Returns:
        Compiled regex pattern

    Raises:
        ValueError: If pattern is unsafe or invalid

    Example:
        try:
            compiled = create_safe_regex(r"^[a-z]+$")
            # Use compiled pattern...
        except ValueError as e:
            print(f"Unsafe pattern: {e}")
    """
    checker = RegexSafetyChecker(config)
    return checker.validate_and_compile(pattern, flags)


def safe_match(
    pattern: str,
    string: str,
    timeout: float = 1.0,
    flags: int = 0,
) -> re.Match | None:
    """Execute regex match with timeout protection.

    Args:
        pattern: Regex pattern
        string: String to match
        timeout: Maximum execution time in seconds
        flags: Regex flags

    Returns:
        Match object or None

    Raises:
        TimeoutError: If matching exceeds timeout
        ValueError: If input is too long

    Example:
        result = safe_match(r"^[a-z]+$", "hello", timeout=0.5)
        if result:
            print("Matched!")
    """
    executor = SafeRegexExecutor(timeout_seconds=timeout)
    return executor.match(pattern, string, flags)


def safe_search(
    pattern: str,
    string: str,
    timeout: float = 1.0,
    flags: int = 0,
) -> re.Match | None:
    """Execute regex search with timeout protection.

    Args:
        pattern: Regex pattern
        string: String to search
        timeout: Maximum execution time in seconds
        flags: Regex flags

    Returns:
        Match object or None

    Raises:
        TimeoutError: If search exceeds timeout
    """
    executor = SafeRegexExecutor(timeout_seconds=timeout)
    return executor.search(pattern, string, flags)
