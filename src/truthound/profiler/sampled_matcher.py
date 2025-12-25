"""Memory-safe pattern matcher with integrated sampling.

This module provides a pattern matcher that integrates sampling strategies
to prevent OOM errors while maintaining statistical accuracy.

Key features:
- Configurable sampling strategies
- Memory-aware processing
- Statistical confidence reporting
- Graceful degradation on failures
- Telemetry integration

Example:
    from truthound.profiler.sampled_matcher import (
        SampledPatternMatcher,
        SampledMatcherConfig,
    )

    matcher = SampledPatternMatcher(
        sampling_config=SamplingConfig.for_accuracy("high"),
    )

    results = matcher.match_column(lf, "email")
    print(f"Confidence: {results.confidence:.2%}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import polars as pl

from truthound.profiler.base import DataType, PatternMatch
from truthound.profiler.native_patterns import (
    BUILTIN_PATTERNS,
    NativePatternMatcher,
    PatternMatchResult,
    PatternRegistry,
    PatternSpec,
)
from truthound.profiler.sampling import (
    DEFAULT_SAMPLING_CONFIG,
    DataSizeEstimator,
    Sampler,
    SamplingConfig,
    SamplingMetrics,
    SamplingMethod,
    SamplingResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sampled Match Result
# =============================================================================


@dataclass
class SampledPatternMatchResult:
    """Pattern match result with sampling metadata.

    Extends PatternMatchResult with statistical confidence
    information from sampling.

    Attributes:
        pattern: The matched pattern specification
        match_count: Number of matches in sample
        total_count: Total non-null values in sample
        match_ratio: Ratio of matches in sample
        sample_matches: Example matching values
        sample_non_matches: Example non-matching values
        sampling_metrics: Metrics from sampling operation
        estimated_population_matches: Extrapolated matches in full data
        confidence_interval: (lower, upper) bounds for match ratio
    """

    pattern: PatternSpec
    match_count: int
    total_count: int
    match_ratio: float
    sample_matches: tuple[str, ...] = field(default_factory=tuple)
    sample_non_matches: tuple[str, ...] = field(default_factory=tuple)
    sampling_metrics: SamplingMetrics | None = None
    estimated_population_matches: int = 0
    confidence_interval: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        if self.sampling_metrics and self.sampling_metrics.original_size > 0:
            # Extrapolate to full population
            self.estimated_population_matches = int(
                self.match_ratio * self.sampling_metrics.original_size
            )

            # Calculate confidence interval using Wilson score
            self._calculate_confidence_interval()

    def _calculate_confidence_interval(self) -> None:
        """Calculate Wilson score confidence interval."""
        if self.total_count == 0:
            self.confidence_interval = (0.0, 1.0)
            return

        n = self.total_count
        p = self.match_ratio

        # Z-score for confidence level (default 95%)
        z = 1.96
        if self.sampling_metrics:
            z = self._z_from_confidence(self.sampling_metrics.confidence_level)

        # Wilson score interval
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator

        lower = max(0.0, center - spread)
        upper = min(1.0, center + spread)

        self.confidence_interval = (lower, upper)

    @staticmethod
    def _z_from_confidence(confidence: float) -> float:
        """Get Z-score from confidence level."""
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
            0.999: 3.291,
        }
        return z_scores.get(round(confidence, 3), 1.96)

    @property
    def confidence(self) -> float:
        """Get confidence in the match ratio estimate."""
        if self.sampling_metrics:
            return self.sampling_metrics.confidence_level
        return 1.0  # No sampling = full confidence

    @property
    def is_sampled(self) -> bool:
        """Check if result is from sampled data."""
        return self.sampling_metrics is not None and self.sampling_metrics.is_full_scan is False

    @property
    def margin_of_error(self) -> float:
        """Get margin of error for match ratio."""
        lower, upper = self.confidence_interval
        return (upper - lower) / 2

    def to_pattern_match(self) -> PatternMatch:
        """Convert to legacy PatternMatch format."""
        return PatternMatch(
            pattern=self.pattern.name,
            regex=self.pattern.regex,
            match_ratio=self.match_ratio,
            sample_matches=self.sample_matches,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_name": self.pattern.name,
            "pattern_regex": self.pattern.regex,
            "match_count": self.match_count,
            "total_count": self.total_count,
            "match_ratio": self.match_ratio,
            "confidence": self.confidence,
            "confidence_interval": list(self.confidence_interval),
            "margin_of_error": self.margin_of_error,
            "is_sampled": self.is_sampled,
            "estimated_population_matches": self.estimated_population_matches,
            "sample_matches": list(self.sample_matches),
            "sampling_metrics": (
                self.sampling_metrics.to_dict() if self.sampling_metrics else None
            ),
        }


@dataclass
class SampledColumnMatchResult:
    """Complete result for a column including all matches and metadata."""

    column: str
    matches: list[SampledPatternMatchResult]
    sampling_metrics: SamplingMetrics | None
    processing_time_ms: float
    inferred_type: DataType | None = None

    @property
    def has_matches(self) -> bool:
        """Check if any patterns matched."""
        return len(self.matches) > 0

    @property
    def best_match(self) -> SampledPatternMatchResult | None:
        """Get the best (highest ratio) match."""
        if not self.matches:
            return None
        return self.matches[0]

    @property
    def is_sampled(self) -> bool:
        """Check if sampling was applied."""
        return (
            self.sampling_metrics is not None
            and not self.sampling_metrics.is_full_scan
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "matches": [m.to_dict() for m in self.matches],
            "sampling_metrics": (
                self.sampling_metrics.to_dict() if self.sampling_metrics else None
            ),
            "processing_time_ms": self.processing_time_ms,
            "inferred_type": self.inferred_type.value if self.inferred_type else None,
            "is_sampled": self.is_sampled,
        }


# =============================================================================
# Sampled Pattern Matcher Configuration
# =============================================================================


@dataclass
class SampledMatcherConfig:
    """Configuration for SampledPatternMatcher.

    Attributes:
        sampling_config: Sampling configuration
        patterns: Pattern registry to use
        min_match_ratio: Minimum ratio to consider a match
        sample_size: Number of sample values to collect
        include_non_matches: Whether to collect non-matching samples
        parallel_threshold: Row count above which to use parallel processing
        fallback_on_error: Whether to fallback to head sampling on error
        cache_sampling_decisions: Cache sampling decisions for same data
    """

    sampling_config: SamplingConfig = field(default_factory=lambda: DEFAULT_SAMPLING_CONFIG)
    patterns: PatternRegistry | None = None
    min_match_ratio: float = 0.8
    sample_size: int = 5
    include_non_matches: bool = False
    parallel_threshold: int = 100_000
    fallback_on_error: bool = True
    cache_sampling_decisions: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.min_match_ratio <= 1.0:
            raise ValueError(
                f"min_match_ratio must be between 0 and 1, got {self.min_match_ratio}"
            )
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be non-negative, got {self.sample_size}")

    @classmethod
    def fast(cls) -> "SampledMatcherConfig":
        """Create config optimized for speed."""
        return cls(
            sampling_config=SamplingConfig.for_speed(),
            min_match_ratio=0.7,
            sample_size=3,
        )

    @classmethod
    def accurate(cls) -> "SampledMatcherConfig":
        """Create config optimized for accuracy."""
        return cls(
            sampling_config=SamplingConfig.for_accuracy("high"),
            min_match_ratio=0.85,
            sample_size=10,
        )

    @classmethod
    def balanced(cls) -> "SampledMatcherConfig":
        """Create balanced config (default)."""
        return cls(
            sampling_config=SamplingConfig.for_accuracy("medium"),
            min_match_ratio=0.8,
            sample_size=5,
        )


# =============================================================================
# Sampled Pattern Matcher
# =============================================================================


class SampledPatternMatcher:
    """Memory-safe pattern matcher with integrated sampling.

    This is the recommended pattern matcher for production use.
    It automatically samples large datasets to prevent OOM errors
    while providing statistical confidence metrics.

    Example:
        # Basic usage
        matcher = SampledPatternMatcher()
        results = matcher.match_column(lf, "email")

        for result in results.matches:
            print(f"{result.pattern.name}: {result.match_ratio:.2%} "
                  f"(±{result.margin_of_error:.2%})")

        # Custom configuration
        config = SampledMatcherConfig(
            sampling_config=SamplingConfig(
                strategy=SamplingMethod.RANDOM,
                max_rows=50_000,
                confidence_level=0.99,
            ),
            min_match_ratio=0.9,
        )
        matcher = SampledPatternMatcher(config=config)

        # Memory-constrained environment
        matcher = SampledPatternMatcher(
            config=SampledMatcherConfig(
                sampling_config=SamplingConfig.for_memory(max_memory_mb=100)
            )
        )
    """

    def __init__(
        self,
        config: SampledMatcherConfig | None = None,
        sampling_config: SamplingConfig | None = None,
        patterns: PatternRegistry | None = None,
    ):
        """Initialize the sampled pattern matcher.

        Args:
            config: Full matcher configuration
            sampling_config: Override sampling config (convenience)
            patterns: Override pattern registry (convenience)
        """
        self.config = config or SampledMatcherConfig.balanced()

        # Allow convenience overrides
        if sampling_config is not None:
            self.config.sampling_config = sampling_config
        if patterns is not None:
            self.config.patterns = patterns

        # Initialize components
        self._sampler = Sampler(self.config.sampling_config)
        self._size_estimator = DataSizeEstimator()
        self._patterns = self.config.patterns or BUILTIN_PATTERNS

        # Internal matcher for actual pattern matching
        self._matcher = NativePatternMatcher(
            patterns=self._patterns,
            min_match_ratio=self.config.min_match_ratio,
            sample_size=self.config.sample_size,
            include_non_matches=self.config.include_non_matches,
        )

    @property
    def patterns(self) -> PatternRegistry:
        """Get the pattern registry."""
        return self._patterns

    @property
    def sampling_config(self) -> SamplingConfig:
        """Get the sampling configuration."""
        return self.config.sampling_config

    def match_column(
        self,
        lf: pl.LazyFrame,
        column: str,
        *,
        patterns: Sequence[PatternSpec] | None = None,
        sampling_config: SamplingConfig | None = None,
    ) -> SampledColumnMatchResult:
        """Match patterns against a column with automatic sampling.

        This is the main entry point. It will:
        1. Estimate data size
        2. Apply appropriate sampling strategy
        3. Run pattern matching on sample
        4. Calculate statistical confidence

        Args:
            lf: LazyFrame containing the data
            column: Column name to analyze
            patterns: Optional specific patterns to check
            sampling_config: Override sampling config for this call

        Returns:
            SampledColumnMatchResult with matches and metrics
        """
        start_time = time.perf_counter()

        # Use override or default config
        config = sampling_config or self.config.sampling_config

        # Step 1: Sample the data
        try:
            sampling_result = self._sample_column(lf, column, config)
        except Exception as e:
            logger.error(f"Sampling failed for column '{column}': {e}")
            if self.config.fallback_on_error:
                # Fallback to simple head sampling
                sampling_result = self._fallback_sample(lf, column, config)
            else:
                raise

        # Step 2: Run pattern matching on sampled data
        try:
            pattern_results = self._match_on_sample(
                sampling_result.data,
                column,
                patterns,
            )
        except Exception as e:
            logger.error(f"Pattern matching failed for column '{column}': {e}")
            pattern_results = []

        # Step 3: Convert to sampled results with confidence
        sampled_results = self._enhance_results(
            pattern_results,
            sampling_result.metrics,
        )

        # Step 4: Infer type from best match
        inferred_type = None
        if sampled_results:
            inferred_type = sampled_results[0].pattern.data_type

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SampledColumnMatchResult(
            column=column,
            matches=sampled_results,
            sampling_metrics=sampling_result.metrics,
            processing_time_ms=elapsed_ms,
            inferred_type=inferred_type,
        )

    def match_all_columns(
        self,
        lf: pl.LazyFrame,
        *,
        string_columns_only: bool = True,
        sampling_config: SamplingConfig | None = None,
    ) -> dict[str, SampledColumnMatchResult]:
        """Match patterns against all applicable columns.

        Args:
            lf: LazyFrame to analyze
            string_columns_only: Only analyze string columns
            sampling_config: Override sampling configuration

        Returns:
            Dictionary mapping column names to their results
        """
        schema = lf.collect_schema()
        results: dict[str, SampledColumnMatchResult] = {}

        for col_name, dtype in schema.items():
            if string_columns_only:
                if dtype not in {pl.String, pl.Utf8}:
                    continue

            result = self.match_column(
                lf,
                col_name,
                sampling_config=sampling_config,
            )

            if result.has_matches:
                results[col_name] = result

        return results

    def infer_type(
        self,
        lf: pl.LazyFrame,
        column: str,
        *,
        min_match_ratio: float | None = None,
    ) -> DataType | None:
        """Infer semantic type for a column.

        Args:
            lf: LazyFrame containing the data
            column: Column name to analyze
            min_match_ratio: Override minimum match ratio

        Returns:
            Inferred DataType or None
        """
        result = self.match_column(lf, column)
        return result.inferred_type

    def _sample_column(
        self,
        lf: pl.LazyFrame,
        column: str,
        config: SamplingConfig,
    ) -> SamplingResult:
        """Sample a single column for pattern matching."""
        # Select only the needed column for efficiency
        column_lf = lf.select(pl.col(column))
        return self._sampler.sample(column_lf, config)

    def _fallback_sample(
        self,
        lf: pl.LazyFrame,
        column: str,
        config: SamplingConfig,
    ) -> SamplingResult:
        """Fallback sampling using simple head."""
        fallback_config = SamplingConfig(
            strategy=config.fallback_strategy,
            max_rows=config.max_rows or 10_000,
            confidence_level=config.confidence_level,
            margin_of_error=config.margin_of_error,
        )
        column_lf = lf.select(pl.col(column))
        return self._sampler.sample(column_lf, fallback_config)

    def _match_on_sample(
        self,
        sampled_lf: pl.LazyFrame,
        column: str,
        patterns: Sequence[PatternSpec] | None,
    ) -> list[PatternMatchResult]:
        """Run pattern matching on sampled data."""
        return self._matcher.match_column(
            sampled_lf,
            column,
            patterns=patterns,
            limit=None,  # Already sampled
        )

    def _enhance_results(
        self,
        results: list[PatternMatchResult],
        sampling_metrics: SamplingMetrics,
    ) -> list[SampledPatternMatchResult]:
        """Enhance pattern results with sampling metadata."""
        enhanced = []

        for result in results:
            enhanced.append(
                SampledPatternMatchResult(
                    pattern=result.pattern,
                    match_count=result.match_count,
                    total_count=result.total_count,
                    match_ratio=result.match_ratio,
                    sample_matches=result.sample_matches,
                    sample_non_matches=result.sample_non_matches,
                    sampling_metrics=sampling_metrics,
                )
            )

        return enhanced


# =============================================================================
# Factory Functions
# =============================================================================


def create_sampled_matcher(
    strategy: str | SamplingMethod = "adaptive",
    max_rows: int = 100_000,
    min_match_ratio: float = 0.8,
    **kwargs: Any,
) -> SampledPatternMatcher:
    """Create a sampled pattern matcher with common options.

    Args:
        strategy: Sampling strategy
        max_rows: Maximum rows to sample
        min_match_ratio: Minimum match ratio threshold
        **kwargs: Additional SamplingConfig options

    Returns:
        Configured SampledPatternMatcher

    Example:
        matcher = create_sampled_matcher(
            strategy="random",
            max_rows=50_000,
            confidence_level=0.99,
        )
    """
    if isinstance(strategy, str):
        strategy = SamplingMethod(strategy)

    sampling_config = SamplingConfig(
        strategy=strategy,
        max_rows=max_rows,
        **kwargs,
    )

    config = SampledMatcherConfig(
        sampling_config=sampling_config,
        min_match_ratio=min_match_ratio,
    )

    return SampledPatternMatcher(config=config)


def match_patterns_safe(
    data: pl.LazyFrame | pl.DataFrame,
    column: str,
    *,
    max_rows: int = 100_000,
    min_ratio: float = 0.8,
) -> SampledColumnMatchResult:
    """Convenience function for safe pattern matching.

    Always applies sampling to prevent OOM.

    Args:
        data: DataFrame or LazyFrame
        column: Column name to analyze
        max_rows: Maximum rows to sample
        min_ratio: Minimum match ratio

    Returns:
        SampledColumnMatchResult

    Example:
        import polars as pl
        from truthound.profiler.sampled_matcher import match_patterns_safe

        df = pl.read_parquet("large_file.parquet")
        result = match_patterns_safe(df.lazy(), "email_column")

        print(f"Best match: {result.best_match.pattern.name}")
        print(f"Confidence: {result.best_match.confidence:.2%}")
    """
    if isinstance(data, pl.DataFrame):
        data = data.lazy()

    matcher = create_sampled_matcher(
        max_rows=max_rows,
        min_match_ratio=min_ratio,
    )

    return matcher.match_column(data, column)


def infer_column_type_safe(
    data: pl.LazyFrame | pl.DataFrame,
    column: str,
    *,
    max_rows: int = 100_000,
    min_ratio: float = 0.9,
) -> DataType | None:
    """Convenience function for safe type inference.

    Args:
        data: DataFrame or LazyFrame
        column: Column name
        max_rows: Maximum rows to sample
        min_ratio: Minimum match ratio for inference

    Returns:
        Inferred DataType or None

    Example:
        from truthound.profiler.sampled_matcher import infer_column_type_safe

        dtype = infer_column_type_safe(df, "mystery_column")
        if dtype:
            print(f"Detected type: {dtype.value}")
    """
    if isinstance(data, pl.DataFrame):
        data = data.lazy()

    matcher = create_sampled_matcher(
        max_rows=max_rows,
        min_match_ratio=min_ratio,
    )

    return matcher.infer_type(data, column)


# =============================================================================
# Integration with NativePatternMatcher (Backward Compatibility)
# =============================================================================


class SafeNativePatternMatcher(NativePatternMatcher):
    """Drop-in replacement for NativePatternMatcher with sampling.

    This class extends NativePatternMatcher to add automatic
    sampling, making it safe for use with large datasets.

    It maintains the same API as NativePatternMatcher but
    adds sampling configuration options.

    Example:
        # Drop-in replacement
        matcher = SafeNativePatternMatcher(max_rows=50_000)
        results = matcher.match_column(lf, "email")

        # Same API as before, but now memory-safe
    """

    def __init__(
        self,
        patterns: PatternRegistry | None = None,
        *,
        min_match_ratio: float = 0.8,
        sample_size: int = 5,
        include_non_matches: bool = False,
        # New sampling options
        max_rows: int = 100_000,
        sampling_strategy: SamplingMethod = SamplingMethod.ADAPTIVE,
        confidence_level: float = 0.95,
    ):
        """Initialize with sampling options.

        Args:
            patterns: Pattern registry
            min_match_ratio: Minimum match ratio
            sample_size: Number of sample values
            include_non_matches: Include non-matching samples
            max_rows: Maximum rows to process
            sampling_strategy: Sampling strategy to use
            confidence_level: Statistical confidence level
        """
        super().__init__(
            patterns=patterns,
            min_match_ratio=min_match_ratio,
            sample_size=sample_size,
            include_non_matches=include_non_matches,
        )

        self._sampling_config = SamplingConfig(
            strategy=sampling_strategy,
            max_rows=max_rows,
            confidence_level=confidence_level,
        )
        self._sampler = Sampler(self._sampling_config)

    def match_column(
        self,
        lf: pl.LazyFrame,
        column: str,
        *,
        patterns: Sequence[PatternSpec] | None = None,
        limit: int | None = None,  # Now uses sampling instead
    ) -> list[PatternMatchResult]:
        """Match patterns with automatic sampling.

        Overrides parent to add sampling before matching.

        Args:
            lf: LazyFrame containing the data
            column: Column name to analyze
            patterns: Optional patterns to check
            limit: Ignored (uses sampling config instead)

        Returns:
            List of PatternMatchResult
        """
        # Apply sampling
        column_lf = lf.select(pl.col(column))
        sampling_result = self._sampler.sample(column_lf)

        # Log sampling decision
        if sampling_result.is_sampled:
            logger.debug(
                f"Sampled column '{column}': "
                f"{sampling_result.metrics.sample_size:,} of "
                f"{sampling_result.metrics.original_size:,} rows "
                f"({sampling_result.metrics.sampling_ratio:.1%})"
            )

        # Run parent's match_column on sampled data
        return super().match_column(
            sampling_result.data,
            column,
            patterns=patterns,
            limit=None,  # Already sampled
        )
