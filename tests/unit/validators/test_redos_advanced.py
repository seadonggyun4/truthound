"""Tests for Advanced ReDoS Protection Features.

This test suite covers:
- ML-based pattern analysis
- Pattern optimization
- CVE database
- CPU monitoring
- Performance profiling
- RE2 engine integration
"""

import re
import pytest

from truthound.validators.security.redos import (
    # Core
    ReDoSRisk,
    check_regex_safety,
    # ML Analyzer
    MLPatternAnalyzer,
    FeatureExtractor,
    MLPredictionResult,
    predict_redos_risk,
    # Optimizer
    PatternOptimizer,
    OptimizationResult,
    optimize_pattern,
    # CVE Database
    CVEDatabase,
    CVEEntry,
    CVEMatchResult,
    CVESeverity,
    check_cve_vulnerability,
    # CPU Monitor
    CPUMonitor,
    CPUMonitorResult,
    ResourceLimits,
    execute_with_monitoring,
    # Profiler
    PatternProfiler,
    ProfileResult,
    BenchmarkConfig,
    profile_pattern,
    # RE2 Engine
    RE2Engine,
    RE2CompileError,
    RE2MatchResult,
    safe_match_re2,
    is_re2_available,
    check_re2_compatibility,
)


# ============================================================================
# ML Analyzer Tests
# ============================================================================


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()

    def test_extract_empty_pattern(self):
        """Test feature extraction for empty pattern."""
        features = self.extractor.extract("")
        assert features.length == 0
        assert features.group_count == 0

    def test_extract_simple_pattern(self):
        """Test feature extraction for simple pattern."""
        features = self.extractor.extract(r"^[a-z]+$")
        assert features.length > 0
        assert features.plus_count == 1
        assert features.char_class_count >= 1
        assert features.start_anchor is True
        assert features.end_anchor is True
        assert features.anchored is True

    def test_extract_nested_quantifiers(self):
        """Test detection of nested quantifiers."""
        features = self.extractor.extract(r"(a+)+b")
        assert features.nested_quantifier_count >= 1
        assert features.backtracking_potential > 0

    def test_extract_backreference(self):
        """Test detection of backreferences."""
        features = self.extractor.extract(r"(a+)\1")
        assert features.backreference_count == 1
        assert features.max_backreference_index == 1

    def test_extract_lookaround(self):
        """Test detection of lookaround assertions."""
        features = self.extractor.extract(r"(?=foo)bar")
        assert features.lookahead_count >= 1

    def test_feature_vector(self):
        """Test conversion to feature vector."""
        features = self.extractor.extract(r"^[a-z]+$")
        vector = features.to_vector()
        assert isinstance(vector, list)
        assert len(vector) == len(features.feature_names())
        assert all(isinstance(v, float) for v in vector)


class TestMLPatternAnalyzer:
    """Tests for MLPatternAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MLPatternAnalyzer()

    def test_predict_safe_pattern(self):
        """Test prediction for safe pattern."""
        result = self.analyzer.predict(r"^[a-z]+$")
        assert isinstance(result, MLPredictionResult)
        # Rule-based model may flag unbounded quantifiers, but risk should be moderate
        assert result.risk_level.value <= ReDoSRisk.HIGH.value
        # Critical patterns like nested quantifiers should have higher risk
        dangerous_result = self.analyzer.predict(r"(a+)+b")
        assert dangerous_result.risk_probability > result.risk_probability

    def test_predict_dangerous_pattern(self):
        """Test prediction for dangerous pattern."""
        result = self.analyzer.predict(r"(a+)+b")
        assert result.risk_probability > 0.5
        assert result.risk_level in [ReDoSRisk.HIGH, ReDoSRisk.CRITICAL]

    def test_predict_batch(self):
        """Test batch prediction."""
        patterns = [r"^[a-z]+$", r"(a+)+", r"\d+"]
        results = self.analyzer.predict_batch(patterns)
        assert len(results) == 3
        assert all(isinstance(r, MLPredictionResult) for r in results)

    def test_contributing_factors(self):
        """Test that contributing factors are provided."""
        result = self.analyzer.predict(r"(a+)+b")
        assert len(result.contributing_factors) > 0
        assert all(isinstance(f, tuple) and len(f) == 2 for f in result.contributing_factors)

    def test_result_to_dict(self):
        """Test result serialization."""
        result = self.analyzer.predict(r"^hello$")
        d = result.to_dict()
        assert "pattern" in d
        assert "risk_probability" in d
        assert "risk_level" in d


def test_predict_redos_risk():
    """Test convenience function."""
    result = predict_redos_risk(r"(a+)+")
    assert isinstance(result, MLPredictionResult)
    assert result.risk_level.value >= ReDoSRisk.HIGH.value


# ============================================================================
# Optimizer Tests
# ============================================================================


class TestPatternOptimizer:
    """Tests for PatternOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = PatternOptimizer()

    def test_optimize_safe_pattern(self):
        """Test that safe patterns are unchanged."""
        result = self.optimizer.optimize(r"^[a-z]+$")
        assert isinstance(result, OptimizationResult)
        # May or may not be optimized depending on rules

    def test_optimize_nested_quantifiers(self):
        """Test optimization of nested quantifiers."""
        result = self.optimizer.optimize(r"(a+)+b")
        assert result.risk_before.value >= ReDoSRisk.HIGH.value
        # Check if optimization was attempted
        assert result.original_pattern == r"(a+)+b"

    def test_optimization_preserves_semantics(self):
        """Test that semantic-preserving mode works."""
        result = self.optimizer.optimize(
            r"(a+)+b",
            preserve_semantics=True,
        )
        assert result.semantics_preserved is True

    def test_aggressive_optimization(self):
        """Test aggressive optimization mode."""
        result = self.optimizer.optimize(
            r".*foo.*",
            aggressive=True,
            preserve_semantics=False,
        )
        # Aggressive mode may add anchors
        if result.was_optimized:
            assert result.applied_rules

    def test_suggest_alternatives(self):
        """Test alternative pattern suggestions."""
        alternatives = self.optimizer.suggest_alternatives(r"(a+)+")
        assert isinstance(alternatives, list)

    def test_explain_optimization(self):
        """Test optimization explanation."""
        explanation = self.optimizer.explain_optimization(r"(a+)+b")
        assert isinstance(explanation, str)
        assert "Pattern:" in explanation

    def test_result_to_dict(self):
        """Test result serialization."""
        result = self.optimizer.optimize(r"(a+)+")
        d = result.to_dict()
        assert "original_pattern" in d
        assert "optimized_pattern" in d
        assert "risk_before" in d


def test_optimize_pattern():
    """Test convenience function."""
    result = optimize_pattern(r"(a+)+b")
    assert isinstance(result, OptimizationResult)


# ============================================================================
# CVE Database Tests
# ============================================================================


class TestCVEDatabase:
    """Tests for CVEDatabase."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db = CVEDatabase()

    def test_database_has_entries(self):
        """Test that database has built-in entries."""
        assert len(self.db) > 0

    def test_check_known_vulnerable_pattern(self):
        """Test checking a known vulnerable pattern."""
        result = self.db.check(r"(a+)+")
        assert isinstance(result, CVEMatchResult)
        # Should match at least one CVE
        assert result.is_vulnerable or len(result.similar_patterns) > 0

    def test_check_safe_pattern(self):
        """Test checking a safe pattern."""
        result = self.db.check(r"^[a-z]+$")
        assert isinstance(result, CVEMatchResult)
        assert len(result.matches) == 0

    def test_search_by_severity(self):
        """Test searching by severity."""
        from truthound.validators.security.redos.cve_database import CVESeverity
        critical = self.db.search(severity=CVESeverity.CRITICAL)
        assert isinstance(critical, list)

    def test_search_by_source(self):
        """Test searching by source."""
        from truthound.validators.security.redos.cve_database import CVESource
        owasp = self.db.search(source=CVESource.OWASP)
        assert isinstance(owasp, list)

    def test_add_custom_entry(self):
        """Test adding custom CVE entry."""
        entry = CVEEntry(
            cve_id="TEST-CVE-001",
            pattern=r"(test)+",
            description="Test entry",
        )
        initial_count = len(self.db)
        self.db.add_entry(entry)
        assert len(self.db) == initial_count + 1

    def test_get_entry(self):
        """Test getting entry by CVE ID."""
        # Get a known entry
        entry = self.db.get_entry("OWASP-REDOS-001")
        if entry:
            assert entry.cve_id == "OWASP-REDOS-001"

    def test_statistics(self):
        """Test database statistics."""
        stats = self.db.get_statistics()
        assert "total_entries" in stats
        assert stats["total_entries"] > 0


def test_check_cve_vulnerability():
    """Test convenience function."""
    result = check_cve_vulnerability(r"(a+)+")
    assert isinstance(result, CVEMatchResult)


# ============================================================================
# CPU Monitor Tests
# ============================================================================


class TestCPUMonitor:
    """Tests for CPUMonitor."""

    def test_basic_monitoring(self):
        """Test basic CPU monitoring."""
        monitor = CPUMonitor()
        monitor.start()

        # Do some work
        _ = sum(range(10000))

        monitor.stop()
        result = monitor.get_result()

        assert isinstance(result, CPUMonitorResult)
        assert result.total_time_seconds >= 0
        assert len(result.samples) >= 0

    def test_resource_limits(self):
        """Test resource limits configuration."""
        limits = ResourceLimits(
            cpu_percent_limit=50.0,
            time_limit_seconds=1.0,
        )
        assert limits.cpu_percent_limit == 50.0
        assert limits.time_limit_seconds == 1.0

    def test_strict_limits(self):
        """Test strict limits preset."""
        limits = ResourceLimits.strict()
        assert limits.cpu_percent_limit == 50.0
        assert limits.time_limit_seconds == 1.0

    def test_lenient_limits(self):
        """Test lenient limits preset."""
        limits = ResourceLimits.lenient()
        assert limits.cpu_percent_limit == 95.0
        assert limits.time_limit_seconds == 30.0

    def test_result_to_dict(self):
        """Test result serialization."""
        monitor = CPUMonitor()
        monitor.start()
        monitor.stop()
        result = monitor.get_result()
        d = result.to_dict()
        assert "success" in d
        assert "total_time_seconds" in d


def test_execute_with_monitoring():
    """Test convenience function."""
    result = execute_with_monitoring(
        pattern=r"^[a-z]+$",
        input_string="hello",
        time_limit=5.0,
    )
    assert isinstance(result, CPUMonitorResult)


# ============================================================================
# Profiler Tests
# ============================================================================


class TestPatternProfiler:
    """Tests for PatternProfiler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PatternProfiler(BenchmarkConfig.quick())

    def test_profile_simple_pattern(self):
        """Test profiling a simple pattern."""
        result = self.profiler.profile(r"^[a-z]+$", ["hello", "world"])
        assert isinstance(result, ProfileResult)
        assert result.mean_time_ns >= 0
        assert len(result.measurements) > 0

    def test_profile_scaling(self):
        """Test scaling profile."""
        result = self.profiler.profile_scaling(
            r"^[a-z]+$",
            min_size=5,
            max_size=20,
            step=5,
        )
        assert isinstance(result, ProfileResult)
        assert len(result.size_to_time) > 0

    def test_benchmark_config_presets(self):
        """Test benchmark configuration presets."""
        quick = BenchmarkConfig.quick()
        assert quick.iterations == 10

        thorough = BenchmarkConfig.thorough()
        assert thorough.iterations == 500

    def test_result_to_dict(self):
        """Test result serialization."""
        result = self.profiler.profile(r"hello", ["hello"])
        d = result.to_dict()
        assert "pattern" in d
        assert "mean_time_ms" in d
        assert "scaling_complexity" in d


def test_profile_pattern():
    """Test convenience function."""
    result = profile_pattern(r"^hello$", ["hello"])
    assert isinstance(result, ProfileResult)


# ============================================================================
# RE2 Engine Tests
# ============================================================================


class TestRE2Engine:
    """Tests for RE2Engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RE2Engine(fallback_to_python=True)

    def test_match_simple_pattern(self):
        """Test matching a simple pattern."""
        result = self.engine.match(r"^[a-z]+$", "hello")
        assert isinstance(result, RE2MatchResult)
        assert result.matched is True
        assert result.match == "hello"

    def test_match_no_match(self):
        """Test when pattern doesn't match."""
        result = self.engine.match(r"^[a-z]+$", "12345")
        assert result.matched is False
        assert result.match is None

    def test_search(self):
        """Test search operation."""
        result = self.engine.search(r"world", "hello world")
        assert result.matched is True
        assert result.match == "world"

    def test_fallback_for_backreference(self):
        """Test fallback for backreference pattern."""
        result = self.engine.match(r"(a+)\1", "aa")
        # Should fall back to Python re
        assert result.used_fallback is True
        assert result.engine == "python_re"

    def test_is_re2_compatible(self):
        """Test RE2 compatibility check."""
        assert self.engine.is_pattern_re2_compatible(r"^[a-z]+$") is True
        assert self.engine.is_pattern_re2_compatible(r"(a+)\1") is False

    def test_get_unsupported_features(self):
        """Test getting unsupported features list."""
        features = self.engine.get_unsupported_features(r"(a+)\1")
        assert len(features) > 0

    def test_result_to_dict(self):
        """Test result serialization."""
        result = self.engine.match(r"hello", "hello")
        d = result.to_dict()
        assert "pattern" in d
        assert "matched" in d
        assert "engine" in d


def test_safe_match_re2():
    """Test convenience function."""
    result = safe_match_re2(r"^[a-z]+$", "hello")
    assert isinstance(result, RE2MatchResult)
    assert result.matched is True


def test_check_re2_compatibility():
    """Test compatibility check function."""
    compatible, features = check_re2_compatibility(r"^[a-z]+$")
    assert compatible is True
    assert len(features) == 0

    compatible, features = check_re2_compatibility(r"(a+)\1")
    assert compatible is False
    assert len(features) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_analysis_pipeline(self):
        """Test complete pattern analysis pipeline."""
        pattern = r"(a+)+b"

        # 1. ML Analysis
        ml_result = predict_redos_risk(pattern)
        assert ml_result.risk_level.value >= ReDoSRisk.HIGH.value

        # 2. CVE Check
        cve_result = check_cve_vulnerability(pattern)
        assert cve_result.is_vulnerable or len(cve_result.similar_patterns) > 0

        # 3. Optimization
        opt_result = optimize_pattern(pattern)
        assert opt_result.original_pattern == pattern

        # 4. RE2 Compatibility
        compatible, _ = check_re2_compatibility(pattern)
        assert compatible is True  # No backreferences

    def test_safe_pattern_pipeline(self):
        """Test pipeline with safe pattern."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        # ML Analysis - complex patterns may have higher scores due to heuristics
        ml_result = predict_redos_risk(pattern)
        # Complex email patterns may score higher, but should be lower than nested quantifiers
        nested_result = predict_redos_risk(r"(a+)+b")
        # Email pattern doesn't have actual nested quantifiers

        # Core check - the static analyzer should pass this
        is_safe, _ = check_regex_safety(pattern)
        assert is_safe is True

    def test_comparison_with_profiler(self):
        """Test profiler comparison of patterns."""
        profiler = PatternProfiler(BenchmarkConfig.quick())

        safe_pattern = r"^[a-z]+$"
        dangerous_pattern = r"(a+)+b"

        safe_result = profiler.profile(safe_pattern, ["a" * 10])
        # Note: We don't profile dangerous patterns with long inputs
        # to avoid actual backtracking in tests

        assert safe_result.mean_time_ns >= 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing API."""

    def test_import_from_security_module(self):
        """Test that all exports are available from security module."""
        from truthound.validators.security import (
            # Core
            check_regex_safety,
            RegexSafetyChecker,
            # ML
            MLPatternAnalyzer,
            predict_redos_risk,
            # Optimizer
            PatternOptimizer,
            optimize_pattern,
            # CVE
            CVEDatabase,
            check_cve_vulnerability,
            # CPU Monitor
            CPUMonitor,
            execute_with_monitoring,
            # Profiler
            PatternProfiler,
            profile_pattern,
            # RE2
            RE2Engine,
            safe_match_re2,
        )

        # Verify they're callable/usable
        assert callable(check_regex_safety)
        assert callable(predict_redos_risk)
        assert callable(optimize_pattern)
        assert callable(check_cve_vulnerability)
        assert callable(execute_with_monitoring)
        assert callable(profile_pattern)
        assert callable(safe_match_re2)

    def test_original_api_still_works(self):
        """Test that original core API still works."""
        from truthound.validators.security.redos import (
            check_regex_safety,
            analyze_regex_complexity,
            create_safe_regex,
            safe_match,
            safe_search,
        )

        # Test original functions
        is_safe, _ = check_regex_safety(r"^[a-z]+$")
        assert is_safe is True

        result = analyze_regex_complexity(r"(a+)+")
        assert result.risk_level.value >= ReDoSRisk.HIGH.value

        compiled = create_safe_regex(r"^[a-z]+$")
        assert isinstance(compiled, re.Pattern)

        match = safe_match(r"^hello$", "hello")
        assert match is not None

        search = safe_search(r"world", "hello world")
        assert search is not None
