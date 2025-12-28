"""Advanced ReDoS (Regular Expression Denial of Service) Protection.

This package provides comprehensive protection against ReDoS attacks:
- Static analysis of regex patterns for dangerous constructs
- ML-based pattern risk prediction
- Automatic pattern optimization (dangerous -> safe)
- CVE database for known vulnerable patterns
- Real-time CPU monitoring during execution
- Pattern performance profiling
- RE2 engine option for linear-time guarantees

Architecture:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                      ReDoS Protection Framework                           │
    └──────────────────────────────────────────────────────────────────────────┘
                                        │
    ┌───────────┬───────────┬───────────┼───────────┬───────────┬───────────┐
    │           │           │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Static │ │   ML   │ │ Pattern│ │  CVE   │ │  CPU   │ │ Perf   │ │  RE2   │
│Analysis│ │Analyzer│ │Optimizer│ │  DB    │ │Monitor │ │Profiler│ │ Engine │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘

Usage:
    from truthound.validators.security.redos import (
        # Core analysis
        check_regex_safety,
        analyze_regex_complexity,
        create_safe_regex,

        # ML-based prediction
        MLPatternAnalyzer,
        predict_redos_risk,

        # Pattern optimization
        PatternOptimizer,
        optimize_pattern,

        # CVE database
        CVEDatabase,
        check_cve_vulnerability,

        # CPU monitoring
        CPUMonitor,
        execute_with_monitoring,

        # Performance profiling
        PatternProfiler,
        profile_pattern,

        # RE2 engine
        RE2Engine,
        safe_match_re2,
    )
"""

from __future__ import annotations

# Re-export from core module (backward compatibility)
from truthound.validators.security.redos.core import (
    # Enums and configs
    ReDoSRisk,
    SafeRegexConfig,
    # Result types
    RegexAnalysisResult,
    # Analyzers
    RegexComplexityAnalyzer,
    RegexSafetyChecker,
    SafeRegexExecutor,
    # Convenience functions
    check_regex_safety,
    analyze_regex_complexity,
    create_safe_regex,
    safe_match,
    safe_search,
)

# ML-based pattern analysis
from truthound.validators.security.redos.ml_analyzer import (
    MLPatternAnalyzer,
    MLPredictionResult,
    FeatureExtractor,
    predict_redos_risk,
)

# Pattern optimization
from truthound.validators.security.redos.optimizer import (
    PatternOptimizer,
    OptimizationResult,
    OptimizationRule,
    optimize_pattern,
)

# CVE database
from truthound.validators.security.redos.cve_database import (
    CVEDatabase,
    CVEEntry,
    CVEMatchResult,
    CVESeverity,
    CVESource,
    check_cve_vulnerability,
)

# CPU monitoring
from truthound.validators.security.redos.cpu_monitor import (
    CPUMonitor,
    CPUMonitorResult,
    ResourceLimits,
    execute_with_monitoring,
)

# Performance profiling
from truthound.validators.security.redos.profiler import (
    PatternProfiler,
    ProfileResult,
    BenchmarkConfig,
    profile_pattern,
)

# RE2 engine
from truthound.validators.security.redos.re2_engine import (
    RE2Engine,
    RE2CompileError,
    RE2MatchResult,
    RE2UnsupportedFeature,
    safe_match_re2,
    safe_search_re2,
    is_re2_available,
    check_re2_compatibility,
)

__all__ = [
    # Core (backward compatible)
    "ReDoSRisk",
    "SafeRegexConfig",
    "RegexAnalysisResult",
    "RegexComplexityAnalyzer",
    "RegexSafetyChecker",
    "SafeRegexExecutor",
    "check_regex_safety",
    "analyze_regex_complexity",
    "create_safe_regex",
    "safe_match",
    "safe_search",
    # ML Analyzer
    "MLPatternAnalyzer",
    "MLPredictionResult",
    "FeatureExtractor",
    "predict_redos_risk",
    # Optimizer
    "PatternOptimizer",
    "OptimizationResult",
    "OptimizationRule",
    "optimize_pattern",
    # CVE Database
    "CVEDatabase",
    "CVEEntry",
    "CVEMatchResult",
    "CVESeverity",
    "CVESource",
    "check_cve_vulnerability",
    # CPU Monitor
    "CPUMonitor",
    "CPUMonitorResult",
    "ResourceLimits",
    "execute_with_monitoring",
    # Profiler
    "PatternProfiler",
    "ProfileResult",
    "BenchmarkConfig",
    "profile_pattern",
    # RE2 Engine
    "RE2Engine",
    "RE2CompileError",
    "RE2MatchResult",
    "safe_match_re2",
    "safe_search_re2",
    "is_re2_available",
    "check_re2_compatibility",
    "RE2UnsupportedFeature",
]
