"""Security module for SQL injection protection, query sanitization, and ReDoS protection.

This module provides comprehensive security features:
- SQL query validation and sanitization
- Parameterized query support
- Whitelist-based validation
- Pluggable security policies
- ReDoS (Regular Expression Denial of Service) protection

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Security Module                              │
    │  (Extensible SQL injection protection and query sanitization)   │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬─────────────────┐
    │               │               │               │                 │
    ▼               ▼               ▼               ▼                 ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌─────────┐
│ Query   │   │ Param   │    │ Whitelist│   │ Security │    │ Audit   │
│Validator│   │ Builder │    │ Validator│   │ Policy   │    │ Logger  │
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └─────────┘

Usage:
    from truthound.validators.security import (
        SecureSQLBuilder,
        SecurityPolicy,
        QueryAuditLogger,
    )

    # Create secure query with parameters
    builder = SecureSQLBuilder(allowed_tables=["orders", "customers"])
    query = builder.select("orders").where("amount > :min_amount").build()
    result = builder.execute(ctx, query, {"min_amount": 100})

    # Custom security policy
    policy = SecurityPolicy(
        max_query_length=5000,
        allow_joins=True,
        blocked_functions=["SLEEP", "BENCHMARK"],
    )
"""

from truthound.validators.security.sql_security import (
    # Core security classes
    SQLSecurityError,
    SQLInjectionError,
    QueryValidationError,
    # Query validation
    SQLQueryValidator,
    validate_sql_query,
    # Parameterized queries
    SecureSQLBuilder,
    ParameterizedQuery,
    # Whitelist validation
    WhitelistValidator,
    SchemaWhitelist,
    # Security policies
    SecurityPolicy,
    SecurityLevel,
    # Mixins
    SecureQueryMixin,
    # Audit
    QueryAuditLogger,
    AuditEntry,
)

from truthound.validators.security.redos import (
    # Core ReDoS classes
    RegexSafetyChecker,
    RegexComplexityAnalyzer,
    SafeRegexConfig,
    SafeRegexExecutor,
    RegexAnalysisResult,
    ReDoSRisk,
    # Convenience functions
    check_regex_safety,
    analyze_regex_complexity,
    create_safe_regex,
    safe_match,
    safe_search,
    # ML Analyzer
    MLPatternAnalyzer,
    MLPredictionResult,
    FeatureExtractor,
    predict_redos_risk,
    # Optimizer
    PatternOptimizer,
    OptimizationResult,
    OptimizationRule,
    optimize_pattern,
    # CVE Database
    CVEDatabase,
    CVEEntry,
    CVEMatchResult,
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
    safe_search_re2,
)

__all__ = [
    # SQL Security - Errors
    "SQLSecurityError",
    "SQLInjectionError",
    "QueryValidationError",
    # SQL Security - Validation
    "SQLQueryValidator",
    "validate_sql_query",
    # SQL Security - Parameterized queries
    "SecureSQLBuilder",
    "ParameterizedQuery",
    # SQL Security - Whitelist
    "WhitelistValidator",
    "SchemaWhitelist",
    # SQL Security - Policies
    "SecurityPolicy",
    "SecurityLevel",
    # SQL Security - Mixins
    "SecureQueryMixin",
    # SQL Security - Audit
    "QueryAuditLogger",
    "AuditEntry",
    # ReDoS Protection - Classes
    "RegexSafetyChecker",
    "RegexComplexityAnalyzer",
    "SafeRegexConfig",
    "SafeRegexExecutor",
    "RegexAnalysisResult",
    "ReDoSRisk",
    # ReDoS Protection - Functions
    "check_regex_safety",
    "analyze_regex_complexity",
    "create_safe_regex",
    "safe_match",
    "safe_search",
    # ReDoS - ML Analyzer
    "MLPatternAnalyzer",
    "MLPredictionResult",
    "FeatureExtractor",
    "predict_redos_risk",
    # ReDoS - Optimizer
    "PatternOptimizer",
    "OptimizationResult",
    "OptimizationRule",
    "optimize_pattern",
    # ReDoS - CVE Database
    "CVEDatabase",
    "CVEEntry",
    "CVEMatchResult",
    "check_cve_vulnerability",
    # ReDoS - CPU Monitor
    "CPUMonitor",
    "CPUMonitorResult",
    "ResourceLimits",
    "execute_with_monitoring",
    # ReDoS - Profiler
    "PatternProfiler",
    "ProfileResult",
    "BenchmarkConfig",
    "profile_pattern",
    # ReDoS - RE2 Engine
    "RE2Engine",
    "RE2CompileError",
    "RE2MatchResult",
    "safe_match_re2",
    "safe_search_re2",
]
