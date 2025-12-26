"""Security module for SQL injection protection and query sanitization.

This module provides comprehensive SQL security features:
- Query validation and sanitization
- Parameterized query support
- Whitelist-based validation
- Pluggable security policies

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

__all__ = [
    # Errors
    "SQLSecurityError",
    "SQLInjectionError",
    "QueryValidationError",
    # Validation
    "SQLQueryValidator",
    "validate_sql_query",
    # Parameterized queries
    "SecureSQLBuilder",
    "ParameterizedQuery",
    # Whitelist
    "WhitelistValidator",
    "SchemaWhitelist",
    # Policies
    "SecurityPolicy",
    "SecurityLevel",
    # Mixins
    "SecureQueryMixin",
    # Audit
    "QueryAuditLogger",
    "AuditEntry",
]
