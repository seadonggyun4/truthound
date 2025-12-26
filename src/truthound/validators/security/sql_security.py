"""Comprehensive SQL security module.

Provides extensible SQL injection protection with:
- Multi-level security policies
- Parameterized query support
- Whitelist-based validation
- Query audit logging
- Pluggable security rules

Security Levels:
    STRICT: Maximum security, minimal allowed operations
    STANDARD: Balanced security for typical use cases (default)
    PERMISSIVE: Relaxed security for trusted environments
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Iterator

import polars as pl


# =============================================================================
# Exceptions
# =============================================================================


class SQLSecurityError(Exception):
    """Base exception for SQL security issues."""

    pass


class SQLInjectionError(SQLSecurityError):
    """Raised when potential SQL injection is detected."""

    def __init__(self, message: str, pattern: str | None = None, query: str | None = None):
        super().__init__(message)
        self.pattern = pattern
        self.query = query[:100] + "..." if query and len(query) > 100 else query


class QueryValidationError(SQLSecurityError):
    """Raised when query validation fails."""

    pass


# =============================================================================
# Security Levels and Policies
# =============================================================================


class SecurityLevel(Enum):
    """Security level presets."""

    STRICT = auto()  # Maximum security
    STANDARD = auto()  # Balanced (default)
    PERMISSIVE = auto()  # Relaxed for trusted environments


@dataclass
class SecurityPolicy:
    """Configurable security policy for SQL validation.

    Defines what operations are allowed and blocked.

    Example:
        # Custom policy for analytics queries
        policy = SecurityPolicy(
            level=SecurityLevel.STANDARD,
            max_query_length=20000,
            allow_joins=True,
            allow_subqueries=True,
            allow_aggregations=True,
            blocked_functions=["SLEEP", "BENCHMARK", "LOAD_FILE"],
        )
    """

    # Basic settings
    level: SecurityLevel = SecurityLevel.STANDARD
    max_query_length: int = 10000
    max_identifier_length: int = 128

    # Structural permissions
    allow_joins: bool = True
    allow_subqueries: bool = True
    allow_aggregations: bool = True
    allow_window_functions: bool = True
    allow_cte: bool = True  # Common Table Expressions (WITH clause)
    allow_union: bool = False  # UNION can be used for injection

    # Statement types
    allowed_statements: set[str] = field(
        default_factory=lambda: {"SELECT", "WITH"}
    )

    # Blocked patterns (regex)
    blocked_patterns: list[str] = field(default_factory=list)

    # Blocked SQL functions
    blocked_functions: list[str] = field(
        default_factory=lambda: [
            "SLEEP",
            "BENCHMARK",
            "LOAD_FILE",
            "INTO OUTFILE",
            "INTO DUMPFILE",
        ]
    )

    # Allowed tables/columns (if empty, all are allowed)
    allowed_tables: set[str] = field(default_factory=set)
    allowed_columns: set[str] = field(default_factory=set)

    # Callbacks
    on_violation: Callable[[str, str], None] | None = None

    @classmethod
    def strict(cls) -> "SecurityPolicy":
        """Create a strict security policy."""
        return cls(
            level=SecurityLevel.STRICT,
            max_query_length=5000,
            allow_joins=False,
            allow_subqueries=False,
            allow_union=False,
            allow_cte=False,
        )

    @classmethod
    def standard(cls) -> "SecurityPolicy":
        """Create a standard security policy."""
        return cls(level=SecurityLevel.STANDARD)

    @classmethod
    def permissive(cls) -> "SecurityPolicy":
        """Create a permissive security policy."""
        return cls(
            level=SecurityLevel.PERMISSIVE,
            max_query_length=50000,
            allow_joins=True,
            allow_subqueries=True,
            allow_union=True,
            allow_cte=True,
        )


# =============================================================================
# Pattern-based Validation
# =============================================================================


@dataclass
class DangerousPattern:
    """A dangerous SQL pattern to detect."""

    name: str
    pattern: str
    severity: str = "HIGH"  # HIGH, MEDIUM, LOW
    description: str = ""


class PatternRegistry:
    """Registry of dangerous SQL patterns.

    Extensible registry for SQL injection patterns.

    Example:
        registry = PatternRegistry()
        registry.register(DangerousPattern(
            name="time_based_injection",
            pattern=r"WAITFOR\s+DELAY",
            severity="HIGH",
            description="Time-based SQL injection"
        ))
    """

    # Default dangerous patterns
    DEFAULT_PATTERNS = [
        # DDL statements
        DangerousPattern(
            "ddl_create",
            r"\b(CREATE)\s+(TABLE|DATABASE|INDEX|VIEW|SCHEMA|PROCEDURE|FUNCTION)\b",
            "HIGH",
            "DDL CREATE statement",
        ),
        DangerousPattern(
            "ddl_alter",
            r"\b(ALTER)\s+(TABLE|DATABASE|INDEX|VIEW|SCHEMA)\b",
            "HIGH",
            "DDL ALTER statement",
        ),
        DangerousPattern(
            "ddl_drop",
            r"\b(DROP)\s+(TABLE|DATABASE|INDEX|VIEW|SCHEMA)\b",
            "HIGH",
            "DDL DROP statement",
        ),
        DangerousPattern(
            "ddl_truncate",
            r"\bTRUNCATE\s+TABLE\b",
            "HIGH",
            "DDL TRUNCATE statement",
        ),
        # DCL statements
        DangerousPattern(
            "dcl_grant",
            r"\b(GRANT|REVOKE|DENY)\b",
            "HIGH",
            "DCL statement",
        ),
        # DML modification
        DangerousPattern(
            "dml_insert",
            r"\bINSERT\s+INTO\b",
            "HIGH",
            "INSERT statement",
        ),
        DangerousPattern(
            "dml_update",
            r"\bUPDATE\s+\w+\s+SET\b",
            "HIGH",
            "UPDATE statement",
        ),
        DangerousPattern(
            "dml_delete",
            r"\bDELETE\s+FROM\b",
            "HIGH",
            "DELETE statement",
        ),
        # Transaction control
        DangerousPattern(
            "transaction",
            r"\b(COMMIT|ROLLBACK|SAVEPOINT|BEGIN\s+TRANSACTION)\b",
            "MEDIUM",
            "Transaction control",
        ),
        # System/Exec
        DangerousPattern(
            "exec",
            r"\b(EXEC|EXECUTE|CALL)\s*\(",
            "HIGH",
            "Execute/call statement",
        ),
        # File operations
        DangerousPattern(
            "file_ops",
            r"\b(LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)\b",
            "HIGH",
            "File operation",
        ),
        # Comment injection
        DangerousPattern(
            "line_comment",
            r"--\s*$",
            "MEDIUM",
            "Line comment at end (potential injection)",
        ),
        DangerousPattern(
            "block_comment",
            r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/",
            "LOW",
            "Block comment",
        ),
        # Stacked queries
        DangerousPattern(
            "stacked_query",
            r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)",
            "HIGH",
            "Stacked query",
        ),
        # Union injection
        DangerousPattern(
            "union_select",
            r"\bUNION\s+(ALL\s+)?SELECT\b",
            "MEDIUM",
            "UNION SELECT (potential injection)",
        ),
        # Time-based injection
        DangerousPattern(
            "sleep",
            r"\b(SLEEP|WAITFOR\s+DELAY|BENCHMARK)\s*\(",
            "HIGH",
            "Time-based injection",
        ),
        # Error-based injection
        DangerousPattern(
            "extractvalue",
            r"\b(EXTRACTVALUE|UPDATEXML|EXP|FLOOR\s*\(\s*RAND)\b",
            "MEDIUM",
            "Error-based injection function",
        ),
        # Boolean-based injection patterns
        DangerousPattern(
            "always_true",
            r"(?:OR|AND)\s+['\"0-9]+\s*=\s*['\"0-9]+",
            "MEDIUM",
            "Always true/false condition",
        ),
        DangerousPattern(
            "or_1_eq_1",
            r"\bOR\s+1\s*=\s*1\b",
            "HIGH",
            "Classic OR 1=1 injection",
        ),
    ]

    def __init__(self) -> None:
        self._patterns: list[DangerousPattern] = []
        self._compiled: list[tuple[DangerousPattern, re.Pattern]] = []

        # Register default patterns
        for pattern in self.DEFAULT_PATTERNS:
            self.register(pattern)

    def register(self, pattern: DangerousPattern) -> None:
        """Register a new dangerous pattern."""
        self._patterns.append(pattern)
        compiled = re.compile(pattern.pattern, re.IGNORECASE | re.MULTILINE)
        self._compiled.append((pattern, compiled))

    def unregister(self, name: str) -> bool:
        """Unregister a pattern by name."""
        for i, p in enumerate(self._patterns):
            if p.name == name:
                del self._patterns[i]
                del self._compiled[i]
                return True
        return False

    def check(self, query: str) -> list[tuple[DangerousPattern, str]]:
        """Check query against all patterns.

        Returns:
            List of (pattern, matched_text) tuples
        """
        matches = []
        for pattern, compiled in self._compiled:
            match = compiled.search(query)
            if match:
                matches.append((pattern, match.group()))
        return matches

    def __iter__(self) -> Iterator[DangerousPattern]:
        return iter(self._patterns)


# =============================================================================
# Core SQL Validator
# =============================================================================


class SQLQueryValidator:
    """Enhanced SQL query validator with pluggable policies.

    Validates SQL queries for security issues using configurable policies
    and pattern-based detection.

    Example:
        # With default policy
        validator = SQLQueryValidator()
        validator.validate("SELECT * FROM users")  # OK

        # With custom policy
        policy = SecurityPolicy.strict()
        validator = SQLQueryValidator(policy=policy)
        validator.validate("SELECT * FROM users JOIN orders")  # Raises error
    """

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        pattern_registry: PatternRegistry | None = None,
        audit_logger: "QueryAuditLogger | None" = None,
    ):
        """Initialize SQL query validator.

        Args:
            policy: Security policy to use (default: STANDARD)
            pattern_registry: Custom pattern registry
            audit_logger: Optional audit logger
        """
        self.policy = policy or SecurityPolicy.standard()
        self.pattern_registry = pattern_registry or PatternRegistry()
        self.audit_logger = audit_logger

        # Apply policy-specific patterns
        self._apply_policy_patterns()

    def _apply_policy_patterns(self) -> None:
        """Apply additional patterns based on policy."""
        # Block UNION if not allowed
        if not self.policy.allow_union:
            self.pattern_registry.register(
                DangerousPattern(
                    "policy_union",
                    r"\bUNION\b",
                    "MEDIUM",
                    "UNION blocked by policy",
                )
            )

        # Block subqueries if not allowed
        if not self.policy.allow_subqueries:
            self.pattern_registry.register(
                DangerousPattern(
                    "policy_subquery",
                    r"\(\s*SELECT\b",
                    "MEDIUM",
                    "Subquery blocked by policy",
                )
            )

        # Block joins if not allowed
        if not self.policy.allow_joins:
            self.pattern_registry.register(
                DangerousPattern(
                    "policy_join",
                    r"\b(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\b",
                    "MEDIUM",
                    "JOIN blocked by policy",
                )
            )

        # Add custom blocked patterns
        for i, pattern in enumerate(self.policy.blocked_patterns):
            self.pattern_registry.register(
                DangerousPattern(
                    f"custom_blocked_{i}",
                    pattern,
                    "HIGH",
                    "Custom blocked pattern",
                )
            )

        # Add blocked functions
        for func in self.policy.blocked_functions:
            self.pattern_registry.register(
                DangerousPattern(
                    f"blocked_func_{func.lower()}",
                    rf"\b{re.escape(func)}\s*\(",
                    "HIGH",
                    f"Blocked function: {func}",
                )
            )

    def validate(self, query: str) -> None:
        """Validate a SQL query for security issues.

        Args:
            query: SQL query string to validate

        Raises:
            QueryValidationError: If query fails basic validation
            SQLInjectionError: If potential injection is detected
        """
        if not query or not query.strip():
            raise QueryValidationError("Empty query")

        # Check length
        if len(query) > self.policy.max_query_length:
            raise QueryValidationError(
                f"Query exceeds maximum length of {self.policy.max_query_length}"
            )

        normalized = query.strip()

        # Check statement type
        self._validate_statement_type(normalized)

        # Check for multiple statements
        self._check_multiple_statements(normalized)

        # Check against pattern registry
        matches = self.pattern_registry.check(normalized)
        if matches:
            pattern, matched = matches[0]
            if self.policy.on_violation:
                self.policy.on_violation(pattern.name, matched)
            raise SQLInjectionError(
                f"Dangerous pattern detected: {pattern.description}",
                pattern=pattern.pattern,
                query=query,
            )

        # Validate table names
        if self.policy.allowed_tables:
            self._validate_table_names(normalized)

        # Log successful validation
        if self.audit_logger:
            self.audit_logger.log_query(query, success=True)

    def _validate_statement_type(self, query: str) -> None:
        """Validate statement type is allowed."""
        match = re.match(r"^\s*(\w+)", query, re.IGNORECASE)
        if not match:
            raise QueryValidationError("Could not determine SQL statement type")

        statement_type = match.group(1).upper()
        if statement_type not in self.policy.allowed_statements:
            raise QueryValidationError(
                f"Statement type '{statement_type}' not allowed. "
                f"Allowed: {', '.join(self.policy.allowed_statements)}"
            )

    def _check_multiple_statements(self, query: str) -> None:
        """Check for multiple statements."""
        # Remove string literals
        cleaned = re.sub(r"'[^']*'", "", query)
        cleaned = re.sub(r'"[^"]*"', "", cleaned)

        if re.search(r";\s*\S", cleaned):
            raise SQLInjectionError(
                "Multiple statements detected",
                pattern="stacked_query",
                query=query,
            )

    def _validate_table_names(self, query: str) -> None:
        """Validate table names against whitelist."""
        table_pattern = r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b"
        matches = re.findall(table_pattern, query, re.IGNORECASE)

        allowed_lower = {t.lower() for t in self.policy.allowed_tables}
        for table in matches:
            if table.lower() not in allowed_lower:
                raise QueryValidationError(
                    f"Table '{table}' not in allowed list: "
                    f"{', '.join(self.policy.allowed_tables)}"
                )


def validate_sql_query(
    query: str,
    policy: SecurityPolicy | None = None,
    allowed_tables: list[str] | None = None,
) -> None:
    """Convenience function to validate SQL query.

    Args:
        query: SQL query to validate
        policy: Optional security policy
        allowed_tables: Optional table whitelist

    Raises:
        SQLSecurityError: If validation fails
    """
    if policy is None:
        policy = SecurityPolicy.standard()

    if allowed_tables:
        policy.allowed_tables = set(allowed_tables)

    validator = SQLQueryValidator(policy=policy)
    validator.validate(query)


# =============================================================================
# Whitelist Validation
# =============================================================================


@dataclass
class SchemaWhitelist:
    """Schema-aware whitelist for tables and columns.

    Example:
        whitelist = SchemaWhitelist()
        whitelist.add_table("orders", ["id", "customer_id", "amount", "status"])
        whitelist.add_table("customers", ["id", "name", "email"])

        whitelist.validate_table("orders")  # OK
        whitelist.validate_column("orders", "amount")  # OK
        whitelist.validate_column("orders", "password")  # Raises error
    """

    tables: dict[str, set[str]] = field(default_factory=dict)
    allow_all_columns: bool = False

    def add_table(self, table: str, columns: list[str] | None = None) -> None:
        """Add a table to the whitelist.

        Args:
            table: Table name
            columns: Allowed columns (None = all columns allowed)
        """
        self.tables[table.lower()] = set(c.lower() for c in columns) if columns else set()

    def remove_table(self, table: str) -> None:
        """Remove a table from the whitelist."""
        self.tables.pop(table.lower(), None)

    def validate_table(self, table: str) -> None:
        """Validate table is in whitelist."""
        if table.lower() not in self.tables:
            raise QueryValidationError(
                f"Table '{table}' not in whitelist. "
                f"Allowed: {', '.join(self.tables.keys())}"
            )

    def validate_column(self, table: str, column: str) -> None:
        """Validate column is in whitelist for table."""
        self.validate_table(table)

        columns = self.tables[table.lower()]
        if columns and column.lower() not in columns:
            raise QueryValidationError(
                f"Column '{column}' not allowed for table '{table}'. "
                f"Allowed: {', '.join(columns)}"
            )

    def get_tables(self) -> list[str]:
        """Get list of allowed tables."""
        return list(self.tables.keys())

    def get_columns(self, table: str) -> list[str]:
        """Get list of allowed columns for table."""
        return list(self.tables.get(table.lower(), []))


class WhitelistValidator:
    """Validates queries against schema whitelist.

    Example:
        whitelist = SchemaWhitelist()
        whitelist.add_table("orders", ["id", "amount"])

        validator = WhitelistValidator(whitelist)
        validator.validate_query("SELECT id, amount FROM orders")  # OK
        validator.validate_query("SELECT password FROM users")  # Raises error
    """

    def __init__(self, schema: SchemaWhitelist):
        self.schema = schema

    def validate_query(self, query: str) -> None:
        """Validate query against whitelist."""
        # Extract table references
        table_pattern = r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b"
        tables = re.findall(table_pattern, query, re.IGNORECASE)

        for table in tables:
            self.schema.validate_table(table)

        # Extract column references (simplified)
        # Note: Full SQL parsing would require a proper parser
        select_pattern = r"SELECT\s+(.+?)\s+FROM"
        match = re.search(select_pattern, query, re.IGNORECASE | re.DOTALL)
        if match and tables:
            columns_str = match.group(1)
            if columns_str.strip() != "*":
                # Parse column list
                columns = [c.strip().split(".")[-1] for c in columns_str.split(",")]
                for col in columns:
                    # Remove aliases
                    col = re.sub(r"\s+AS\s+\w+$", "", col, flags=re.IGNORECASE).strip()
                    if col and not col.startswith("("):
                        # Validate against first table (simplified)
                        self.schema.validate_column(tables[0], col)


# =============================================================================
# Parameterized Queries
# =============================================================================


@dataclass
class ParameterizedQuery:
    """A parameterized SQL query.

    Stores query template and parameters separately for safe execution.

    Example:
        query = ParameterizedQuery(
            template="SELECT * FROM orders WHERE amount > :min_amount",
            parameters={"min_amount": 100}
        )
    """

    template: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate template and parameters."""
        # Find all parameter placeholders
        placeholders = set(re.findall(r":(\w+)", self.template))

        # Check all parameters are provided
        missing = placeholders - set(self.parameters.keys())
        if missing:
            raise QueryValidationError(
                f"Missing parameters: {', '.join(missing)}"
            )

    def render(self) -> str:
        """Render the query with parameters.

        Note: For Polars SQL, parameters are substituted directly.
        Values are escaped to prevent injection.
        """
        result = self.template
        for key, value in self.parameters.items():
            placeholder = f":{key}"
            escaped_value = self._escape_value(value)
            result = result.replace(placeholder, escaped_value)
        return result

    def _escape_value(self, value: Any) -> str:
        """Escape a parameter value for SQL."""
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(value, (list, tuple)):
            escaped_items = [self._escape_value(v) for v in value]
            return f"({', '.join(escaped_items)})"
        else:
            raise QueryValidationError(
                f"Unsupported parameter type: {type(value)}"
            )


class SecureSQLBuilder:
    """Builder for secure SQL queries with parameterization.

    Provides a fluent interface for building secure SQL queries
    with automatic parameter escaping and validation.

    Example:
        builder = SecureSQLBuilder(allowed_tables=["orders", "customers"])

        query = (
            builder
            .select("orders", ["id", "amount", "status"])
            .where("amount > :min_amount")
            .where("status = :status")
            .order_by("amount", desc=True)
            .limit(100)
            .build({"min_amount": 100, "status": "pending"})
        )

        # Execute with context
        result = builder.execute(ctx, query)
    """

    def __init__(
        self,
        allowed_tables: list[str] | None = None,
        policy: SecurityPolicy | None = None,
    ):
        self.allowed_tables = set(allowed_tables) if allowed_tables else None
        self.policy = policy or SecurityPolicy.standard()
        self.validator = SQLQueryValidator(policy=self.policy)

        # Query parts
        self._select_table: str | None = None
        self._select_columns: list[str] = []
        self._joins: list[str] = []
        self._where_clauses: list[str] = []
        self._group_by: list[str] = []
        self._having_clauses: list[str] = []
        self._order_by: list[str] = []
        self._limit_value: int | None = None
        self._offset_value: int | None = None

    def select(
        self,
        table: str,
        columns: list[str] | None = None,
    ) -> "SecureSQLBuilder":
        """Set SELECT table and columns.

        Args:
            table: Table name
            columns: Columns to select (None = all)
        """
        self._validate_identifier(table)
        if self.allowed_tables and table not in self.allowed_tables:
            raise QueryValidationError(
                f"Table '{table}' not in allowed list"
            )

        self._select_table = table

        if columns:
            for col in columns:
                self._validate_identifier(col)
            self._select_columns = columns
        else:
            self._select_columns = ["*"]

        return self

    def join(
        self,
        table: str,
        on: str,
        join_type: str = "INNER",
    ) -> "SecureSQLBuilder":
        """Add a JOIN clause.

        Args:
            table: Table to join
            on: Join condition
            join_type: Type of join (INNER, LEFT, RIGHT, etc.)
        """
        if not self.policy.allow_joins:
            raise QueryValidationError("JOINs not allowed by policy")

        self._validate_identifier(table)
        if self.allowed_tables and table not in self.allowed_tables:
            raise QueryValidationError(
                f"Table '{table}' not in allowed list"
            )

        join_type = join_type.upper()
        if join_type not in {"INNER", "LEFT", "RIGHT", "FULL", "CROSS"}:
            raise QueryValidationError(f"Invalid join type: {join_type}")

        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self

    def where(self, condition: str) -> "SecureSQLBuilder":
        """Add a WHERE condition.

        Args:
            condition: WHERE condition (can include :param placeholders)
        """
        self._where_clauses.append(condition)
        return self

    def group_by(self, *columns: str) -> "SecureSQLBuilder":
        """Add GROUP BY columns."""
        for col in columns:
            self._validate_identifier(col.split(".")[-1])
        self._group_by.extend(columns)
        return self

    def having(self, condition: str) -> "SecureSQLBuilder":
        """Add HAVING condition."""
        self._having_clauses.append(condition)
        return self

    def order_by(self, column: str, desc: bool = False) -> "SecureSQLBuilder":
        """Add ORDER BY column."""
        self._validate_identifier(column.split(".")[-1])
        direction = "DESC" if desc else "ASC"
        self._order_by.append(f"{column} {direction}")
        return self

    def limit(self, n: int) -> "SecureSQLBuilder":
        """Set LIMIT."""
        if n < 0:
            raise QueryValidationError("LIMIT must be non-negative")
        self._limit_value = n
        return self

    def offset(self, n: int) -> "SecureSQLBuilder":
        """Set OFFSET."""
        if n < 0:
            raise QueryValidationError("OFFSET must be non-negative")
        self._offset_value = n
        return self

    def build(self, parameters: dict[str, Any] | None = None) -> ParameterizedQuery:
        """Build the parameterized query.

        Args:
            parameters: Query parameters

        Returns:
            ParameterizedQuery ready for execution
        """
        if not self._select_table:
            raise QueryValidationError("No table selected")

        parts = []

        # SELECT
        columns_str = ", ".join(self._select_columns)
        parts.append(f"SELECT {columns_str}")

        # FROM
        parts.append(f"FROM {self._select_table}")

        # JOINs
        for join in self._joins:
            parts.append(join)

        # WHERE
        if self._where_clauses:
            conditions = " AND ".join(f"({c})" for c in self._where_clauses)
            parts.append(f"WHERE {conditions}")

        # GROUP BY
        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")

        # HAVING
        if self._having_clauses:
            conditions = " AND ".join(f"({c})" for c in self._having_clauses)
            parts.append(f"HAVING {conditions}")

        # ORDER BY
        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")

        # LIMIT
        if self._limit_value is not None:
            parts.append(f"LIMIT {self._limit_value}")

        # OFFSET
        if self._offset_value is not None:
            parts.append(f"OFFSET {self._offset_value}")

        template = " ".join(parts)

        return ParameterizedQuery(
            template=template,
            parameters=parameters or {},
        )

    def execute(
        self,
        ctx: pl.SQLContext,
        query: ParameterizedQuery,
    ) -> pl.DataFrame:
        """Execute a parameterized query.

        Args:
            ctx: Polars SQL context
            query: Parameterized query to execute

        Returns:
            Query result as DataFrame
        """
        rendered = query.render()

        # Validate the rendered query
        self.validator.validate(rendered)

        return ctx.execute(rendered).collect()

    def reset(self) -> "SecureSQLBuilder":
        """Reset builder state."""
        self._select_table = None
        self._select_columns = []
        self._joins = []
        self._where_clauses = []
        self._group_by = []
        self._having_clauses = []
        self._order_by = []
        self._limit_value = None
        self._offset_value = None
        return self

    def _validate_identifier(self, identifier: str) -> None:
        """Validate SQL identifier."""
        if not identifier:
            raise QueryValidationError("Empty identifier")

        if len(identifier) > self.policy.max_identifier_length:
            raise QueryValidationError(
                f"Identifier too long: {len(identifier)} > {self.policy.max_identifier_length}"
            )

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            if identifier != "*":
                raise QueryValidationError(
                    f"Invalid identifier '{identifier}': must be alphanumeric with underscores"
                )


# =============================================================================
# Secure Query Mixin
# =============================================================================


class SecureQueryMixin:
    """Mixin providing secure query execution for validators.

    Use in validators that need to execute SQL queries safely.

    Example:
        class MyValidator(BaseValidator, SecureQueryMixin):
            def validate(self, lf):
                query = self.build_secure_query(
                    table="data",
                    columns=["id", "value"],
                    where="value > :threshold",
                    parameters={"threshold": 100},
                )
                result = self.execute_secure_query(lf, query)
                return self.process_result(result)
    """

    _security_policy: SecurityPolicy = SecurityPolicy.standard()
    _sql_validator: SQLQueryValidator | None = None

    def set_security_policy(self, policy: SecurityPolicy) -> None:
        """Set security policy for query execution."""
        self._security_policy = policy
        self._sql_validator = SQLQueryValidator(policy=policy)

    def get_sql_validator(self) -> SQLQueryValidator:
        """Get or create SQL validator."""
        if self._sql_validator is None:
            self._sql_validator = SQLQueryValidator(policy=self._security_policy)
        return self._sql_validator

    def validate_query(self, query: str) -> None:
        """Validate a SQL query for security.

        Args:
            query: Query to validate

        Raises:
            SQLSecurityError: If validation fails
        """
        self.get_sql_validator().validate(query)

    def build_secure_query(
        self,
        table: str,
        columns: list[str] | None = None,
        where: str | None = None,
        parameters: dict[str, Any] | None = None,
        allowed_tables: list[str] | None = None,
    ) -> ParameterizedQuery:
        """Build a secure parameterized query.

        Args:
            table: Table name
            columns: Columns to select
            where: WHERE clause with :param placeholders
            parameters: Parameter values
            allowed_tables: Optional table whitelist

        Returns:
            ParameterizedQuery
        """
        builder = SecureSQLBuilder(
            allowed_tables=allowed_tables,
            policy=self._security_policy,
        )

        builder.select(table, columns)
        if where:
            builder.where(where)

        return builder.build(parameters)

    def execute_secure_query(
        self,
        lf: pl.LazyFrame,
        query: ParameterizedQuery,
        table_name: str = "data",
    ) -> pl.DataFrame:
        """Execute a parameterized query securely.

        Args:
            lf: LazyFrame to query
            query: Parameterized query
            table_name: Name for table in SQL context

        Returns:
            Query result
        """
        rendered = query.render()
        self.validate_query(rendered)

        ctx = pl.SQLContext()
        ctx.register(table_name, lf)
        return ctx.execute(rendered).collect()


# =============================================================================
# Audit Logging
# =============================================================================


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: datetime
    query_hash: str
    query_preview: str
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    user: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


class QueryAuditLogger:
    """Audit logger for SQL query execution.

    Logs all query attempts for security monitoring.

    Example:
        logger = QueryAuditLogger()
        logger.log_query("SELECT * FROM users", success=True)

        # Get recent entries
        for entry in logger.get_recent(10):
            print(f"{entry.timestamp}: {entry.query_preview}")

        # Export to file
        logger.export_to_file("audit.log")
    """

    def __init__(
        self,
        max_entries: int = 10000,
        log_full_queries: bool = False,
        python_logger: logging.Logger | None = None,
    ):
        """Initialize audit logger.

        Args:
            max_entries: Maximum entries to keep in memory
            log_full_queries: Whether to log full query text
            python_logger: Optional Python logger for external logging
        """
        self.max_entries = max_entries
        self.log_full_queries = log_full_queries
        self.python_logger = python_logger
        self._entries: list[AuditEntry] = []

    def log_query(
        self,
        query: str,
        success: bool,
        error: Exception | None = None,
        user: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log a query execution attempt.

        Args:
            query: SQL query
            success: Whether execution succeeded
            error: Optional error that occurred
            user: Optional user identifier
            context: Optional additional context
        """
        # Create hash of query
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        # Create preview (first 100 chars)
        preview = query[:100] + "..." if len(query) > 100 else query
        if not self.log_full_queries:
            preview = re.sub(r"'[^']*'", "'***'", preview)  # Mask string values

        entry = AuditEntry(
            timestamp=datetime.now(),
            query_hash=query_hash,
            query_preview=preview,
            success=success,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            user=user,
            context=context or {},
        )

        self._entries.append(entry)

        # Trim if over limit
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

        # Log to Python logger if configured
        if self.python_logger:
            log_level = logging.INFO if success else logging.WARNING
            self.python_logger.log(
                log_level,
                f"SQL {'OK' if success else 'FAIL'} [{query_hash}]: {preview}",
            )

    def get_recent(self, n: int = 100) -> list[AuditEntry]:
        """Get recent audit entries."""
        return self._entries[-n:]

    def get_failures(self, n: int = 100) -> list[AuditEntry]:
        """Get recent failed queries."""
        failures = [e for e in self._entries if not e.success]
        return failures[-n:]

    def get_by_hash(self, query_hash: str) -> list[AuditEntry]:
        """Get entries by query hash."""
        return [e for e in self._entries if e.query_hash == query_hash]

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    def export_to_file(self, filepath: str) -> None:
        """Export audit log to file.

        Args:
            filepath: Output file path
        """
        import json

        with open(filepath, "w") as f:
            for entry in self._entries:
                record = {
                    "timestamp": entry.timestamp.isoformat(),
                    "query_hash": entry.query_hash,
                    "query_preview": entry.query_preview,
                    "success": entry.success,
                    "error_type": entry.error_type,
                    "error_message": entry.error_message,
                    "user": entry.user,
                    "context": entry.context,
                }
                f.write(json.dumps(record) + "\n")

    def get_stats(self) -> dict[str, Any]:
        """Get audit statistics."""
        total = len(self._entries)
        successes = sum(1 for e in self._entries if e.success)
        failures = total - successes

        return {
            "total_queries": total,
            "successful": successes,
            "failed": failures,
            "success_rate": successes / total if total > 0 else 1.0,
            "unique_queries": len(set(e.query_hash for e in self._entries)),
        }
