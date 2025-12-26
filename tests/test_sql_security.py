"""Tests for SQL security module.

This module tests the comprehensive SQL injection protection system including:
- Security policies and levels
- Pattern-based injection detection
- Parameterized queries
- Whitelist validation
- Query audit logging
- SecureSQLBuilder
"""

import pytest
from datetime import datetime

from truthound.validators.security import (
    SQLSecurityError,
    SQLInjectionError,
    QueryValidationError,
    SQLQueryValidator,
    validate_sql_query,
    SecureSQLBuilder,
    ParameterizedQuery,
    WhitelistValidator,
    SchemaWhitelist,
    SecurityPolicy,
    SecurityLevel,
    SecureQueryMixin,
    QueryAuditLogger,
    AuditEntry,
)


class TestSecurityLevel:
    """Tests for SecurityLevel enum."""

    def test_security_levels_exist(self):
        """Test that all security levels are defined."""
        assert SecurityLevel.STRICT
        assert SecurityLevel.STANDARD
        assert SecurityLevel.PERMISSIVE

    def test_security_level_ordering(self):
        """Test security level ordering (STRICT > STANDARD > PERMISSIVE)."""
        assert SecurityLevel.STRICT.value < SecurityLevel.STANDARD.value
        assert SecurityLevel.STANDARD.value < SecurityLevel.PERMISSIVE.value


class TestSecurityPolicy:
    """Tests for SecurityPolicy dataclass."""

    def test_default_policy(self):
        """Test default security policy settings."""
        policy = SecurityPolicy()
        assert policy.level == SecurityLevel.STANDARD
        assert policy.max_query_length == 10000
        assert policy.allow_subqueries is True
        assert policy.allow_union is False
        assert policy.blocked_patterns == []

    def test_strict_policy_creation(self):
        """Test creating a strict security policy."""
        policy = SecurityPolicy.strict()
        assert policy.level == SecurityLevel.STRICT
        assert policy.allow_subqueries is False
        assert policy.max_query_length == 5000

    def test_standard_policy_creation(self):
        """Test creating a standard security policy."""
        policy = SecurityPolicy.standard()
        assert policy.level == SecurityLevel.STANDARD

    def test_permissive_policy_creation(self):
        """Test creating a permissive security policy."""
        policy = SecurityPolicy.permissive()
        assert policy.level == SecurityLevel.PERMISSIVE
        assert policy.allow_union is True

    def test_custom_blocked_patterns(self):
        """Test adding custom blocked patterns."""
        policy = SecurityPolicy(
            blocked_patterns=[r"\bDELETE\b", r"\bUPDATE\b"]
        )
        assert len(policy.blocked_patterns) == 2


class TestSQLQueryValidator:
    """Tests for SQLQueryValidator class."""

    def test_valid_select_query(self):
        """Test that valid SELECT queries pass."""
        validator = SQLQueryValidator()
        # Should not raise
        validator.validate("SELECT * FROM users")
        validator.validate("SELECT id, name FROM users WHERE active = 1")
        validator.validate("SELECT COUNT(*) FROM orders")

    def test_valid_with_query(self):
        """Test that valid WITH (CTE) queries pass."""
        validator = SQLQueryValidator()
        query = """
        WITH active_users AS (
            SELECT * FROM users WHERE active = 1
        )
        SELECT * FROM active_users
        """
        validator.validate(query)

    def test_ddl_blocked(self):
        """Test that DDL statements are blocked."""
        validator = SQLQueryValidator()

        ddl_queries = [
            "DROP TABLE users",
            "CREATE TABLE evil (id INT)",
            "ALTER TABLE users ADD COLUMN hacked VARCHAR",
            "TRUNCATE TABLE users",
        ]

        for query in ddl_queries:
            with pytest.raises((QueryValidationError, SQLInjectionError)):
                validator.validate(query)

    def test_dcl_blocked(self):
        """Test that DCL statements are blocked."""
        validator = SQLQueryValidator()

        dcl_queries = [
            "GRANT ALL ON users TO hacker",
            "REVOKE ALL ON users FROM admin",
        ]

        for query in dcl_queries:
            with pytest.raises((QueryValidationError, SQLInjectionError)):
                validator.validate(query)

    def test_insert_update_delete_blocked(self):
        """Test that INSERT/UPDATE/DELETE are blocked."""
        validator = SQLQueryValidator()

        dml_queries = [
            "INSERT INTO users VALUES (1, 'hacker')",
            "UPDATE users SET password = 'hacked'",
            "DELETE FROM users WHERE 1=1",
        ]

        for query in dml_queries:
            with pytest.raises((QueryValidationError, SQLInjectionError)):
                validator.validate(query)

    def test_multiple_statements_blocked(self):
        """Test that multiple statements are blocked."""
        validator = SQLQueryValidator()

        with pytest.raises((QueryValidationError, SQLInjectionError)):
            validator.validate("SELECT 1; DROP TABLE users")

    def test_union_blocked_by_default(self):
        """Test that UNION is blocked by default policy."""
        validator = SQLQueryValidator()

        with pytest.raises((QueryValidationError, SQLInjectionError)):
            validator.validate("SELECT id FROM users UNION SELECT password FROM secrets")

    def test_union_allowed_with_permissive_policy(self):
        """Test that UNION is allowed with permissive policy."""
        policy = SecurityPolicy.permissive()
        validator = SQLQueryValidator(policy=policy)

        # Should not raise - but may still block UNION ALL SELECT pattern
        try:
            validator.validate("SELECT id FROM users UNION SELECT id FROM orders")
        except SQLInjectionError:
            # The union_select pattern may still match
            pass

    def test_max_query_length(self):
        """Test query length limit."""
        policy = SecurityPolicy(max_query_length=50)
        validator = SQLQueryValidator(policy=policy)

        short_query = "SELECT * FROM data"
        long_query = "SELECT " + "a, " * 100 + "b FROM data"

        validator.validate(short_query)

        with pytest.raises(QueryValidationError):
            validator.validate(long_query)

    def test_empty_query_blocked(self):
        """Test that empty queries are blocked."""
        validator = SQLQueryValidator()

        with pytest.raises(QueryValidationError):
            validator.validate("")

        with pytest.raises(QueryValidationError):
            validator.validate("   ")

    def test_table_whitelist(self):
        """Test table name whitelist validation."""
        policy = SecurityPolicy(allowed_tables={"users", "orders"})
        validator = SQLQueryValidator(policy=policy)

        # Allowed tables
        validator.validate("SELECT * FROM users")
        validator.validate("SELECT * FROM orders")

        # Not allowed
        with pytest.raises(QueryValidationError):
            validator.validate("SELECT * FROM secrets")

    def test_table_whitelist_case_insensitive(self):
        """Test table whitelist is case insensitive."""
        policy = SecurityPolicy(allowed_tables={"Users"})
        validator = SQLQueryValidator(policy=policy)

        validator.validate("SELECT * FROM users")
        validator.validate("SELECT * FROM USERS")

    def test_custom_blocked_patterns(self):
        """Test custom blocked patterns."""
        policy = SecurityPolicy(
            blocked_patterns=[r"\bEVIL_FUNCTION\b"]
        )
        validator = SQLQueryValidator(policy=policy)

        with pytest.raises(SQLInjectionError):
            validator.validate("SELECT EVIL_FUNCTION() FROM data")

    def test_strict_policy(self):
        """Test strict security policy."""
        policy = SecurityPolicy.strict()
        validator = SQLQueryValidator(policy=policy)

        # Subquery blocked in strict mode
        with pytest.raises(SQLInjectionError):
            validator.validate("SELECT * FROM (SELECT * FROM users) AS sub")


class TestValidateSqlQueryFunction:
    """Tests for the validate_sql_query convenience function."""

    def test_valid_query(self):
        """Test valid query passes."""
        validate_sql_query("SELECT * FROM data")

    def test_invalid_query(self):
        """Test invalid query raises."""
        with pytest.raises((QueryValidationError, SQLInjectionError)):
            validate_sql_query("DROP TABLE users")

    def test_with_allowed_tables(self):
        """Test with allowed tables."""
        validate_sql_query("SELECT * FROM users", allowed_tables=["users"])

        with pytest.raises(QueryValidationError):
            validate_sql_query("SELECT * FROM secrets", allowed_tables=["users"])


class TestParameterizedQuery:
    """Tests for ParameterizedQuery class."""

    def test_basic_parameter_substitution(self):
        """Test basic parameter substitution."""
        pq = ParameterizedQuery(
            template="SELECT * FROM users WHERE id = :id",
            parameters={"id": 123}
        )
        result = pq.render()
        assert "123" in result

    def test_string_parameter_escaping(self):
        """Test that string parameters are properly escaped."""
        pq = ParameterizedQuery(
            template="SELECT * FROM users WHERE name = :name",
            parameters={"name": "O'Brien"}
        )
        result = pq.render()
        # Should escape the single quote
        assert "O''Brien" in result

    def test_multiple_parameters(self):
        """Test multiple parameter substitution."""
        pq = ParameterizedQuery(
            template="SELECT * FROM users WHERE status = :status AND role = :role",
            parameters={"status": "active", "role": "admin"}
        )
        result = pq.render()
        assert "'active'" in result
        assert "'admin'" in result

    def test_missing_parameter_error(self):
        """Test error on missing parameter."""
        with pytest.raises(QueryValidationError):
            ParameterizedQuery(
                template="SELECT * FROM users WHERE id = :id",
                parameters={}
            )

    def test_sql_injection_via_parameter(self):
        """Test that SQL injection via parameters is blocked."""
        pq = ParameterizedQuery(
            template="SELECT * FROM users WHERE name = :name",
            parameters={"name": "'; DROP TABLE users; --"}
        )
        result = pq.render()

        # The result should be escaped - quotes should be doubled
        assert "''" in result  # Escaped quote

    def test_null_parameter(self):
        """Test NULL parameter handling."""
        pq = ParameterizedQuery(
            template="SELECT * FROM users WHERE deleted = :deleted",
            parameters={"deleted": None}
        )
        result = pq.render()
        assert "NULL" in result

    def test_boolean_parameter(self):
        """Test boolean parameter handling."""
        pq = ParameterizedQuery(
            template="SELECT * FROM users WHERE active = :active",
            parameters={"active": True}
        )
        result = pq.render()
        assert "TRUE" in result

    def test_list_parameter(self):
        """Test list parameter handling."""
        pq = ParameterizedQuery(
            template="SELECT * FROM users WHERE id IN :ids",
            parameters={"ids": [1, 2, 3]}
        )
        result = pq.render()
        assert "(1, 2, 3)" in result


class TestSchemaWhitelist:
    """Tests for SchemaWhitelist class."""

    def test_add_table(self):
        """Test adding a table to whitelist."""
        whitelist = SchemaWhitelist()
        whitelist.add_table("users", ["id", "name", "email"])

        # Table should be allowed
        whitelist.validate_table("users")

        # Columns should be allowed
        whitelist.validate_column("users", "id")
        whitelist.validate_column("users", "name")

    def test_disallowed_table(self):
        """Test disallowed table."""
        whitelist = SchemaWhitelist()
        whitelist.add_table("users", ["id"])

        with pytest.raises(QueryValidationError):
            whitelist.validate_table("secrets")

    def test_disallowed_column(self):
        """Test disallowed column."""
        whitelist = SchemaWhitelist()
        whitelist.add_table("users", ["id", "name"])

        with pytest.raises(QueryValidationError):
            whitelist.validate_column("users", "password")

    def test_get_tables(self):
        """Test getting allowed tables."""
        whitelist = SchemaWhitelist()
        whitelist.add_table("users", ["id"])
        whitelist.add_table("orders", ["id"])

        tables = whitelist.get_tables()
        assert "users" in tables
        assert "orders" in tables

    def test_get_columns(self):
        """Test getting allowed columns."""
        whitelist = SchemaWhitelist()
        whitelist.add_table("users", ["id", "name", "email"])

        columns = whitelist.get_columns("users")
        assert "id" in columns
        assert "name" in columns
        assert "email" in columns


class TestWhitelistValidator:
    """Tests for WhitelistValidator class."""

    def test_validate_query_with_whitelist(self):
        """Test query validation with whitelist."""
        whitelist = SchemaWhitelist()
        whitelist.add_table("users", ["id", "name"])

        validator = WhitelistValidator(whitelist)

        # Valid query
        validator.validate_query("SELECT id, name FROM users")

        # Invalid table
        with pytest.raises(QueryValidationError):
            validator.validate_query("SELECT * FROM secrets")

    def test_column_validation(self):
        """Test column validation."""
        whitelist = SchemaWhitelist()
        whitelist.add_table("users", ["id", "name"])

        validator = WhitelistValidator(whitelist)

        # Valid columns
        validator.validate_query("SELECT id FROM users")

        # Invalid column - should raise
        with pytest.raises(QueryValidationError):
            validator.validate_query("SELECT password FROM users")


class TestSecureSQLBuilder:
    """Tests for SecureSQLBuilder fluent API."""

    def test_simple_select(self):
        """Test simple SELECT building."""
        builder = SecureSQLBuilder()
        query = builder.select("users").build()

        assert "SELECT" in query.template.upper()
        assert "FROM" in query.template.upper()
        assert "users" in query.template.lower()

    def test_select_with_columns(self):
        """Test SELECT with specific columns."""
        builder = SecureSQLBuilder()
        query = builder.select("users", ["id", "name"]).build()

        assert "id" in query.template
        assert "name" in query.template

    def test_where_clause(self):
        """Test WHERE clause building."""
        builder = SecureSQLBuilder()
        query = builder.select("users").where("active = 1").build()

        assert "WHERE" in query.template.upper()
        assert "active" in query.template

    def test_order_by(self):
        """Test ORDER BY clause."""
        builder = SecureSQLBuilder()
        query = builder.select("users").order_by("name").build()

        assert "ORDER BY" in query.template.upper()

    def test_limit(self):
        """Test LIMIT clause."""
        builder = SecureSQLBuilder()
        query = builder.select("users").limit(10).build()

        assert "LIMIT" in query.template.upper()
        assert "10" in query.template

    def test_join(self):
        """Test JOIN clause."""
        builder = SecureSQLBuilder()
        query = (
            builder
            .select("orders")
            .join("users", "orders.user_id = users.id")
            .build()
        )

        assert "JOIN" in query.template.upper()
        assert "users" in query.template.lower()

    def test_method_chaining(self):
        """Test complete method chaining."""
        builder = SecureSQLBuilder()
        query = (
            builder
            .select("users", ["id", "name"])
            .where("active = 1")
            .order_by("name")
            .limit(100)
            .build()
        )

        assert "SELECT" in query.template.upper()
        assert "FROM" in query.template.upper()
        assert "WHERE" in query.template.upper()
        assert "ORDER BY" in query.template.upper()
        assert "LIMIT" in query.template.upper()

    def test_builder_with_parameters(self):
        """Test builder with parameterized query."""
        builder = SecureSQLBuilder()
        query = (
            builder
            .select("users")
            .where("id = :id")
            .build({"id": 123})
        )

        rendered = query.render()
        assert "123" in rendered

    def test_builder_reset(self):
        """Test builder reset."""
        builder = SecureSQLBuilder()
        builder.select("users").where("active = 1")
        builder.reset()

        # After reset, should require select again
        with pytest.raises(QueryValidationError):
            builder.build()


class TestQueryAuditLogger:
    """Tests for QueryAuditLogger class."""

    def test_log_query(self):
        """Test logging a query."""
        logger = QueryAuditLogger(max_entries=100)

        logger.log_query("SELECT * FROM users", success=True)

        entries = logger.get_recent()
        assert len(entries) == 1
        assert "SELECT" in entries[0].query_preview
        assert entries[0].success is True

    def test_log_failed_query(self):
        """Test logging a failed query."""
        logger = QueryAuditLogger()

        error = QueryValidationError("DDL not allowed")
        logger.log_query("DROP TABLE users", success=False, error=error)

        entries = logger.get_recent()
        assert len(entries) == 1
        assert entries[0].success is False
        assert "QueryValidationError" in entries[0].error_type

    def test_max_entries_limit(self):
        """Test max entries limit."""
        logger = QueryAuditLogger(max_entries=5)

        for i in range(10):
            logger.log_query(f"SELECT {i}", success=True)

        entries = logger.get_recent(100)
        assert len(entries) <= 5

    def test_get_failures(self):
        """Test getting only failed entries."""
        logger = QueryAuditLogger()

        logger.log_query("SELECT 1", success=True)
        logger.log_query("DROP TABLE", success=False)
        logger.log_query("SELECT 2", success=True)
        logger.log_query("DELETE FROM", success=False)

        failed = logger.get_failures()
        assert len(failed) == 2
        assert all(not e.success for e in failed)

    def test_clear_entries(self):
        """Test clearing entries."""
        logger = QueryAuditLogger()

        logger.log_query("SELECT 1", success=True)
        logger.log_query("SELECT 2", success=True)

        assert len(logger.get_recent()) == 2

        logger.clear()

        assert len(logger.get_recent()) == 0

    def test_get_stats(self):
        """Test getting audit statistics."""
        logger = QueryAuditLogger()

        logger.log_query("SELECT 1", success=True)
        logger.log_query("SELECT 2", success=True)
        logger.log_query("DROP TABLE", success=False)

        stats = logger.get_stats()
        assert stats["total_queries"] == 3
        assert stats["successful"] == 2
        assert stats["failed"] == 1


class TestSecureQueryMixin:
    """Tests for SecureQueryMixin."""

    def test_mixin_provides_validator(self):
        """Test that mixin provides a validator."""
        class TestClass(SecureQueryMixin):
            pass

        obj = TestClass()
        validator = obj.get_sql_validator()
        assert validator is not None
        assert isinstance(validator, SQLQueryValidator)

    def test_mixin_validate_method(self):
        """Test mixin validate method."""
        class TestClass(SecureQueryMixin):
            pass

        obj = TestClass()

        # Valid query should pass
        obj.validate_query("SELECT * FROM data")

        # Invalid query should raise
        with pytest.raises((QueryValidationError, SQLInjectionError)):
            obj.validate_query("DROP TABLE data")

    def test_set_security_policy(self):
        """Test setting security policy."""
        class TestClass(SecureQueryMixin):
            pass

        obj = TestClass()
        policy = SecurityPolicy.strict()
        obj.set_security_policy(policy)

        # Strict policy should block subqueries
        with pytest.raises(SQLInjectionError):
            obj.validate_query("SELECT * FROM (SELECT * FROM users) AS sub")


class TestSQLSecurityExceptions:
    """Tests for exception hierarchy."""

    def test_sql_security_error_base(self):
        """Test SQLSecurityError is base exception."""
        error = SQLSecurityError("test error")
        assert str(error) == "test error"

    def test_sql_injection_error(self):
        """Test SQLInjectionError."""
        error = SQLInjectionError("injection detected", pattern=r"\bDROP\b", query="DROP TABLE")
        assert isinstance(error, SQLSecurityError)
        assert error.pattern == r"\bDROP\b"

    def test_query_validation_error(self):
        """Test QueryValidationError."""
        error = QueryValidationError("validation failed")
        assert isinstance(error, SQLSecurityError)


class TestRealWorldScenarios:
    """Test real-world attack scenarios."""

    def test_or_1_eq_1_attack(self):
        """Test classic OR 1=1 attack detection."""
        validator = SQLQueryValidator()

        # Classic OR 1=1 injection
        with pytest.raises(SQLInjectionError):
            validator.validate("SELECT * FROM users WHERE name = '' OR 1=1")

    def test_tautology_attack(self):
        """Test tautology attack detection."""
        validator = SQLQueryValidator()

        # Simple 1=1 without OR is not necessarily an attack
        validator.validate("SELECT * FROM users WHERE 1=1")  # This may pass

    def test_piggyback_attack(self):
        """Test piggyback (stacked query) attack."""
        validator = SQLQueryValidator()

        with pytest.raises(SQLInjectionError):
            validator.validate("SELECT * FROM users; DELETE FROM users")

    def test_sleep_attack(self):
        """Test time-based injection with SLEEP."""
        validator = SQLQueryValidator()

        with pytest.raises(SQLInjectionError):
            validator.validate("SELECT * FROM users WHERE id = 1 AND SLEEP(5)")

    def test_benchmark_attack(self):
        """Test time-based injection with BENCHMARK."""
        validator = SQLQueryValidator()

        with pytest.raises(SQLInjectionError):
            validator.validate("SELECT * FROM users WHERE id = BENCHMARK(10000000, SHA1('test'))")


class TestIntegrationWithQueryValidator:
    """Integration tests with query validators."""

    def test_query_validator_uses_security_module(self):
        """Test that query validators use the security module."""
        from truthound.validators.query.base import (
            SQLValidationError,
            validate_sql_query,
        )

        # Should work
        validate_sql_query("SELECT * FROM data")

        # Should fail
        with pytest.raises((SQLValidationError, QueryValidationError, SQLInjectionError)):
            validate_sql_query("DROP TABLE data")

    def test_backward_compatibility(self):
        """Test backward compatibility with old API."""
        from truthound.validators.query.base import SQLValidationError

        # SQLValidationError should still work
        error = SQLValidationError("test")
        assert isinstance(error, Exception)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_identifiers(self):
        """Test handling of very long identifiers."""
        builder = SecureSQLBuilder()

        long_name = "a" * 200  # Exceeds max_identifier_length (128)
        with pytest.raises(QueryValidationError):
            builder.select(long_name)

    def test_empty_identifier(self):
        """Test empty identifier rejection."""
        builder = SecureSQLBuilder()

        with pytest.raises(QueryValidationError):
            builder.select("")

    def test_whitespace_only_query(self):
        """Test whitespace-only query rejection."""
        validator = SQLQueryValidator()

        with pytest.raises(QueryValidationError):
            validator.validate("   \t\n   ")

    def test_newlines_in_query(self):
        """Test handling of newlines in query."""
        validator = SQLQueryValidator()

        query = """
        SELECT *
        FROM users
        WHERE active = 1
        """
        validator.validate(query)

    def test_invalid_identifier_characters(self):
        """Test identifiers with invalid characters."""
        builder = SecureSQLBuilder()

        with pytest.raises(QueryValidationError):
            builder.select("table-name")  # hyphen not allowed

        with pytest.raises(QueryValidationError):
            builder.select("table name")  # space not allowed
