"""Integration tests for the Query Pushdown framework.

These tests verify the complete pushdown workflow from query building
to SQL generation across different dialects.
"""

from __future__ import annotations

import pytest

from truthound.execution.pushdown.ast import (
    Column,
    Literal,
    Star,
    Table,
    SelectStatement,
    FromClause,
    WhereClause,
    GroupByClause,
    OrderByClause,
    OrderByItem,
    LimitClause,
    BinaryExpression,
    AggregateFunction,
    ComparisonOp,
    SortOrder,
)
from truthound.execution.pushdown.dialects import (
    SQLDialect,
    get_dialect_generator,
)
from truthound.execution.pushdown.builder import (
    QueryBuilder,
    col,
    literal,
    func,
    and_,
    or_,
    not_,
)
from truthound.execution.pushdown.optimizer import (
    PushdownAnalyzer,
    QueryOptimizer,
)


class TestQueryBuilderIntegration:
    """Integration tests for QueryBuilder."""

    def test_simple_select_all(self):
        """Test simple SELECT * query."""
        query = QueryBuilder("users").select("*")
        stmt = query.build()

        assert stmt is not None
        assert len(stmt.select_items) == 1
        assert stmt.from_clause is not None

    def test_select_with_columns(self):
        """Test SELECT with specific columns."""
        query = QueryBuilder("users").select("id", "name", "email")
        stmt = query.build()

        assert len(stmt.select_items) == 3

    def test_select_with_where(self):
        """Test SELECT with WHERE clause."""
        query = (
            QueryBuilder("users")
            .select("*")
            .where(col("age") > 18)
        )
        stmt = query.build()

        assert stmt.where_clause is not None

    def test_select_with_multiple_conditions(self):
        """Test SELECT with multiple WHERE conditions."""
        query = (
            QueryBuilder("users")
            .select("*")
            .where(
                col("age") > 18,
                col("status") == "active",
            )
        )
        stmt = query.build()

        assert stmt.where_clause is not None

    def test_select_with_group_by(self):
        """Test SELECT with GROUP BY."""
        query = (
            QueryBuilder("orders")
            .select(
                col("customer_id"),
                func.count("*").alias("order_count"),
            )
            .group_by("customer_id")
        )
        stmt = query.build()

        assert stmt.group_by_clause is not None

    def test_select_with_order_by(self):
        """Test SELECT with ORDER BY."""
        query = (
            QueryBuilder("users")
            .select("*")
            .order_by("name")
            .order_by("created_at", desc=True)
        )
        stmt = query.build()

        assert stmt.order_by_clause is not None
        assert len(stmt.order_by_clause.items) == 2

    def test_select_with_limit_offset(self):
        """Test SELECT with LIMIT and OFFSET."""
        query = (
            QueryBuilder("users")
            .select("*")
            .limit(10)
            .offset(20)
        )
        stmt = query.build()

        assert stmt.limit_clause is not None
        assert stmt.offset_clause is not None

    def test_complex_query(self):
        """Test complex query with multiple clauses."""
        query = (
            QueryBuilder("orders", alias="o")
            .select(
                col("o.customer_id"),
                func.count("*").alias("order_count"),
                func.sum("o.amount").alias("total"),
            )
            .where(col("o.status") == "completed")
            .group_by("o.customer_id")
            .having(func.count("*") > 5)
            .order_by("total", desc=True)
            .limit(100)
        )
        stmt = query.build()

        assert stmt.from_clause is not None
        assert stmt.where_clause is not None
        assert stmt.group_by_clause is not None
        assert stmt.having_clause is not None
        assert stmt.order_by_clause is not None
        assert stmt.limit_clause is not None


class TestDialectSQLGeneration:
    """Tests for SQL generation across dialects."""

    @pytest.mark.parametrize("dialect,quote_char", [
        (SQLDialect.POSTGRESQL, '"'),
        (SQLDialect.MYSQL, "`"),
        (SQLDialect.SQLITE, '"'),
        (SQLDialect.BIGQUERY, "`"),
        (SQLDialect.SNOWFLAKE, '"'),
    ])
    def test_identifier_quoting(self, dialect, quote_char):
        """Test identifier quoting varies by dialect."""
        gen = get_dialect_generator(dialect)
        col = Column("name")
        sql = gen.generate(col)

        assert quote_char in sql

    def test_postgresql_sql_generation(self):
        """Test PostgreSQL SQL generation."""
        query = (
            QueryBuilder("users")
            .select("id", "name")
            .where(col("age") > 18)
            .limit(10)
        )
        sql = query.to_sql(SQLDialect.POSTGRESQL)

        assert "SELECT" in sql
        assert "FROM" in sql
        assert "WHERE" in sql
        assert "LIMIT" in sql
        assert '"users"' in sql

    def test_mysql_sql_generation(self):
        """Test MySQL SQL generation."""
        query = (
            QueryBuilder("users")
            .select("id", "name")
            .where(col("active") == True)
            .limit(10)
        )
        sql = query.to_sql(SQLDialect.MYSQL)

        assert "SELECT" in sql
        assert "`users`" in sql
        assert "LIMIT" in sql

    def test_bigquery_sql_generation(self):
        """Test BigQuery SQL generation."""
        query = (
            QueryBuilder("users")
            .select(col("department"), func.count("*").alias("cnt"))
            .group_by("department")
        )
        sql = query.to_sql(SQLDialect.BIGQUERY)

        assert "SELECT" in sql
        assert "GROUP BY" in sql
        assert "`department`" in sql


class TestExpressionOperators:
    """Test expression operator functionality."""

    def test_comparison_operators(self):
        """Test comparison operators."""
        c = col("age")

        eq_expr = c == 18
        assert isinstance(eq_expr, BinaryExpression)
        assert eq_expr.operator == ComparisonOp.EQ

        ne_expr = c != 18
        assert ne_expr.operator == ComparisonOp.NE

        lt_expr = c < 18
        assert lt_expr.operator == ComparisonOp.LT

        gt_expr = c > 18
        assert gt_expr.operator == ComparisonOp.GT

    def test_logical_operators(self):
        """Test logical operators."""
        a = col("x") == 1
        b = col("y") == 2

        and_expr = a & b
        assert isinstance(and_expr, BinaryExpression)

        or_expr = a | b
        assert isinstance(or_expr, BinaryExpression)

    def test_arithmetic_operators(self):
        """Test arithmetic operators."""
        c = col("price")

        add_expr = c + 10
        assert isinstance(add_expr, BinaryExpression)

        sub_expr = c - 10
        assert isinstance(sub_expr, BinaryExpression)

        mul_expr = c * 2
        assert isinstance(mul_expr, BinaryExpression)

        div_expr = c / 2
        assert isinstance(div_expr, BinaryExpression)

    def test_in_expression(self):
        """Test IN expression."""
        expr = col("status").in_(["active", "pending"])
        assert expr is not None
        assert not expr.negated

    def test_between_expression(self):
        """Test BETWEEN expression."""
        expr = col("age").between(18, 65)
        assert expr is not None
        assert not expr.negated

    def test_like_expression(self):
        """Test LIKE expression."""
        expr = col("name").like("%Smith%")
        assert isinstance(expr, BinaryExpression)
        assert expr.operator == ComparisonOp.LIKE

    def test_null_check(self):
        """Test IS NULL and IS NOT NULL."""
        c = col("deleted_at")

        is_null = c.is_null()
        assert is_null is not None

        is_not_null = c.is_not_null()
        assert is_not_null is not None


class TestAggregateFunctions:
    """Test aggregate function building."""

    def test_count_star(self):
        """Test COUNT(*)."""
        agg = func.count("*")
        assert agg.name == "COUNT"
        assert agg.argument is None

    def test_count_column(self):
        """Test COUNT(column)."""
        agg = func.count("id")
        assert agg.name == "COUNT"
        assert agg.argument is not None

    def test_count_distinct(self):
        """Test COUNT(DISTINCT column)."""
        agg = func.count_distinct("email")
        assert agg.distinct is True

    def test_sum(self):
        """Test SUM."""
        agg = func.sum("amount")
        assert agg.name == "SUM"

    def test_avg(self):
        """Test AVG."""
        agg = func.avg("price")
        assert agg.name == "AVG"

    def test_min_max(self):
        """Test MIN and MAX."""
        min_agg = func.min("date")
        assert min_agg.name == "MIN"

        max_agg = func.max("date")
        assert max_agg.name == "MAX"


class TestWindowFunctions:
    """Test window function building."""

    def test_row_number(self):
        """Test ROW_NUMBER()."""
        f = func.row_number()
        assert f.name == "ROW_NUMBER"

    def test_rank(self):
        """Test RANK()."""
        f = func.rank()
        assert f.name == "RANK"

    def test_lead_lag(self):
        """Test LEAD and LAG."""
        lead = func.lead("value", 1)
        assert lead.name == "LEAD"

        lag = func.lag("value", 1, 0)
        assert lag.name == "LAG"


class TestPushdownAnalysis:
    """Test pushdown analysis."""

    def test_simple_select_can_pushdown(self):
        """Test simple SELECT can be pushed down."""
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
        )

        analyzer = PushdownAnalyzer(SQLDialect.POSTGRESQL)
        decision = analyzer.analyze(stmt)

        assert decision.can_pushdown is True

    def test_aggregation_can_pushdown(self):
        """Test aggregation can be pushed down."""
        stmt = SelectStatement(
            select_items=[
                Column("department"),
                AggregateFunction("COUNT", None),
            ],
            from_clause=FromClause(Table("employees")),
            group_by_clause=GroupByClause([Column("department")]),
        )

        analyzer = PushdownAnalyzer(SQLDialect.POSTGRESQL)
        decision = analyzer.analyze(stmt)

        assert decision.can_pushdown is True


class TestQueryOptimization:
    """Test query optimization."""

    def test_optimizer_creation(self):
        """Test optimizer can be created."""
        optimizer = QueryOptimizer()
        assert optimizer is not None

    def test_optimize_simple_query(self):
        """Test optimizing a simple query."""
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            where_clause=WhereClause(
                BinaryExpression(Column("age"), ComparisonOp.GT, Literal(18))
            ),
        )

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(stmt)

        assert isinstance(optimized, SelectStatement)


class TestUnionOperations:
    """Test UNION and set operations."""

    def test_union(self):
        """Test UNION."""
        q1 = QueryBuilder("users").select("id", "name")
        q2 = QueryBuilder("admins").select("id", "name")

        union = q1.union(q2)
        stmt = union.build()

        assert stmt is not None

    def test_union_all(self):
        """Test UNION ALL."""
        q1 = QueryBuilder("users").select("id")
        q2 = QueryBuilder("admins").select("id")

        union = q1.union(q2, all=True)
        sql = union.to_sql(SQLDialect.POSTGRESQL)

        assert "UNION ALL" in sql


class TestJoinOperations:
    """Test JOIN operations."""

    def test_inner_join(self):
        """Test INNER JOIN."""
        query = (
            QueryBuilder("users")
            .select(col("users.name"), col("orders.amount"))
            .inner_join("orders", col("users.id") == col("orders.user_id"))
        )
        stmt = query.build()

        assert stmt.from_clause is not None

    def test_left_join(self):
        """Test LEFT JOIN."""
        query = (
            QueryBuilder("users")
            .select("*")
            .left_join("orders", col("users.id") == col("orders.user_id"))
        )
        stmt = query.build()

        assert stmt.from_clause is not None

    def test_multiple_joins(self):
        """Test multiple JOINs."""
        query = (
            QueryBuilder("users")
            .select("*")
            .inner_join("orders", col("users.id") == col("orders.user_id"))
            .left_join("products", col("orders.product_id") == col("products.id"))
        )
        stmt = query.build()

        assert stmt.from_clause is not None


class TestLogicalOperatorHelpers:
    """Test logical operator helper functions."""

    def test_and_single(self):
        """Test and_ with single condition."""
        cond = col("a") == 1
        result = and_(cond)
        assert result is cond

    def test_and_multiple(self):
        """Test and_ with multiple conditions."""
        result = and_(
            col("a") == 1,
            col("b") == 2,
            col("c") == 3,
        )
        assert isinstance(result, BinaryExpression)

    def test_or_multiple(self):
        """Test or_ with multiple conditions."""
        result = or_(
            col("status") == "active",
            col("status") == "pending",
        )
        assert isinstance(result, BinaryExpression)

    def test_empty_and_raises(self):
        """Test and_ with no conditions raises."""
        with pytest.raises(ValueError):
            and_()

    def test_empty_or_raises(self):
        """Test or_ with no conditions raises."""
        with pytest.raises(ValueError):
            or_()


class TestDistinctAndAlias:
    """Test DISTINCT and alias functionality."""

    def test_select_distinct(self):
        """Test SELECT DISTINCT."""
        query = QueryBuilder("users").select("country").distinct()
        stmt = query.build()

        assert stmt.distinct is True

    def test_column_alias(self):
        """Test column alias."""
        query = QueryBuilder("users").select(
            col("first_name").alias("name")
        )
        stmt = query.build()

        assert len(stmt.select_items) == 1

    def test_aggregate_alias(self):
        """Test aggregate with alias."""
        query = QueryBuilder("orders").select(
            func.sum("amount").alias("total")
        )
        stmt = query.build()

        assert len(stmt.select_items) == 1
