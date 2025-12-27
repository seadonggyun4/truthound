"""Tests for QueryBuilder and expression builders."""

from __future__ import annotations

import pytest

from truthound.execution.pushdown.builder import (
    QueryBuilder,
    col,
    literal,
    star,
    func,
    and_,
    or_,
    not_,
    case,
    when,
    cast,
    window,
)
from truthound.execution.pushdown.ast import (
    Column,
    Literal,
    Star,
    BinaryExpression,
    UnaryExpression,
    InExpression,
    BetweenExpression,
    CaseExpression,
    CastExpression,
    AggregateFunction,
    WindowFunction,
    SelectStatement,
    ComparisonOp,
    LogicalOp,
    SortOrder,
    JoinType,
)
from truthound.execution.pushdown.dialects import SQLDialect


class TestColumnBuilder:
    """Tests for col() builder function."""

    def test_simple_column(self):
        """Test simple column creation."""
        c = col("name")
        assert isinstance(c, Column)
        assert c.name == "name"
        assert c.table is None

    def test_qualified_column(self):
        """Test qualified column with dot notation."""
        c = col("users.name")
        assert c.name == "name"
        assert c.table == "users"

    def test_column_with_table_arg(self):
        """Test column with explicit table argument."""
        c = col("name", "users")
        assert c.name == "name"
        assert c.table == "users"

    def test_three_part_name(self):
        """Test three-part column name."""
        c = col("public.users.name")
        assert c.name == "name"
        assert c.table == "users"
        assert c.schema == "public"


class TestLiteralBuilder:
    """Tests for literal() builder function."""

    def test_number_literal(self):
        """Test numeric literal."""
        lit = literal(42)
        assert isinstance(lit, Literal)
        assert lit.value == 42

    def test_string_literal(self):
        """Test string literal."""
        lit = literal("hello")
        assert lit.value == "hello"

    def test_none_literal(self):
        """Test None becomes NULL."""
        from truthound.execution.pushdown.ast import NullLiteral
        lit = literal(None)
        assert isinstance(lit, NullLiteral)


class TestFunctionBuilder:
    """Tests for func builder."""

    def test_count_star(self):
        """Test COUNT(*)."""
        agg = func.count("*")
        assert isinstance(agg, AggregateFunction)
        assert agg.name == "COUNT"
        assert agg.argument is None

    def test_count_column(self):
        """Test COUNT(column)."""
        agg = func.count("id")
        assert agg.argument is not None

    def test_count_distinct(self):
        """Test COUNT(DISTINCT column)."""
        agg = func.count_distinct("email")
        assert agg.distinct

    def test_sum(self):
        """Test SUM."""
        agg = func.sum("amount")
        assert agg.name == "SUM"

    def test_avg(self):
        """Test AVG."""
        agg = func.avg("price")
        assert agg.name == "AVG"

    def test_min(self):
        """Test MIN."""
        agg = func.min("date")
        assert agg.name == "MIN"

    def test_max(self):
        """Test MAX."""
        agg = func.max("date")
        assert agg.name == "MAX"

    def test_row_number(self):
        """Test ROW_NUMBER."""
        agg = func.row_number()
        assert agg.name == "ROW_NUMBER"

    def test_rank(self):
        """Test RANK."""
        agg = func.rank()
        assert agg.name == "RANK"

    def test_lead(self):
        """Test LEAD."""
        f = func.lead("value", 1)
        assert f.name == "LEAD"

    def test_lag(self):
        """Test LAG."""
        f = func.lag("value", 1, 0)
        assert f.name == "LAG"
        assert len(f.arguments) == 3

    def test_coalesce(self):
        """Test COALESCE."""
        f = func.coalesce("a", "b", "default")
        assert f.name == "COALESCE"
        assert len(f.arguments) == 3

    def test_upper(self):
        """Test UPPER."""
        f = func.upper("name")
        assert f.name == "UPPER"

    def test_lower(self):
        """Test LOWER."""
        f = func.lower("name")
        assert f.name == "LOWER"

    def test_concat(self):
        """Test CONCAT."""
        f = func.concat("first", " ", "last")
        assert f.name == "CONCAT"
        assert len(f.arguments) == 3

    def test_round(self):
        """Test ROUND."""
        f = func.round("value", 2)
        assert f.name == "ROUND"


class TestLogicalOperators:
    """Tests for logical operator functions."""

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
        assert result.operator == LogicalOp.AND

    def test_or_single(self):
        """Test or_ with single condition."""
        cond = col("a") == 1
        result = or_(cond)
        assert result is cond

    def test_or_multiple(self):
        """Test or_ with multiple conditions."""
        result = or_(
            col("status") == "active",
            col("status") == "pending",
        )
        assert isinstance(result, BinaryExpression)
        assert result.operator == LogicalOp.OR

    def test_not(self):
        """Test not_ function."""
        result = not_(col("active"))
        assert isinstance(result, UnaryExpression)

    def test_empty_and_raises(self):
        """Test and_ with no conditions raises."""
        with pytest.raises(ValueError):
            and_()

    def test_empty_or_raises(self):
        """Test or_ with no conditions raises."""
        with pytest.raises(ValueError):
            or_()


class TestCaseBuilder:
    """Tests for CASE expression builder."""

    def test_simple_case(self):
        """Test simple CASE expression."""
        result = case(
            when(col("status") == "active", "Active"),
            when(col("status") == "pending", "Pending"),
            else_="Unknown",
        )
        assert isinstance(result, CaseExpression)
        assert len(result.when_clauses) == 2
        assert result.else_result is not None


class TestCast:
    """Tests for CAST builder."""

    def test_cast_column(self):
        """Test CAST on column."""
        result = cast(col("id"), "VARCHAR(10)")
        assert isinstance(result, CastExpression)
        assert result.target_type == "VARCHAR(10)"


class TestWindowBuilder:
    """Tests for window specification builder."""

    def test_partition_by(self):
        """Test PARTITION BY."""
        spec = window().partition_by("department").build()
        assert spec.partition_by is not None
        assert len(spec.partition_by) == 1

    def test_order_by(self):
        """Test ORDER BY in window."""
        spec = window().order_by("salary", desc=True).build()
        assert spec.order_by is not None
        assert len(spec.order_by) == 1

    def test_partition_and_order(self):
        """Test PARTITION BY with ORDER BY."""
        spec = (
            window()
            .partition_by("department")
            .order_by("salary", desc=True)
            .build()
        )
        assert spec.partition_by is not None
        assert spec.order_by is not None

    def test_frame_specification(self):
        """Test frame specification."""
        spec = (
            window()
            .order_by("date")
            .rows()
            .unbounded_preceding()
            .current_row()
            .build()
        )
        assert spec.frame_type is not None
        assert spec.frame_start is not None


class TestQueryBuilder:
    """Tests for QueryBuilder."""

    def test_simple_select(self):
        """Test simple SELECT *."""
        query = QueryBuilder("users").select_all()
        stmt = query.build()
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.select_items) == 1
        assert isinstance(stmt.select_items[0], Star)

    def test_select_columns(self):
        """Test SELECT with specific columns."""
        query = QueryBuilder("users").select("id", "name", "email")
        stmt = query.build()
        assert len(stmt.select_items) == 3

    def test_select_with_expressions(self):
        """Test SELECT with expressions."""
        query = QueryBuilder("users").select(
            col("name"),
            (col("age") + 1).alias("next_age"),
        )
        stmt = query.build()
        assert len(stmt.select_items) == 2

    def test_select_distinct(self):
        """Test SELECT DISTINCT."""
        query = QueryBuilder("users").select("country").distinct()
        stmt = query.build()
        assert stmt.distinct

    def test_where_simple(self):
        """Test simple WHERE clause."""
        query = QueryBuilder("users").select("*").where(col("age") > 18)
        stmt = query.build()
        assert stmt.where_clause is not None

    def test_where_multiple_conditions(self):
        """Test WHERE with multiple conditions (ANDed)."""
        query = (
            QueryBuilder("users")
            .select("*")
            .where(col("age") > 18, col("status") == "active")
        )
        stmt = query.build()
        assert stmt.where_clause is not None

    def test_group_by(self):
        """Test GROUP BY."""
        query = (
            QueryBuilder("orders")
            .select(col("customer_id"), func.count("*"))
            .group_by("customer_id")
        )
        stmt = query.build()
        assert stmt.group_by_clause is not None

    def test_having(self):
        """Test HAVING."""
        query = (
            QueryBuilder("orders")
            .select(col("customer_id"), func.sum("amount"))
            .group_by("customer_id")
            .having(func.sum("amount") > 1000)
        )
        stmt = query.build()
        assert stmt.having_clause is not None

    def test_order_by(self):
        """Test ORDER BY."""
        query = (
            QueryBuilder("users")
            .select("*")
            .order_by("name")
            .order_by("created_at", desc=True)
        )
        stmt = query.build()
        assert stmt.order_by_clause is not None
        assert len(stmt.order_by_clause.items) == 2

    def test_limit(self):
        """Test LIMIT."""
        query = QueryBuilder("users").select("*").limit(100)
        stmt = query.build()
        assert stmt.limit_clause is not None
        assert stmt.limit_clause.count == 100

    def test_offset(self):
        """Test OFFSET."""
        query = QueryBuilder("users").select("*").limit(100).offset(50)
        stmt = query.build()
        assert stmt.offset_clause is not None
        assert stmt.offset_clause.offset == 50

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

    def test_complex_query(self):
        """Test complex query with multiple clauses."""
        query = (
            QueryBuilder("orders", alias="o")
            .select(
                col("o.customer_id"),
                func.count("*").alias("order_count"),
                func.sum("o.amount").alias("total_amount"),
            )
            .inner_join("customers", col("o.customer_id") == col("customers.id"), alias="c")
            .where(
                col("o.status") == "completed",
                col("o.created_at") > "2024-01-01",
            )
            .group_by("o.customer_id")
            .having(func.count("*") > 5)
            .order_by(func.sum("o.amount"), desc=True)
            .limit(100)
        )
        stmt = query.build()

        assert stmt.from_clause is not None
        assert stmt.where_clause is not None
        assert stmt.group_by_clause is not None
        assert stmt.having_clause is not None
        assert stmt.order_by_clause is not None
        assert stmt.limit_clause is not None


class TestQueryBuilderToSQL:
    """Tests for QueryBuilder.to_sql() method."""

    def test_simple_select_postgresql(self):
        """Test simple SELECT generates valid PostgreSQL."""
        query = QueryBuilder("users").select("*")
        sql = query.to_sql(SQLDialect.POSTGRESQL)

        assert "SELECT" in sql
        assert "*" in sql
        assert '"users"' in sql

    def test_select_with_where(self):
        """Test SELECT with WHERE generates valid SQL."""
        query = QueryBuilder("users").select("name").where(col("age") > 18)
        sql = query.to_sql(SQLDialect.POSTGRESQL)

        assert "SELECT" in sql
        assert "WHERE" in sql
        assert ">" in sql

    def test_select_with_aggregation(self):
        """Test SELECT with aggregation generates valid SQL."""
        query = (
            QueryBuilder("orders")
            .select(col("customer_id"), func.sum("amount").alias("total"))
            .group_by("customer_id")
        )
        sql = query.to_sql(SQLDialect.POSTGRESQL)

        assert "SELECT" in sql
        assert "SUM" in sql
        assert "GROUP BY" in sql

    def test_dialect_differences(self):
        """Test different dialects generate different SQL."""
        query = QueryBuilder("users").select("*").limit(10)

        pg_sql = query.to_sql(SQLDialect.POSTGRESQL)
        mysql_sql = query.to_sql(SQLDialect.MYSQL)

        # Both should have LIMIT
        assert "LIMIT" in pg_sql
        assert "LIMIT" in mysql_sql

        # But identifier quoting is different
        assert '"users"' in pg_sql
        assert "`users`" in mysql_sql


class TestSetOperations:
    """Tests for set operations (UNION, INTERSECT, EXCEPT)."""

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
