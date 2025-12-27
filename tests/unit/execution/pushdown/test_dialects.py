"""Tests for SQL dialect generators."""

from __future__ import annotations

import pytest

from truthound.execution.pushdown.ast import (
    Column,
    Literal,
    NullLiteral,
    BooleanLiteral,
    ArrayLiteral,
    Star,
    Table,
    Alias,
    BinaryExpression,
    UnaryExpression,
    InExpression,
    BetweenExpression,
    CaseExpression,
    WhenClause,
    CastExpression,
    FunctionCall,
    AggregateFunction,
    WindowFunction,
    WindowSpec,
    SelectStatement,
    FromClause,
    JoinClause,
    WhereClause,
    GroupByClause,
    HavingClause,
    OrderByClause,
    OrderByItem,
    LimitClause,
    OffsetClause,
    SetOperationStatement,
    SetOperation,
    CTEClause,
    ComparisonOp,
    LogicalOp,
    ArithmeticOp,
    UnaryOp,
    SortOrder,
    NullsPosition,
    JoinType,
    FrameType,
    FrameBound,
)
from truthound.execution.pushdown.dialects import (
    SQLDialect,
    DialectConfig,
    get_dialect_generator,
    register_dialect_generator,
    BaseDialectGenerator,
)
from truthound.execution.pushdown.builder import (
    QueryBuilder,
    col,
    literal,
    func,
    and_,
    or_,
)


class TestDialectConfig:
    """Tests for DialectConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DialectConfig()
        assert config.identifier_quote == '"'
        assert config.string_quote == "'"
        assert config.supports_cte is True
        assert config.supports_window_functions is True
        assert config.supports_lateral is True

    def test_mysql_style_config(self):
        """Test MySQL-style configuration."""
        config = DialectConfig(
            identifier_quote="`",
        )
        assert config.identifier_quote == "`"


class TestLiteralGeneration:
    """Tests for literal value SQL generation."""

    def test_integer_literal(self):
        """Test integer literal generation."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        lit = Literal(42)
        assert gen.generate(lit) == "42"

    def test_float_literal(self):
        """Test float literal generation."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        lit = Literal(3.14159)
        assert gen.generate(lit) == "3.14159"

    def test_string_literal(self):
        """Test string literal with proper quoting."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        lit = Literal("hello")
        assert gen.generate(lit) == "'hello'"

    def test_string_escape(self):
        """Test string with single quote escaping."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        lit = Literal("it's")
        assert gen.generate(lit) == "'it''s'"

    def test_null_literal(self):
        """Test NULL literal."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        lit = NullLiteral()
        assert gen.generate(lit) == "NULL"

    def test_boolean_literal_postgresql(self):
        """Test boolean literal in PostgreSQL."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        assert gen.generate(BooleanLiteral(True)) == "TRUE"
        assert gen.generate(BooleanLiteral(False)) == "FALSE"

    def test_boolean_literal_mysql(self):
        """Test boolean literal in MySQL (uses 1/0)."""
        gen = get_dialect_generator(SQLDialect.MYSQL)
        # MySQL supports TRUE/FALSE keywords
        assert gen.generate(BooleanLiteral(True)) == "TRUE"
        assert gen.generate(BooleanLiteral(False)) == "FALSE"

    def test_array_literal_postgresql(self):
        """Test array literal in PostgreSQL."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        arr = ArrayLiteral([Literal(1), Literal(2), Literal(3)])
        assert gen.generate(arr) == "ARRAY[1, 2, 3]"

    def test_array_literal_bigquery(self):
        """Test array literal in BigQuery."""
        gen = get_dialect_generator(SQLDialect.BIGQUERY)
        arr = ArrayLiteral([Literal(1), Literal(2), Literal(3)])
        assert gen.generate(arr) == "[1, 2, 3]"


class TestIdentifierQuoting:
    """Tests for identifier quoting across dialects."""

    def test_postgresql_quotes(self):
        """Test PostgreSQL uses double quotes."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        col = Column("name")
        assert gen.generate(col) == '"name"'

    def test_mysql_quotes(self):
        """Test MySQL uses backticks."""
        gen = get_dialect_generator(SQLDialect.MYSQL)
        col = Column("name")
        assert gen.generate(col) == "`name`"

    def test_sqlite_quotes(self):
        """Test SQLite uses double quotes."""
        gen = get_dialect_generator(SQLDialect.SQLITE)
        col = Column("name")
        assert gen.generate(col) == '"name"'

    def test_bigquery_quotes(self):
        """Test BigQuery uses backticks."""
        gen = get_dialect_generator(SQLDialect.BIGQUERY)
        col = Column("name")
        assert gen.generate(col) == "`name`"

    def test_snowflake_quotes(self):
        """Test Snowflake uses double quotes."""
        gen = get_dialect_generator(SQLDialect.SNOWFLAKE)
        col = Column("name")
        assert gen.generate(col) == '"name"'

    def test_sqlserver_quotes(self):
        """Test SQL Server uses square brackets."""
        gen = get_dialect_generator(SQLDialect.SQLSERVER)
        col = Column("name")
        assert gen.generate(col) == "[name]"

    def test_qualified_column(self):
        """Test qualified column name."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        col = Column("name", table="users")
        assert gen.generate(col) == '"users"."name"'

    def test_fully_qualified_column(self):
        """Test fully qualified column with schema."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        col = Column("name", table="users", schema="public")
        assert gen.generate(col) == '"public"."users"."name"'


class TestExpressionGeneration:
    """Tests for expression SQL generation."""

    def test_comparison_equal(self):
        """Test equality comparison."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=Column("age"),
            operator=ComparisonOp.EQ,
            right=Literal(18),
        )
        assert gen.generate(expr) == '"age" = 18'

    def test_comparison_not_equal(self):
        """Test not equal comparison."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=Column("status"),
            operator=ComparisonOp.NE,
            right=Literal("inactive"),
        )
        assert gen.generate(expr) == "\"status\" <> 'inactive'"

    def test_comparison_greater_than(self):
        """Test greater than comparison."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=Column("price"),
            operator=ComparisonOp.GT,
            right=Literal(100),
        )
        assert gen.generate(expr) == '"price" > 100'

    def test_like_operator(self):
        """Test LIKE operator."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=Column("name"),
            operator=ComparisonOp.LIKE,
            right=Literal("%Smith%"),
        )
        assert gen.generate(expr) == "\"name\" LIKE '%Smith%'"

    def test_is_null(self):
        """Test IS NULL."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=Column("deleted_at"),
            operator=ComparisonOp.IS,
            right=NullLiteral(),
        )
        assert gen.generate(expr) == '"deleted_at" IS NULL'

    def test_logical_and(self):
        """Test AND operator."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=BinaryExpression(Column("a"), ComparisonOp.EQ, Literal(1)),
            operator=LogicalOp.AND,
            right=BinaryExpression(Column("b"), ComparisonOp.EQ, Literal(2)),
        )
        result = gen.generate(expr)
        assert "AND" in result
        assert '"a" = 1' in result
        assert '"b" = 2' in result

    def test_logical_or(self):
        """Test OR operator."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=BinaryExpression(Column("status"), ComparisonOp.EQ, Literal("active")),
            operator=LogicalOp.OR,
            right=BinaryExpression(Column("status"), ComparisonOp.EQ, Literal("pending")),
        )
        result = gen.generate(expr)
        assert "OR" in result

    def test_unary_not(self):
        """Test NOT operator."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = UnaryExpression(
            operator=UnaryOp.NOT,
            operand=Column("active"),
        )
        assert gen.generate(expr) == 'NOT "active"'

    def test_unary_negative(self):
        """Test negation operator."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = UnaryExpression(
            operator=UnaryOp.NEGATIVE,
            operand=Column("amount"),
        )
        assert gen.generate(expr) == '-"amount"'

    def test_arithmetic_addition(self):
        """Test addition."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=Column("price"),
            operator=ArithmeticOp.ADD,
            right=Column("tax"),
        )
        assert gen.generate(expr) == '"price" + "tax"'

    def test_arithmetic_multiplication(self):
        """Test multiplication."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BinaryExpression(
            left=Column("quantity"),
            operator=ArithmeticOp.MUL,
            right=Column("unit_price"),
        )
        assert gen.generate(expr) == '"quantity" * "unit_price"'


class TestInExpression:
    """Tests for IN expression generation."""

    def test_in_list(self):
        """Test IN with list of values."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = InExpression(
            expression=Column("status"),
            values=[Literal("active"), Literal("pending"), Literal("review")],
        )
        result = gen.generate(expr)
        assert '"status" IN' in result
        assert "'active'" in result
        assert "'pending'" in result
        assert "'review'" in result

    def test_not_in(self):
        """Test NOT IN."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = InExpression(
            expression=Column("status"),
            values=[Literal("deleted"), Literal("archived")],
            negated=True,
        )
        result = gen.generate(expr)
        assert "NOT IN" in result


class TestBetweenExpression:
    """Tests for BETWEEN expression generation."""

    def test_between(self):
        """Test BETWEEN expression."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BetweenExpression(
            expression=Column("age"),
            low=Literal(18),
            high=Literal(65),
        )
        assert gen.generate(expr) == '"age" BETWEEN 18 AND 65'

    def test_not_between(self):
        """Test NOT BETWEEN."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = BetweenExpression(
            expression=Column("score"),
            low=Literal(0),
            high=Literal(50),
            negated=True,
        )
        assert "NOT BETWEEN" in gen.generate(expr)


class TestCaseExpression:
    """Tests for CASE expression generation."""

    def test_simple_case(self):
        """Test simple CASE expression."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        case = CaseExpression(
            when_clauses=[
                WhenClause(
                    condition=BinaryExpression(Column("status"), ComparisonOp.EQ, Literal("A")),
                    result=Literal("Active"),
                ),
                WhenClause(
                    condition=BinaryExpression(Column("status"), ComparisonOp.EQ, Literal("I")),
                    result=Literal("Inactive"),
                ),
            ],
            else_result=Literal("Unknown"),
        )
        result = gen.generate(case)
        assert "CASE" in result
        assert "WHEN" in result
        assert "THEN" in result
        assert "ELSE" in result
        assert "END" in result


class TestCastExpression:
    """Tests for CAST expression generation."""

    def test_cast_to_varchar(self):
        """Test CAST to VARCHAR."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = CastExpression(
            expression=Column("id"),
            target_type="VARCHAR(10)",
        )
        assert gen.generate(expr) == 'CAST("id" AS VARCHAR(10))'

    def test_cast_to_integer(self):
        """Test CAST to INTEGER."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        expr = CastExpression(
            expression=Literal("42"),
            target_type="INTEGER",
        )
        assert gen.generate(expr) == "CAST('42' AS INTEGER)"


class TestFunctionGeneration:
    """Tests for function SQL generation."""

    def test_simple_function(self):
        """Test simple function call."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        f = FunctionCall(
            name="UPPER",
            arguments=[Column("name")],
        )
        assert gen.generate(f) == 'UPPER("name")'

    def test_function_multiple_args(self):
        """Test function with multiple arguments."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        f = FunctionCall(
            name="COALESCE",
            arguments=[Column("name"), Literal("Unknown")],
        )
        assert gen.generate(f) == "COALESCE(\"name\", 'Unknown')"

    def test_aggregate_count_star(self):
        """Test COUNT(*)."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        agg = AggregateFunction(name="COUNT", argument=None)
        assert gen.generate(agg) == "COUNT(*)"

    def test_aggregate_count_column(self):
        """Test COUNT(column)."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        agg = AggregateFunction(name="COUNT", argument=Column("id"))
        assert gen.generate(agg) == 'COUNT("id")'

    def test_aggregate_count_distinct(self):
        """Test COUNT(DISTINCT column)."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        agg = AggregateFunction(name="COUNT", argument=Column("email"), distinct=True)
        assert gen.generate(agg) == 'COUNT(DISTINCT "email")'

    def test_aggregate_sum(self):
        """Test SUM."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        agg = AggregateFunction(name="SUM", argument=Column("amount"))
        assert gen.generate(agg) == 'SUM("amount")'

    def test_aggregate_with_filter(self):
        """Test aggregate with FILTER clause (PostgreSQL)."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        agg = AggregateFunction(
            name="COUNT",
            argument=Column("id"),
            filter_clause=BinaryExpression(Column("status"), ComparisonOp.EQ, Literal("active")),
        )
        result = gen.generate(agg)
        assert "COUNT" in result
        assert "FILTER" in result
        assert "WHERE" in result


class TestWindowFunctionGeneration:
    """Tests for window function SQL generation."""

    def test_row_number(self):
        """Test ROW_NUMBER()."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        wf = WindowFunction(
            name="ROW_NUMBER",
            arguments=[],
            over=WindowSpec(
                partition_by=[Column("department")],
                order_by=[OrderByItem(Column("salary"), order=SortOrder.DESC)],
            ),
        )
        result = gen.generate(wf)
        assert "ROW_NUMBER()" in result
        assert "OVER" in result
        assert "PARTITION BY" in result
        assert "ORDER BY" in result

    def test_rank(self):
        """Test RANK()."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        wf = WindowFunction(
            name="RANK",
            arguments=[],
            over=WindowSpec(
                order_by=[OrderByItem(Column("score"), order=SortOrder.DESC)],
            ),
        )
        result = gen.generate(wf)
        assert "RANK()" in result
        assert "OVER" in result
        assert "ORDER BY" in result

    def test_lead(self):
        """Test LEAD()."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        wf = WindowFunction(
            name="LEAD",
            arguments=[Column("value"), Literal(1)],
            over=WindowSpec(
                order_by=[OrderByItem(Column("date"))],
            ),
        )
        result = gen.generate(wf)
        assert "LEAD(" in result

    def test_sum_window_with_frame(self):
        """Test SUM with frame specification."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        wf = WindowFunction(
            name="SUM",
            arguments=[Column("amount")],
            over=WindowSpec(
                partition_by=[Column("customer_id")],
                order_by=[OrderByItem(Column("date"))],
                frame_type=FrameType.ROWS,
                frame_start=FrameBound.UNBOUNDED_PRECEDING,
                frame_end=FrameBound.CURRENT_ROW,
            ),
        )
        result = gen.generate(wf)
        assert "SUM" in result
        assert "ROWS" in result
        assert "UNBOUNDED PRECEDING" in result
        assert "CURRENT ROW" in result


class TestSelectStatementGeneration:
    """Tests for SELECT statement generation."""

    def test_simple_select_star(self):
        """Test SELECT * FROM table."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
        )
        result = gen.generate(stmt)
        assert "SELECT" in result
        assert "*" in result
        assert "FROM" in result
        assert '"users"' in result

    def test_select_columns(self):
        """Test SELECT with specific columns."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Column("id"), Column("name"), Column("email")],
            from_clause=FromClause(Table("users")),
        )
        result = gen.generate(stmt)
        assert '"id"' in result
        assert '"name"' in result
        assert '"email"' in result

    def test_select_distinct(self):
        """Test SELECT DISTINCT."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Column("country")],
            from_clause=FromClause(Table("users")),
            distinct=True,
        )
        result = gen.generate(stmt)
        assert "SELECT DISTINCT" in result

    def test_select_with_alias(self):
        """Test SELECT with column alias."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[
                Alias(Column("first_name"), "name"),
            ],
            from_clause=FromClause(Table("users")),
        )
        result = gen.generate(stmt)
        assert "AS" in result
        assert '"name"' in result

    def test_select_with_where(self):
        """Test SELECT with WHERE clause."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            where_clause=WhereClause(
                BinaryExpression(Column("age"), ComparisonOp.GE, Literal(18))
            ),
        )
        result = gen.generate(stmt)
        assert "WHERE" in result
        assert ">=" in result

    def test_select_with_group_by(self):
        """Test SELECT with GROUP BY."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[
                Column("department"),
                Alias(AggregateFunction("COUNT", None), "count"),
            ],
            from_clause=FromClause(Table("employees")),
            group_by_clause=GroupByClause([Column("department")]),
        )
        result = gen.generate(stmt)
        assert "GROUP BY" in result

    def test_select_with_having(self):
        """Test SELECT with HAVING."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[
                Column("department"),
                Alias(AggregateFunction("COUNT", None), "count"),
            ],
            from_clause=FromClause(Table("employees")),
            group_by_clause=GroupByClause([Column("department")]),
            having_clause=HavingClause(
                BinaryExpression(
                    AggregateFunction("COUNT", None),
                    ComparisonOp.GT,
                    Literal(5),
                )
            ),
        )
        result = gen.generate(stmt)
        assert "HAVING" in result
        assert "COUNT(*)" in result
        assert "> 5" in result

    def test_select_with_order_by(self):
        """Test SELECT with ORDER BY."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            order_by_clause=OrderByClause([
                OrderByItem(Column("created_at"), order=SortOrder.DESC),
                OrderByItem(Column("name"), order=SortOrder.ASC),
            ]),
        )
        result = gen.generate(stmt)
        assert "ORDER BY" in result
        assert "DESC" in result

    def test_select_with_order_by_nulls(self):
        """Test ORDER BY with NULLS FIRST/LAST."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            order_by_clause=OrderByClause([
                OrderByItem(
                    Column("score"),
                    order=SortOrder.DESC,
                    nulls=NullsPosition.LAST,
                ),
            ]),
        )
        result = gen.generate(stmt)
        assert "NULLS LAST" in result


class TestLimitOffsetGeneration:
    """Tests for LIMIT/OFFSET generation across dialects."""

    def test_postgresql_limit(self):
        """Test PostgreSQL LIMIT."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            limit_clause=LimitClause(100),
        )
        result = gen.generate(stmt)
        assert "LIMIT 100" in result

    def test_postgresql_offset(self):
        """Test PostgreSQL OFFSET."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            limit_clause=LimitClause(100),
            offset_clause=OffsetClause(50),
        )
        result = gen.generate(stmt)
        assert "LIMIT 100" in result
        assert "OFFSET 50" in result

    def test_mysql_limit(self):
        """Test MySQL LIMIT."""
        gen = get_dialect_generator(SQLDialect.MYSQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            limit_clause=LimitClause(100),
            offset_clause=OffsetClause(50),
        )
        result = gen.generate(stmt)
        assert "LIMIT" in result

    def test_sqlserver_top(self):
        """Test SQL Server uses TOP (when no OFFSET)."""
        gen = get_dialect_generator(SQLDialect.SQLSERVER)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            limit_clause=LimitClause(100),
        )
        result = gen.generate(stmt)
        # SQL Server should use TOP or FETCH
        assert "TOP" in result or "FETCH" in result

    def test_oracle_fetch(self):
        """Test Oracle uses FETCH FIRST."""
        gen = get_dialect_generator(SQLDialect.ORACLE)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            limit_clause=LimitClause(100),
        )
        result = gen.generate(stmt)
        # Oracle 12c+ uses FETCH FIRST
        assert "FETCH" in result or "ROWNUM" in result


class TestJoinGeneration:
    """Tests for JOIN clause generation."""

    def test_inner_join(self):
        """Test INNER JOIN."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(
                source=Table("users"),
                joins=[
                    JoinClause(
                        join_type=JoinType.INNER,
                        table=Table("orders"),
                        condition=BinaryExpression(
                            Column("id", table="users"),
                            ComparisonOp.EQ,
                            Column("user_id", table="orders"),
                        ),
                    ),
                ],
            ),
        )
        result = gen.generate(stmt)
        assert "INNER JOIN" in result
        assert "ON" in result

    def test_left_join(self):
        """Test LEFT JOIN."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(
                source=Table("users"),
                joins=[
                    JoinClause(
                        join_type=JoinType.LEFT,
                        table=Table("orders"),
                        condition=BinaryExpression(
                            Column("id", table="users"),
                            ComparisonOp.EQ,
                            Column("user_id", table="orders"),
                        ),
                    ),
                ],
            ),
        )
        result = gen.generate(stmt)
        assert "LEFT" in result
        assert "JOIN" in result

    def test_multiple_joins(self):
        """Test multiple JOINs."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(
                source=Table("users"),
                joins=[
                    JoinClause(
                        join_type=JoinType.INNER,
                        table=Table("orders"),
                        condition=BinaryExpression(
                            Column("id", table="users"),
                            ComparisonOp.EQ,
                            Column("user_id", table="orders"),
                        ),
                    ),
                    JoinClause(
                        join_type=JoinType.LEFT,
                        table=Table("products"),
                        condition=BinaryExpression(
                            Column("product_id", table="orders"),
                            ComparisonOp.EQ,
                            Column("id", table="products"),
                        ),
                    ),
                ],
            ),
        )
        result = gen.generate(stmt)
        assert "INNER JOIN" in result
        assert "LEFT" in result
        assert result.count("JOIN") >= 2


class TestCTEGeneration:
    """Tests for CTE (Common Table Expression) generation."""

    def test_simple_cte(self):
        """Test simple WITH clause."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)

        # CTE definition
        cte_query = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("orders")),
            where_clause=WhereClause(
                BinaryExpression(Column("status"), ComparisonOp.EQ, Literal("completed"))
            ),
        )

        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("completed_orders")),
            cte_clause=CTEClause(
                ctes=[(Table("completed_orders"), cte_query)],
            ),
        )
        result = gen.generate(stmt)
        assert "WITH" in result
        assert '"completed_orders"' in result
        assert "AS" in result

    def test_recursive_cte(self):
        """Test RECURSIVE CTE."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)

        cte_query = SelectStatement(
            select_items=[Column("id"), Column("parent_id")],
            from_clause=FromClause(Table("categories")),
        )

        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("category_tree")),
            cte_clause=CTEClause(
                ctes=[(Table("category_tree"), cte_query)],
                recursive=True,
            ),
        )
        result = gen.generate(stmt)
        assert "WITH RECURSIVE" in result


class TestSetOperations:
    """Tests for set operations (UNION, INTERSECT, EXCEPT)."""

    def test_union(self):
        """Test UNION."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)

        q1 = SelectStatement(
            select_items=[Column("id"), Column("name")],
            from_clause=FromClause(Table("users")),
        )
        q2 = SelectStatement(
            select_items=[Column("id"), Column("name")],
            from_clause=FromClause(Table("admins")),
        )

        stmt = SetOperationStatement(
            left=q1,
            operation=SetOperation.UNION,
            right=q2,
        )
        result = gen.generate(stmt)
        assert "UNION" in result
        assert "UNION ALL" not in result

    def test_union_all(self):
        """Test UNION ALL."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)

        q1 = SelectStatement(
            select_items=[Column("id")],
            from_clause=FromClause(Table("t1")),
        )
        q2 = SelectStatement(
            select_items=[Column("id")],
            from_clause=FromClause(Table("t2")),
        )

        stmt = SetOperationStatement(
            left=q1,
            operation=SetOperation.UNION,
            right=q2,
            all=True,
        )
        result = gen.generate(stmt)
        assert "UNION ALL" in result

    def test_intersect(self):
        """Test INTERSECT."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)

        q1 = SelectStatement(
            select_items=[Column("id")],
            from_clause=FromClause(Table("t1")),
        )
        q2 = SelectStatement(
            select_items=[Column("id")],
            from_clause=FromClause(Table("t2")),
        )

        stmt = SetOperationStatement(
            left=q1,
            operation=SetOperation.INTERSECT,
            right=q2,
        )
        result = gen.generate(stmt)
        assert "INTERSECT" in result

    def test_except(self):
        """Test EXCEPT."""
        gen = get_dialect_generator(SQLDialect.POSTGRESQL)

        q1 = SelectStatement(
            select_items=[Column("id")],
            from_clause=FromClause(Table("all_users")),
        )
        q2 = SelectStatement(
            select_items=[Column("id")],
            from_clause=FromClause(Table("blocked_users")),
        )

        stmt = SetOperationStatement(
            left=q1,
            operation=SetOperation.EXCEPT,
            right=q2,
        )
        result = gen.generate(stmt)
        assert "EXCEPT" in result


class TestQueryBuilderToDialect:
    """Tests for QueryBuilder generating different dialects."""

    def test_query_builder_postgresql(self):
        """Test QueryBuilder to PostgreSQL."""
        query = (
            QueryBuilder("users")
            .select("id", "name")
            .where(col("age") > 18)
            .order_by("name")
            .limit(10)
        )
        sql = query.to_sql(SQLDialect.POSTGRESQL)

        assert 'SELECT "id", "name"' in sql
        assert 'FROM "users"' in sql
        assert "WHERE" in sql
        assert "ORDER BY" in sql
        assert "LIMIT 10" in sql

    def test_query_builder_mysql(self):
        """Test QueryBuilder to MySQL."""
        query = (
            QueryBuilder("users")
            .select("id", "name")
            .where(col("status") == "active")
        )
        sql = query.to_sql(SQLDialect.MYSQL)

        assert "`id`" in sql
        assert "`name`" in sql
        assert "`users`" in sql

    def test_query_builder_bigquery(self):
        """Test QueryBuilder to BigQuery."""
        query = (
            QueryBuilder("users")
            .select(col("id"), func.count("*").alias("cnt"))
            .group_by("id")
        )
        sql = query.to_sql(SQLDialect.BIGQUERY)

        assert "`id`" in sql
        assert "COUNT(*)" in sql
        assert "GROUP BY" in sql


class TestCustomDialectRegistration:
    """Tests for custom dialect registration."""

    def test_register_custom_generator(self):
        """Test registering a custom dialect generator."""

        class CustomGenerator(BaseDialectGenerator):
            """Custom generator for testing."""

            def __init__(self):
                super().__init__(DialectConfig(identifier_quote="$"))

        # Register it
        register_dialect_generator("custom_test", CustomGenerator)

        # Use it
        gen = get_dialect_generator("custom_test")
        col = Column("name")
        assert gen.generate(col) == "$name$"

    def test_invalid_dialect_raises(self):
        """Test that invalid dialect raises error."""
        with pytest.raises((KeyError, ValueError)):
            get_dialect_generator("nonexistent_dialect")
