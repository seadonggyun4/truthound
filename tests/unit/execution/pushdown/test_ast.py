"""Tests for SQL AST nodes."""

from __future__ import annotations

import pytest

from truthound.execution.pushdown.ast import (
    # Literals
    Literal,
    NullLiteral,
    BooleanLiteral,
    ArrayLiteral,
    # Identifiers
    Identifier,
    Column,
    Table,
    Alias,
    Star,
    # Operators
    ComparisonOp,
    LogicalOp,
    ArithmeticOp,
    UnaryOp,
    JoinType,
    SortOrder,
    # Expressions
    BinaryExpression,
    UnaryExpression,
    InExpression,
    BetweenExpression,
    CastExpression,
    CaseExpression,
    WhenClause,
    FunctionCall,
    AggregateFunction,
    WindowFunction,
    WindowSpec,
    FrameBound,
    FrameBoundType,
    FrameType,
    # Clauses
    SelectItem,
    FromClause,
    JoinClause,
    WhereClause,
    GroupByClause,
    HavingClause,
    OrderByItem,
    OrderByClause,
    LimitClause,
    OffsetClause,
    # Statements
    SelectStatement,
)


class TestLiterals:
    """Tests for literal AST nodes."""

    def test_literal_number(self):
        """Test numeric literal."""
        lit = Literal(42)
        assert lit.value == 42

    def test_literal_string(self):
        """Test string literal."""
        lit = Literal("hello")
        assert lit.value == "hello"

    def test_null_literal(self):
        """Test NULL literal."""
        null = NullLiteral()
        assert repr(null) == "NULL"

    def test_boolean_literal(self):
        """Test boolean literals."""
        true = BooleanLiteral(True)
        false = BooleanLiteral(False)
        assert true.value is True
        assert false.value is False

    def test_array_literal(self):
        """Test array literal."""
        arr = ArrayLiteral((Literal(1), Literal(2), Literal(3)))
        assert len(arr.elements) == 3


class TestIdentifiers:
    """Tests for identifier AST nodes."""

    def test_simple_column(self):
        """Test simple column reference."""
        col = Column("name")
        assert col.name == "name"
        assert col.table is None

    def test_qualified_column(self):
        """Test qualified column reference."""
        col = Column("name", "users")
        assert col.name == "name"
        assert col.table == "users"

    def test_full_qualified_column(self):
        """Test fully qualified column reference."""
        col = Column("name", "users", "public")
        assert col.name == "name"
        assert col.table == "users"
        assert col.schema == "public"

    def test_table_reference(self):
        """Test table reference."""
        tbl = Table("users")
        assert tbl.name == "users"

    def test_table_with_schema(self):
        """Test table with schema."""
        tbl = Table("users", schema="public")
        assert tbl.schema == "public"

    def test_table_with_alias(self):
        """Test table with alias."""
        tbl = Table("users", alias="u")
        assert tbl.alias == "u"

    def test_alias_expression(self):
        """Test aliased expression."""
        expr = Alias(Column("name"), "n")
        assert expr.alias == "n"

    def test_star(self):
        """Test star expression."""
        star = Star()
        assert star.table is None

        qualified = Star("users")
        assert qualified.table == "users"


class TestExpressions:
    """Tests for expression AST nodes."""

    def test_binary_comparison(self):
        """Test binary comparison expressions."""
        expr = BinaryExpression(
            Column("age"),
            ComparisonOp.GT,
            Literal(18),
        )
        assert expr.operator == ComparisonOp.GT

    def test_logical_and(self):
        """Test logical AND expression."""
        expr = BinaryExpression(
            BinaryExpression(Column("a"), ComparisonOp.EQ, Literal(1)),
            LogicalOp.AND,
            BinaryExpression(Column("b"), ComparisonOp.EQ, Literal(2)),
        )
        assert expr.operator == LogicalOp.AND

    def test_arithmetic(self):
        """Test arithmetic expressions."""
        expr = BinaryExpression(
            Column("price"),
            ArithmeticOp.MUL,
            Column("quantity"),
        )
        assert expr.operator == ArithmeticOp.MUL

    def test_unary_not(self):
        """Test unary NOT expression."""
        expr = UnaryExpression(
            UnaryOp.NOT,
            Column("active"),
        )
        assert expr.operator == UnaryOp.NOT

    def test_unary_is_null(self):
        """Test IS NULL expression."""
        expr = UnaryExpression(
            UnaryOp.IS_NULL,
            Column("email"),
        )
        assert expr.operator == UnaryOp.IS_NULL

    def test_in_expression(self):
        """Test IN expression."""
        expr = InExpression(
            Column("status"),
            [Literal("active"), Literal("pending")],
        )
        assert len(expr.values) == 2
        assert not expr.negated

    def test_not_in_expression(self):
        """Test NOT IN expression."""
        expr = InExpression(
            Column("status"),
            [Literal("deleted")],
            negated=True,
        )
        assert expr.negated

    def test_between_expression(self):
        """Test BETWEEN expression."""
        expr = BetweenExpression(
            Column("age"),
            Literal(18),
            Literal(65),
        )
        assert not expr.negated

    def test_cast_expression(self):
        """Test CAST expression."""
        expr = CastExpression(Column("id"), "VARCHAR(10)")
        assert expr.target_type == "VARCHAR(10)"

    def test_case_expression(self):
        """Test CASE expression."""
        expr = CaseExpression(
            when_clauses=[
                WhenClause(
                    BinaryExpression(Column("status"), ComparisonOp.EQ, Literal("active")),
                    Literal("Active"),
                ),
                WhenClause(
                    BinaryExpression(Column("status"), ComparisonOp.EQ, Literal("pending")),
                    Literal("Pending"),
                ),
            ],
            else_result=Literal("Unknown"),
        )
        assert len(expr.when_clauses) == 2
        assert expr.else_result is not None


class TestExpressionOperators:
    """Tests for expression operator overloading."""

    def test_eq_operator(self):
        """Test equality operator."""
        col = Column("age")
        expr = col == 18
        assert isinstance(expr, BinaryExpression)
        assert expr.operator == ComparisonOp.EQ

    def test_ne_operator(self):
        """Test inequality operator."""
        col = Column("status")
        expr = col != "deleted"
        assert expr.operator == ComparisonOp.NE

    def test_lt_operator(self):
        """Test less than operator."""
        col = Column("age")
        expr = col < 18
        assert expr.operator == ComparisonOp.LT

    def test_gt_operator(self):
        """Test greater than operator."""
        col = Column("age")
        expr = col > 18
        assert expr.operator == ComparisonOp.GT

    def test_and_operator(self):
        """Test AND operator."""
        a = Column("a") == 1
        b = Column("b") == 2
        expr = a & b
        assert isinstance(expr, BinaryExpression)
        assert expr.operator == LogicalOp.AND

    def test_or_operator(self):
        """Test OR operator."""
        a = Column("a") == 1
        b = Column("b") == 2
        expr = a | b
        assert expr.operator == LogicalOp.OR

    def test_not_operator(self):
        """Test NOT operator."""
        col = Column("active")
        expr = ~col
        assert isinstance(expr, UnaryExpression)
        assert expr.operator == UnaryOp.NOT

    def test_add_operator(self):
        """Test addition operator."""
        col = Column("price")
        expr = col + 10
        assert expr.operator == ArithmeticOp.ADD

    def test_sub_operator(self):
        """Test subtraction operator."""
        col = Column("price")
        expr = col - 10
        assert expr.operator == ArithmeticOp.SUB

    def test_mul_operator(self):
        """Test multiplication operator."""
        col = Column("price")
        expr = col * 2
        assert expr.operator == ArithmeticOp.MUL

    def test_div_operator(self):
        """Test division operator."""
        col = Column("total")
        expr = col / 2
        assert expr.operator == ArithmeticOp.DIV

    def test_is_null_method(self):
        """Test is_null method."""
        col = Column("email")
        expr = col.is_null()
        assert isinstance(expr, UnaryExpression)
        assert expr.operator == UnaryOp.IS_NULL

    def test_in_method(self):
        """Test in_ method."""
        col = Column("status")
        expr = col.in_(["active", "pending"])
        assert isinstance(expr, InExpression)
        assert not expr.negated

    def test_between_method(self):
        """Test between method."""
        col = Column("age")
        expr = col.between(18, 65)
        assert isinstance(expr, BetweenExpression)

    def test_like_method(self):
        """Test like method."""
        col = Column("name")
        expr = col.like("%john%")
        assert isinstance(expr, BinaryExpression)
        assert expr.operator == ComparisonOp.LIKE

    def test_alias_method(self):
        """Test alias method."""
        col = Column("first_name")
        aliased = col.alias("name")
        assert isinstance(aliased, Alias)
        assert aliased.alias == "name"


class TestFunctions:
    """Tests for function AST nodes."""

    def test_simple_function(self):
        """Test simple function call."""
        func = FunctionCall("UPPER", [Column("name")])
        assert func.name == "UPPER"
        assert len(func.arguments) == 1

    def test_function_with_multiple_args(self):
        """Test function with multiple arguments."""
        func = FunctionCall("COALESCE", [Column("a"), Column("b"), Literal("default")])
        assert len(func.arguments) == 3

    def test_aggregate_function(self):
        """Test aggregate function."""
        agg = AggregateFunction("COUNT", Column("id"))
        assert agg.name == "COUNT"
        assert not agg.distinct

    def test_aggregate_count_star(self):
        """Test COUNT(*) aggregate."""
        agg = AggregateFunction("COUNT", None)
        assert agg.argument is None

    def test_aggregate_distinct(self):
        """Test aggregate with DISTINCT."""
        agg = AggregateFunction("COUNT", Column("email"), distinct=True)
        assert agg.distinct

    def test_window_function(self):
        """Test window function."""
        agg = AggregateFunction("ROW_NUMBER", None)
        window_spec = WindowSpec(
            partition_by=[Column("department")],
            order_by=[OrderByItem(Column("salary"), SortOrder.DESC)],
        )
        wf = WindowFunction(agg, window_spec)
        assert isinstance(wf.window_spec, WindowSpec)

    def test_window_spec_with_frame(self):
        """Test window spec with frame."""
        spec = WindowSpec(
            order_by=[OrderByItem(Column("date"))],
            frame_type=FrameType.ROWS,
            frame_start=FrameBound(FrameBoundType.UNBOUNDED_PRECEDING),
            frame_end=FrameBound(FrameBoundType.CURRENT_ROW),
        )
        assert spec.frame_type == FrameType.ROWS


class TestClauses:
    """Tests for SQL clause AST nodes."""

    def test_select_item(self):
        """Test select item."""
        item = SelectItem(Column("name"), alias="user_name")
        assert item.alias == "user_name"

    def test_from_clause(self):
        """Test FROM clause."""
        from_clause = FromClause(Table("users"))
        assert isinstance(from_clause.source, Table)

    def test_join_clause(self):
        """Test JOIN clause."""
        join = JoinClause(
            Table("users"),
            Table("orders"),
            JoinType.INNER,
            condition=BinaryExpression(
                Column("id", "users"),
                ComparisonOp.EQ,
                Column("user_id", "orders"),
            ),
        )
        assert join.join_type == JoinType.INNER

    def test_where_clause(self):
        """Test WHERE clause."""
        where = WhereClause(
            BinaryExpression(Column("active"), ComparisonOp.EQ, BooleanLiteral(True))
        )
        assert isinstance(where.condition, BinaryExpression)

    def test_group_by_clause(self):
        """Test GROUP BY clause."""
        group_by = GroupByClause([Column("department"), Column("region")])
        assert len(group_by.expressions) == 2

    def test_having_clause(self):
        """Test HAVING clause."""
        having = HavingClause(
            BinaryExpression(
                AggregateFunction("COUNT", Star()),
                ComparisonOp.GT,
                Literal(10),
            )
        )
        assert isinstance(having.condition, BinaryExpression)

    def test_order_by_clause(self):
        """Test ORDER BY clause."""
        order_by = OrderByClause([
            OrderByItem(Column("name"), SortOrder.ASC),
            OrderByItem(Column("created_at"), SortOrder.DESC),
        ])
        assert len(order_by.items) == 2

    def test_limit_clause(self):
        """Test LIMIT clause."""
        limit = LimitClause(100)
        assert limit.count == 100

    def test_offset_clause(self):
        """Test OFFSET clause."""
        offset = OffsetClause(50)
        assert offset.offset == 50


class TestSelectStatement:
    """Tests for SELECT statement AST node."""

    def test_simple_select(self):
        """Test simple SELECT statement."""
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
        )
        assert len(stmt.select_items) == 1
        assert stmt.from_clause is not None

    def test_select_with_where(self):
        """Test SELECT with WHERE."""
        stmt = SelectStatement(
            select_items=[Column("name"), Column("email")],
            from_clause=FromClause(Table("users")),
            where_clause=WhereClause(
                BinaryExpression(Column("active"), ComparisonOp.EQ, BooleanLiteral(True))
            ),
        )
        assert stmt.where_clause is not None

    def test_select_with_group_by(self):
        """Test SELECT with GROUP BY."""
        stmt = SelectStatement(
            select_items=[
                Column("department"),
                AggregateFunction("COUNT", Star()),
            ],
            from_clause=FromClause(Table("employees")),
            group_by_clause=GroupByClause([Column("department")]),
        )
        assert stmt.group_by_clause is not None

    def test_select_with_having(self):
        """Test SELECT with HAVING."""
        stmt = SelectStatement(
            select_items=[
                Column("department"),
                AggregateFunction("AVG", Column("salary")),
            ],
            from_clause=FromClause(Table("employees")),
            group_by_clause=GroupByClause([Column("department")]),
            having_clause=HavingClause(
                BinaryExpression(
                    AggregateFunction("AVG", Column("salary")),
                    ComparisonOp.GT,
                    Literal(50000),
                )
            ),
        )
        assert stmt.having_clause is not None

    def test_select_with_order_limit(self):
        """Test SELECT with ORDER BY and LIMIT."""
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            order_by_clause=OrderByClause([
                OrderByItem(Column("created_at"), SortOrder.DESC)
            ]),
            limit_clause=LimitClause(10),
        )
        assert stmt.order_by_clause is not None
        assert stmt.limit_clause is not None

    def test_select_distinct(self):
        """Test SELECT DISTINCT."""
        stmt = SelectStatement(
            select_items=[Column("country")],
            from_clause=FromClause(Table("users")),
            distinct=True,
        )
        assert stmt.distinct

    def test_select_children(self):
        """Test children() method."""
        stmt = SelectStatement(
            select_items=[Column("name")],
            from_clause=FromClause(Table("users")),
            where_clause=WhereClause(
                BinaryExpression(Column("id"), ComparisonOp.EQ, Literal(1))
            ),
        )
        children = stmt.children()
        assert len(children) >= 2  # select_items, from_clause, where_clause
