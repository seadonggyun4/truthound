"""Fluent Query Builder for SQL Pushdown.

This module provides a fluent API for building SQL queries using
the AST nodes. It offers a more intuitive interface for constructing
complex queries.

Example:
    >>> from truthound.execution.pushdown import (
    ...     QueryBuilder,
    ...     col,
    ...     func,
    ...     and_,
    ...     or_,
    ... )
    >>>
    >>> # Simple query
    >>> query = (
    ...     QueryBuilder("users")
    ...     .select("name", "email", col("age"))
    ...     .where(col("age") > 18)
    ...     .order_by("name")
    ...     .limit(100)
    ... )
    >>>
    >>> # Complex query with aggregation
    >>> query = (
    ...     QueryBuilder("orders")
    ...     .select(
    ...         col("customer_id"),
    ...         func.sum("amount").alias("total"),
    ...         func.count("*").alias("order_count"),
    ...     )
    ...     .where(col("status") == "completed")
    ...     .group_by("customer_id")
    ...     .having(func.sum("amount") > 1000)
    ...     .order_by(col("total").desc())
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Union

from truthound.execution.pushdown.ast import (
    # Base
    Expression,
    SQLNode,
    # Literals
    Literal,
    NullLiteral,
    BooleanLiteral,
    # Identifiers
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
    NullsPosition,
    FrameType,
    FrameBoundType,
    SetOperation,
    # Expressions
    BinaryExpression,
    UnaryExpression,
    InExpression,
    BetweenExpression,
    ExistsExpression,
    SubqueryExpression,
    CastExpression,
    WhenClause,
    CaseExpression,
    FunctionCall,
    AggregateFunction,
    FrameBound,
    WindowSpec,
    WindowFunction,
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
    CTEClause,
    # Statements
    SelectStatement,
    SetOperationStatement,
)
from truthound.execution.pushdown.dialects import (
    SQLDialect,
    get_dialect_generator,
)


# Type alias for expression-like values
ExprLike = Union[Expression, str, int, float, bool, None]


# =============================================================================
# Expression Builders
# =============================================================================


def _to_expr(value: ExprLike) -> Expression:
    """Convert a value to an Expression.

    Args:
        value: Value to convert.

    Returns:
        Expression representing the value.
    """
    if isinstance(value, Expression):
        return value
    if isinstance(value, str):
        # Check if it's a column reference (simple heuristic)
        if "." in value or value.isidentifier():
            return col(value)
        return Literal(value)
    if value is None:
        return NullLiteral()
    if isinstance(value, bool):
        return BooleanLiteral(value)
    return Literal(value)


class ExpressionBuilder(Expression):
    """Builder for creating SQL expressions with a fluent API.

    This class wraps an Expression and provides additional methods
    for building complex expressions.
    """

    def __init__(self, expr: Expression) -> None:
        """Initialize expression builder.

        Args:
            expr: The underlying expression.
        """
        self._expr = expr

    @property
    def expression(self) -> Expression:
        """Get the underlying expression."""
        return self._expr

    def accept(self, visitor: Any) -> Any:
        """Accept a visitor."""
        return self._expr.accept(visitor)

    def children(self) -> Sequence[SQLNode]:
        """Get child nodes."""
        return self._expr.children()

    # Delegate comparison operators to underlying expression
    def __eq__(self, other: Any) -> BinaryExpression:  # type: ignore[override]
        return self._expr.__eq__(other)

    def __ne__(self, other: Any) -> BinaryExpression:  # type: ignore[override]
        return self._expr.__ne__(other)

    def __lt__(self, other: Any) -> BinaryExpression:
        return self._expr.__lt__(other)

    def __le__(self, other: Any) -> BinaryExpression:
        return self._expr.__le__(other)

    def __gt__(self, other: Any) -> BinaryExpression:
        return self._expr.__gt__(other)

    def __ge__(self, other: Any) -> BinaryExpression:
        return self._expr.__ge__(other)

    def __add__(self, other: Any) -> BinaryExpression:
        return self._expr.__add__(other)

    def __sub__(self, other: Any) -> BinaryExpression:
        return self._expr.__sub__(other)

    def __mul__(self, other: Any) -> BinaryExpression:
        return self._expr.__mul__(other)

    def __truediv__(self, other: Any) -> BinaryExpression:
        return self._expr.__truediv__(other)

    def __mod__(self, other: Any) -> BinaryExpression:
        return self._expr.__mod__(other)

    def __and__(self, other: Expression) -> BinaryExpression:
        return self._expr.__and__(other)

    def __or__(self, other: Expression) -> BinaryExpression:
        return self._expr.__or__(other)

    def __invert__(self) -> UnaryExpression:
        return self._expr.__invert__()

    def __neg__(self) -> UnaryExpression:
        return self._expr.__neg__()


# =============================================================================
# Column Builder
# =============================================================================


def col(name: str, table: str | None = None) -> Column:
    """Create a column reference.

    Args:
        name: Column name. Can include table prefix (e.g., "users.id").
        table: Optional table name.

    Returns:
        Column expression.

    Examples:
        >>> col("name")
        Column(name)
        >>> col("users.name")
        Column(users.name)
        >>> col("name", "users")
        Column(users.name)
    """
    if "." in name and table is None:
        parts = name.split(".")
        if len(parts) == 2:
            return Column(parts[1], parts[0])
        elif len(parts) == 3:
            return Column(parts[2], parts[1], parts[0])
    return Column(name, table)


def literal(value: Any) -> Literal:
    """Create a literal value.

    Args:
        value: The literal value.

    Returns:
        Literal expression.
    """
    if value is None:
        return NullLiteral()  # type: ignore
    return Literal(value)


def star(table: str | None = None) -> Star:
    """Create a * expression.

    Args:
        table: Optional table name for table.* syntax.

    Returns:
        Star expression.
    """
    return Star(table)


# =============================================================================
# Function Builder
# =============================================================================


class FunctionBuilder:
    """Builder for SQL function calls.

    Provides shortcuts for common SQL functions.

    Example:
        >>> func.count("*")
        COUNT(*)
        >>> func.sum("amount")
        SUM(amount)
        >>> func.avg("price").over(partition_by="category")
        AVG(price) OVER (PARTITION BY category)
    """

    def __call__(self, name: str, *args: ExprLike) -> FunctionCall:
        """Create a generic function call.

        Args:
            name: Function name.
            args: Function arguments.

        Returns:
            FunctionCall expression.
        """
        exprs = [_to_expr(a) for a in args]
        return FunctionCall(name, exprs)

    # -------------------------------------------------------------------------
    # Aggregate Functions
    # -------------------------------------------------------------------------

    def count(
        self,
        column: ExprLike = "*",
        distinct: bool = False,
    ) -> AggregateFunction:
        """COUNT aggregate function.

        Args:
            column: Column to count. Use "*" for COUNT(*).
            distinct: Whether to count distinct values.

        Returns:
            AggregateFunction.
        """
        if column == "*":
            return AggregateFunction("COUNT", None, distinct=distinct)
        return AggregateFunction("COUNT", _to_expr(column), distinct=distinct)

    def count_distinct(self, column: ExprLike) -> AggregateFunction:
        """COUNT(DISTINCT column) aggregate function."""
        return self.count(column, distinct=True)

    def sum(self, column: ExprLike, distinct: bool = False) -> AggregateFunction:
        """SUM aggregate function."""
        return AggregateFunction("SUM", _to_expr(column), distinct=distinct)

    def avg(self, column: ExprLike, distinct: bool = False) -> AggregateFunction:
        """AVG aggregate function."""
        return AggregateFunction("AVG", _to_expr(column), distinct=distinct)

    def min(self, column: ExprLike) -> AggregateFunction:
        """MIN aggregate function."""
        return AggregateFunction("MIN", _to_expr(column))

    def max(self, column: ExprLike) -> AggregateFunction:
        """MAX aggregate function."""
        return AggregateFunction("MAX", _to_expr(column))

    def stddev(self, column: ExprLike) -> AggregateFunction:
        """STDDEV aggregate function."""
        return AggregateFunction("STDDEV", _to_expr(column))

    def stddev_pop(self, column: ExprLike) -> AggregateFunction:
        """STDDEV_POP aggregate function."""
        return AggregateFunction("STDDEV_POP", _to_expr(column))

    def stddev_samp(self, column: ExprLike) -> AggregateFunction:
        """STDDEV_SAMP aggregate function."""
        return AggregateFunction("STDDEV_SAMP", _to_expr(column))

    def variance(self, column: ExprLike) -> AggregateFunction:
        """VARIANCE aggregate function."""
        return AggregateFunction("VARIANCE", _to_expr(column))

    def var_pop(self, column: ExprLike) -> AggregateFunction:
        """VAR_POP aggregate function."""
        return AggregateFunction("VAR_POP", _to_expr(column))

    def var_samp(self, column: ExprLike) -> AggregateFunction:
        """VAR_SAMP aggregate function."""
        return AggregateFunction("VAR_SAMP", _to_expr(column))

    def array_agg(
        self,
        column: ExprLike,
        distinct: bool = False,
        order_by: Sequence[OrderByItem] | None = None,
    ) -> AggregateFunction:
        """ARRAY_AGG aggregate function."""
        return AggregateFunction(
            "ARRAY_AGG", _to_expr(column), distinct=distinct, order_by=order_by
        )

    def string_agg(
        self,
        column: ExprLike,
        separator: str = ",",
        order_by: Sequence[OrderByItem] | None = None,
    ) -> FunctionCall:
        """STRING_AGG (or GROUP_CONCAT) function."""
        return FunctionCall("STRING_AGG", [_to_expr(column), Literal(separator)])

    def group_concat(
        self,
        column: ExprLike,
        separator: str = ",",
    ) -> FunctionCall:
        """GROUP_CONCAT function (MySQL style)."""
        return FunctionCall("GROUP_CONCAT", [_to_expr(column)])

    def listagg(
        self,
        column: ExprLike,
        separator: str = ",",
    ) -> FunctionCall:
        """LISTAGG function (Oracle style)."""
        return FunctionCall("LISTAGG", [_to_expr(column), Literal(separator)])

    # -------------------------------------------------------------------------
    # Window Functions
    # -------------------------------------------------------------------------

    def row_number(self) -> AggregateFunction:
        """ROW_NUMBER window function."""
        return AggregateFunction("ROW_NUMBER", None)

    def rank(self) -> AggregateFunction:
        """RANK window function."""
        return AggregateFunction("RANK", None)

    def dense_rank(self) -> AggregateFunction:
        """DENSE_RANK window function."""
        return AggregateFunction("DENSE_RANK", None)

    def ntile(self, n: int) -> FunctionCall:
        """NTILE window function."""
        return FunctionCall("NTILE", [Literal(n)])

    def percent_rank(self) -> AggregateFunction:
        """PERCENT_RANK window function."""
        return AggregateFunction("PERCENT_RANK", None)

    def cume_dist(self) -> AggregateFunction:
        """CUME_DIST window function."""
        return AggregateFunction("CUME_DIST", None)

    def lead(
        self,
        column: ExprLike,
        offset: int = 1,
        default: ExprLike | None = None,
    ) -> FunctionCall:
        """LEAD window function."""
        args = [_to_expr(column), Literal(offset)]
        if default is not None:
            args.append(_to_expr(default))
        return FunctionCall("LEAD", args)

    def lag(
        self,
        column: ExprLike,
        offset: int = 1,
        default: ExprLike | None = None,
    ) -> FunctionCall:
        """LAG window function."""
        args = [_to_expr(column), Literal(offset)]
        if default is not None:
            args.append(_to_expr(default))
        return FunctionCall("LAG", args)

    def first_value(self, column: ExprLike) -> FunctionCall:
        """FIRST_VALUE window function."""
        return FunctionCall("FIRST_VALUE", [_to_expr(column)])

    def last_value(self, column: ExprLike) -> FunctionCall:
        """LAST_VALUE window function."""
        return FunctionCall("LAST_VALUE", [_to_expr(column)])

    def nth_value(self, column: ExprLike, n: int) -> FunctionCall:
        """NTH_VALUE window function."""
        return FunctionCall("NTH_VALUE", [_to_expr(column), Literal(n)])

    # -------------------------------------------------------------------------
    # String Functions
    # -------------------------------------------------------------------------

    def length(self, column: ExprLike) -> FunctionCall:
        """LENGTH function."""
        return FunctionCall("LENGTH", [_to_expr(column)])

    def upper(self, column: ExprLike) -> FunctionCall:
        """UPPER function."""
        return FunctionCall("UPPER", [_to_expr(column)])

    def lower(self, column: ExprLike) -> FunctionCall:
        """LOWER function."""
        return FunctionCall("LOWER", [_to_expr(column)])

    def trim(self, column: ExprLike) -> FunctionCall:
        """TRIM function."""
        return FunctionCall("TRIM", [_to_expr(column)])

    def ltrim(self, column: ExprLike) -> FunctionCall:
        """LTRIM function."""
        return FunctionCall("LTRIM", [_to_expr(column)])

    def rtrim(self, column: ExprLike) -> FunctionCall:
        """RTRIM function."""
        return FunctionCall("RTRIM", [_to_expr(column)])

    def substr(
        self,
        column: ExprLike,
        start: int,
        length: int | None = None,
    ) -> FunctionCall:
        """SUBSTR/SUBSTRING function."""
        args = [_to_expr(column), Literal(start)]
        if length is not None:
            args.append(Literal(length))
        return FunctionCall("SUBSTRING", args)

    def concat(self, *args: ExprLike) -> FunctionCall:
        """CONCAT function."""
        exprs = [_to_expr(a) for a in args]
        return FunctionCall("CONCAT", exprs)

    def replace(
        self,
        column: ExprLike,
        old: str,
        new: str,
    ) -> FunctionCall:
        """REPLACE function."""
        return FunctionCall(
            "REPLACE", [_to_expr(column), Literal(old), Literal(new)]
        )

    def regexp_replace(
        self,
        column: ExprLike,
        pattern: str,
        replacement: str,
    ) -> FunctionCall:
        """REGEXP_REPLACE function."""
        return FunctionCall(
            "REGEXP_REPLACE", [_to_expr(column), Literal(pattern), Literal(replacement)]
        )

    def split(self, column: ExprLike, delimiter: str) -> FunctionCall:
        """SPLIT function."""
        return FunctionCall("SPLIT", [_to_expr(column), Literal(delimiter)])

    # -------------------------------------------------------------------------
    # Date/Time Functions
    # -------------------------------------------------------------------------

    def now(self) -> FunctionCall:
        """NOW/CURRENT_TIMESTAMP function."""
        return FunctionCall("NOW", [])

    def current_date(self) -> FunctionCall:
        """CURRENT_DATE function."""
        return FunctionCall("CURRENT_DATE", [])

    def current_timestamp(self) -> FunctionCall:
        """CURRENT_TIMESTAMP function."""
        return FunctionCall("CURRENT_TIMESTAMP", [])

    def date_trunc(self, part: str, column: ExprLike) -> FunctionCall:
        """DATE_TRUNC function."""
        return FunctionCall("DATE_TRUNC", [Literal(part), _to_expr(column)])

    def date_part(self, part: str, column: ExprLike) -> FunctionCall:
        """DATE_PART/EXTRACT function."""
        return FunctionCall("DATE_PART", [Literal(part), _to_expr(column)])

    def extract(self, part: str, column: ExprLike) -> FunctionCall:
        """EXTRACT function."""
        return FunctionCall("EXTRACT", [Literal(part), _to_expr(column)])

    def date_add(
        self,
        column: ExprLike,
        interval: int,
        unit: str = "DAY",
    ) -> FunctionCall:
        """DATE_ADD function."""
        return FunctionCall("DATE_ADD", [_to_expr(column), Literal(interval)])

    def date_diff(
        self,
        end: ExprLike,
        start: ExprLike,
        unit: str = "DAY",
    ) -> FunctionCall:
        """DATEDIFF function."""
        return FunctionCall("DATEDIFF", [_to_expr(end), _to_expr(start)])

    # -------------------------------------------------------------------------
    # Null Handling Functions
    # -------------------------------------------------------------------------

    def coalesce(self, *args: ExprLike) -> FunctionCall:
        """COALESCE function."""
        exprs = [_to_expr(a) for a in args]
        return FunctionCall("COALESCE", exprs)

    def nullif(self, expr1: ExprLike, expr2: ExprLike) -> FunctionCall:
        """NULLIF function."""
        return FunctionCall("NULLIF", [_to_expr(expr1), _to_expr(expr2)])

    def ifnull(self, column: ExprLike, default: ExprLike) -> FunctionCall:
        """IFNULL/NVL function."""
        return FunctionCall("IFNULL", [_to_expr(column), _to_expr(default)])

    # -------------------------------------------------------------------------
    # Math Functions
    # -------------------------------------------------------------------------

    def abs(self, column: ExprLike) -> FunctionCall:
        """ABS function."""
        return FunctionCall("ABS", [_to_expr(column)])

    def round(self, column: ExprLike, decimals: int = 0) -> FunctionCall:
        """ROUND function."""
        return FunctionCall("ROUND", [_to_expr(column), Literal(decimals)])

    def floor(self, column: ExprLike) -> FunctionCall:
        """FLOOR function."""
        return FunctionCall("FLOOR", [_to_expr(column)])

    def ceil(self, column: ExprLike) -> FunctionCall:
        """CEIL function."""
        return FunctionCall("CEIL", [_to_expr(column)])

    def power(self, column: ExprLike, exponent: ExprLike) -> FunctionCall:
        """POWER function."""
        return FunctionCall("POWER", [_to_expr(column), _to_expr(exponent)])

    def sqrt(self, column: ExprLike) -> FunctionCall:
        """SQRT function."""
        return FunctionCall("SQRT", [_to_expr(column)])

    def log(self, column: ExprLike) -> FunctionCall:
        """LOG function (natural logarithm)."""
        return FunctionCall("LN", [_to_expr(column)])

    def log10(self, column: ExprLike) -> FunctionCall:
        """LOG10 function."""
        return FunctionCall("LOG10", [_to_expr(column)])

    def exp(self, column: ExprLike) -> FunctionCall:
        """EXP function."""
        return FunctionCall("EXP", [_to_expr(column)])

    # -------------------------------------------------------------------------
    # Conditional Functions
    # -------------------------------------------------------------------------

    def if_(
        self,
        condition: Expression,
        true_value: ExprLike,
        false_value: ExprLike,
    ) -> FunctionCall:
        """IF function (or CASE-based equivalent)."""
        return FunctionCall(
            "IF", [condition, _to_expr(true_value), _to_expr(false_value)]
        )


# Singleton function builder instance
func = FunctionBuilder()


# =============================================================================
# Logical Operators
# =============================================================================


def and_(*conditions: Expression) -> Expression:
    """Create an AND expression from multiple conditions.

    Args:
        conditions: Conditions to AND together.

    Returns:
        Combined AND expression.

    Example:
        >>> and_(col("age") > 18, col("status") == "active")
        (age > 18 AND status = 'active')
    """
    if not conditions:
        raise ValueError("and_() requires at least one condition")
    if len(conditions) == 1:
        return conditions[0]

    result = conditions[0]
    for cond in conditions[1:]:
        result = BinaryExpression(result, LogicalOp.AND, cond)
    return result


def or_(*conditions: Expression) -> Expression:
    """Create an OR expression from multiple conditions.

    Args:
        conditions: Conditions to OR together.

    Returns:
        Combined OR expression.

    Example:
        >>> or_(col("status") == "active", col("status") == "pending")
        (status = 'active' OR status = 'pending')
    """
    if not conditions:
        raise ValueError("or_() requires at least one condition")
    if len(conditions) == 1:
        return conditions[0]

    result = conditions[0]
    for cond in conditions[1:]:
        result = BinaryExpression(result, LogicalOp.OR, cond)
    return result


def not_(condition: Expression) -> UnaryExpression:
    """Create a NOT expression.

    Args:
        condition: Condition to negate.

    Returns:
        NOT expression.
    """
    return UnaryExpression(UnaryOp.NOT, condition)


def exists(subquery: "QueryBuilder | SelectStatement") -> ExistsExpression:
    """Create an EXISTS expression.

    Args:
        subquery: Subquery to check.

    Returns:
        EXISTS expression.
    """
    if isinstance(subquery, QueryBuilder):
        subquery = subquery.build()
    return ExistsExpression(subquery)


def not_exists(subquery: "QueryBuilder | SelectStatement") -> ExistsExpression:
    """Create a NOT EXISTS expression.

    Args:
        subquery: Subquery to check.

    Returns:
        NOT EXISTS expression.
    """
    if isinstance(subquery, QueryBuilder):
        subquery = subquery.build()
    return ExistsExpression(subquery, negated=True)


def cast(column: ExprLike, target_type: str) -> CastExpression:
    """Create a CAST expression.

    Args:
        column: Expression to cast.
        target_type: Target SQL type.

    Returns:
        CAST expression.
    """
    return CastExpression(_to_expr(column), target_type)


# =============================================================================
# CASE Expression Builder
# =============================================================================


class CaseBuilder:
    """Builder for CASE expressions.

    Example:
        >>> case(
        ...     when(col("status") == "active", "Active"),
        ...     when(col("status") == "pending", "Pending"),
        ...     else_="Unknown",
        ... )
    """

    def __init__(self, operand: Expression | None = None) -> None:
        """Initialize CASE builder.

        Args:
            operand: Optional operand for simple CASE.
        """
        self._operand = operand
        self._when_clauses: list[WhenClause] = []
        self._else_result: Expression | None = None

    def when(self, condition: ExprLike, result: ExprLike) -> "CaseBuilder":
        """Add a WHEN clause.

        Args:
            condition: WHEN condition.
            result: THEN result.

        Returns:
            Self for chaining.
        """
        self._when_clauses.append(
            WhenClause(_to_expr(condition), _to_expr(result))
        )
        return self

    def else_(self, result: ExprLike) -> "CaseBuilder":
        """Add an ELSE clause.

        Args:
            result: ELSE result.

        Returns:
            Self for chaining.
        """
        self._else_result = _to_expr(result)
        return self

    def build(self) -> CaseExpression:
        """Build the CASE expression.

        Returns:
            CaseExpression.
        """
        return CaseExpression(
            when_clauses=self._when_clauses,
            operand=self._operand,
            else_result=self._else_result,
        )


def case(*args: WhenClause, operand: Expression | None = None, else_: ExprLike | None = None) -> CaseExpression:
    """Create a CASE expression.

    Args:
        args: WHEN clauses (use when() to create).
        operand: Optional operand for simple CASE.
        else_: Optional ELSE result.

    Returns:
        CaseExpression.
    """
    return CaseExpression(
        when_clauses=args,
        operand=operand,
        else_result=_to_expr(else_) if else_ is not None else None,
    )


def when(condition: ExprLike, result: ExprLike) -> WhenClause:
    """Create a WHEN clause for CASE expression.

    Args:
        condition: WHEN condition.
        result: THEN result.

    Returns:
        WhenClause.
    """
    return WhenClause(_to_expr(condition), _to_expr(result))


# =============================================================================
# Window Specification Builder
# =============================================================================


class WindowBuilder:
    """Builder for window specifications.

    Example:
        >>> window().partition_by("department").order_by("salary", desc=True)
    """

    def __init__(self) -> None:
        """Initialize window builder."""
        self._partition_by: list[Expression] = []
        self._order_by: list[OrderByItem] = []
        self._frame_type: FrameType | None = None
        self._frame_start: FrameBound | None = None
        self._frame_end: FrameBound | None = None
        self._name: str | None = None

    def partition_by(self, *columns: ExprLike) -> "WindowBuilder":
        """Add PARTITION BY clause.

        Args:
            columns: Columns to partition by.

        Returns:
            Self for chaining.
        """
        self._partition_by.extend(_to_expr(c) for c in columns)
        return self

    def order_by(
        self,
        column: ExprLike,
        desc: bool = False,
        descending: bool | None = None,
        nulls: NullsPosition | None = None,
    ) -> "WindowBuilder":
        """Add ORDER BY clause.

        Args:
            column: Column to order by.
            desc: Whether to sort descending (short form).
            descending: Whether to sort descending (long form, alias for desc).
            nulls: Position of nulls.

        Returns:
            Self for chaining.
        """
        is_descending = descending if descending is not None else desc
        order = SortOrder.DESC if is_descending else SortOrder.ASC
        self._order_by.append(OrderByItem(_to_expr(column), order, nulls))
        return self

    def rows(self) -> "WindowBuilder":
        """Use ROWS frame type."""
        self._frame_type = FrameType.ROWS
        return self

    def range(self) -> "WindowBuilder":
        """Use RANGE frame type."""
        self._frame_type = FrameType.RANGE
        return self

    def unbounded_preceding(self) -> "WindowBuilder":
        """Set frame start to UNBOUNDED PRECEDING."""
        self._frame_start = FrameBound(FrameBoundType.UNBOUNDED_PRECEDING)
        return self

    def preceding(self, n: int) -> "WindowBuilder":
        """Set frame start to N PRECEDING."""
        self._frame_start = FrameBound(FrameBoundType.PRECEDING, n)
        return self

    def current_row(self) -> "WindowBuilder":
        """Set frame start/end to CURRENT ROW."""
        if self._frame_start is None:
            self._frame_start = FrameBound(FrameBoundType.CURRENT_ROW)
        else:
            self._frame_end = FrameBound(FrameBoundType.CURRENT_ROW)
        return self

    def following(self, n: int) -> "WindowBuilder":
        """Set frame end to N FOLLOWING."""
        self._frame_end = FrameBound(FrameBoundType.FOLLOWING, n)
        return self

    def unbounded_following(self) -> "WindowBuilder":
        """Set frame end to UNBOUNDED FOLLOWING."""
        self._frame_end = FrameBound(FrameBoundType.UNBOUNDED_FOLLOWING)
        return self

    def between(
        self,
        start: FrameBound,
        end: FrameBound,
    ) -> "WindowBuilder":
        """Set frame BETWEEN bounds."""
        self._frame_start = start
        self._frame_end = end
        return self

    def name(self, window_name: str) -> "WindowBuilder":
        """Set window name."""
        self._name = window_name
        return self

    def build(self) -> WindowSpec:
        """Build the window specification.

        Returns:
            WindowSpec.
        """
        return WindowSpec(
            partition_by=self._partition_by or None,
            order_by=self._order_by or None,
            frame_type=self._frame_type,
            frame_start=self._frame_start,
            frame_end=self._frame_end,
            name=self._name,
        )


def window() -> WindowBuilder:
    """Create a window specification builder.

    Returns:
        WindowBuilder.
    """
    return WindowBuilder()


# =============================================================================
# Query Builder
# =============================================================================


class QueryBuilder:
    """Fluent builder for SELECT queries.

    This class provides a fluent API for building SQL queries,
    which are compiled into AST nodes.

    Example:
        >>> query = (
        ...     QueryBuilder("users")
        ...     .select("id", "name", "email")
        ...     .where(col("age") > 18)
        ...     .order_by("name")
        ...     .limit(100)
        ... )
        >>> sql = query.to_sql(SQLDialect.POSTGRESQL)
    """

    def __init__(
        self,
        table: str | Table | "QueryBuilder" | None = None,
        schema: str | None = None,
        alias: str | None = None,
    ) -> None:
        """Initialize query builder.

        Args:
            table: Table name, Table object, or subquery.
            schema: Optional schema name.
            alias: Optional table alias.
        """
        self._from_source: Table | SelectStatement | JoinClause | None = None
        self._select_items: list[SelectItem | Expression] = []
        self._where_conditions: list[Expression] = []
        self._group_by_exprs: list[Expression] = []
        self._having_conditions: list[Expression] = []
        self._order_by_items: list[OrderByItem] = []
        self._limit_value: int | Expression | None = None
        self._offset_value: int | Expression | None = None
        self._distinct: bool = False
        self._ctes: list[CTEClause] = []
        self._group_by_rollup: bool = False
        self._group_by_cube: bool = False
        self._grouping_sets: list[tuple[Expression, ...]] | None = None
        self._windows: dict[str, WindowSpec] = {}

        if table is not None:
            if isinstance(table, Table):
                self._from_source = table
            elif isinstance(table, QueryBuilder):
                self._from_source = table.build()
            else:
                self._from_source = Table(table, schema=schema, alias=alias)

    def with_cte(
        self,
        name: str,
        query: "QueryBuilder | SelectStatement",
        columns: Sequence[str] | None = None,
        recursive: bool = False,
    ) -> "QueryBuilder":
        """Add a Common Table Expression (CTE).

        Args:
            name: CTE name.
            query: CTE query.
            columns: Optional column names.
            recursive: Whether this is a recursive CTE.

        Returns:
            Self for chaining.
        """
        if isinstance(query, QueryBuilder):
            query = query.build()
        self._ctes.append(CTEClause(name, query, columns, recursive))
        return self

    def select(self, *columns: ExprLike | Star) -> "QueryBuilder":
        """Set SELECT columns.

        Args:
            columns: Columns or expressions to select.

        Returns:
            Self for chaining.
        """
        for c in columns:
            if isinstance(c, str):
                if c == "*":
                    self._select_items.append(Star())
                else:
                    self._select_items.append(col(c))
            elif isinstance(c, (Expression, Star)):
                self._select_items.append(c)
            else:
                self._select_items.append(Literal(c))
        return self

    def select_all(self) -> "QueryBuilder":
        """Select all columns (SELECT *).

        Returns:
            Self for chaining.
        """
        self._select_items.append(Star())
        return self

    def distinct(self, value: bool = True) -> "QueryBuilder":
        """Enable SELECT DISTINCT.

        Args:
            value: Whether to enable DISTINCT.

        Returns:
            Self for chaining.
        """
        self._distinct = value
        return self

    def from_(
        self,
        table: str | Table | "QueryBuilder",
        schema: str | None = None,
        alias: str | None = None,
    ) -> "QueryBuilder":
        """Set FROM clause.

        Args:
            table: Table name, Table object, or subquery.
            schema: Optional schema name.
            alias: Optional table alias.

        Returns:
            Self for chaining.
        """
        if isinstance(table, Table):
            self._from_source = table
        elif isinstance(table, QueryBuilder):
            self._from_source = table.build()
        else:
            self._from_source = Table(table, schema=schema, alias=alias)
        return self

    def join(
        self,
        table: str | Table | "QueryBuilder",
        on: Expression | None = None,
        using: Sequence[str] | None = None,
        join_type: JoinType = JoinType.INNER,
        alias: str | None = None,
    ) -> "QueryBuilder":
        """Add a JOIN clause.

        Args:
            table: Table to join.
            on: Join condition.
            using: USING columns.
            join_type: Type of join.
            alias: Optional table alias.

        Returns:
            Self for chaining.
        """
        if isinstance(table, str):
            right: Table | SelectStatement = Table(table, alias=alias)
        elif isinstance(table, QueryBuilder):
            right = table.build()
        else:
            right = table

        if self._from_source is None:
            raise ValueError("Cannot JOIN without a FROM clause")

        self._from_source = JoinClause(
            self._from_source,
            right,
            join_type,
            on,
            using,
        )
        return self

    def inner_join(
        self,
        table: str | Table | "QueryBuilder",
        on: Expression | None = None,
        using: Sequence[str] | None = None,
        alias: str | None = None,
    ) -> "QueryBuilder":
        """Add an INNER JOIN."""
        return self.join(table, on, using, JoinType.INNER, alias)

    def left_join(
        self,
        table: str | Table | "QueryBuilder",
        on: Expression | None = None,
        using: Sequence[str] | None = None,
        alias: str | None = None,
    ) -> "QueryBuilder":
        """Add a LEFT JOIN."""
        return self.join(table, on, using, JoinType.LEFT, alias)

    def right_join(
        self,
        table: str | Table | "QueryBuilder",
        on: Expression | None = None,
        using: Sequence[str] | None = None,
        alias: str | None = None,
    ) -> "QueryBuilder":
        """Add a RIGHT JOIN."""
        return self.join(table, on, using, JoinType.RIGHT, alias)

    def full_join(
        self,
        table: str | Table | "QueryBuilder",
        on: Expression | None = None,
        using: Sequence[str] | None = None,
        alias: str | None = None,
    ) -> "QueryBuilder":
        """Add a FULL OUTER JOIN."""
        return self.join(table, on, using, JoinType.FULL_OUTER, alias)

    def cross_join(
        self,
        table: str | Table | "QueryBuilder",
        alias: str | None = None,
    ) -> "QueryBuilder":
        """Add a CROSS JOIN."""
        return self.join(table, None, None, JoinType.CROSS, alias)

    def where(self, *conditions: Expression) -> "QueryBuilder":
        """Add WHERE conditions (ANDed together).

        Args:
            conditions: Filter conditions.

        Returns:
            Self for chaining.
        """
        self._where_conditions.extend(conditions)
        return self

    def or_where(self, *conditions: Expression) -> "QueryBuilder":
        """Add WHERE conditions with OR (ORed with existing conditions).

        Args:
            conditions: Filter conditions.

        Returns:
            Self for chaining.
        """
        if self._where_conditions:
            existing = and_(*self._where_conditions)
            new = and_(*conditions)
            self._where_conditions = [or_(existing, new)]
        else:
            self._where_conditions.extend(conditions)
        return self

    def group_by(self, *columns: ExprLike) -> "QueryBuilder":
        """Add GROUP BY clause.

        Args:
            columns: Columns to group by.

        Returns:
            Self for chaining.
        """
        self._group_by_exprs.extend(_to_expr(c) for c in columns)
        return self

    def group_by_rollup(self, *columns: ExprLike) -> "QueryBuilder":
        """Add GROUP BY with ROLLUP.

        Args:
            columns: Columns to group by.

        Returns:
            Self for chaining.
        """
        self._group_by_exprs.extend(_to_expr(c) for c in columns)
        self._group_by_rollup = True
        return self

    def group_by_cube(self, *columns: ExprLike) -> "QueryBuilder":
        """Add GROUP BY with CUBE.

        Args:
            columns: Columns to group by.

        Returns:
            Self for chaining.
        """
        self._group_by_exprs.extend(_to_expr(c) for c in columns)
        self._group_by_cube = True
        return self

    def grouping_sets(
        self,
        *sets: Sequence[ExprLike],
    ) -> "QueryBuilder":
        """Add GROUPING SETS.

        Args:
            sets: Grouping sets.

        Returns:
            Self for chaining.
        """
        self._grouping_sets = [
            tuple(_to_expr(c) for c in s)
            for s in sets
        ]
        return self

    def having(self, *conditions: Expression) -> "QueryBuilder":
        """Add HAVING conditions.

        Args:
            conditions: Filter conditions for groups.

        Returns:
            Self for chaining.
        """
        self._having_conditions.extend(conditions)
        return self

    def order_by(
        self,
        column: ExprLike | OrderByItem,
        desc: bool = False,
        descending: bool | None = None,
        nulls: NullsPosition | None = None,
    ) -> "QueryBuilder":
        """Add ORDER BY clause.

        Args:
            column: Column or expression to order by.
            desc: Whether to sort descending (short form).
            descending: Whether to sort descending (long form, alias for desc).
            nulls: Position of nulls (NULLS FIRST/LAST).

        Returns:
            Self for chaining.

        Example:
            >>> query.order_by("name")  # ASC
            >>> query.order_by("age", desc=True)  # DESC
            >>> query.order_by("created_at", descending=True)  # DESC (alias)
        """
        if isinstance(column, OrderByItem):
            self._order_by_items.append(column)
        else:
            # Support both 'desc' and 'descending' parameters
            is_descending = descending if descending is not None else desc
            order = SortOrder.DESC if is_descending else SortOrder.ASC
            self._order_by_items.append(OrderByItem(_to_expr(column), order, nulls))
        return self

    def limit(self, count: int | Expression) -> "QueryBuilder":
        """Set LIMIT clause.

        Args:
            count: Maximum number of rows.

        Returns:
            Self for chaining.
        """
        self._limit_value = count
        return self

    def offset(self, offset: int | Expression) -> "QueryBuilder":
        """Set OFFSET clause.

        Args:
            offset: Number of rows to skip.

        Returns:
            Self for chaining.
        """
        self._offset_value = offset
        return self

    def window(self, name: str, spec: WindowSpec | WindowBuilder) -> "QueryBuilder":
        """Define a named window.

        Args:
            name: Window name.
            spec: Window specification.

        Returns:
            Self for chaining.
        """
        if isinstance(spec, WindowBuilder):
            spec = spec.build()
        self._windows[name] = spec
        return self

    def union(
        self,
        other: "QueryBuilder | SelectStatement",
        all: bool = False,
    ) -> "SetOperationBuilder":
        """Create a UNION with another query.

        Args:
            other: Other query.
            all: Whether to use UNION ALL.

        Returns:
            SetOperationBuilder for further operations.
        """
        operation = SetOperation.UNION_ALL if all else SetOperation.UNION
        return SetOperationBuilder(self.build(), operation, other)

    def intersect(
        self,
        other: "QueryBuilder | SelectStatement",
        all: bool = False,
    ) -> "SetOperationBuilder":
        """Create an INTERSECT with another query."""
        operation = SetOperation.INTERSECT_ALL if all else SetOperation.INTERSECT
        return SetOperationBuilder(self.build(), operation, other)

    def except_(
        self,
        other: "QueryBuilder | SelectStatement",
        all: bool = False,
    ) -> "SetOperationBuilder":
        """Create an EXCEPT with another query."""
        operation = SetOperation.EXCEPT_ALL if all else SetOperation.EXCEPT
        return SetOperationBuilder(self.build(), operation, other)

    def build(self) -> SelectStatement:
        """Build the SelectStatement AST node.

        Returns:
            SelectStatement.
        """
        # Handle empty select - default to *
        select_items = self._select_items if self._select_items else [Star()]

        # Build FROM clause
        from_clause = None
        if self._from_source is not None:
            from_clause = FromClause(self._from_source)

        # Build WHERE clause
        where_clause = None
        if self._where_conditions:
            if len(self._where_conditions) == 1:
                where_clause = WhereClause(self._where_conditions[0])
            else:
                where_clause = WhereClause(and_(*self._where_conditions))

        # Build GROUP BY clause
        group_by_clause = None
        if self._group_by_exprs or self._grouping_sets:
            group_by_clause = GroupByClause(
                expressions=self._group_by_exprs,
                with_rollup=self._group_by_rollup,
                with_cube=self._group_by_cube,
                grouping_sets=self._grouping_sets,
            )

        # Build HAVING clause
        having_clause = None
        if self._having_conditions:
            if len(self._having_conditions) == 1:
                having_clause = HavingClause(self._having_conditions[0])
            else:
                having_clause = HavingClause(and_(*self._having_conditions))

        # Build ORDER BY clause
        order_by_clause = None
        if self._order_by_items:
            order_by_clause = OrderByClause(self._order_by_items)

        # Build LIMIT/OFFSET clauses
        limit_clause = None
        if self._limit_value is not None:
            limit_clause = LimitClause(self._limit_value)

        offset_clause = None
        if self._offset_value is not None:
            offset_clause = OffsetClause(self._offset_value)

        return SelectStatement(
            select_items=select_items,
            from_clause=from_clause,
            where_clause=where_clause,
            group_by_clause=group_by_clause,
            having_clause=having_clause,
            order_by_clause=order_by_clause,
            limit_clause=limit_clause,
            offset_clause=offset_clause,
            distinct=self._distinct,
            ctes=self._ctes if self._ctes else None,
        )

    def to_sql(self, dialect: SQLDialect = SQLDialect.GENERIC) -> str:
        """Generate SQL string for the specified dialect.

        Args:
            dialect: SQL dialect.

        Returns:
            SQL string.
        """
        generator = get_dialect_generator(dialect)
        return generator.generate(self.build())


class SetOperationBuilder:
    """Builder for set operations (UNION, INTERSECT, EXCEPT)."""

    def __init__(
        self,
        left: SelectStatement,
        operation: SetOperation,
        right: "QueryBuilder | SelectStatement",
    ) -> None:
        """Initialize set operation builder."""
        if isinstance(right, QueryBuilder):
            right = right.build()
        self._statement = SetOperationStatement(left, right, operation)

    def union(
        self,
        other: "QueryBuilder | SelectStatement",
        all: bool = False,
    ) -> "SetOperationBuilder":
        """Add another UNION."""
        operation = SetOperation.UNION_ALL if all else SetOperation.UNION
        if isinstance(other, QueryBuilder):
            other = other.build()
        self._statement = SetOperationStatement(self._statement, other, operation)
        return self

    def build(self) -> SetOperationStatement:
        """Build the set operation statement."""
        return self._statement

    def to_sql(self, dialect: SQLDialect = SQLDialect.GENERIC) -> str:
        """Generate SQL string."""
        generator = get_dialect_generator(dialect)
        return generator.generate(self.build())
