"""SQL Abstract Syntax Tree (AST) for Query Pushdown.

This module defines the AST nodes that represent SQL queries in a
database-agnostic way. The AST can be traversed and transformed
into dialect-specific SQL strings.

Design Principles:
    - Immutable nodes for thread safety
    - Visitor pattern support for traversal
    - Type-safe construction
    - Full SQL expression coverage

Example:
    >>> from truthound.execution.pushdown.ast import (
    ...     SelectStatement,
    ...     Column,
    ...     Table,
    ...     BinaryExpression,
    ...     ComparisonOp,
    ...     Literal,
    ... )
    >>>
    >>> # Build: SELECT * FROM users WHERE age > 18
    >>> stmt = SelectStatement(
    ...     select_items=[Column("*")],
    ...     from_clause=FromClause(Table("users")),
    ...     where_clause=WhereClause(
    ...         BinaryExpression(
    ...             Column("age"),
    ...             ComparisonOp.GT,
    ...             Literal(18),
    ...         )
    ...     ),
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Generic, Sequence, TypeVar


# =============================================================================
# Base Classes
# =============================================================================


class SQLNode(ABC):
    """Base class for all SQL AST nodes.

    All AST nodes inherit from this class, providing a common
    interface for traversal and transformation.
    """

    @abstractmethod
    def accept(self, visitor: "SQLVisitor") -> Any:
        """Accept a visitor for traversal.

        Args:
            visitor: The visitor to accept.

        Returns:
            Result from the visitor.
        """
        pass

    def children(self) -> Sequence["SQLNode"]:
        """Get child nodes for traversal.

        Returns:
            Sequence of child nodes.
        """
        return []


class Expression(SQLNode):
    """Base class for SQL expressions.

    Expressions are nodes that evaluate to a value, such as:
    - Column references
    - Literals
    - Function calls
    - Binary/Unary operations
    """

    def __and__(self, other: "Expression") -> "BinaryExpression":
        """AND operator."""
        return BinaryExpression(self, LogicalOp.AND, other)

    def __or__(self, other: "Expression") -> "BinaryExpression":
        """OR operator."""
        return BinaryExpression(self, LogicalOp.OR, other)

    def __invert__(self) -> "UnaryExpression":
        """NOT operator."""
        return UnaryExpression(UnaryOp.NOT, self)

    def __eq__(self, other: Any) -> "BinaryExpression":  # type: ignore[override]
        """Equality comparison."""
        return BinaryExpression(self, ComparisonOp.EQ, _to_expression(other))

    def __ne__(self, other: Any) -> "BinaryExpression":  # type: ignore[override]
        """Inequality comparison."""
        return BinaryExpression(self, ComparisonOp.NE, _to_expression(other))

    def __lt__(self, other: Any) -> "BinaryExpression":
        """Less than comparison."""
        return BinaryExpression(self, ComparisonOp.LT, _to_expression(other))

    def __le__(self, other: Any) -> "BinaryExpression":
        """Less than or equal comparison."""
        return BinaryExpression(self, ComparisonOp.LE, _to_expression(other))

    def __gt__(self, other: Any) -> "BinaryExpression":
        """Greater than comparison."""
        return BinaryExpression(self, ComparisonOp.GT, _to_expression(other))

    def __ge__(self, other: Any) -> "BinaryExpression":
        """Greater than or equal comparison."""
        return BinaryExpression(self, ComparisonOp.GE, _to_expression(other))

    def __add__(self, other: Any) -> "BinaryExpression":
        """Addition."""
        return BinaryExpression(self, ArithmeticOp.ADD, _to_expression(other))

    def __sub__(self, other: Any) -> "BinaryExpression":
        """Subtraction."""
        return BinaryExpression(self, ArithmeticOp.SUB, _to_expression(other))

    def __mul__(self, other: Any) -> "BinaryExpression":
        """Multiplication."""
        return BinaryExpression(self, ArithmeticOp.MUL, _to_expression(other))

    def __truediv__(self, other: Any) -> "BinaryExpression":
        """Division."""
        return BinaryExpression(self, ArithmeticOp.DIV, _to_expression(other))

    def __mod__(self, other: Any) -> "BinaryExpression":
        """Modulo."""
        return BinaryExpression(self, ArithmeticOp.MOD, _to_expression(other))

    def __neg__(self) -> "UnaryExpression":
        """Negation."""
        return UnaryExpression(UnaryOp.MINUS, self)

    def alias(self, name: str) -> "Alias":
        """Create an alias for this expression."""
        return Alias(self, name)

    def asc(self) -> "OrderByItem":
        """Create ascending order by item."""
        return OrderByItem(self, SortOrder.ASC)

    def desc(self) -> "OrderByItem":
        """Create descending order by item."""
        return OrderByItem(self, SortOrder.DESC)

    def is_null(self) -> "UnaryExpression":
        """IS NULL check."""
        return UnaryExpression(UnaryOp.IS_NULL, self)

    def is_not_null(self) -> "UnaryExpression":
        """IS NOT NULL check."""
        return UnaryExpression(UnaryOp.IS_NOT_NULL, self)

    def in_(self, values: Sequence[Any]) -> "InExpression":
        """IN operator."""
        exprs = [_to_expression(v) for v in values]
        return InExpression(self, exprs, negated=False)

    def not_in(self, values: Sequence[Any]) -> "InExpression":
        """NOT IN operator."""
        exprs = [_to_expression(v) for v in values]
        return InExpression(self, exprs, negated=True)

    def between(self, low: Any, high: Any) -> "BetweenExpression":
        """BETWEEN operator."""
        return BetweenExpression(
            self,
            _to_expression(low),
            _to_expression(high),
            negated=False,
        )

    def not_between(self, low: Any, high: Any) -> "BetweenExpression":
        """NOT BETWEEN operator."""
        return BetweenExpression(
            self,
            _to_expression(low),
            _to_expression(high),
            negated=True,
        )

    def like(self, pattern: str) -> "BinaryExpression":
        """LIKE operator."""
        return BinaryExpression(self, ComparisonOp.LIKE, Literal(pattern))

    def not_like(self, pattern: str) -> "BinaryExpression":
        """NOT LIKE operator."""
        return BinaryExpression(self, ComparisonOp.NOT_LIKE, Literal(pattern))

    def ilike(self, pattern: str) -> "BinaryExpression":
        """ILIKE operator (case-insensitive LIKE)."""
        return BinaryExpression(self, ComparisonOp.ILIKE, Literal(pattern))

    def regexp(self, pattern: str) -> "BinaryExpression":
        """Regular expression match."""
        return BinaryExpression(self, ComparisonOp.REGEXP, Literal(pattern))


class Statement(SQLNode):
    """Base class for SQL statements (SELECT, INSERT, etc.)."""
    pass


# =============================================================================
# Enums
# =============================================================================


class QueryType(Enum):
    """Type of SQL query."""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"


class ComparisonOp(Enum):
    """SQL comparison operators."""

    EQ = "="
    NE = "<>"
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    ILIKE = "ILIKE"  # Case-insensitive LIKE
    REGEXP = "REGEXP"
    SIMILAR_TO = "SIMILAR TO"


class LogicalOp(Enum):
    """SQL logical operators."""

    AND = "AND"
    OR = "OR"


class ArithmeticOp(Enum):
    """SQL arithmetic operators."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    POWER = "^"


class UnaryOp(Enum):
    """SQL unary operators."""

    NOT = "NOT"
    MINUS = "-"
    PLUS = "+"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT EXISTS"


class BinaryOp(Enum):
    """Combined binary operator enum for convenience."""

    # Comparison
    EQ = "="
    NE = "<>"
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    # Logical
    AND = "AND"
    OR = "OR"
    # Arithmetic
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    # String
    LIKE = "LIKE"
    ILIKE = "ILIKE"
    CONCAT = "||"


class JoinType(Enum):
    """SQL JOIN types."""

    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    LEFT_OUTER = "LEFT OUTER JOIN"
    RIGHT = "RIGHT JOIN"
    RIGHT_OUTER = "RIGHT OUTER JOIN"
    FULL = "FULL JOIN"
    FULL_OUTER = "FULL OUTER JOIN"
    CROSS = "CROSS JOIN"
    NATURAL = "NATURAL JOIN"


class SortOrder(Enum):
    """SQL sort order."""

    ASC = "ASC"
    DESC = "DESC"


class NullsPosition(Enum):
    """Position of NULLs in ORDER BY."""

    FIRST = "NULLS FIRST"
    LAST = "NULLS LAST"


class FrameType(Enum):
    """Window frame type."""

    ROWS = "ROWS"
    RANGE = "RANGE"
    GROUPS = "GROUPS"


class FrameBoundType(Enum):
    """Window frame bound type."""

    UNBOUNDED_PRECEDING = "UNBOUNDED PRECEDING"
    PRECEDING = "PRECEDING"
    CURRENT_ROW = "CURRENT ROW"
    FOLLOWING = "FOLLOWING"
    UNBOUNDED_FOLLOWING = "UNBOUNDED FOLLOWING"


class SetOperation(Enum):
    """SQL set operations."""

    UNION = "UNION"
    UNION_ALL = "UNION ALL"
    INTERSECT = "INTERSECT"
    INTERSECT_ALL = "INTERSECT ALL"
    EXCEPT = "EXCEPT"
    EXCEPT_ALL = "EXCEPT ALL"


# =============================================================================
# Literals
# =============================================================================


@dataclass(frozen=True, eq=False)
class Literal(Expression):
    """A literal value (number, string, boolean, etc.).

    Attributes:
        value: The literal value.
    """

    value: Any

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_literal(self)

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


@dataclass(frozen=True, eq=False)
class NullLiteral(Expression):
    """SQL NULL literal."""

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_null_literal(self)

    def __repr__(self) -> str:
        return "NULL"


@dataclass(frozen=True, eq=False)
class BooleanLiteral(Expression):
    """SQL boolean literal (TRUE/FALSE)."""

    value: bool

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_boolean_literal(self)

    def __repr__(self) -> str:
        return "TRUE" if self.value else "FALSE"


@dataclass(frozen=True, eq=False)
class ArrayLiteral(Expression):
    """SQL array literal."""

    elements: tuple[Expression, ...]

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_array_literal(self)

    def children(self) -> Sequence[SQLNode]:
        return self.elements


# =============================================================================
# Identifiers
# =============================================================================


@dataclass(frozen=True, eq=False)
class Identifier(Expression):
    """A SQL identifier (table name, column name, etc.).

    Attributes:
        name: The identifier name.
        quoted: Whether the identifier should be quoted.
    """

    name: str
    quoted: bool = False

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_identifier(self)

    def __repr__(self) -> str:
        return f"Identifier({self.name!r})"


@dataclass(frozen=True, eq=False)
class Column(Expression):
    """A column reference.

    Attributes:
        name: Column name.
        table: Optional table/alias name.
        schema: Optional schema name.
    """

    name: str
    table: str | None = None
    schema: str | None = None

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_column(self)

    def __repr__(self) -> str:
        parts = []
        if self.schema:
            parts.append(self.schema)
        if self.table:
            parts.append(self.table)
        parts.append(self.name)
        return f"Column({'.'.join(parts)})"


@dataclass(frozen=True)
class Table(SQLNode):
    """A table reference.

    Attributes:
        name: Table name.
        schema: Optional schema name.
        catalog: Optional catalog/database name.
        alias: Optional table alias.
    """

    name: str
    schema: str | None = None
    catalog: str | None = None
    alias: str | None = None

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_table(self)

    def __repr__(self) -> str:
        parts = []
        if self.catalog:
            parts.append(self.catalog)
        if self.schema:
            parts.append(self.schema)
        parts.append(self.name)
        result = '.'.join(parts)
        if self.alias:
            result += f" AS {self.alias}"
        return f"Table({result})"


@dataclass(frozen=True, eq=False)
class Alias(Expression):
    """An aliased expression.

    Attributes:
        expression: The expression being aliased.
        alias: The alias name.
    """

    expression: Expression
    alias: str

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_alias(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.expression]

    def __repr__(self) -> str:
        return f"Alias({self.expression!r} AS {self.alias})"


@dataclass(frozen=True, eq=False)
class Star(Expression):
    """SELECT * or table.* expression.

    Attributes:
        table: Optional table name for table.* syntax.
    """

    table: str | None = None

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_star(self)

    def __repr__(self) -> str:
        if self.table:
            return f"Star({self.table}.*)"
        return "Star(*)"


# =============================================================================
# Expressions
# =============================================================================


@dataclass(frozen=True, eq=False)
class BinaryExpression(Expression):
    """A binary expression (left op right).

    Attributes:
        left: Left operand.
        operator: Binary operator.
        right: Right operand.
    """

    left: Expression
    operator: ComparisonOp | LogicalOp | ArithmeticOp | BinaryOp
    right: Expression

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_binary_expression(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.left, self.right]

    def __repr__(self) -> str:
        op = self.operator.value if isinstance(self.operator, Enum) else self.operator
        return f"({self.left!r} {op} {self.right!r})"


@dataclass(frozen=True, eq=False)
class UnaryExpression(Expression):
    """A unary expression (op operand).

    Attributes:
        operator: Unary operator.
        operand: The operand.
    """

    operator: UnaryOp
    operand: Expression

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_unary_expression(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.operand]

    def __repr__(self) -> str:
        return f"({self.operator.value} {self.operand!r})"


@dataclass(frozen=True, eq=False)
class InExpression(Expression):
    """IN expression (expr IN (values)).

    Attributes:
        expression: The expression to test.
        values: List of values or subquery.
        negated: Whether this is NOT IN.
    """

    expression: Expression
    values: tuple[Expression, ...] | "SelectStatement"
    negated: bool = False

    def __init__(
        self,
        expression: Expression,
        values: Sequence[Expression] | "SelectStatement",
        negated: bool = False,
    ):
        object.__setattr__(self, "expression", expression)
        if isinstance(values, SelectStatement):
            object.__setattr__(self, "values", values)
        else:
            object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "negated", negated)

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_in_expression(self)

    def children(self) -> Sequence[SQLNode]:
        children: list[SQLNode] = [self.expression]
        if isinstance(self.values, SelectStatement):
            children.append(self.values)
        else:
            children.extend(self.values)
        return children

    def __repr__(self) -> str:
        op = "NOT IN" if self.negated else "IN"
        return f"({self.expression!r} {op} {self.values!r})"


@dataclass(frozen=True, eq=False)
class BetweenExpression(Expression):
    """BETWEEN expression (expr BETWEEN low AND high).

    Attributes:
        expression: The expression to test.
        low: Lower bound.
        high: Upper bound.
        negated: Whether this is NOT BETWEEN.
    """

    expression: Expression
    low: Expression
    high: Expression
    negated: bool = False

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_between_expression(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.expression, self.low, self.high]

    def __repr__(self) -> str:
        op = "NOT BETWEEN" if self.negated else "BETWEEN"
        return f"({self.expression!r} {op} {self.low!r} AND {self.high!r})"


@dataclass(frozen=True, eq=False)
class ExistsExpression(Expression):
    """EXISTS expression.

    Attributes:
        subquery: The subquery to test.
        negated: Whether this is NOT EXISTS.
    """

    subquery: "SelectStatement"
    negated: bool = False

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_exists_expression(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.subquery]

    def __repr__(self) -> str:
        op = "NOT EXISTS" if self.negated else "EXISTS"
        return f"({op} {self.subquery!r})"


@dataclass(frozen=True, eq=False)
class SubqueryExpression(Expression):
    """A subquery used as an expression.

    Attributes:
        subquery: The subquery.
    """

    subquery: "SelectStatement"

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_subquery_expression(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.subquery]


@dataclass(frozen=True, eq=False)
class CastExpression(Expression):
    """CAST expression (CAST(expr AS type)).

    Attributes:
        expression: The expression to cast.
        target_type: The target SQL type.
    """

    expression: Expression
    target_type: str

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_cast_expression(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.expression]

    def __repr__(self) -> str:
        return f"CAST({self.expression!r} AS {self.target_type})"


@dataclass(frozen=True)
class WhenClause(SQLNode):
    """WHEN clause in CASE expression.

    Attributes:
        condition: The condition to test.
        result: The result if condition is true.
    """

    condition: Expression
    result: Expression

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_when_clause(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.condition, self.result]


@dataclass(frozen=True, eq=False)
class CaseExpression(Expression):
    """CASE expression.

    Attributes:
        operand: Optional operand for simple CASE.
        when_clauses: List of WHEN clauses.
        else_result: Optional ELSE result.
    """

    when_clauses: tuple[WhenClause, ...]
    operand: Expression | None = None
    else_result: Expression | None = None

    def __init__(
        self,
        when_clauses: Sequence[WhenClause],
        operand: Expression | None = None,
        else_result: Expression | None = None,
    ):
        object.__setattr__(self, "when_clauses", tuple(when_clauses))
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "else_result", else_result)

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_case_expression(self)

    def children(self) -> Sequence[SQLNode]:
        children: list[SQLNode] = []
        if self.operand:
            children.append(self.operand)
        children.extend(self.when_clauses)
        if self.else_result:
            children.append(self.else_result)
        return children


# =============================================================================
# Function Calls
# =============================================================================


@dataclass(frozen=True, eq=False)
class FunctionCall(Expression):
    """A SQL function call.

    Attributes:
        name: Function name.
        arguments: Function arguments.
        distinct: Whether DISTINCT is applied (for aggregates).
        filter_clause: Optional FILTER clause.
    """

    name: str
    arguments: tuple[Expression, ...]
    distinct: bool = False
    filter_clause: Expression | None = None

    def __init__(
        self,
        name: str,
        arguments: Sequence[Expression] | None = None,
        distinct: bool = False,
        filter_clause: Expression | None = None,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", tuple(arguments or []))
        object.__setattr__(self, "distinct", distinct)
        object.__setattr__(self, "filter_clause", filter_clause)

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_function_call(self)

    def children(self) -> Sequence[SQLNode]:
        children: list[SQLNode] = list(self.arguments)
        if self.filter_clause:
            children.append(self.filter_clause)
        return children

    def __repr__(self) -> str:
        args = ", ".join(repr(a) for a in self.arguments)
        distinct = "DISTINCT " if self.distinct else ""
        return f"{self.name}({distinct}{args})"


@dataclass(frozen=True, eq=False)
class AggregateFunction(Expression):
    """An aggregate function call (COUNT, SUM, AVG, etc.).

    Attributes:
        name: Function name (COUNT, SUM, AVG, MIN, MAX, etc.).
        argument: The argument to aggregate.
        distinct: Whether DISTINCT is applied.
        filter_clause: Optional FILTER (WHERE ...) clause.
        order_by: Optional ORDER BY for ordered-set aggregates.
    """

    name: str
    argument: Expression | None
    distinct: bool = False
    filter_clause: Expression | None = None
    order_by: tuple["OrderByItem", ...] | None = None

    def __init__(
        self,
        name: str,
        argument: Expression | None = None,
        distinct: bool = False,
        filter_clause: Expression | None = None,
        order_by: Sequence["OrderByItem"] | None = None,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "argument", argument)
        object.__setattr__(self, "distinct", distinct)
        object.__setattr__(self, "filter_clause", filter_clause)
        object.__setattr__(
            self, "order_by", tuple(order_by) if order_by else None
        )

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_aggregate_function(self)

    def children(self) -> Sequence[SQLNode]:
        children: list[SQLNode] = []
        if self.argument:
            children.append(self.argument)
        if self.filter_clause:
            children.append(self.filter_clause)
        if self.order_by:
            children.extend(self.order_by)
        return children

    def over(self, window: "WindowSpec | str | None" = None) -> "WindowFunction":
        """Add OVER clause to create a window function."""
        return WindowFunction(self, window)

    def __repr__(self) -> str:
        if self.argument:
            distinct = "DISTINCT " if self.distinct else ""
            return f"{self.name}({distinct}{self.argument!r})"
        return f"{self.name}(*)"


@dataclass(frozen=True)
class FrameBound(SQLNode):
    """Window frame bound.

    Attributes:
        bound_type: Type of bound.
        offset: Optional numeric offset.
    """

    bound_type: FrameBoundType
    offset: int | None = None

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_frame_bound(self)

    def __repr__(self) -> str:
        if self.offset is not None:
            if self.bound_type == FrameBoundType.PRECEDING:
                return f"{self.offset} PRECEDING"
            elif self.bound_type == FrameBoundType.FOLLOWING:
                return f"{self.offset} FOLLOWING"
        return self.bound_type.value


@dataclass(frozen=True)
class WindowSpec(SQLNode):
    """Window specification for window functions.

    Attributes:
        partition_by: Optional PARTITION BY columns.
        order_by: Optional ORDER BY items.
        frame_type: Optional frame type (ROWS, RANGE, GROUPS).
        frame_start: Optional frame start bound.
        frame_end: Optional frame end bound.
        name: Optional window name (for WINDOW clause).
    """

    partition_by: tuple[Expression, ...] | None = None
    order_by: tuple["OrderByItem", ...] | None = None
    frame_type: FrameType | None = None
    frame_start: FrameBound | None = None
    frame_end: FrameBound | None = None
    name: str | None = None

    def __init__(
        self,
        partition_by: Sequence[Expression] | None = None,
        order_by: Sequence["OrderByItem"] | None = None,
        frame_type: FrameType | None = None,
        frame_start: FrameBound | None = None,
        frame_end: FrameBound | None = None,
        name: str | None = None,
    ):
        object.__setattr__(
            self, "partition_by", tuple(partition_by) if partition_by else None
        )
        object.__setattr__(
            self, "order_by", tuple(order_by) if order_by else None
        )
        object.__setattr__(self, "frame_type", frame_type)
        object.__setattr__(self, "frame_start", frame_start)
        object.__setattr__(self, "frame_end", frame_end)
        object.__setattr__(self, "name", name)

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_window_spec(self)

    def children(self) -> Sequence[SQLNode]:
        children: list[SQLNode] = []
        if self.partition_by:
            children.extend(self.partition_by)
        if self.order_by:
            children.extend(self.order_by)
        if self.frame_start:
            children.append(self.frame_start)
        if self.frame_end:
            children.append(self.frame_end)
        return children


@dataclass(frozen=True, eq=False)
class WindowFunction(Expression):
    """A window function expression.

    Attributes:
        function: The aggregate or window function.
        window_spec: The window specification or named window.
    """

    function: AggregateFunction | FunctionCall
    window_spec: WindowSpec | str | None

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_window_function(self)

    def children(self) -> Sequence[SQLNode]:
        children: list[SQLNode] = [self.function]
        if isinstance(self.window_spec, WindowSpec):
            children.append(self.window_spec)
        return children

    def __repr__(self) -> str:
        if self.window_spec:
            return f"{self.function!r} OVER ({self.window_spec!r})"
        return f"{self.function!r} OVER ()"


# =============================================================================
# SELECT Clauses
# =============================================================================


@dataclass(frozen=True)
class SelectItem(SQLNode):
    """A single item in SELECT clause.

    Attributes:
        expression: The expression.
        alias: Optional alias.
    """

    expression: Expression
    alias: str | None = None

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_select_item(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.expression]


@dataclass(frozen=True)
class FromClause(SQLNode):
    """FROM clause.

    Attributes:
        source: Table, subquery, or joined tables.
    """

    source: Table | "SelectStatement" | "JoinClause"

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_from_clause(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.source]


@dataclass(frozen=True)
class JoinClause(SQLNode):
    """A JOIN clause.

    Attributes:
        left: Left side of join.
        right: Right side of join.
        join_type: Type of join.
        condition: Optional join condition (for ON clause).
        using_columns: Optional USING columns.
    """

    left: Table | "SelectStatement" | "JoinClause"
    right: Table | "SelectStatement"
    join_type: JoinType
    condition: Expression | None = None
    using_columns: tuple[str, ...] | None = None

    def __init__(
        self,
        left: Table | "SelectStatement" | "JoinClause",
        right: Table | "SelectStatement",
        join_type: JoinType,
        condition: Expression | None = None,
        using_columns: Sequence[str] | None = None,
    ):
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "join_type", join_type)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(
            self, "using_columns", tuple(using_columns) if using_columns else None
        )

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_join_clause(self)

    def children(self) -> Sequence[SQLNode]:
        children: list[SQLNode] = [self.left, self.right]
        if self.condition:
            children.append(self.condition)
        return children


@dataclass(frozen=True)
class WhereClause(SQLNode):
    """WHERE clause.

    Attributes:
        condition: The filter condition.
    """

    condition: Expression

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_where_clause(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.condition]


@dataclass(frozen=True)
class GroupByClause(SQLNode):
    """GROUP BY clause.

    Attributes:
        expressions: Grouping expressions.
        with_rollup: Whether to include ROLLUP.
        with_cube: Whether to include CUBE.
        grouping_sets: Optional grouping sets.
    """

    expressions: tuple[Expression, ...]
    with_rollup: bool = False
    with_cube: bool = False
    grouping_sets: tuple[tuple[Expression, ...], ...] | None = None

    def __init__(
        self,
        expressions: Sequence[Expression],
        with_rollup: bool = False,
        with_cube: bool = False,
        grouping_sets: Sequence[Sequence[Expression]] | None = None,
    ):
        object.__setattr__(self, "expressions", tuple(expressions))
        object.__setattr__(self, "with_rollup", with_rollup)
        object.__setattr__(self, "with_cube", with_cube)
        if grouping_sets:
            object.__setattr__(
                self,
                "grouping_sets",
                tuple(tuple(gs) for gs in grouping_sets),
            )
        else:
            object.__setattr__(self, "grouping_sets", None)

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_group_by_clause(self)

    def children(self) -> Sequence[SQLNode]:
        return list(self.expressions)


@dataclass(frozen=True)
class HavingClause(SQLNode):
    """HAVING clause.

    Attributes:
        condition: The filter condition (applied after GROUP BY).
    """

    condition: Expression

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_having_clause(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.condition]


@dataclass(frozen=True)
class OrderByItem(SQLNode):
    """A single item in ORDER BY clause.

    Attributes:
        expression: The expression to sort by.
        order: Sort order (ASC/DESC).
        nulls: Position of NULLs.
    """

    expression: Expression
    order: SortOrder = SortOrder.ASC
    nulls: NullsPosition | None = None

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_order_by_item(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.expression]


@dataclass(frozen=True)
class OrderByClause(SQLNode):
    """ORDER BY clause.

    Attributes:
        items: Order by items.
    """

    items: tuple[OrderByItem, ...]

    def __init__(self, items: Sequence[OrderByItem]):
        object.__setattr__(self, "items", tuple(items))

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_order_by_clause(self)

    def children(self) -> Sequence[SQLNode]:
        return list(self.items)


@dataclass(frozen=True)
class LimitClause(SQLNode):
    """LIMIT clause.

    Attributes:
        count: Maximum number of rows.
    """

    count: int | Expression

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_limit_clause(self)

    def children(self) -> Sequence[SQLNode]:
        if isinstance(self.count, Expression):
            return [self.count]
        return []


@dataclass(frozen=True)
class OffsetClause(SQLNode):
    """OFFSET clause.

    Attributes:
        offset: Number of rows to skip.
    """

    offset: int | Expression

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_offset_clause(self)

    def children(self) -> Sequence[SQLNode]:
        if isinstance(self.offset, Expression):
            return [self.offset]
        return []


# =============================================================================
# Statements
# =============================================================================


@dataclass(frozen=True)
class SelectStatement(Statement):
    """A complete SELECT statement.

    Attributes:
        select_items: Items to select.
        from_clause: Optional FROM clause.
        where_clause: Optional WHERE clause.
        group_by_clause: Optional GROUP BY clause.
        having_clause: Optional HAVING clause.
        order_by_clause: Optional ORDER BY clause.
        limit_clause: Optional LIMIT clause.
        offset_clause: Optional OFFSET clause.
        distinct: Whether to use SELECT DISTINCT.
        ctes: Optional Common Table Expressions.
    """

    select_items: tuple[SelectItem | Expression, ...]
    from_clause: FromClause | None = None
    where_clause: WhereClause | None = None
    group_by_clause: GroupByClause | None = None
    having_clause: HavingClause | None = None
    order_by_clause: OrderByClause | None = None
    limit_clause: LimitClause | None = None
    offset_clause: OffsetClause | None = None
    distinct: bool = False
    ctes: tuple["CTEClause", ...] | None = None

    def __init__(
        self,
        select_items: Sequence[SelectItem | Expression],
        from_clause: FromClause | None = None,
        where_clause: WhereClause | None = None,
        group_by_clause: GroupByClause | None = None,
        having_clause: HavingClause | None = None,
        order_by_clause: OrderByClause | None = None,
        limit_clause: LimitClause | None = None,
        offset_clause: OffsetClause | None = None,
        distinct: bool = False,
        ctes: Sequence["CTEClause"] | None = None,
    ):
        object.__setattr__(self, "select_items", tuple(select_items))
        object.__setattr__(self, "from_clause", from_clause)
        object.__setattr__(self, "where_clause", where_clause)
        object.__setattr__(self, "group_by_clause", group_by_clause)
        object.__setattr__(self, "having_clause", having_clause)
        object.__setattr__(self, "order_by_clause", order_by_clause)
        object.__setattr__(self, "limit_clause", limit_clause)
        object.__setattr__(self, "offset_clause", offset_clause)
        object.__setattr__(self, "distinct", distinct)
        object.__setattr__(self, "ctes", tuple(ctes) if ctes else None)

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_select_statement(self)

    def children(self) -> Sequence[SQLNode]:
        children: list[SQLNode] = list(self.select_items)
        if self.ctes:
            children.extend(self.ctes)
        if self.from_clause:
            children.append(self.from_clause)
        if self.where_clause:
            children.append(self.where_clause)
        if self.group_by_clause:
            children.append(self.group_by_clause)
        if self.having_clause:
            children.append(self.having_clause)
        if self.order_by_clause:
            children.append(self.order_by_clause)
        if self.limit_clause:
            children.append(self.limit_clause)
        if self.offset_clause:
            children.append(self.offset_clause)
        return children


@dataclass(frozen=True)
class CTEClause(SQLNode):
    """Common Table Expression (WITH clause).

    Attributes:
        name: CTE name.
        query: The CTE query.
        columns: Optional column names.
        recursive: Whether this is a recursive CTE.
    """

    name: str
    query: SelectStatement
    columns: tuple[str, ...] | None = None
    recursive: bool = False

    def __init__(
        self,
        name: str,
        query: SelectStatement,
        columns: Sequence[str] | None = None,
        recursive: bool = False,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "columns", tuple(columns) if columns else None)
        object.__setattr__(self, "recursive", recursive)

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_cte_clause(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.query]


@dataclass(frozen=True)
class SetOperationStatement(Statement):
    """A set operation (UNION, INTERSECT, EXCEPT).

    Attributes:
        left: Left query.
        right: Right query.
        operation: Set operation type.
    """

    left: SelectStatement | "SetOperationStatement"
    right: SelectStatement | "SetOperationStatement"
    operation: SetOperation

    def accept(self, visitor: "SQLVisitor") -> Any:
        return visitor.visit_set_operation(self)

    def children(self) -> Sequence[SQLNode]:
        return [self.left, self.right]


# =============================================================================
# Visitor Pattern
# =============================================================================


class SQLVisitor(ABC):
    """Abstract visitor for SQL AST traversal.

    Implement this class to transform or analyze SQL AST nodes.
    """

    @abstractmethod
    def visit_literal(self, node: Literal) -> Any:
        pass

    @abstractmethod
    def visit_null_literal(self, node: NullLiteral) -> Any:
        pass

    @abstractmethod
    def visit_boolean_literal(self, node: BooleanLiteral) -> Any:
        pass

    @abstractmethod
    def visit_array_literal(self, node: ArrayLiteral) -> Any:
        pass

    @abstractmethod
    def visit_identifier(self, node: Identifier) -> Any:
        pass

    @abstractmethod
    def visit_column(self, node: Column) -> Any:
        pass

    @abstractmethod
    def visit_table(self, node: Table) -> Any:
        pass

    @abstractmethod
    def visit_alias(self, node: Alias) -> Any:
        pass

    @abstractmethod
    def visit_star(self, node: Star) -> Any:
        pass

    @abstractmethod
    def visit_binary_expression(self, node: BinaryExpression) -> Any:
        pass

    @abstractmethod
    def visit_unary_expression(self, node: UnaryExpression) -> Any:
        pass

    @abstractmethod
    def visit_in_expression(self, node: InExpression) -> Any:
        pass

    @abstractmethod
    def visit_between_expression(self, node: BetweenExpression) -> Any:
        pass

    @abstractmethod
    def visit_exists_expression(self, node: ExistsExpression) -> Any:
        pass

    @abstractmethod
    def visit_subquery_expression(self, node: SubqueryExpression) -> Any:
        pass

    @abstractmethod
    def visit_cast_expression(self, node: CastExpression) -> Any:
        pass

    @abstractmethod
    def visit_when_clause(self, node: WhenClause) -> Any:
        pass

    @abstractmethod
    def visit_case_expression(self, node: CaseExpression) -> Any:
        pass

    @abstractmethod
    def visit_function_call(self, node: FunctionCall) -> Any:
        pass

    @abstractmethod
    def visit_aggregate_function(self, node: AggregateFunction) -> Any:
        pass

    @abstractmethod
    def visit_frame_bound(self, node: FrameBound) -> Any:
        pass

    @abstractmethod
    def visit_window_spec(self, node: WindowSpec) -> Any:
        pass

    @abstractmethod
    def visit_window_function(self, node: WindowFunction) -> Any:
        pass

    @abstractmethod
    def visit_select_item(self, node: SelectItem) -> Any:
        pass

    @abstractmethod
    def visit_from_clause(self, node: FromClause) -> Any:
        pass

    @abstractmethod
    def visit_join_clause(self, node: JoinClause) -> Any:
        pass

    @abstractmethod
    def visit_where_clause(self, node: WhereClause) -> Any:
        pass

    @abstractmethod
    def visit_group_by_clause(self, node: GroupByClause) -> Any:
        pass

    @abstractmethod
    def visit_having_clause(self, node: HavingClause) -> Any:
        pass

    @abstractmethod
    def visit_order_by_item(self, node: OrderByItem) -> Any:
        pass

    @abstractmethod
    def visit_order_by_clause(self, node: OrderByClause) -> Any:
        pass

    @abstractmethod
    def visit_limit_clause(self, node: LimitClause) -> Any:
        pass

    @abstractmethod
    def visit_offset_clause(self, node: OffsetClause) -> Any:
        pass

    @abstractmethod
    def visit_select_statement(self, node: SelectStatement) -> Any:
        pass

    @abstractmethod
    def visit_cte_clause(self, node: CTEClause) -> Any:
        pass

    @abstractmethod
    def visit_set_operation(self, node: SetOperationStatement) -> Any:
        pass


# =============================================================================
# Helper Functions
# =============================================================================


def _to_expression(value: Any) -> Expression:
    """Convert a value to an Expression.

    Args:
        value: Value to convert.

    Returns:
        Expression representing the value.
    """
    if isinstance(value, Expression):
        return value
    if value is None:
        return NullLiteral()
    if isinstance(value, bool):
        return BooleanLiteral(value)
    return Literal(value)
