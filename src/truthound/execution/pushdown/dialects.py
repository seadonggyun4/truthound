"""SQL Dialect-specific Code Generators.

This module provides dialect-specific SQL generators that convert
AST nodes into valid SQL strings for each database platform.

Supported Dialects:
    - PostgreSQL
    - MySQL
    - SQLite
    - BigQuery
    - Snowflake
    - Redshift
    - Databricks (Spark SQL)
    - Oracle
    - SQL Server (T-SQL)

Example:
    >>> from truthound.execution.pushdown.dialects import (
    ...     SQLDialect,
    ...     get_dialect_generator,
    ... )
    >>> from truthound.execution.pushdown.ast import SelectStatement, Column, Table
    >>>
    >>> stmt = SelectStatement(...)
    >>> generator = get_dialect_generator(SQLDialect.POSTGRESQL)
    >>> sql = generator.generate(stmt)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from truthound.execution.pushdown.ast import (
    # Base
    SQLNode,
    SQLVisitor,
    Expression,
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
    BinaryOp,
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


# =============================================================================
# SQL Dialect Enum
# =============================================================================


class SQLDialect(Enum):
    """Supported SQL dialects."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    DATABRICKS = "databricks"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    DUCKDB = "duckdb"
    CLICKHOUSE = "clickhouse"
    PRESTO = "presto"
    TRINO = "trino"
    HIVE = "hive"
    GENERIC = "generic"


# =============================================================================
# Dialect Configuration
# =============================================================================


@dataclass
class DialectConfig:
    """Configuration for SQL dialect.

    Attributes:
        identifier_quote: Character(s) to quote identifiers.
        string_quote: Character to quote strings.
        escape_char: Character for escaping.
        supports_nulls_ordering: Whether NULLS FIRST/LAST is supported.
        supports_limit_offset: Whether LIMIT/OFFSET syntax is supported.
        supports_fetch_first: Whether FETCH FIRST syntax is supported.
        supports_top: Whether TOP syntax is supported (SQL Server).
        supports_window_functions: Whether window functions are supported.
        supports_cte: Whether WITH clause (CTEs) are supported.
        supports_lateral: Whether LATERAL joins are supported.
        supports_json: Whether JSON functions are supported.
        supports_arrays: Whether array types are supported.
        supports_filter_clause: Whether FILTER (WHERE ...) clause is supported.
        supports_grouping_sets: Whether GROUPING SETS are supported.
        max_identifier_length: Maximum identifier length.
        case_sensitive_identifiers: Whether identifiers are case-sensitive.
        boolean_as_int: Whether booleans are represented as integers.
        concat_operator: Operator for string concatenation.
        date_format: Default date format string.
        timestamp_format: Default timestamp format string.
    """

    identifier_quote: str = '"'
    string_quote: str = "'"
    escape_char: str = "\\"
    supports_nulls_ordering: bool = True
    supports_limit_offset: bool = True
    supports_fetch_first: bool = False
    supports_top: bool = False
    supports_window_functions: bool = True
    supports_cte: bool = True
    supports_lateral: bool = False
    supports_json: bool = True
    supports_arrays: bool = False
    supports_filter_clause: bool = False
    supports_grouping_sets: bool = True
    max_identifier_length: int = 128
    case_sensitive_identifiers: bool = False
    boolean_as_int: bool = False
    concat_operator: str = "||"
    date_format: str = "YYYY-MM-DD"
    timestamp_format: str = "YYYY-MM-DD HH24:MI:SS"


# Predefined configs for common dialects
DIALECT_CONFIGS: dict[SQLDialect, DialectConfig] = {
    SQLDialect.POSTGRESQL: DialectConfig(
        identifier_quote='"',
        supports_nulls_ordering=True,
        supports_filter_clause=True,
        supports_lateral=True,
        supports_arrays=True,
    ),
    SQLDialect.MYSQL: DialectConfig(
        identifier_quote="`",
        supports_nulls_ordering=False,
        supports_filter_clause=False,
        concat_operator="CONCAT",
    ),
    SQLDialect.SQLITE: DialectConfig(
        identifier_quote='"',
        supports_nulls_ordering=False,
        supports_cte=True,
        supports_window_functions=True,
        supports_lateral=False,
    ),
    SQLDialect.BIGQUERY: DialectConfig(
        identifier_quote="`",
        supports_nulls_ordering=True,
        supports_arrays=True,
        supports_lateral=True,
    ),
    SQLDialect.SNOWFLAKE: DialectConfig(
        identifier_quote='"',
        supports_nulls_ordering=True,
        supports_lateral=True,
        case_sensitive_identifiers=False,
    ),
    SQLDialect.REDSHIFT: DialectConfig(
        identifier_quote='"',
        supports_nulls_ordering=True,
        supports_lateral=False,
        max_identifier_length=127,
    ),
    SQLDialect.DATABRICKS: DialectConfig(
        identifier_quote="`",
        supports_nulls_ordering=True,
        supports_arrays=True,
        supports_lateral=True,
    ),
    SQLDialect.ORACLE: DialectConfig(
        identifier_quote='"',
        supports_limit_offset=False,
        supports_fetch_first=True,
        supports_nulls_ordering=True,
        max_identifier_length=30,
    ),
    SQLDialect.SQLSERVER: DialectConfig(
        identifier_quote="[",
        supports_limit_offset=False,
        supports_top=True,
        supports_nulls_ordering=False,
        concat_operator="+",
    ),
    SQLDialect.DUCKDB: DialectConfig(
        identifier_quote='"',
        supports_nulls_ordering=True,
        supports_filter_clause=True,
        supports_arrays=True,
    ),
    SQLDialect.GENERIC: DialectConfig(),
}


# =============================================================================
# Base Dialect Generator
# =============================================================================


class BaseDialectGenerator(SQLVisitor, ABC):
    """Base class for SQL dialect generators.

    This class implements the visitor pattern to convert AST nodes
    into SQL strings. Subclasses can override methods to customize
    SQL generation for specific dialects.

    Attributes:
        config: Dialect configuration.
        indent_level: Current indentation level.
        indent_str: String used for indentation.
    """

    dialect: SQLDialect = SQLDialect.GENERIC

    def __init__(
        self,
        config: DialectConfig | None = None,
        indent_str: str = "  ",
    ) -> None:
        """Initialize generator.

        Args:
            config: Optional dialect configuration.
            indent_str: String used for indentation.
        """
        self.config = config or DIALECT_CONFIGS.get(
            self.dialect, DialectConfig()
        )
        self.indent_level = 0
        self.indent_str = indent_str
        self._function_mappings: dict[str, str] = {}

    def generate(self, node: SQLNode) -> str:
        """Generate SQL string from AST node.

        Args:
            node: AST node to generate SQL for.

        Returns:
            SQL string.
        """
        return node.accept(self)

    def _indent(self) -> str:
        """Get current indentation string."""
        return self.indent_str * self.indent_level

    def _quote_identifier(self, name: str) -> str:
        """Quote an identifier.

        Args:
            name: Identifier name.

        Returns:
            Quoted identifier.
        """
        quote = self.config.identifier_quote
        if quote == "[":
            return f"[{name}]"
        return f"{quote}{name}{quote}"

    def _quote_string(self, value: str) -> str:
        """Quote a string literal.

        Args:
            value: String value.

        Returns:
            Quoted string literal.
        """
        quote = self.config.string_quote
        escaped = value.replace(quote, quote + quote)
        return f"{quote}{escaped}{quote}"

    def _format_literal(self, value: Any) -> str:
        """Format a literal value.

        Args:
            value: Value to format.

        Returns:
            SQL literal string.
        """
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            if self.config.boolean_as_int:
                return "1" if value else "0"
            return "TRUE" if value else "FALSE"
        if isinstance(value, str):
            return self._quote_string(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, bytes):
            return f"X'{value.hex()}'"
        return self._quote_string(str(value))

    def _map_function(self, name: str) -> str:
        """Map a function name to dialect-specific name.

        Args:
            name: Function name.

        Returns:
            Dialect-specific function name.
        """
        return self._function_mappings.get(name.upper(), name)

    # -------------------------------------------------------------------------
    # Visitor Methods - Literals
    # -------------------------------------------------------------------------

    def visit_literal(self, node: Literal) -> str:
        return self._format_literal(node.value)

    def visit_null_literal(self, node: NullLiteral) -> str:
        return "NULL"

    def visit_boolean_literal(self, node: BooleanLiteral) -> str:
        if self.config.boolean_as_int:
            return "1" if node.value else "0"
        return "TRUE" if node.value else "FALSE"

    def visit_array_literal(self, node: ArrayLiteral) -> str:
        elements = ", ".join(e.accept(self) for e in node.elements)
        return f"ARRAY[{elements}]"

    # -------------------------------------------------------------------------
    # Visitor Methods - Identifiers
    # -------------------------------------------------------------------------

    def visit_identifier(self, node: Identifier) -> str:
        if node.quoted:
            return self._quote_identifier(node.name)
        return node.name

    def visit_column(self, node: Column) -> str:
        parts = []
        if node.schema:
            parts.append(self._quote_identifier(node.schema))
        if node.table:
            parts.append(self._quote_identifier(node.table))

        if node.name == "*":
            parts.append("*")
        else:
            parts.append(self._quote_identifier(node.name))

        return ".".join(parts)

    def visit_table(self, node: Table) -> str:
        parts = []
        if node.catalog:
            parts.append(self._quote_identifier(node.catalog))
        if node.schema:
            parts.append(self._quote_identifier(node.schema))
        parts.append(self._quote_identifier(node.name))

        result = ".".join(parts)
        if node.alias:
            result += f" AS {self._quote_identifier(node.alias)}"
        return result

    def visit_alias(self, node: Alias) -> str:
        expr = node.expression.accept(self)
        return f"{expr} AS {self._quote_identifier(node.alias)}"

    def visit_star(self, node: Star) -> str:
        if node.table:
            return f"{self._quote_identifier(node.table)}.*"
        return "*"

    # -------------------------------------------------------------------------
    # Visitor Methods - Expressions
    # -------------------------------------------------------------------------

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        left = node.left.accept(self)
        right = node.right.accept(self)

        # Get operator string
        if isinstance(node.operator, Enum):
            op = node.operator.value
        else:
            op = str(node.operator)

        # Handle special cases
        if op == "||" and self.config.concat_operator == "CONCAT":
            return f"CONCAT({left}, {right})"

        return f"({left} {op} {right})"

    def visit_unary_expression(self, node: UnaryExpression) -> str:
        operand = node.operand.accept(self)

        if node.operator in (UnaryOp.IS_NULL, UnaryOp.IS_NOT_NULL):
            return f"({operand} {node.operator.value})"

        if node.operator in (UnaryOp.EXISTS, UnaryOp.NOT_EXISTS):
            return f"{node.operator.value} ({operand})"

        return f"({node.operator.value} {operand})"

    def visit_in_expression(self, node: InExpression) -> str:
        expr = node.expression.accept(self)
        op = "NOT IN" if node.negated else "IN"

        if isinstance(node.values, SelectStatement):
            subquery = node.values.accept(self)
            return f"{expr} {op} ({subquery})"
        else:
            values = ", ".join(v.accept(self) for v in node.values)
            return f"{expr} {op} ({values})"

    def visit_between_expression(self, node: BetweenExpression) -> str:
        expr = node.expression.accept(self)
        low = node.low.accept(self)
        high = node.high.accept(self)
        op = "NOT BETWEEN" if node.negated else "BETWEEN"
        return f"{expr} {op} {low} AND {high}"

    def visit_exists_expression(self, node: ExistsExpression) -> str:
        subquery = node.subquery.accept(self)
        op = "NOT EXISTS" if node.negated else "EXISTS"
        return f"{op} ({subquery})"

    def visit_subquery_expression(self, node: SubqueryExpression) -> str:
        subquery = node.subquery.accept(self)
        return f"({subquery})"

    def visit_cast_expression(self, node: CastExpression) -> str:
        expr = node.expression.accept(self)
        return f"CAST({expr} AS {node.target_type})"

    def visit_when_clause(self, node: WhenClause) -> str:
        condition = node.condition.accept(self)
        result = node.result.accept(self)
        return f"WHEN {condition} THEN {result}"

    def visit_case_expression(self, node: CaseExpression) -> str:
        parts = ["CASE"]

        if node.operand:
            parts.append(node.operand.accept(self))

        for when in node.when_clauses:
            parts.append(when.accept(self))

        if node.else_result:
            parts.append(f"ELSE {node.else_result.accept(self)}")

        parts.append("END")
        return " ".join(parts)

    # -------------------------------------------------------------------------
    # Visitor Methods - Functions
    # -------------------------------------------------------------------------

    def visit_function_call(self, node: FunctionCall) -> str:
        name = self._map_function(node.name)
        distinct = "DISTINCT " if node.distinct else ""

        if node.arguments:
            args = ", ".join(a.accept(self) for a in node.arguments)
        else:
            args = ""

        result = f"{name}({distinct}{args})"

        if node.filter_clause and self.config.supports_filter_clause:
            filter_expr = node.filter_clause.accept(self)
            result += f" FILTER (WHERE {filter_expr})"

        return result

    def visit_aggregate_function(self, node: AggregateFunction) -> str:
        name = self._map_function(node.name)
        distinct = "DISTINCT " if node.distinct else ""

        if node.argument:
            arg = node.argument.accept(self)
        else:
            arg = "*"

        # Handle ordered-set aggregates
        if node.order_by:
            order_items = ", ".join(o.accept(self) for o in node.order_by)
            result = f"{name}({distinct}{arg} ORDER BY {order_items})"
        else:
            result = f"{name}({distinct}{arg})"

        # Add FILTER clause
        if node.filter_clause and self.config.supports_filter_clause:
            filter_expr = node.filter_clause.accept(self)
            result += f" FILTER (WHERE {filter_expr})"

        return result

    def visit_frame_bound(self, node: FrameBound) -> str:
        if node.offset is not None:
            if node.bound_type == FrameBoundType.PRECEDING:
                return f"{node.offset} PRECEDING"
            elif node.bound_type == FrameBoundType.FOLLOWING:
                return f"{node.offset} FOLLOWING"
        return node.bound_type.value

    def visit_window_spec(self, node: WindowSpec) -> str:
        parts = []

        # PARTITION BY
        if node.partition_by:
            partition = ", ".join(e.accept(self) for e in node.partition_by)
            parts.append(f"PARTITION BY {partition}")

        # ORDER BY
        if node.order_by:
            order = ", ".join(o.accept(self) for o in node.order_by)
            parts.append(f"ORDER BY {order}")

        # Frame specification
        if node.frame_type:
            frame_parts = [node.frame_type.value]

            if node.frame_start and node.frame_end:
                start = node.frame_start.accept(self)
                end = node.frame_end.accept(self)
                frame_parts.append(f"BETWEEN {start} AND {end}")
            elif node.frame_start:
                frame_parts.append(node.frame_start.accept(self))

            parts.append(" ".join(frame_parts))

        return " ".join(parts)

    def visit_window_function(self, node: WindowFunction) -> str:
        func = node.function.accept(self)

        if isinstance(node.window_spec, str):
            return f"{func} OVER {node.window_spec}"
        elif node.window_spec:
            spec = node.window_spec.accept(self)
            return f"{func} OVER ({spec})"
        else:
            return f"{func} OVER ()"

    # -------------------------------------------------------------------------
    # Visitor Methods - Clauses
    # -------------------------------------------------------------------------

    def visit_select_item(self, node: SelectItem) -> str:
        expr = node.expression.accept(self)
        if node.alias:
            return f"{expr} AS {self._quote_identifier(node.alias)}"
        return expr

    def visit_from_clause(self, node: FromClause) -> str:
        return node.source.accept(self)

    def visit_join_clause(self, node: JoinClause) -> str:
        left = node.left.accept(self)
        right = node.right.accept(self)

        result = f"{left} {node.join_type.value} {right}"

        if node.condition:
            condition = node.condition.accept(self)
            result += f" ON {condition}"
        elif node.using_columns:
            cols = ", ".join(self._quote_identifier(c) for c in node.using_columns)
            result += f" USING ({cols})"

        return result

    def visit_where_clause(self, node: WhereClause) -> str:
        return node.condition.accept(self)

    def visit_group_by_clause(self, node: GroupByClause) -> str:
        exprs = ", ".join(e.accept(self) for e in node.expressions)

        if node.grouping_sets and self.config.supports_grouping_sets:
            sets = []
            for gs in node.grouping_sets:
                gs_exprs = ", ".join(e.accept(self) for e in gs)
                sets.append(f"({gs_exprs})")
            return f"GROUPING SETS ({', '.join(sets)})"

        result = exprs
        if node.with_rollup:
            result += " WITH ROLLUP"
        elif node.with_cube:
            result += " WITH CUBE"

        return result

    def visit_having_clause(self, node: HavingClause) -> str:
        return node.condition.accept(self)

    def visit_order_by_item(self, node: OrderByItem) -> str:
        expr = node.expression.accept(self)
        parts = [expr, node.order.value]

        if node.nulls and self.config.supports_nulls_ordering:
            parts.append(node.nulls.value)

        return " ".join(parts)

    def visit_order_by_clause(self, node: OrderByClause) -> str:
        return ", ".join(item.accept(self) for item in node.items)

    def visit_limit_clause(self, node: LimitClause) -> str:
        if isinstance(node.count, Expression):
            return node.count.accept(self)
        return str(node.count)

    def visit_offset_clause(self, node: OffsetClause) -> str:
        if isinstance(node.offset, Expression):
            return node.offset.accept(self)
        return str(node.offset)

    def visit_cte_clause(self, node: CTEClause) -> str:
        name = self._quote_identifier(node.name)

        if node.columns:
            cols = ", ".join(self._quote_identifier(c) for c in node.columns)
            name = f"{name}({cols})"

        query = node.query.accept(self)
        return f"{name} AS ({query})"

    # -------------------------------------------------------------------------
    # Visitor Methods - Statements
    # -------------------------------------------------------------------------

    def visit_select_statement(self, node: SelectStatement) -> str:
        parts = []

        # WITH clause (CTEs)
        if node.ctes:
            has_recursive = any(cte.recursive for cte in node.ctes)
            cte_keyword = "WITH RECURSIVE" if has_recursive else "WITH"
            cte_parts = [cte.accept(self) for cte in node.ctes]
            parts.append(f"{cte_keyword} {', '.join(cte_parts)}")

        # SELECT
        distinct = "DISTINCT " if node.distinct else ""
        select_items = ", ".join(
            item.accept(self) if isinstance(item, SelectItem) else item.accept(self)
            for item in node.select_items
        )

        # Handle TOP for SQL Server
        if self.config.supports_top and node.limit_clause and not node.offset_clause:
            limit = self.visit_limit_clause(node.limit_clause)
            parts.append(f"SELECT {distinct}TOP {limit} {select_items}")
        else:
            parts.append(f"SELECT {distinct}{select_items}")

        # FROM
        if node.from_clause:
            parts.append(f"FROM {node.from_clause.accept(self)}")

        # WHERE
        if node.where_clause:
            parts.append(f"WHERE {node.where_clause.accept(self)}")

        # GROUP BY
        if node.group_by_clause:
            parts.append(f"GROUP BY {node.group_by_clause.accept(self)}")

        # HAVING
        if node.having_clause:
            parts.append(f"HAVING {node.having_clause.accept(self)}")

        # ORDER BY
        if node.order_by_clause:
            parts.append(f"ORDER BY {node.order_by_clause.accept(self)}")

        # LIMIT/OFFSET or FETCH FIRST
        if node.limit_clause and not self.config.supports_top:
            if self.config.supports_fetch_first:
                limit = self.visit_limit_clause(node.limit_clause)
                if node.offset_clause:
                    offset = self.visit_offset_clause(node.offset_clause)
                    parts.append(f"OFFSET {offset} ROWS FETCH FIRST {limit} ROWS ONLY")
                else:
                    parts.append(f"FETCH FIRST {limit} ROWS ONLY")
            elif self.config.supports_limit_offset:
                limit = self.visit_limit_clause(node.limit_clause)
                parts.append(f"LIMIT {limit}")
                if node.offset_clause:
                    offset = self.visit_offset_clause(node.offset_clause)
                    parts.append(f"OFFSET {offset}")

        return "\n".join(parts)

    def visit_set_operation(self, node: SetOperationStatement) -> str:
        left = node.left.accept(self)
        right = node.right.accept(self)
        return f"({left})\n{node.operation.value}\n({right})"


# =============================================================================
# Dialect-Specific Generators
# =============================================================================


class PostgreSQLGenerator(BaseDialectGenerator):
    """SQL generator for PostgreSQL."""

    dialect = SQLDialect.POSTGRESQL

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "LENGTH": "LENGTH",
            "SUBSTR": "SUBSTRING",
            "NVL": "COALESCE",
            "IFNULL": "COALESCE",
            "NOW": "NOW",
            "GETDATE": "NOW",
        }

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        # Handle ILIKE (PostgreSQL native)
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.ILIKE:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"({left} ILIKE {right})"

        # Handle REGEXP
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.REGEXP:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"({left} ~ {right})"

        return super().visit_binary_expression(node)


class MySQLGenerator(BaseDialectGenerator):
    """SQL generator for MySQL."""

    dialect = SQLDialect.MYSQL

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "LENGTH": "CHAR_LENGTH",
            "NVL": "IFNULL",
            "COALESCE": "COALESCE",
            "NOW": "NOW",
            "CURRENT_TIMESTAMP": "NOW",
        }

    def visit_boolean_literal(self, node: BooleanLiteral) -> str:
        # MySQL uses TRUE/FALSE
        return "TRUE" if node.value else "FALSE"

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        # Handle ILIKE (convert to case-insensitive LIKE)
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.ILIKE:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"(LOWER({left}) LIKE LOWER({right}))"

        # Handle REGEXP
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.REGEXP:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"({left} REGEXP {right})"

        return super().visit_binary_expression(node)


class SQLiteGenerator(BaseDialectGenerator):
    """SQL generator for SQLite."""

    dialect = SQLDialect.SQLITE

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "NVL": "IFNULL",
            "COALESCE": "COALESCE",
            "NOW": "DATETIME",
            "LEN": "LENGTH",
        }

    def visit_boolean_literal(self, node: BooleanLiteral) -> str:
        # SQLite uses 1/0
        return "1" if node.value else "0"


class BigQueryGenerator(BaseDialectGenerator):
    """SQL generator for Google BigQuery."""

    dialect = SQLDialect.BIGQUERY

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "NVL": "IFNULL",
            "LENGTH": "LENGTH",
            "SUBSTR": "SUBSTR",
            "NOW": "CURRENT_TIMESTAMP",
        }

    def visit_array_literal(self, node: ArrayLiteral) -> str:
        elements = ", ".join(e.accept(self) for e in node.elements)
        return f"[{elements}]"

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        # Handle REGEXP
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.REGEXP:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"REGEXP_CONTAINS({left}, {right})"

        return super().visit_binary_expression(node)


class SnowflakeGenerator(BaseDialectGenerator):
    """SQL generator for Snowflake."""

    dialect = SQLDialect.SNOWFLAKE

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "LENGTH": "LENGTH",
            "SUBSTR": "SUBSTR",
            "NVL": "NVL",
            "NOW": "CURRENT_TIMESTAMP",
        }

    def _quote_identifier(self, name: str) -> str:
        # Snowflake preserves case with quotes
        quote = self.config.identifier_quote
        return f'{quote}{name}{quote}'

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        # Handle REGEXP
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.REGEXP:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"REGEXP_LIKE({left}, {right})"

        # Handle ILIKE (Snowflake native)
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.ILIKE:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"({left} ILIKE {right})"

        return super().visit_binary_expression(node)


class RedshiftGenerator(BaseDialectGenerator):
    """SQL generator for Amazon Redshift."""

    dialect = SQLDialect.REDSHIFT

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "LENGTH": "LEN",
            "SUBSTR": "SUBSTRING",
            "NVL": "NVL",
            "NOW": "GETDATE",
        }

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        # Handle REGEXP (Redshift uses SIMILAR TO or REGEXP_SUBSTR)
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.REGEXP:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"({left} ~ {right})"

        # Handle ILIKE (Redshift native)
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.ILIKE:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"({left} ILIKE {right})"

        return super().visit_binary_expression(node)


class DatabricksGenerator(BaseDialectGenerator):
    """SQL generator for Databricks (Spark SQL)."""

    dialect = SQLDialect.DATABRICKS

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "LENGTH": "LENGTH",
            "SUBSTR": "SUBSTRING",
            "NVL": "NVL",
            "IFNULL": "COALESCE",
            "NOW": "CURRENT_TIMESTAMP",
        }

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        # Handle REGEXP
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.REGEXP:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"({left} RLIKE {right})"

        return super().visit_binary_expression(node)


class OracleGenerator(BaseDialectGenerator):
    """SQL generator for Oracle Database."""

    dialect = SQLDialect.ORACLE

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "LENGTH": "LENGTH",
            "SUBSTR": "SUBSTR",
            "IFNULL": "NVL",
            "COALESCE": "COALESCE",
            "NOW": "SYSDATE",
            "CURRENT_TIMESTAMP": "SYSTIMESTAMP",
        }

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        # Handle REGEXP
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.REGEXP:
            left = node.left.accept(self)
            right = node.right.accept(self)
            return f"REGEXP_LIKE({left}, {right})"

        return super().visit_binary_expression(node)


class SQLServerGenerator(BaseDialectGenerator):
    """SQL generator for Microsoft SQL Server."""

    dialect = SQLDialect.SQLSERVER

    def __init__(self, config: DialectConfig | None = None) -> None:
        super().__init__(config)
        self._function_mappings = {
            "LENGTH": "LEN",
            "SUBSTR": "SUBSTRING",
            "NVL": "ISNULL",
            "IFNULL": "ISNULL",
            "NOW": "GETDATE",
        }

    def _quote_identifier(self, name: str) -> str:
        return f"[{name}]"

    def visit_boolean_literal(self, node: BooleanLiteral) -> str:
        return "1" if node.value else "0"

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        # Handle string concatenation
        if isinstance(node.operator, (ArithmeticOp, BinaryOp)):
            if node.operator in (ArithmeticOp.ADD, BinaryOp.ADD):
                # Check if this might be string concatenation
                pass

        # SQL Server doesn't have native REGEXP - would need CLR
        if isinstance(node.operator, ComparisonOp) and node.operator == ComparisonOp.REGEXP:
            raise NotImplementedError(
                "SQL Server does not support native regular expressions. "
                "Consider using LIKE patterns or CLR functions."
            )

        return super().visit_binary_expression(node)


# =============================================================================
# Generator Registry
# =============================================================================


_GENERATOR_REGISTRY: dict[SQLDialect, type[BaseDialectGenerator]] = {
    SQLDialect.POSTGRESQL: PostgreSQLGenerator,
    SQLDialect.MYSQL: MySQLGenerator,
    SQLDialect.SQLITE: SQLiteGenerator,
    SQLDialect.BIGQUERY: BigQueryGenerator,
    SQLDialect.SNOWFLAKE: SnowflakeGenerator,
    SQLDialect.REDSHIFT: RedshiftGenerator,
    SQLDialect.DATABRICKS: DatabricksGenerator,
    SQLDialect.ORACLE: OracleGenerator,
    SQLDialect.SQLSERVER: SQLServerGenerator,
    SQLDialect.GENERIC: BaseDialectGenerator,
}


def get_dialect_generator(
    dialect: SQLDialect,
    config: DialectConfig | None = None,
) -> BaseDialectGenerator:
    """Get a dialect generator instance.

    Args:
        dialect: SQL dialect.
        config: Optional dialect configuration.

    Returns:
        Dialect generator instance.
    """
    generator_class = _GENERATOR_REGISTRY.get(dialect, BaseDialectGenerator)
    return generator_class(config)


def register_dialect_generator(
    dialect: SQLDialect,
    generator_class: type[BaseDialectGenerator],
) -> None:
    """Register a custom dialect generator.

    Args:
        dialect: SQL dialect.
        generator_class: Generator class to register.
    """
    _GENERATOR_REGISTRY[dialect] = generator_class
