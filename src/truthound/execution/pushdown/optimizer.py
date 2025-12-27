"""Query Optimizer for SQL Pushdown.

This module provides query optimization capabilities including:
- Pushdown analysis: Determining which operations can be pushed to SQL
- Query optimization: Rewriting queries for better performance
- Cost estimation: Estimating the cost of different execution plans

Example:
    >>> from truthound.execution.pushdown import (
    ...     QueryOptimizer,
    ...     PushdownAnalyzer,
    ...     QueryBuilder,
    ... )
    >>>
    >>> query = QueryBuilder("users").select("*").where(col("age") > 18)
    >>> analyzer = PushdownAnalyzer()
    >>> decision = analyzer.analyze(query.build())
    >>> print(decision.can_pushdown)
    True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence, TypeVar

from truthound.execution.pushdown.ast import (
    # Base
    SQLNode,
    SQLVisitor,
    Expression,
    Statement,
    # Expressions
    BinaryExpression,
    UnaryExpression,
    FunctionCall,
    AggregateFunction,
    WindowFunction,
    CaseExpression,
    InExpression,
    BetweenExpression,
    ExistsExpression,
    SubqueryExpression,
    CastExpression,
    # Identifiers
    Column,
    Table,
    Star,
    Literal,
    NullLiteral,
    BooleanLiteral,
    ArrayLiteral,
    Identifier,
    Alias,
    # Operators
    ComparisonOp,
    LogicalOp,
    UnaryOp,
    JoinType,
    # Clauses
    SelectItem,
    FromClause,
    JoinClause,
    WhereClause,
    GroupByClause,
    HavingClause,
    OrderByClause,
    OrderByItem,
    LimitClause,
    OffsetClause,
    WindowSpec,
    FrameBound,
    WhenClause,
    CTEClause,
    # Statements
    SelectStatement,
    SetOperationStatement,
)
from truthound.execution.pushdown.dialects import SQLDialect, DialectConfig, DIALECT_CONFIGS


# =============================================================================
# Pushdown Capability
# =============================================================================


class PushdownCapability(Enum):
    """Capabilities that affect pushdown decisions."""

    # Basic capabilities
    BASIC_SELECT = auto()
    FILTER = auto()
    AGGREGATION = auto()
    GROUP_BY = auto()
    ORDER_BY = auto()
    LIMIT_OFFSET = auto()
    DISTINCT = auto()

    # Join capabilities
    INNER_JOIN = auto()
    LEFT_JOIN = auto()
    RIGHT_JOIN = auto()
    FULL_JOIN = auto()
    CROSS_JOIN = auto()

    # Advanced capabilities
    WINDOW_FUNCTIONS = auto()
    CTE = auto()
    SUBQUERY = auto()
    SET_OPERATIONS = auto()

    # Function capabilities
    STRING_FUNCTIONS = auto()
    DATE_FUNCTIONS = auto()
    MATH_FUNCTIONS = auto()
    JSON_FUNCTIONS = auto()
    ARRAY_FUNCTIONS = auto()
    REGEX_FUNCTIONS = auto()

    # Expression capabilities
    CASE_EXPRESSION = auto()
    CAST_EXPRESSION = auto()
    IN_EXPRESSION = auto()
    BETWEEN_EXPRESSION = auto()
    EXISTS_EXPRESSION = auto()

    # Grouping capabilities
    GROUPING_SETS = auto()
    ROLLUP = auto()
    CUBE = auto()


# Standard capability sets for common database types
STANDARD_SQL_CAPABILITIES: set[PushdownCapability] = {
    PushdownCapability.BASIC_SELECT,
    PushdownCapability.FILTER,
    PushdownCapability.AGGREGATION,
    PushdownCapability.GROUP_BY,
    PushdownCapability.ORDER_BY,
    PushdownCapability.LIMIT_OFFSET,
    PushdownCapability.DISTINCT,
    PushdownCapability.INNER_JOIN,
    PushdownCapability.LEFT_JOIN,
    PushdownCapability.CROSS_JOIN,
    PushdownCapability.STRING_FUNCTIONS,
    PushdownCapability.DATE_FUNCTIONS,
    PushdownCapability.MATH_FUNCTIONS,
    PushdownCapability.CASE_EXPRESSION,
    PushdownCapability.CAST_EXPRESSION,
    PushdownCapability.IN_EXPRESSION,
    PushdownCapability.BETWEEN_EXPRESSION,
}

FULL_SQL_CAPABILITIES: set[PushdownCapability] = STANDARD_SQL_CAPABILITIES | {
    PushdownCapability.RIGHT_JOIN,
    PushdownCapability.FULL_JOIN,
    PushdownCapability.WINDOW_FUNCTIONS,
    PushdownCapability.CTE,
    PushdownCapability.SUBQUERY,
    PushdownCapability.SET_OPERATIONS,
    PushdownCapability.JSON_FUNCTIONS,
    PushdownCapability.ARRAY_FUNCTIONS,
    PushdownCapability.REGEX_FUNCTIONS,
    PushdownCapability.EXISTS_EXPRESSION,
    PushdownCapability.GROUPING_SETS,
    PushdownCapability.ROLLUP,
    PushdownCapability.CUBE,
}

DIALECT_CAPABILITIES: dict[SQLDialect, set[PushdownCapability]] = {
    SQLDialect.POSTGRESQL: FULL_SQL_CAPABILITIES,
    SQLDialect.MYSQL: STANDARD_SQL_CAPABILITIES | {
        PushdownCapability.WINDOW_FUNCTIONS,
        PushdownCapability.CTE,
        PushdownCapability.SUBQUERY,
        PushdownCapability.JSON_FUNCTIONS,
        PushdownCapability.REGEX_FUNCTIONS,
        PushdownCapability.ROLLUP,
    },
    SQLDialect.SQLITE: STANDARD_SQL_CAPABILITIES | {
        PushdownCapability.WINDOW_FUNCTIONS,
        PushdownCapability.CTE,
        PushdownCapability.SUBQUERY,
    },
    SQLDialect.BIGQUERY: FULL_SQL_CAPABILITIES,
    SQLDialect.SNOWFLAKE: FULL_SQL_CAPABILITIES,
    SQLDialect.REDSHIFT: STANDARD_SQL_CAPABILITIES | {
        PushdownCapability.WINDOW_FUNCTIONS,
        PushdownCapability.CTE,
        PushdownCapability.SUBQUERY,
        PushdownCapability.SET_OPERATIONS,
        PushdownCapability.JSON_FUNCTIONS,
    },
    SQLDialect.DATABRICKS: FULL_SQL_CAPABILITIES,
    SQLDialect.ORACLE: FULL_SQL_CAPABILITIES,
    SQLDialect.SQLSERVER: FULL_SQL_CAPABILITIES - {PushdownCapability.REGEX_FUNCTIONS},
}


# =============================================================================
# Pushdown Decision
# =============================================================================


class PushdownReason(Enum):
    """Reasons for pushdown decisions."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED_FUNCTION = "unsupported_function"
    UNSUPPORTED_EXPRESSION = "unsupported_expression"
    UNSUPPORTED_JOIN = "unsupported_join"
    UNSUPPORTED_WINDOW = "unsupported_window"
    UNSUPPORTED_CTE = "unsupported_cte"
    UNSUPPORTED_SUBQUERY = "unsupported_subquery"
    MISSING_CAPABILITY = "missing_capability"
    PERFORMANCE_CONCERN = "performance_concern"
    DATA_SIZE_CONCERN = "data_size_concern"


@dataclass
class PushdownIssue:
    """An issue preventing or affecting pushdown.

    Attributes:
        node: The AST node with the issue.
        reason: Reason for the issue.
        message: Human-readable message.
        severity: Issue severity (error, warning, info).
        required_capability: Required capability if applicable.
    """

    node: SQLNode | None
    reason: PushdownReason
    message: str
    severity: str = "error"  # error, warning, info
    required_capability: PushdownCapability | None = None


@dataclass
class PushdownDecision:
    """Result of pushdown analysis.

    Attributes:
        can_pushdown: Whether the query can be fully pushed down.
        partial_pushdown: Whether partial pushdown is possible.
        issues: List of issues found during analysis.
        required_capabilities: Set of required capabilities.
        estimated_cost: Estimated cost (0-100, lower is better).
        recommendations: List of optimization recommendations.
    """

    can_pushdown: bool
    partial_pushdown: bool = False
    issues: list[PushdownIssue] = field(default_factory=list)
    required_capabilities: set[PushdownCapability] = field(default_factory=set)
    estimated_cost: float = 0.0
    recommendations: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.severity == "warning" for issue in self.issues)

    def add_issue(
        self,
        node: SQLNode | None,
        reason: PushdownReason,
        message: str,
        severity: str = "error",
        required_capability: PushdownCapability | None = None,
    ) -> None:
        """Add an issue to the decision."""
        self.issues.append(
            PushdownIssue(node, reason, message, severity, required_capability)
        )
        if severity == "error":
            self.can_pushdown = False

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(recommendation)


# =============================================================================
# Pushdown Analyzer
# =============================================================================


class PushdownAnalyzer(SQLVisitor):
    """Analyzes queries to determine pushdown feasibility.

    This analyzer traverses the AST to identify:
    - Required capabilities for the query
    - Potential issues preventing pushdown
    - Optimization opportunities

    Example:
        >>> analyzer = PushdownAnalyzer(SQLDialect.POSTGRESQL)
        >>> decision = analyzer.analyze(query)
        >>> if decision.can_pushdown:
        ...     print("Query can be fully pushed down")
    """

    def __init__(
        self,
        dialect: SQLDialect = SQLDialect.GENERIC,
        available_capabilities: set[PushdownCapability] | None = None,
    ) -> None:
        """Initialize analyzer.

        Args:
            dialect: SQL dialect to analyze for.
            available_capabilities: Override available capabilities.
        """
        self.dialect = dialect
        self.available_capabilities = (
            available_capabilities
            if available_capabilities is not None
            else DIALECT_CAPABILITIES.get(dialect, STANDARD_SQL_CAPABILITIES)
        )
        self._decision: PushdownDecision | None = None
        self._function_registry: dict[str, set[SQLDialect]] = self._build_function_registry()

    def _build_function_registry(self) -> dict[str, set[SQLDialect]]:
        """Build registry of supported functions per dialect."""
        # Common functions supported by all dialects
        common_functions = {
            "COUNT", "SUM", "AVG", "MIN", "MAX",
            "COALESCE", "NULLIF",
            "UPPER", "LOWER", "LENGTH", "TRIM",
            "ROUND", "ABS", "FLOOR", "CEIL",
            "CAST",
        }

        registry: dict[str, set[SQLDialect]] = {}

        # Add common functions to all dialects
        for func in common_functions:
            registry[func] = set(SQLDialect)

        # PostgreSQL-specific
        pg_functions = {
            "ARRAY_AGG", "STRING_AGG", "JSONB_BUILD_OBJECT",
            "REGEXP_REPLACE", "REGEXP_MATCHES",
        }
        for func in pg_functions:
            registry[func] = {SQLDialect.POSTGRESQL}

        # BigQuery-specific
        bq_functions = {
            "SAFE_DIVIDE", "IFNULL", "REGEXP_CONTAINS",
            "STRUCT", "ARRAY", "UNNEST",
        }
        for func in bq_functions:
            registry[func] = {SQLDialect.BIGQUERY}

        return registry

    def analyze(self, statement: Statement) -> PushdownDecision:
        """Analyze a statement for pushdown feasibility.

        Args:
            statement: Statement to analyze.

        Returns:
            PushdownDecision with analysis results.
        """
        self._decision = PushdownDecision(can_pushdown=True)
        statement.accept(self)

        # Check if all required capabilities are available
        missing = self._decision.required_capabilities - self.available_capabilities
        if missing:
            for cap in missing:
                self._decision.add_issue(
                    None,
                    PushdownReason.MISSING_CAPABILITY,
                    f"Missing capability: {cap.name}",
                    required_capability=cap,
                )

        return self._decision

    def _require_capability(self, capability: PushdownCapability) -> None:
        """Mark a capability as required."""
        if self._decision:
            self._decision.required_capabilities.add(capability)

    def _check_function(self, func_name: str, node: SQLNode) -> None:
        """Check if a function is supported."""
        func_upper = func_name.upper()
        if func_upper in self._function_registry:
            supported_dialects = self._function_registry[func_upper]
            if self.dialect not in supported_dialects and SQLDialect not in supported_dialects:
                if self._decision:
                    self._decision.add_issue(
                        node,
                        PushdownReason.UNSUPPORTED_FUNCTION,
                        f"Function {func_name} may not be supported in {self.dialect.value}",
                        severity="warning",
                    )

    # -------------------------------------------------------------------------
    # Visitor Methods
    # -------------------------------------------------------------------------

    def visit_literal(self, node: Literal) -> Any:
        return None

    def visit_null_literal(self, node: NullLiteral) -> Any:
        return None

    def visit_boolean_literal(self, node: BooleanLiteral) -> Any:
        return None

    def visit_array_literal(self, node: ArrayLiteral) -> Any:
        self._require_capability(PushdownCapability.ARRAY_FUNCTIONS)
        for elem in node.elements:
            elem.accept(self)
        return None

    def visit_identifier(self, node: Identifier) -> Any:
        return None

    def visit_column(self, node: Column) -> Any:
        return None

    def visit_table(self, node: Table) -> Any:
        return None

    def visit_alias(self, node: Alias) -> Any:
        node.expression.accept(self)
        return None

    def visit_star(self, node: Star) -> Any:
        return None

    def visit_binary_expression(self, node: BinaryExpression) -> Any:
        node.left.accept(self)
        node.right.accept(self)

        # Check for regex operations
        if isinstance(node.operator, ComparisonOp):
            if node.operator in (ComparisonOp.REGEXP, ComparisonOp.SIMILAR_TO):
                self._require_capability(PushdownCapability.REGEX_FUNCTIONS)

        return None

    def visit_unary_expression(self, node: UnaryExpression) -> Any:
        node.operand.accept(self)
        return None

    def visit_in_expression(self, node: InExpression) -> Any:
        self._require_capability(PushdownCapability.IN_EXPRESSION)
        node.expression.accept(self)
        if isinstance(node.values, SelectStatement):
            self._require_capability(PushdownCapability.SUBQUERY)
            node.values.accept(self)
        else:
            for v in node.values:
                v.accept(self)
        return None

    def visit_between_expression(self, node: BetweenExpression) -> Any:
        self._require_capability(PushdownCapability.BETWEEN_EXPRESSION)
        node.expression.accept(self)
        node.low.accept(self)
        node.high.accept(self)
        return None

    def visit_exists_expression(self, node: ExistsExpression) -> Any:
        self._require_capability(PushdownCapability.EXISTS_EXPRESSION)
        self._require_capability(PushdownCapability.SUBQUERY)
        node.subquery.accept(self)
        return None

    def visit_subquery_expression(self, node: SubqueryExpression) -> Any:
        self._require_capability(PushdownCapability.SUBQUERY)
        node.subquery.accept(self)
        return None

    def visit_cast_expression(self, node: CastExpression) -> Any:
        self._require_capability(PushdownCapability.CAST_EXPRESSION)
        node.expression.accept(self)
        return None

    def visit_when_clause(self, node: WhenClause) -> Any:
        node.condition.accept(self)
        node.result.accept(self)
        return None

    def visit_case_expression(self, node: CaseExpression) -> Any:
        self._require_capability(PushdownCapability.CASE_EXPRESSION)
        if node.operand:
            node.operand.accept(self)
        for when in node.when_clauses:
            when.accept(self)
        if node.else_result:
            node.else_result.accept(self)
        return None

    def visit_function_call(self, node: FunctionCall) -> Any:
        self._check_function(node.name, node)
        for arg in node.arguments:
            arg.accept(self)
        if node.filter_clause:
            node.filter_clause.accept(self)
        return None

    def visit_aggregate_function(self, node: AggregateFunction) -> Any:
        self._require_capability(PushdownCapability.AGGREGATION)
        self._check_function(node.name, node)
        if node.argument:
            node.argument.accept(self)
        if node.filter_clause:
            node.filter_clause.accept(self)
        if node.order_by:
            for item in node.order_by:
                item.accept(self)
        return None

    def visit_frame_bound(self, node: FrameBound) -> Any:
        return None

    def visit_window_spec(self, node: WindowSpec) -> Any:
        if node.partition_by:
            for expr in node.partition_by:
                expr.accept(self)
        if node.order_by:
            for item in node.order_by:
                item.accept(self)
        if node.frame_start:
            node.frame_start.accept(self)
        if node.frame_end:
            node.frame_end.accept(self)
        return None

    def visit_window_function(self, node: WindowFunction) -> Any:
        self._require_capability(PushdownCapability.WINDOW_FUNCTIONS)
        node.function.accept(self)
        if isinstance(node.window_spec, WindowSpec):
            node.window_spec.accept(self)
        return None

    def visit_select_item(self, node: SelectItem) -> Any:
        node.expression.accept(self)
        return None

    def visit_from_clause(self, node: FromClause) -> Any:
        node.source.accept(self)
        return None

    def visit_join_clause(self, node: JoinClause) -> Any:
        node.left.accept(self)
        node.right.accept(self)

        # Check join type capabilities
        join_capability_map = {
            JoinType.INNER: PushdownCapability.INNER_JOIN,
            JoinType.LEFT: PushdownCapability.LEFT_JOIN,
            JoinType.LEFT_OUTER: PushdownCapability.LEFT_JOIN,
            JoinType.RIGHT: PushdownCapability.RIGHT_JOIN,
            JoinType.RIGHT_OUTER: PushdownCapability.RIGHT_JOIN,
            JoinType.FULL: PushdownCapability.FULL_JOIN,
            JoinType.FULL_OUTER: PushdownCapability.FULL_JOIN,
            JoinType.CROSS: PushdownCapability.CROSS_JOIN,
        }

        capability = join_capability_map.get(node.join_type)
        if capability:
            self._require_capability(capability)

        if node.condition:
            node.condition.accept(self)

        return None

    def visit_where_clause(self, node: WhereClause) -> Any:
        self._require_capability(PushdownCapability.FILTER)
        node.condition.accept(self)
        return None

    def visit_group_by_clause(self, node: GroupByClause) -> Any:
        self._require_capability(PushdownCapability.GROUP_BY)

        for expr in node.expressions:
            expr.accept(self)

        if node.with_rollup:
            self._require_capability(PushdownCapability.ROLLUP)
        if node.with_cube:
            self._require_capability(PushdownCapability.CUBE)
        if node.grouping_sets:
            self._require_capability(PushdownCapability.GROUPING_SETS)

        return None

    def visit_having_clause(self, node: HavingClause) -> Any:
        self._require_capability(PushdownCapability.GROUP_BY)
        node.condition.accept(self)
        return None

    def visit_order_by_item(self, node: OrderByItem) -> Any:
        node.expression.accept(self)
        return None

    def visit_order_by_clause(self, node: OrderByClause) -> Any:
        self._require_capability(PushdownCapability.ORDER_BY)
        for item in node.items:
            item.accept(self)
        return None

    def visit_limit_clause(self, node: LimitClause) -> Any:
        self._require_capability(PushdownCapability.LIMIT_OFFSET)
        if isinstance(node.count, Expression):
            node.count.accept(self)
        return None

    def visit_offset_clause(self, node: OffsetClause) -> Any:
        self._require_capability(PushdownCapability.LIMIT_OFFSET)
        if isinstance(node.offset, Expression):
            node.offset.accept(self)
        return None

    def visit_cte_clause(self, node: CTEClause) -> Any:
        self._require_capability(PushdownCapability.CTE)
        node.query.accept(self)
        return None

    def visit_select_statement(self, node: SelectStatement) -> Any:
        self._require_capability(PushdownCapability.BASIC_SELECT)

        if node.distinct:
            self._require_capability(PushdownCapability.DISTINCT)

        if node.ctes:
            for cte in node.ctes:
                cte.accept(self)

        for item in node.select_items:
            if isinstance(item, SelectItem):
                item.accept(self)
            else:
                item.accept(self)

        if node.from_clause:
            node.from_clause.accept(self)
        if node.where_clause:
            node.where_clause.accept(self)
        if node.group_by_clause:
            node.group_by_clause.accept(self)
        if node.having_clause:
            node.having_clause.accept(self)
        if node.order_by_clause:
            node.order_by_clause.accept(self)
        if node.limit_clause:
            node.limit_clause.accept(self)
        if node.offset_clause:
            node.offset_clause.accept(self)

        return None

    def visit_set_operation(self, node: SetOperationStatement) -> Any:
        self._require_capability(PushdownCapability.SET_OPERATIONS)
        node.left.accept(self)
        node.right.accept(self)
        return None


# =============================================================================
# Optimization Rules
# =============================================================================


class OptimizationRule(ABC):
    """Base class for query optimization rules.

    Optimization rules transform queries to improve performance.
    Each rule checks if it applies and transforms the query.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name."""
        pass

    @property
    def priority(self) -> int:
        """Rule priority (higher runs first)."""
        return 0

    @abstractmethod
    def applies(self, node: SQLNode) -> bool:
        """Check if this rule applies to the node.

        Args:
            node: Node to check.

        Returns:
            True if rule applies.
        """
        pass

    @abstractmethod
    def transform(self, node: SQLNode) -> SQLNode:
        """Transform the node.

        Args:
            node: Node to transform.

        Returns:
            Transformed node.
        """
        pass


class PredicatePushdownRule(OptimizationRule):
    """Push predicates closer to data sources.

    This rule attempts to move WHERE conditions as early as possible
    in the query execution, potentially pushing them into subqueries.
    """

    @property
    def name(self) -> str:
        return "predicate_pushdown"

    @property
    def priority(self) -> int:
        return 100

    def applies(self, node: SQLNode) -> bool:
        # Check if there's a WHERE clause that can be pushed down
        if isinstance(node, SelectStatement):
            return (
                node.where_clause is not None
                and node.from_clause is not None
                and isinstance(node.from_clause.source, (JoinClause, SelectStatement))
            )
        return False

    def transform(self, node: SQLNode) -> SQLNode:
        # For now, return unchanged - actual implementation would
        # analyze predicates and push them into subqueries
        return node


class ProjectionPushdownRule(OptimizationRule):
    """Push projections closer to data sources.

    This rule reduces the number of columns fetched by pushing
    SELECT column lists into subqueries.
    """

    @property
    def name(self) -> str:
        return "projection_pushdown"

    @property
    def priority(self) -> int:
        return 90

    def applies(self, node: SQLNode) -> bool:
        if isinstance(node, SelectStatement):
            # Check if we're selecting specific columns from a subquery
            if (
                node.from_clause is not None
                and isinstance(node.from_clause.source, SelectStatement)
            ):
                # Check if subquery has SELECT *
                subquery = node.from_clause.source
                return any(
                    isinstance(item, Star) or (isinstance(item, SelectItem) and isinstance(item.expression, Star))
                    for item in subquery.select_items
                )
        return False

    def transform(self, node: SQLNode) -> SQLNode:
        return node


class ConstantFoldingRule(OptimizationRule):
    """Fold constant expressions at compile time.

    This rule evaluates constant expressions like `1 + 1` to `2`.
    """

    @property
    def name(self) -> str:
        return "constant_folding"

    @property
    def priority(self) -> int:
        return 50

    def applies(self, node: SQLNode) -> bool:
        if isinstance(node, BinaryExpression):
            return (
                isinstance(node.left, Literal)
                and isinstance(node.right, Literal)
            )
        return False

    def transform(self, node: SQLNode) -> SQLNode:
        if isinstance(node, BinaryExpression):
            if isinstance(node.left, Literal) and isinstance(node.right, Literal):
                # Could evaluate the expression here
                pass
        return node


# =============================================================================
# Query Optimizer
# =============================================================================


class QueryOptimizer:
    """Optimizes SQL queries for better performance.

    The optimizer applies a series of transformation rules to
    rewrite queries into more efficient forms.

    Example:
        >>> optimizer = QueryOptimizer()
        >>> optimized = optimizer.optimize(query)
    """

    def __init__(
        self,
        rules: Sequence[OptimizationRule] | None = None,
        dialect: SQLDialect = SQLDialect.GENERIC,
    ) -> None:
        """Initialize optimizer.

        Args:
            rules: Custom optimization rules. If None, uses default rules.
            dialect: SQL dialect for dialect-specific optimizations.
        """
        self.dialect = dialect
        self._rules = list(rules) if rules else self._default_rules()
        # Sort by priority (descending)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def _default_rules(self) -> list[OptimizationRule]:
        """Get default optimization rules."""
        return [
            PredicatePushdownRule(),
            ProjectionPushdownRule(),
            ConstantFoldingRule(),
        ]

    def add_rule(self, rule: OptimizationRule) -> None:
        """Add an optimization rule.

        Args:
            rule: Rule to add.
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def optimize(
        self,
        statement: Statement,
        max_iterations: int = 10,
    ) -> Statement:
        """Optimize a statement.

        Args:
            statement: Statement to optimize.
            max_iterations: Maximum optimization iterations.

        Returns:
            Optimized statement.
        """
        current = statement
        for _ in range(max_iterations):
            changed = False
            for rule in self._rules:
                if rule.applies(current):
                    new_statement = rule.transform(current)
                    if new_statement is not current:
                        current = new_statement
                        changed = True
            if not changed:
                break
        return current

    def analyze_and_optimize(
        self,
        statement: Statement,
    ) -> tuple[Statement, PushdownDecision]:
        """Analyze and optimize a statement.

        Args:
            statement: Statement to process.

        Returns:
            Tuple of (optimized statement, pushdown decision).
        """
        # First analyze
        analyzer = PushdownAnalyzer(self.dialect)
        decision = analyzer.analyze(statement)

        # Then optimize if pushdown is possible
        if decision.can_pushdown:
            optimized = self.optimize(statement)
            return optimized, decision

        return statement, decision


# =============================================================================
# Cost Estimator
# =============================================================================


@dataclass
class CostEstimate:
    """Estimated cost of a query.

    Attributes:
        rows: Estimated number of rows.
        cpu_cost: Estimated CPU cost (arbitrary units).
        io_cost: Estimated I/O cost (arbitrary units).
        network_cost: Estimated network cost (arbitrary units).
        total_cost: Total estimated cost.
        confidence: Confidence level (0-1).
    """

    rows: int = 0
    cpu_cost: float = 0.0
    io_cost: float = 0.0
    network_cost: float = 0.0
    total_cost: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.total_cost == 0.0:
            self.total_cost = self.cpu_cost + self.io_cost + self.network_cost


class CostEstimator:
    """Estimates the cost of query execution.

    This is a simple heuristic-based cost estimator that provides
    rough cost estimates for query planning purposes.
    """

    def __init__(
        self,
        default_table_rows: int = 10000,
        default_selectivity: float = 0.1,
    ) -> None:
        """Initialize cost estimator.

        Args:
            default_table_rows: Default estimated rows for unknown tables.
            default_selectivity: Default filter selectivity.
        """
        self.default_table_rows = default_table_rows
        self.default_selectivity = default_selectivity
        self._table_stats: dict[str, int] = {}

    def set_table_stats(self, table: str, rows: int) -> None:
        """Set known table statistics.

        Args:
            table: Table name.
            rows: Number of rows.
        """
        self._table_stats[table] = rows

    def estimate(self, statement: Statement) -> CostEstimate:
        """Estimate the cost of a statement.

        Args:
            statement: Statement to estimate.

        Returns:
            CostEstimate.
        """
        if isinstance(statement, SelectStatement):
            return self._estimate_select(statement)
        return CostEstimate(confidence=0.0)

    def _estimate_select(self, statement: SelectStatement) -> CostEstimate:
        """Estimate cost of SELECT statement."""
        # Start with base row estimate
        rows = self.default_table_rows

        # Try to get actual table size
        if statement.from_clause:
            source = statement.from_clause.source
            if isinstance(source, Table):
                rows = self._table_stats.get(source.name, self.default_table_rows)

        # Apply selectivity for WHERE clause
        if statement.where_clause:
            rows = int(rows * self.default_selectivity)

        # Estimate costs
        cpu_cost = rows * 0.01  # Simple scan cost
        io_cost = rows * 0.001  # I/O per row

        # Aggregation increases CPU cost
        if statement.group_by_clause:
            cpu_cost *= 2

        # Sorting increases CPU cost
        if statement.order_by_clause:
            cpu_cost += rows * 0.05 * (rows > 0 and len(bin(rows)) or 1)  # O(n log n)

        # Window functions are expensive
        has_windows = any(
            isinstance(item, SelectItem) and isinstance(item.expression, WindowFunction)
            for item in statement.select_items
        )
        if has_windows:
            cpu_cost *= 3

        # Apply LIMIT
        if statement.limit_clause:
            if isinstance(statement.limit_clause.count, int):
                rows = min(rows, statement.limit_clause.count)

        return CostEstimate(
            rows=rows,
            cpu_cost=cpu_cost,
            io_cost=io_cost,
            network_cost=rows * 0.0001,
            confidence=0.5,  # Low confidence for heuristic estimates
        )
