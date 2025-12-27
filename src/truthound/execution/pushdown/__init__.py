"""Advanced Query Pushdown Framework for SQL Execution.

This module provides a comprehensive query pushdown framework that enables
complex SQL operations to be executed server-side, minimizing data transfer
and leveraging database optimization capabilities.

Key Components:
    - Query AST: Abstract Syntax Tree for building complex queries
    - SQL Dialects: Dialect-specific code generators (PostgreSQL, MySQL, BigQuery, etc.)
    - Expression Builder: Fluent API for building filter expressions
    - Aggregation Support: GROUP BY, HAVING, Window Functions
    - Query Optimizer: Pushdown decision making and query optimization

Example:
    >>> from truthound.execution.pushdown import (
    ...     QueryBuilder,
    ...     SQLDialect,
    ...     col,
    ...     func,
    ... )
    >>>
    >>> # Build a complex query
    >>> query = (
    ...     QueryBuilder("users")
    ...     .select("department", func.count("*").alias("cnt"))
    ...     .where(col("age") > 18)
    ...     .group_by("department")
    ...     .having(func.count("*") > 10)
    ...     .order_by("cnt", descending=True)
    ...     .limit(100)
    ... )
    >>>
    >>> # Generate SQL for specific dialect
    >>> sql = query.to_sql(SQLDialect.POSTGRESQL)
"""

from truthound.execution.pushdown.ast import (
    # Base nodes
    SQLNode,
    Expression,
    Statement,
    # Literals
    Literal,
    NullLiteral,
    # Identifiers
    Identifier,
    Column,
    Table,
    Alias,
    # Operators
    BinaryOp,
    UnaryOp,
    ComparisonOp,
    LogicalOp,
    ArithmeticOp,
    # Expressions
    BinaryExpression,
    UnaryExpression,
    FunctionCall,
    CaseExpression,
    WhenClause,
    InExpression,
    BetweenExpression,
    ExistsExpression,
    SubqueryExpression,
    CastExpression,
    # Aggregation
    AggregateFunction,
    WindowFunction,
    WindowSpec,
    FrameBound,
    FrameType,
    # SELECT components
    SelectItem,
    FromClause,
    JoinClause,
    JoinType,
    WhereClause,
    GroupByClause,
    HavingClause,
    OrderByClause,
    OrderByItem,
    SortOrder,
    LimitClause,
    OffsetClause,
    # Statements
    SelectStatement,
    # Query types
    QueryType,
)

from truthound.execution.pushdown.dialects import (
    SQLDialect,
    DialectConfig,
    BaseDialectGenerator,
    PostgreSQLGenerator,
    MySQLGenerator,
    SQLiteGenerator,
    BigQueryGenerator,
    SnowflakeGenerator,
    RedshiftGenerator,
    DatabricksGenerator,
    OracleGenerator,
    SQLServerGenerator,
    get_dialect_generator,
    register_dialect_generator,
)

from truthound.execution.pushdown.builder import (
    QueryBuilder,
    ExpressionBuilder,
    col,
    literal,
    func,
    case,
    when,
    and_,
    or_,
    not_,
    exists,
    cast,
)

from truthound.execution.pushdown.optimizer import (
    QueryOptimizer,
    OptimizationRule,
    PushdownAnalyzer,
    PushdownDecision,
    PushdownCapability,
)

from truthound.execution.pushdown.executor import (
    PushdownExecutor,
    PushdownResult,
    ExecutionPlan,
    PlanNode,
)

__all__ = [
    # AST - Base nodes
    "SQLNode",
    "Expression",
    "Statement",
    # AST - Literals
    "Literal",
    "NullLiteral",
    # AST - Identifiers
    "Identifier",
    "Column",
    "Table",
    "Alias",
    # AST - Operators
    "BinaryOp",
    "UnaryOp",
    "ComparisonOp",
    "LogicalOp",
    "ArithmeticOp",
    # AST - Expressions
    "BinaryExpression",
    "UnaryExpression",
    "FunctionCall",
    "CaseExpression",
    "WhenClause",
    "InExpression",
    "BetweenExpression",
    "ExistsExpression",
    "SubqueryExpression",
    "CastExpression",
    # AST - Aggregation
    "AggregateFunction",
    "WindowFunction",
    "WindowSpec",
    "FrameBound",
    "FrameType",
    # AST - SELECT components
    "SelectItem",
    "FromClause",
    "JoinClause",
    "JoinType",
    "WhereClause",
    "GroupByClause",
    "HavingClause",
    "OrderByClause",
    "OrderByItem",
    "SortOrder",
    "LimitClause",
    "OffsetClause",
    # AST - Statements
    "SelectStatement",
    "QueryType",
    # Dialects
    "SQLDialect",
    "DialectConfig",
    "BaseDialectGenerator",
    "PostgreSQLGenerator",
    "MySQLGenerator",
    "SQLiteGenerator",
    "BigQueryGenerator",
    "SnowflakeGenerator",
    "RedshiftGenerator",
    "DatabricksGenerator",
    "OracleGenerator",
    "SQLServerGenerator",
    "get_dialect_generator",
    "register_dialect_generator",
    # Builder
    "QueryBuilder",
    "ExpressionBuilder",
    "col",
    "literal",
    "func",
    "case",
    "when",
    "and_",
    "or_",
    "not_",
    "exists",
    "cast",
    # Optimizer
    "QueryOptimizer",
    "OptimizationRule",
    "PushdownAnalyzer",
    "PushdownDecision",
    "PushdownCapability",
    # Executor
    "PushdownExecutor",
    "PushdownResult",
    "ExecutionPlan",
    "PlanNode",
]
