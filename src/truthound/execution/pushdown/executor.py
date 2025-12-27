"""Pushdown Executor for SQL Execution.

This module provides the execution infrastructure for running
pushdown queries against SQL databases.

Example:
    >>> from truthound.execution.pushdown import (
    ...     PushdownExecutor,
    ...     QueryBuilder,
    ...     SQLDialect,
    ... )
    >>>
    >>> executor = PushdownExecutor(datasource, SQLDialect.POSTGRESQL)
    >>> query = QueryBuilder("users").select("*").where(col("age") > 18)
    >>> result = executor.execute(query)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

from truthound.execution.pushdown.ast import (
    SQLNode,
    Statement,
    SelectStatement,
    SetOperationStatement,
)
from truthound.execution.pushdown.dialects import (
    SQLDialect,
    get_dialect_generator,
)
from truthound.execution.pushdown.optimizer import (
    QueryOptimizer,
    PushdownAnalyzer,
    PushdownDecision,
    CostEstimator,
    CostEstimate,
)

if TYPE_CHECKING:
    from truthound.datasources.sql.base import BaseSQLDataSource


# =============================================================================
# Execution Plan
# =============================================================================


class PlanNodeType(Enum):
    """Types of execution plan nodes."""

    # Data access
    TABLE_SCAN = auto()
    INDEX_SCAN = auto()
    INDEX_SEEK = auto()

    # Operations
    FILTER = auto()
    PROJECT = auto()
    AGGREGATE = auto()
    SORT = auto()
    LIMIT = auto()

    # Joins
    NESTED_LOOP_JOIN = auto()
    HASH_JOIN = auto()
    MERGE_JOIN = auto()

    # Set operations
    UNION = auto()
    INTERSECT = auto()
    EXCEPT = auto()

    # Window
    WINDOW = auto()

    # Other
    SUBQUERY = auto()
    CTE = auto()
    MATERIALIZE = auto()


@dataclass
class PlanNode:
    """A node in the execution plan.

    Attributes:
        node_type: Type of plan node.
        description: Human-readable description.
        estimated_rows: Estimated output rows.
        estimated_cost: Estimated cost.
        children: Child plan nodes.
        properties: Additional properties.
    """

    node_type: PlanNodeType
    description: str
    estimated_rows: int = 0
    estimated_cost: float = 0.0
    children: list["PlanNode"] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.node_type.name,
            "description": self.description,
            "estimated_rows": self.estimated_rows,
            "estimated_cost": self.estimated_cost,
            "properties": self.properties,
            "children": [c.to_dict() for c in self.children],
        }

    def __str__(self) -> str:
        """String representation."""
        lines = [f"{self.node_type.name}: {self.description}"]
        lines.append(f"  Rows: {self.estimated_rows}, Cost: {self.estimated_cost:.2f}")
        for key, value in self.properties.items():
            lines.append(f"  {key}: {value}")
        for child in self.children:
            child_lines = str(child).split("\n")
            lines.extend(f"  {line}" for line in child_lines)
        return "\n".join(lines)


@dataclass
class ExecutionPlan:
    """Complete execution plan for a query.

    Attributes:
        root: Root plan node.
        sql: Generated SQL string.
        dialect: SQL dialect used.
        total_cost: Total estimated cost.
        pushdown_decision: Pushdown analysis result.
        created_at: Plan creation timestamp.
    """

    root: PlanNode
    sql: str
    dialect: SQLDialect
    total_cost: float = 0.0
    pushdown_decision: PushdownDecision | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sql": self.sql,
            "dialect": self.dialect.value,
            "total_cost": self.total_cost,
            "created_at": self.created_at.isoformat(),
            "plan": self.root.to_dict(),
        }

    def explain(self, verbose: bool = False) -> str:
        """Generate EXPLAIN-like output.

        Args:
            verbose: Include additional details.

        Returns:
            Formatted plan string.
        """
        lines = [
            "=" * 60,
            "EXECUTION PLAN",
            "=" * 60,
            f"Dialect: {self.dialect.value}",
            f"Total Cost: {self.total_cost:.2f}",
            "",
            "PLAN:",
            "-" * 60,
            str(self.root),
            "",
            "SQL:",
            "-" * 60,
            self.sql,
            "=" * 60,
        ]

        if verbose and self.pushdown_decision:
            lines.extend([
                "",
                "PUSHDOWN ANALYSIS:",
                "-" * 60,
                f"Can Pushdown: {self.pushdown_decision.can_pushdown}",
                f"Required Capabilities: {[c.name for c in self.pushdown_decision.required_capabilities]}",
            ])
            if self.pushdown_decision.issues:
                lines.append("Issues:")
                for issue in self.pushdown_decision.issues:
                    lines.append(f"  - [{issue.severity}] {issue.message}")
            if self.pushdown_decision.recommendations:
                lines.append("Recommendations:")
                for rec in self.pushdown_decision.recommendations:
                    lines.append(f"  - {rec}")

        return "\n".join(lines)


# =============================================================================
# Execution Result
# =============================================================================


@dataclass
class PushdownResult:
    """Result of a pushdown query execution.

    Attributes:
        data: Query result data.
        columns: Column names.
        row_count: Number of rows returned.
        execution_time_ms: Execution time in milliseconds.
        plan: Execution plan used.
        was_pushed_down: Whether the query was pushed down.
        bytes_scanned: Bytes scanned (if available).
        metadata: Additional metadata.
    """

    data: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    execution_time_ms: float = 0.0
    plan: ExecutionPlan | None = None
    was_pushed_down: bool = True
    bytes_scanned: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "columns": self.columns,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "was_pushed_down": self.was_pushed_down,
            "bytes_scanned": self.bytes_scanned,
            "metadata": self.metadata,
        }

    def to_polars(self) -> Any:
        """Convert to Polars DataFrame.

        Returns:
            Polars DataFrame.
        """
        import polars as pl
        return pl.DataFrame(self.data)

    def to_pandas(self) -> Any:
        """Convert to Pandas DataFrame.

        Returns:
            Pandas DataFrame.
        """
        import pandas as pd
        return pd.DataFrame(self.data)

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        """Iterate over rows.

        Yields:
            Row dictionaries.
        """
        yield from self.data

    def first(self) -> dict[str, Any] | None:
        """Get the first row.

        Returns:
            First row or None.
        """
        return self.data[0] if self.data else None

    def scalar(self) -> Any:
        """Get a single scalar value (first column of first row).

        Returns:
            Scalar value.
        """
        if self.data and self.columns:
            return self.data[0].get(self.columns[0])
        return None


# =============================================================================
# Plan Builder
# =============================================================================


class ExecutionPlanBuilder:
    """Builds execution plans from AST nodes.

    This class analyzes SQL AST nodes and builds execution plans
    that describe how the query will be executed.
    """

    def __init__(
        self,
        dialect: SQLDialect = SQLDialect.GENERIC,
        cost_estimator: CostEstimator | None = None,
    ) -> None:
        """Initialize plan builder.

        Args:
            dialect: SQL dialect.
            cost_estimator: Cost estimator instance.
        """
        self.dialect = dialect
        self.cost_estimator = cost_estimator or CostEstimator()
        self.generator = get_dialect_generator(dialect)
        self.analyzer = PushdownAnalyzer(dialect)

    def build_plan(self, statement: Statement) -> ExecutionPlan:
        """Build an execution plan for a statement.

        Args:
            statement: Statement to plan.

        Returns:
            ExecutionPlan.
        """
        # Analyze for pushdown
        pushdown_decision = self.analyzer.analyze(statement)

        # Generate SQL
        sql = self.generator.generate(statement)

        # Build plan tree
        root = self._build_plan_node(statement)

        # Estimate total cost
        cost_estimate = self.cost_estimator.estimate(statement)

        return ExecutionPlan(
            root=root,
            sql=sql,
            dialect=self.dialect,
            total_cost=cost_estimate.total_cost,
            pushdown_decision=pushdown_decision,
        )

    def _build_plan_node(self, node: SQLNode) -> PlanNode:
        """Build a plan node from an AST node."""
        if isinstance(node, SelectStatement):
            return self._build_select_plan(node)
        elif isinstance(node, SetOperationStatement):
            return self._build_set_operation_plan(node)
        else:
            return PlanNode(
                node_type=PlanNodeType.TABLE_SCAN,
                description="Unknown operation",
            )

    def _build_select_plan(self, stmt: SelectStatement) -> PlanNode:
        """Build plan for SELECT statement."""
        # Start with data source
        children: list[PlanNode] = []

        if stmt.from_clause:
            from_node = self._build_from_plan(stmt.from_clause.source)
            children.append(from_node)
            rows = from_node.estimated_rows
        else:
            rows = 1

        # Filter (WHERE)
        if stmt.where_clause:
            filter_node = PlanNode(
                node_type=PlanNodeType.FILTER,
                description="Filter rows",
                estimated_rows=int(rows * 0.1),  # Assume 10% selectivity
                properties={"condition": "WHERE clause"},
            )
            if children:
                filter_node.children = children
                children = [filter_node]
            rows = filter_node.estimated_rows

        # Aggregate (GROUP BY)
        if stmt.group_by_clause:
            agg_node = PlanNode(
                node_type=PlanNodeType.AGGREGATE,
                description="Aggregate rows",
                estimated_rows=max(1, rows // 10),  # Reduce rows
                properties={"type": "GROUP BY"},
            )
            if children:
                agg_node.children = children
                children = [agg_node]
            rows = agg_node.estimated_rows

        # Sort (ORDER BY)
        if stmt.order_by_clause:
            sort_node = PlanNode(
                node_type=PlanNodeType.SORT,
                description="Sort rows",
                estimated_rows=rows,
            )
            if children:
                sort_node.children = children
                children = [sort_node]

        # Limit
        if stmt.limit_clause:
            limit = (
                stmt.limit_clause.count
                if isinstance(stmt.limit_clause.count, int)
                else 100
            )
            limit_node = PlanNode(
                node_type=PlanNodeType.LIMIT,
                description=f"Limit to {limit} rows",
                estimated_rows=min(rows, limit),
            )
            if children:
                limit_node.children = children
                children = [limit_node]
            rows = limit_node.estimated_rows

        # Project (SELECT columns)
        project_node = PlanNode(
            node_type=PlanNodeType.PROJECT,
            description="Project columns",
            estimated_rows=rows,
        )
        if children:
            project_node.children = children

        return project_node

    def _build_from_plan(self, source: Any) -> PlanNode:
        """Build plan for FROM clause source."""
        from truthound.execution.pushdown.ast import Table, JoinClause

        if isinstance(source, Table):
            return PlanNode(
                node_type=PlanNodeType.TABLE_SCAN,
                description=f"Scan table: {source.name}",
                estimated_rows=self.cost_estimator.default_table_rows,
                properties={"table": source.name},
            )
        elif isinstance(source, JoinClause):
            return self._build_join_plan(source)
        elif isinstance(source, SelectStatement):
            return PlanNode(
                node_type=PlanNodeType.SUBQUERY,
                description="Subquery",
                children=[self._build_select_plan(source)],
            )
        else:
            return PlanNode(
                node_type=PlanNodeType.TABLE_SCAN,
                description="Unknown source",
            )

    def _build_join_plan(self, join: Any) -> PlanNode:
        """Build plan for JOIN clause."""
        from truthound.execution.pushdown.ast import JoinClause, JoinType

        left_plan = self._build_from_plan(join.left)
        right_plan = self._build_from_plan(join.right)

        # Estimate join rows
        left_rows = left_plan.estimated_rows
        right_rows = right_plan.estimated_rows

        if join.join_type in (JoinType.INNER,):
            estimated_rows = min(left_rows, right_rows)
        elif join.join_type in (JoinType.LEFT, JoinType.LEFT_OUTER):
            estimated_rows = left_rows
        elif join.join_type in (JoinType.RIGHT, JoinType.RIGHT_OUTER):
            estimated_rows = right_rows
        elif join.join_type in (JoinType.FULL, JoinType.FULL_OUTER):
            estimated_rows = left_rows + right_rows
        elif join.join_type == JoinType.CROSS:
            estimated_rows = left_rows * right_rows
        else:
            estimated_rows = max(left_rows, right_rows)

        return PlanNode(
            node_type=PlanNodeType.HASH_JOIN,
            description=f"{join.join_type.value}",
            estimated_rows=estimated_rows,
            children=[left_plan, right_plan],
        )

    def _build_set_operation_plan(self, stmt: SetOperationStatement) -> PlanNode:
        """Build plan for set operation."""
        left_plan = self._build_plan_node(stmt.left)
        right_plan = self._build_plan_node(stmt.right)

        node_type_map = {
            "UNION": PlanNodeType.UNION,
            "UNION ALL": PlanNodeType.UNION,
            "INTERSECT": PlanNodeType.INTERSECT,
            "INTERSECT ALL": PlanNodeType.INTERSECT,
            "EXCEPT": PlanNodeType.EXCEPT,
            "EXCEPT ALL": PlanNodeType.EXCEPT,
        }

        return PlanNode(
            node_type=node_type_map.get(stmt.operation.value, PlanNodeType.UNION),
            description=stmt.operation.value,
            estimated_rows=left_plan.estimated_rows + right_plan.estimated_rows,
            children=[left_plan, right_plan],
        )


# =============================================================================
# Pushdown Executor
# =============================================================================


class PushdownExecutor:
    """Executes pushdown queries against SQL databases.

    This executor handles:
    - Query planning and optimization
    - SQL generation for the target dialect
    - Query execution
    - Result processing

    Example:
        >>> executor = PushdownExecutor(datasource, SQLDialect.POSTGRESQL)
        >>> result = executor.execute(query)
        >>> print(result.row_count)
    """

    def __init__(
        self,
        datasource: "BaseSQLDataSource",
        dialect: SQLDialect | None = None,
        optimizer: QueryOptimizer | None = None,
        enable_optimization: bool = True,
        enable_caching: bool = True,
    ) -> None:
        """Initialize executor.

        Args:
            datasource: SQL data source.
            dialect: SQL dialect. If None, inferred from datasource.
            optimizer: Query optimizer. If None, creates default.
            enable_optimization: Whether to optimize queries.
            enable_caching: Whether to cache query results.
        """
        self.datasource = datasource
        self.dialect = dialect or self._infer_dialect()
        self.optimizer = optimizer or QueryOptimizer(dialect=self.dialect)
        self.enable_optimization = enable_optimization
        self.enable_caching = enable_caching
        self._cache: dict[str, PushdownResult] = {}
        self._plan_builder = ExecutionPlanBuilder(self.dialect)

    def _infer_dialect(self) -> SQLDialect:
        """Infer dialect from datasource."""
        source_type = self.datasource.source_type.lower()
        dialect_map = {
            "postgresql": SQLDialect.POSTGRESQL,
            "postgres": SQLDialect.POSTGRESQL,
            "mysql": SQLDialect.MYSQL,
            "sqlite": SQLDialect.SQLITE,
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
            "oracle": SQLDialect.ORACLE,
            "sqlserver": SQLDialect.SQLSERVER,
            "mssql": SQLDialect.SQLSERVER,
        }
        return dialect_map.get(source_type, SQLDialect.GENERIC)

    def plan(self, statement: Statement) -> ExecutionPlan:
        """Create an execution plan without executing.

        Args:
            statement: Statement to plan.

        Returns:
            ExecutionPlan.
        """
        # Optimize if enabled
        if self.enable_optimization:
            statement, _ = self.optimizer.analyze_and_optimize(statement)

        return self._plan_builder.build_plan(statement)

    def execute(
        self,
        query: Statement | "QueryBuilder",
        timeout: float | None = None,
        dry_run: bool = False,
    ) -> PushdownResult:
        """Execute a pushdown query.

        Args:
            query: Query to execute (Statement or QueryBuilder).
            timeout: Query timeout in seconds.
            dry_run: If True, only plan without executing.

        Returns:
            PushdownResult with query results.
        """
        from truthound.execution.pushdown.builder import QueryBuilder
        import time

        # Convert QueryBuilder to Statement
        if isinstance(query, QueryBuilder):
            statement = query.build()
        else:
            statement = query

        # Create execution plan
        plan = self.plan(statement)

        # Check cache
        if self.enable_caching and plan.sql in self._cache:
            cached = self._cache[plan.sql]
            cached.metadata["from_cache"] = True
            return cached

        # Dry run - return plan without executing
        if dry_run:
            return PushdownResult(
                data=[],
                columns=[],
                row_count=0,
                plan=plan,
                metadata={"dry_run": True},
            )

        # Check if pushdown is possible
        if plan.pushdown_decision and not plan.pushdown_decision.can_pushdown:
            # Could fall back to local execution here
            raise ValueError(
                f"Query cannot be pushed down: "
                f"{[i.message for i in plan.pushdown_decision.issues if i.severity == 'error']}"
            )

        # Execute query
        start_time = time.time()
        try:
            rows = self.datasource.execute_query(plan.sql)
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}") from e

        execution_time = (time.time() - start_time) * 1000  # ms

        # Extract columns
        columns = list(rows[0].keys()) if rows else []

        # Create result
        result = PushdownResult(
            data=rows,
            columns=columns,
            row_count=len(rows),
            execution_time_ms=execution_time,
            plan=plan,
            was_pushed_down=True,
        )

        # Cache result
        if self.enable_caching:
            self._cache[plan.sql] = result

        return result

    def execute_sql(
        self,
        sql: str,
        timeout: float | None = None,
    ) -> PushdownResult:
        """Execute raw SQL.

        Args:
            sql: SQL query string.
            timeout: Query timeout in seconds.

        Returns:
            PushdownResult.
        """
        import time

        start_time = time.time()
        rows = self.datasource.execute_query(sql)
        execution_time = (time.time() - start_time) * 1000

        columns = list(rows[0].keys()) if rows else []

        return PushdownResult(
            data=rows,
            columns=columns,
            row_count=len(rows),
            execution_time_ms=execution_time,
            was_pushed_down=True,
        )

    def explain(
        self,
        query: Statement | "QueryBuilder",
        verbose: bool = False,
    ) -> str:
        """Get query execution plan explanation.

        Args:
            query: Query to explain.
            verbose: Include additional details.

        Returns:
            Formatted explanation string.
        """
        from truthound.execution.pushdown.builder import QueryBuilder

        if isinstance(query, QueryBuilder):
            statement = query.build()
        else:
            statement = query

        plan = self.plan(statement)
        return plan.explain(verbose=verbose)

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics dictionary.
        """
        return {
            "size": len(self._cache),
            "queries": list(self._cache.keys())[:10],  # First 10
        }


# =============================================================================
# Batch Executor
# =============================================================================


class BatchPushdownExecutor:
    """Executes multiple pushdown queries efficiently.

    This executor optimizes batch query execution by:
    - Combining compatible queries
    - Parallel execution where possible
    - Shared caching
    """

    def __init__(
        self,
        datasource: "BaseSQLDataSource",
        dialect: SQLDialect | None = None,
        max_parallel: int = 4,
    ) -> None:
        """Initialize batch executor.

        Args:
            datasource: SQL data source.
            dialect: SQL dialect.
            max_parallel: Maximum parallel queries.
        """
        self.executor = PushdownExecutor(datasource, dialect)
        self.max_parallel = max_parallel

    def execute_batch(
        self,
        queries: Sequence[Statement | "QueryBuilder"],
        fail_fast: bool = False,
    ) -> list[PushdownResult | Exception]:
        """Execute multiple queries.

        Args:
            queries: Queries to execute.
            fail_fast: Stop on first error.

        Returns:
            List of results or exceptions.
        """
        results: list[PushdownResult | Exception] = []

        for query in queries:
            try:
                result = self.executor.execute(query)
                results.append(result)
            except Exception as e:
                if fail_fast:
                    raise
                results.append(e)

        return results

    def execute_batch_parallel(
        self,
        queries: Sequence[Statement | "QueryBuilder"],
    ) -> list[PushdownResult | Exception]:
        """Execute queries in parallel.

        Args:
            queries: Queries to execute.

        Returns:
            List of results or exceptions.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: dict[int, PushdownResult | Exception] = {}

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(self.executor.execute, q): i
                for i, q in enumerate(queries)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e

        return [results[i] for i in range(len(queries))]
