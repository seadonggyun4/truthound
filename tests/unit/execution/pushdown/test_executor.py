"""Tests for query executor and execution plan."""

from __future__ import annotations

from unittest.mock import Mock, MagicMock, patch
from typing import Any
from datetime import datetime

import pytest

from truthound.execution.pushdown.ast import (
    Column,
    Literal,
    BinaryExpression,
    AggregateFunction,
    SelectStatement,
    FromClause,
    WhereClause,
    GroupByClause,
    OrderByClause,
    OrderByItem,
    LimitClause,
    Table,
    Star,
    ComparisonOp,
    SortOrder,
)
from truthound.execution.pushdown.dialects import SQLDialect
from truthound.execution.pushdown.executor import (
    PlanNode,
    PlanNodeType,
    ExecutionPlan,
    PushdownResult,
    ExecutionPlanBuilder,
    PushdownExecutor,
    BatchPushdownExecutor,
)
from truthound.execution.pushdown.builder import (
    QueryBuilder,
    col,
    func,
)


class TestPlanNode:
    """Tests for execution plan nodes."""

    def test_create_scan_node(self):
        """Test creating a table scan node."""
        node = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan users table",
            estimated_rows=1000,
            estimated_cost=1.0,
        )
        assert node.node_type == PlanNodeType.TABLE_SCAN
        assert node.description == "Scan users table"
        assert node.estimated_cost == 1.0

    def test_create_filter_node(self):
        """Test creating a filter node."""
        scan = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan users",
            estimated_cost=1.0,
        )
        filter_node = PlanNode(
            node_type=PlanNodeType.FILTER,
            description="Filter: age > 18",
            children=[scan],
            estimated_cost=0.5,
        )
        assert filter_node.node_type == PlanNodeType.FILTER
        assert len(filter_node.children) == 1
        assert "age > 18" in filter_node.description

    def test_create_projection_node(self):
        """Test creating a projection node."""
        scan = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan users",
            estimated_cost=1.0,
        )
        proj = PlanNode(
            node_type=PlanNodeType.PROJECT,
            description="Project columns",
            children=[scan],
            properties={"columns": ["id", "name", "email"]},
            estimated_cost=0.1,
        )
        assert proj.node_type == PlanNodeType.PROJECT
        assert "id" in proj.properties["columns"]

    def test_create_aggregate_node(self):
        """Test creating an aggregate node."""
        scan = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan orders",
            estimated_cost=1.0,
        )
        agg = PlanNode(
            node_type=PlanNodeType.AGGREGATE,
            description="Aggregate by customer_id",
            children=[scan],
            properties={
                "group_by": ["customer_id"],
                "aggregates": ["SUM(amount)", "COUNT(*)"],
            },
            estimated_cost=2.0,
        )
        assert agg.node_type == PlanNodeType.AGGREGATE
        assert "customer_id" in agg.properties["group_by"]

    def test_create_sort_node(self):
        """Test creating a sort node."""
        scan = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan users",
            estimated_cost=1.0,
        )
        sort = PlanNode(
            node_type=PlanNodeType.SORT,
            description="Sort by name, created_at",
            children=[scan],
            properties={"order_by": ["name ASC", "created_at DESC"]},
            estimated_cost=1.5,
        )
        assert sort.node_type == PlanNodeType.SORT
        assert len(sort.properties["order_by"]) == 2

    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        scan = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan t",
            estimated_rows=100,
            estimated_cost=1.0,
        )
        result = scan.to_dict()
        assert result["type"] == "TABLE_SCAN"
        assert result["description"] == "Scan t"
        assert result["estimated_rows"] == 100
        assert result["estimated_cost"] == 1.0

    def test_node_str_representation(self):
        """Test string representation of node."""
        scan = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan users",
            estimated_rows=1000,
            estimated_cost=5.0,
        )
        result = str(scan)
        assert "TABLE_SCAN" in result
        assert "Scan users" in result


class TestExecutionPlan:
    """Tests for execution plan."""

    def test_create_execution_plan(self):
        """Test creating an execution plan."""
        root = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan users",
            estimated_cost=1.0,
        )
        plan = ExecutionPlan(
            root=root,
            sql='SELECT * FROM "users"',
            dialect=SQLDialect.POSTGRESQL,
            total_cost=1.0,
        )

        assert plan.root is not None
        assert plan.dialect == SQLDialect.POSTGRESQL
        assert plan.sql == 'SELECT * FROM "users"'

    def test_plan_to_dict(self):
        """Test converting plan to dictionary."""
        root = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan users",
            estimated_cost=1.0,
        )
        plan = ExecutionPlan(
            root=root,
            sql='SELECT * FROM "users"',
            dialect=SQLDialect.POSTGRESQL,
        )

        result = plan.to_dict()
        assert "sql" in result
        assert "dialect" in result
        assert "plan" in result

    def test_plan_explain(self):
        """Test plan explain output."""
        scan = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan users",
            estimated_cost=1.0,
        )
        filter_node = PlanNode(
            node_type=PlanNodeType.FILTER,
            description="Filter: age > 18",
            children=[scan],
            estimated_cost=0.5,
        )
        plan = ExecutionPlan(
            root=filter_node,
            sql='SELECT * FROM "users" WHERE "age" > 18',
            dialect=SQLDialect.POSTGRESQL,
            total_cost=1.5,
        )

        explain = plan.explain()

        assert isinstance(explain, str)
        assert "EXECUTION PLAN" in explain
        assert "FILTER" in explain or "TABLE_SCAN" in explain

    def test_plan_total_cost(self):
        """Test plan total cost."""
        root = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan t",
            estimated_cost=5.0,
        )
        plan = ExecutionPlan(
            root=root,
            sql="SELECT * FROM t",
            dialect=SQLDialect.POSTGRESQL,
            total_cost=5.0,
        )

        assert plan.total_cost == 5.0


class TestPushdownResult:
    """Tests for pushdown result."""

    def test_create_result(self):
        """Test creating a pushdown result."""
        result = PushdownResult(
            data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            columns=["id", "name"],
            row_count=2,
        )

        assert result.data is not None
        assert len(result.data) == 2
        assert result.columns == ["id", "name"]
        assert result.row_count == 2

    def test_result_with_timing(self):
        """Test result with timing information."""
        result = PushdownResult(
            data=[],
            columns=[],
            row_count=0,
            execution_time_ms=150.5,
        )

        assert result.execution_time_ms == 150.5

    def test_result_with_plan(self):
        """Test result with execution plan."""
        root = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            description="Scan t",
            estimated_cost=1.0,
        )
        plan = ExecutionPlan(
            root=root,
            sql="SELECT * FROM t",
            dialect=SQLDialect.POSTGRESQL,
        )

        result = PushdownResult(
            data=[],
            columns=[],
            row_count=0,
            plan=plan,
        )

        assert result.plan is not None
        assert result.plan.root.node_type == PlanNodeType.TABLE_SCAN

    def test_result_empty(self):
        """Test empty result."""
        result = PushdownResult(
            data=[],
            columns=["id", "name"],
            row_count=0,
        )

        assert len(result.data) == 0
        assert result.row_count == 0
        assert len(result.columns) == 2

    def test_result_first(self):
        """Test getting first row."""
        result = PushdownResult(
            data=[{"id": 1}, {"id": 2}],
            columns=["id"],
            row_count=2,
        )
        assert result.first() == {"id": 1}

    def test_result_first_empty(self):
        """Test first on empty result."""
        result = PushdownResult(data=[], columns=[], row_count=0)
        assert result.first() is None

    def test_result_iter_rows(self):
        """Test iterating over rows."""
        result = PushdownResult(
            data=[{"id": 1}, {"id": 2}, {"id": 3}],
            columns=["id"],
            row_count=3,
        )
        rows = list(result.iter_rows())
        assert len(rows) == 3

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = PushdownResult(
            data=[{"id": 1}],
            columns=["id"],
            row_count=1,
            execution_time_ms=10.0,
        )
        d = result.to_dict()
        assert "columns" in d
        assert "row_count" in d
        assert "execution_time_ms" in d


class TestExecutionPlanBuilder:
    """Tests for execution plan builder."""

    def test_build_simple_scan(self):
        """Test building plan for simple SELECT."""
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
        )

        builder = ExecutionPlanBuilder(SQLDialect.POSTGRESQL)
        plan = builder.build(stmt)

        assert plan is not None
        assert plan.root is not None
        assert plan.sql is not None

    def test_build_with_filter(self):
        """Test building plan with WHERE clause."""
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            where_clause=WhereClause(
                BinaryExpression(Column("age"), ComparisonOp.GT, Literal(18))
            ),
        )

        builder = ExecutionPlanBuilder(SQLDialect.POSTGRESQL)
        plan = builder.build(stmt)

        assert plan is not None
        assert "WHERE" in plan.sql

    def test_build_with_aggregation(self):
        """Test building plan with GROUP BY."""
        stmt = SelectStatement(
            select_items=[
                Column("department"),
                AggregateFunction("COUNT", None),
            ],
            from_clause=FromClause(Table("employees")),
            group_by_clause=GroupByClause([Column("department")]),
        )

        builder = ExecutionPlanBuilder(SQLDialect.POSTGRESQL)
        plan = builder.build(stmt)

        assert plan is not None
        assert "GROUP BY" in plan.sql

    def test_build_with_order_by(self):
        """Test building plan with ORDER BY."""
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            order_by_clause=OrderByClause([
                OrderByItem(Column("name"), SortOrder.ASC),
            ]),
        )

        builder = ExecutionPlanBuilder(SQLDialect.POSTGRESQL)
        plan = builder.build(stmt)

        assert plan is not None
        assert "ORDER BY" in plan.sql

    def test_build_with_limit(self):
        """Test building plan with LIMIT."""
        stmt = SelectStatement(
            select_items=[Star()],
            from_clause=FromClause(Table("users")),
            limit_clause=LimitClause(100),
        )

        builder = ExecutionPlanBuilder(SQLDialect.POSTGRESQL)
        plan = builder.build(stmt)

        assert plan is not None
        assert "LIMIT" in plan.sql

    def test_build_complex_query(self):
        """Test building plan for complex query."""
        query = (
            QueryBuilder("orders", alias="o")
            .select(
                col("o.customer_id"),
                func.count("*").alias("order_count"),
                func.sum("o.amount").alias("total"),
            )
            .where(col("o.status") == "completed")
            .group_by("o.customer_id")
            .order_by("total", desc=True)
            .limit(10)
        )

        stmt = query.build()
        builder = ExecutionPlanBuilder(SQLDialect.POSTGRESQL)
        plan = builder.build(stmt)

        assert plan is not None
        assert plan.sql is not None
        assert "GROUP BY" in plan.sql
        assert "ORDER BY" in plan.sql


class TestPushdownExecutor:
    """Tests for pushdown executor."""

    def test_executor_creation(self):
        """Test creating executor."""
        executor = PushdownExecutor(dialect=SQLDialect.POSTGRESQL)
        assert executor is not None
        assert executor.dialect == SQLDialect.POSTGRESQL

    def test_executor_plan(self):
        """Test executor plan method."""
        executor = PushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        query = QueryBuilder("users").select("*").where(col("active") == True)
        stmt = query.build()

        plan = executor.plan(stmt)

        assert plan is not None
        assert isinstance(plan, ExecutionPlan)

    def test_executor_explain(self):
        """Test executor explain method."""
        executor = PushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        query = QueryBuilder("users").select("id", "name")
        stmt = query.build()

        explain = executor.explain(stmt)

        assert isinstance(explain, str)
        assert len(explain) > 0

    def test_executor_generate_sql(self):
        """Test executor generates correct SQL."""
        executor = PushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        query = (
            QueryBuilder("users")
            .select("id", "name")
            .where(col("age") > 18)
            .order_by("name")
            .limit(10)
        )
        stmt = query.build()

        sql = executor.generate_sql(stmt)

        assert "SELECT" in sql
        assert "FROM" in sql
        assert "WHERE" in sql
        assert "ORDER BY" in sql
        assert "LIMIT" in sql

    def test_executor_analyze(self):
        """Test executor analyze method."""
        executor = PushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        query = QueryBuilder("users").select("*")
        stmt = query.build()

        decision = executor.analyze(stmt)

        assert decision is not None
        assert decision.can_pushdown is True


class TestBatchPushdownExecutor:
    """Tests for batch pushdown executor."""

    def test_batch_executor_creation(self):
        """Test creating batch executor."""
        executor = BatchPushdownExecutor(dialect=SQLDialect.POSTGRESQL)
        assert executor is not None

    def test_batch_add_query(self):
        """Test adding queries to batch."""
        executor = BatchPushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        q1 = QueryBuilder("users").select("*")
        q2 = QueryBuilder("orders").select("*")

        executor.add(q1.build())
        executor.add(q2.build())

        assert len(executor.queries) == 2

    def test_batch_clear(self):
        """Test clearing batch."""
        executor = BatchPushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        executor.add(QueryBuilder("users").select("*").build())
        executor.add(QueryBuilder("orders").select("*").build())

        executor.clear()

        assert len(executor.queries) == 0

    def test_batch_plan_all(self):
        """Test planning all queries in batch."""
        executor = BatchPushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        executor.add(QueryBuilder("users").select("*").build())
        executor.add(QueryBuilder("orders").select("*").build())

        plans = executor.plan_all()

        assert len(plans) == 2
        assert all(isinstance(p, ExecutionPlan) for p in plans)


class TestExecutorDialects:
    """Tests for executor with different dialects."""

    @pytest.mark.parametrize("dialect", [
        SQLDialect.POSTGRESQL,
        SQLDialect.MYSQL,
        SQLDialect.SQLITE,
        SQLDialect.BIGQUERY,
        SQLDialect.SNOWFLAKE,
    ])
    def test_executor_supports_dialect(self, dialect):
        """Test executor works with different dialects."""
        executor = PushdownExecutor(dialect=dialect)

        query = (
            QueryBuilder("users")
            .select("id", "name")
            .where(col("active") == True)
            .limit(10)
        )
        stmt = query.build()

        sql = executor.generate_sql(stmt)

        assert sql is not None
        assert "SELECT" in sql
        assert "FROM" in sql

    def test_postgresql_specific_syntax(self):
        """Test PostgreSQL-specific SQL generation."""
        executor = PushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        query = QueryBuilder("users").select("name").limit(10)
        stmt = query.build()

        sql = executor.generate_sql(stmt)

        # PostgreSQL uses double quotes
        assert '"' in sql

    def test_mysql_specific_syntax(self):
        """Test MySQL-specific SQL generation."""
        executor = PushdownExecutor(dialect=SQLDialect.MYSQL)

        query = QueryBuilder("users").select("name").limit(10)
        stmt = query.build()

        sql = executor.generate_sql(stmt)

        # MySQL uses backticks
        assert "`" in sql


class TestExecutorErrorHandling:
    """Tests for executor error handling."""

    def test_invalid_statement_handling(self):
        """Test handling of invalid statements."""
        executor = PushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        # None statement should be handled gracefully
        with pytest.raises((ValueError, TypeError, AttributeError)):
            executor.plan(None)


class TestExecutorOptimization:
    """Tests for executor optimization features."""

    def test_executor_with_optimizer(self):
        """Test executor applies optimizations."""
        executor = PushdownExecutor(
            dialect=SQLDialect.POSTGRESQL,
            optimize=True,
        )

        query = (
            QueryBuilder("users")
            .select("id", "name")
            .where(col("age") > 18)
        )
        stmt = query.build()

        plan = executor.plan(stmt)

        assert plan is not None

    def test_executor_without_optimizer(self):
        """Test executor without optimizations."""
        executor = PushdownExecutor(
            dialect=SQLDialect.POSTGRESQL,
            optimize=False,
        )

        query = QueryBuilder("users").select("*")
        stmt = query.build()

        plan = executor.plan(stmt)

        assert plan is not None

    def test_cost_estimation(self):
        """Test cost estimation in plan."""
        executor = PushdownExecutor(dialect=SQLDialect.POSTGRESQL)

        query = (
            QueryBuilder("users")
            .select("*")
            .where(
                col("age") > 18,
                col("status") == "active",
            )
        )
        stmt = query.build()

        plan = executor.plan(stmt)

        # Plan should have a cost estimate
        assert plan.total_cost >= 0
