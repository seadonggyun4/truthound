"""Query-based validators for flexible data validation.

This module provides 15 query-based validators for SQL and expression-based validation:

Query Result Validators:
- QueryReturnsSingleValueValidator: Check query returns expected value
- QueryReturnsNoRowsValidator: Check query returns no rows
- QueryReturnsRowsValidator: Check query returns rows within bounds
- QueryResultMatchesValidator: Check query result matches expected DataFrame

Row Count Validators:
- QueryRowCountValidator: Check row count from query
- QueryRowCountRatioValidator: Check row count ratio
- QueryRowCountCompareValidator: Compare row counts between queries

Column Validators:
- QueryColumnValuesValidator: Check column values from query
- QueryColumnUniqueValidator: Check column uniqueness in query result
- QueryColumnNotNullValidator: Check no nulls in query column

Aggregate Validators:
- QueryAggregateValidator: Check aggregate value bounds
- QueryGroupAggregateValidator: Check group-level aggregates
- QueryAggregateCompareValidator: Compare aggregates between queries

Expression Validators:
- CustomExpressionValidator: Validate with custom Polars expression
- ConditionalExpressionValidator: IF-THEN validation logic
- MultiConditionValidator: Multiple conditions with AND/OR
- RowLevelValidator: Per-row Python function validation
"""

from truthound.validators.query.base import (
    QueryValidator,
    ExpressionValidator,
)
from truthound.validators.query.result import (
    QueryReturnsSingleValueValidator,
    QueryReturnsNoRowsValidator,
    QueryReturnsRowsValidator,
    QueryResultMatchesValidator,
)
from truthound.validators.query.row_count import (
    QueryRowCountValidator,
    QueryRowCountRatioValidator,
    QueryRowCountCompareValidator,
)
from truthound.validators.query.column import (
    QueryColumnValuesValidator,
    QueryColumnUniqueValidator,
    QueryColumnNotNullValidator,
)
from truthound.validators.query.aggregate import (
    QueryAggregateValidator,
    QueryGroupAggregateValidator,
    QueryAggregateCompareValidator,
)
from truthound.validators.query.expression import (
    CustomExpressionValidator,
    ConditionalExpressionValidator,
    MultiConditionValidator,
    RowLevelValidator,
)

__all__ = [
    # Base classes
    "QueryValidator",
    "ExpressionValidator",
    # Result validators
    "QueryReturnsSingleValueValidator",
    "QueryReturnsNoRowsValidator",
    "QueryReturnsRowsValidator",
    "QueryResultMatchesValidator",
    # Row count validators
    "QueryRowCountValidator",
    "QueryRowCountRatioValidator",
    "QueryRowCountCompareValidator",
    # Column validators
    "QueryColumnValuesValidator",
    "QueryColumnUniqueValidator",
    "QueryColumnNotNullValidator",
    # Aggregate validators
    "QueryAggregateValidator",
    "QueryGroupAggregateValidator",
    "QueryAggregateCompareValidator",
    # Expression validators
    "CustomExpressionValidator",
    "ConditionalExpressionValidator",
    "MultiConditionValidator",
    "RowLevelValidator",
]
