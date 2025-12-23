"""Multi-column compound validators.

This module provides 18 validators for checking relationships across multiple columns:

Arithmetic Validators:
- ColumnSumValidator: Validate sum of columns
- ColumnProductValidator: Validate product of columns
- ColumnDifferenceValidator: Validate difference between columns
- ColumnRatioValidator: Validate ratio between columns
- ColumnPercentageValidator: Validate percentage calculation

Comparison Validators:
- ColumnComparisonValidator: Compare two columns (>, <, ==, etc.)
- ColumnChainComparisonValidator: Chain comparisons (a < b < c)
- ColumnMaxValidator: Validate max of columns
- ColumnMinValidator: Validate min of columns
- ColumnMeanValidator: Validate mean of columns

Consistency Validators:
- ColumnConsistencyValidator: Rule-based consistency checks
- ColumnMutualExclusivityValidator: Only one column can have value
- ColumnCoexistenceValidator: Columns must all be filled or all empty
- ColumnDependencyValidator: Conditional column requirements
- ColumnImplicationValidator: If A then B logic

Statistical Validators:
- ColumnCorrelationValidator: Check correlation between columns
- ColumnCovarianceValidator: Check covariance between columns
- MultiColumnVarianceValidator: Check variance across columns
"""

from truthound.validators.multi_column.base import (
    MultiColumnValidator,
    ColumnArithmeticValidator,
)
from truthound.validators.multi_column.arithmetic import (
    ColumnSumValidator,
    ColumnProductValidator,
    ColumnDifferenceValidator,
    ColumnRatioValidator,
    ColumnPercentageValidator,
)
from truthound.validators.multi_column.comparison import (
    ColumnComparisonValidator,
    ColumnChainComparisonValidator,
    ColumnMaxValidator,
    ColumnMinValidator,
    ColumnMeanValidator,
)
from truthound.validators.multi_column.consistency import (
    ColumnConsistencyValidator,
    ColumnMutualExclusivityValidator,
    ColumnCoexistenceValidator,
    ColumnDependencyValidator,
    ColumnImplicationValidator,
)
from truthound.validators.multi_column.statistical import (
    ColumnCorrelationValidator,
    ColumnCovarianceValidator,
    MultiColumnVarianceValidator,
)

__all__ = [
    # Base classes
    "MultiColumnValidator",
    "ColumnArithmeticValidator",
    # Arithmetic
    "ColumnSumValidator",
    "ColumnProductValidator",
    "ColumnDifferenceValidator",
    "ColumnRatioValidator",
    "ColumnPercentageValidator",
    # Comparison
    "ColumnComparisonValidator",
    "ColumnChainComparisonValidator",
    "ColumnMaxValidator",
    "ColumnMinValidator",
    "ColumnMeanValidator",
    # Consistency
    "ColumnConsistencyValidator",
    "ColumnMutualExclusivityValidator",
    "ColumnCoexistenceValidator",
    "ColumnDependencyValidator",
    "ColumnImplicationValidator",
    # Statistical
    "ColumnCorrelationValidator",
    "ColumnCovarianceValidator",
    "MultiColumnVarianceValidator",
]
