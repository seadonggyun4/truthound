"""Table metadata validators.

This module provides 13 validators for checking table-level properties:

Row Count Validators:
- TableRowCountRangeValidator: Validate row count within range
- TableRowCountExactValidator: Validate exact row count
- TableRowCountCompareValidator: Compare row count with another table
- TableNotEmptyValidator: Validate table is not empty

Column Count Validators:
- TableColumnCountValidator: Validate column count
- TableRequiredColumnsValidator: Validate required columns exist
- TableForbiddenColumnsValidator: Validate forbidden columns don't exist

Freshness Validators:
- TableFreshnessValidator: Validate data is fresh (recent)
- TableDataRecencyValidator: Validate percentage of recent data
- TableUpdateFrequencyValidator: Validate update frequency

Schema Validators:
- TableSchemaMatchValidator: Validate schema matches expected
- TableSchemaCompareValidator: Compare schema with another table
- TableColumnTypesValidator: Validate specific column types

Size Validators:
- TableMemorySizeValidator: Validate memory size bounds
- TableRowToColumnRatioValidator: Validate row/column ratio
- TableDimensionsValidator: Validate dimensions (rows and columns)
"""

from truthound.validators.table.base import (
    TableValidator,
    TableSchemaValidator,
)
from truthound.validators.table.row_count import (
    TableRowCountRangeValidator,
    TableRowCountExactValidator,
    TableRowCountCompareValidator,
    TableNotEmptyValidator,
)
from truthound.validators.table.column_count import (
    TableColumnCountValidator,
    TableRequiredColumnsValidator,
    TableForbiddenColumnsValidator,
)
from truthound.validators.table.freshness import (
    TableFreshnessValidator,
    TableDataRecencyValidator,
    TableUpdateFrequencyValidator,
)
from truthound.validators.table.schema import (
    TableSchemaMatchValidator,
    TableSchemaCompareValidator,
    TableColumnTypesValidator,
)
from truthound.validators.table.size import (
    TableMemorySizeValidator,
    TableRowToColumnRatioValidator,
    TableDimensionsValidator,
)

__all__ = [
    # Base classes
    "TableValidator",
    "TableSchemaValidator",
    # Row count
    "TableRowCountRangeValidator",
    "TableRowCountExactValidator",
    "TableRowCountCompareValidator",
    "TableNotEmptyValidator",
    # Column count
    "TableColumnCountValidator",
    "TableRequiredColumnsValidator",
    "TableForbiddenColumnsValidator",
    # Freshness
    "TableFreshnessValidator",
    "TableDataRecencyValidator",
    "TableUpdateFrequencyValidator",
    # Schema
    "TableSchemaMatchValidator",
    "TableSchemaCompareValidator",
    "TableColumnTypesValidator",
    # Size
    "TableMemorySizeValidator",
    "TableRowToColumnRatioValidator",
    "TableDimensionsValidator",
]
