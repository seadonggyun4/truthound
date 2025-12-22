"""Range validator for numeric values."""

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class RangeValidator(Validator):
    """Detects out-of-range values like negative ages, future dates, etc."""

    name = "range"

    # Common column patterns and their expected ranges
    KNOWN_RANGES: dict[str, tuple[float | None, float | None]] = {
        "age": (0, 150),
        "price": (0, None),
        "quantity": (0, None),
        "count": (0, None),
        "amount": (0, None),
        "percentage": (0, 100),
        "percent": (0, 100),
        "pct": (0, 100),
        "rate": (0, 100),
        "score": (0, 100),
        "rating": (0, 5),
        "year": (1900, 2100),
        "month": (1, 12),
        "day": (1, 31),
        "hour": (0, 23),
        "minute": (0, 59),
        "second": (0, 59),
    }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for out-of-range values in numeric columns.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues for columns with out-of-range values.
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        numeric_types = [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        ]

        for col in schema.names():
            dtype = schema[col]

            if dtype not in numeric_types:
                continue

            col_lower = col.lower()

            # Try to find a known range for this column
            min_val, max_val = None, None
            for pattern, (pmin, pmax) in self.KNOWN_RANGES.items():
                if pattern in col_lower:
                    min_val, max_val = pmin, pmax
                    break

            # If no known range, just check for negative values in columns
            # that shouldn't be negative
            if min_val is None and max_val is None:
                # Check for negative values in columns that likely shouldn't have them
                negative_indicators = ["count", "quantity", "amount", "total", "num", "number"]
                if any(ind in col_lower for ind in negative_indicators):
                    min_val = 0

            if min_val is None and max_val is None:
                continue

            col_data = df.get_column(col)
            out_of_range_count = 0

            if min_val is not None:
                below_min = col_data.filter(col_data < min_val).drop_nulls()
                out_of_range_count += len(below_min)

            if max_val is not None:
                above_max = col_data.filter(col_data > max_val).drop_nulls()
                out_of_range_count += len(above_max)

            if out_of_range_count > 0:
                range_desc = ""
                if min_val is not None and max_val is not None:
                    range_desc = f"expected [{min_val}, {max_val}]"
                elif min_val is not None:
                    range_desc = f"expected >= {min_val}"
                elif max_val is not None:
                    range_desc = f"expected <= {max_val}"

                oor_pct = out_of_range_count / total_rows

                if oor_pct > 0.1:
                    severity = Severity.HIGH
                elif oor_pct > 0.01:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="out_of_range",
                        count=out_of_range_count,
                        severity=severity,
                        details=range_desc,
                    )
                )

        return issues
