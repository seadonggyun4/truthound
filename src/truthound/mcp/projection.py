"""Versioned summaries allow selected metadata, never raw values or diagnostics."""

from __future__ import annotations

from typing import Any

_DTYPE_FAMILIES = frozenset(
    {
        "Array",
        "Binary",
        "Boolean",
        "Categorical",
        "Date",
        "Datetime",
        "Decimal",
        "Duration",
        "Enum",
        "Float16",
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "List",
        "Map",
        "Null",
        "Object",
        "String",
        "Struct",
        "Time",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "UInt128",
        "Unknown",
        "Utf8",
    }
)


def _dtype_family(value: Any) -> str:
    # Core reprs may contain Enum values, nested schemas and timezone strings.
    # Inspect only a bounded prefix; never return parameters or unknown text.
    family = value[:32].partition("(")[0] if isinstance(value, str) else "Unknown"
    return family if family in _DTYPE_FAMILIES else "Unknown"


def _profile_column(info: dict) -> dict:
    column = {
        key: value for key, value in info.items() if key in {"name", "null_pct", "unique_pct"}
    }
    column["dtype"] = _dtype_family(info.get("dtype"))
    return column


def project(operation: str, result: Any, limit: int) -> tuple[dict, bool]:
    if operation == "check":
        data = {
            "row_count": result.row_count,
            "column_count": result.column_count,
            "success": result.success,
            "issue_count": len(result.issues),
            "execution_issue_count": len(result.execution_issues),
            "checks": [
                {"name": c.name, "success": c.success, "issue_count": c.issue_count}
                for c in result.checks[:limit]
            ],
        }
        return data, len(result.checks) > limit
    if operation == "profile":
        # Deliberately omit samples, distinct values, patterns and extrema.
        columns = result.columns
        return {
            "row_count": result.row_count,
            "column_count": result.column_count,
            "columns": [_profile_column(info) for info in columns[:limit]],
        }, len(columns) > limit
    return {
        "row_count": result.row_count,
        "finding_count": len(result.findings),
        "count_semantics": "estimated_from_first_1000_non_null_values",
        "findings": [
            {k: f[k] for k in ("column", "pii_type", "count", "confidence") if k in f}
            for f in result.findings[:limit]
        ],
    }, len(result.findings) > limit


def response(
    operation: str, *, data: dict | None = None, truncated: bool = False, error: str | None = None
) -> dict:
    return {
        "schema_version": 1,
        "operation": operation,
        "ok": error is None,
        "data": data,
        "truncated": truncated,
        "error": None if error is None else {"code": error},
    }
