"""Versioned summaries allow selected metadata, never raw values or diagnostics."""

from __future__ import annotations

from typing import Any


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
        keys = {"name", "dtype", "null_pct", "unique_pct"}
        columns = result.columns
        return {
            "row_count": result.row_count,
            "column_count": result.column_count,
            "columns": [{k: v for k, v in info.items() if k in keys} for info in columns[:limit]],
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
