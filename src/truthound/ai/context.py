"""Read-only context bundle builder for AI suite proposals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from truthound.adapters import to_lazyframe
from truthound.ai.models import InputRef, ValidationSuiteSnapshot
from truthound.ai.suite_diff import (
    build_current_validation_suite,
    build_existing_suite_summary,
    snapshot_validation_suite,
)
from truthound.context import TruthoundContext, get_context
from truthound.schema import Schema, learn

if TYPE_CHECKING:
    from truthound.core.suite import ValidationSuite


SUPPORTED_INTENT_NAMES = (
    "null",
    "not_null",
    "completeness_ratio",
    "unique",
    "unique_ratio",
    "between",
    "in_set",
    "length",
    "format",
    "regex",
    "mean_between",
    "sum_between",
)

NUMERIC_DTYPE_MARKERS = (
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "Decimal",
)
STRING_DTYPE_MARKERS = ("String", "Utf8")


@dataclass(frozen=True)
class ContextBundle:
    source_key: str
    source_ref: str
    workspace_root: str
    input_refs: tuple[InputRef, ...]
    current_suite: "ValidationSuite"
    current_suite_snapshot: ValidationSuiteSnapshot
    schema_summary: dict[str, Any]
    baseline_summary: dict[str, Any] | None
    history_summary: dict[str, Any]
    existing_suite_summary: dict[str, Any]
    prompt_constraints: tuple[str, ...]
    summary_budget: int

    def build_system_prompt(self) -> str:
        intents = ", ".join(SUPPORTED_INTENT_NAMES)
        return (
            "You compile reviewable validation suite proposals for Truthound. "
            "Return one JSON object with keys summary, rationale, proposed_checks, risks, rejected_requests. "
            "Each proposed_checks item uses intent, columns, params, rationale. "
            f"Allowed intents only: {intents}. "
            "Use aggregate schema signals only. "
            "Do not include literal values, table excerpts, write actions, apply actions, or baseline mutations."
        )

    def build_user_prompt(self, prompt: str) -> str:
        sections = [
            f"operator_request: {prompt}",
            f"source_ref: {self.source_ref}",
            f"summary_budget: {self.summary_budget}",
            f"schema_summary: {self._render_schema_summary(self.schema_summary)}",
            f"baseline_summary: {self._render_baseline_summary(self.baseline_summary)}",
            f"history_summary: {self._render_history_summary(self.history_summary)}",
            f"existing_suite_summary: {self._render_existing_suite_summary(self.existing_suite_summary)}",
            f"constraints: {', '.join(self.prompt_constraints)}",
        ]
        return "; ".join(section for section in sections if section)

    def _render_schema_summary(self, summary: dict[str, Any]) -> str:
        columns = summary.get("columns", [])
        parts: list[str] = [
            f"observed_count {summary.get('observed_count', 0)}",
            f"column_count {summary.get('column_count', len(columns))}",
        ]
        for column in columns:
            column_parts = [
                str(column.get("name", "unknown")),
                f"dtype {column.get('dtype', 'unknown')}",
                f"null_ratio {column.get('null_ratio', 0.0)}",
                f"unique_ratio {column.get('unique_ratio', 0.0)}",
            ]
            if "min_value" in column:
                column_parts.append(f"min {column['min_value']}")
            if "max_value" in column:
                column_parts.append(f"max {column['max_value']}")
            if "min_length" in column:
                column_parts.append(f"min_length {column['min_length']}")
            if "max_length" in column:
                column_parts.append(f"max_length {column['max_length']}")
            parts.append(" ".join(column_parts))
        return " | ".join(parts)

    def _render_baseline_summary(self, summary: dict[str, Any] | None) -> str:
        if not summary:
            return "none"
        columns = summary.get("columns", [])
        parts = [f"column_count {summary.get('column_count', len(columns))}"]
        for column in columns:
            column_parts = [
                str(column.get("name", "unknown")),
                f"dtype {column.get('dtype', 'unknown')}",
                f"nullable {column.get('nullable', True)}",
                f"unique {column.get('unique', False)}",
            ]
            if "min_value" in column:
                column_parts.append(f"min {column['min_value']}")
            if "max_value" in column:
                column_parts.append(f"max {column['max_value']}")
            parts.append(" ".join(column_parts))
        return " | ".join(parts)

    def _render_history_summary(self, summary: dict[str, Any]) -> str:
        statuses = ",".join(str(item) for item in summary.get("recent_statuses", ()))
        return (
            f"run_count {summary.get('run_count', 0)} "
            f"failure_count {summary.get('failure_count', 0)} "
            f"recent_statuses {statuses or 'none'} "
            f"latest_run_id {summary.get('latest_run_id', 'none')}"
        )

    def _render_existing_suite_summary(self, summary: dict[str, Any]) -> str:
        checks = summary.get("checks", [])
        parts = [f"check_count {summary.get('check_count', len(checks))}"]
        for check in checks:
            columns = ",".join(check.get("columns", ())) or "none"
            parts.append(
                f"{check.get('validator_name', 'unknown')} columns {columns} key {check.get('check_key', 'unknown')}"
            )
        return " | ".join(parts)


class ContextBundleBuilder:
    """Build a summary-only context bundle without mutating core workspace state."""

    def __init__(self, *, summary_budget: int = 1000) -> None:
        self._summary_budget = max(1, int(summary_budget))

    def build(
        self,
        *,
        data: Any = None,
        source: Any = None,
        context: TruthoundContext | None = None,
    ) -> ContextBundle:
        if data is None and source is None:
            raise ValueError("suggest_suite requires either data or source")

        active_context = context or get_context()
        source_key = active_context.resolve_source_key(data=data, source=source)
        lf = source.to_polars_lazyframe() if source is not None else to_lazyframe(data)
        observed_df = lf.head(self._summary_budget).collect(engine="streaming")
        observed_count = len(observed_df)

        schema_summary = self._build_schema_summary(observed_df, observed_count)
        baseline_schema = self._load_baseline_schema(active_context, source_key)
        baseline_summary = self._build_baseline_summary(baseline_schema)
        history_summary = self._load_history_summary(active_context, source_key)
        current_suite = build_current_validation_suite(
            observed_df=observed_df,
            baseline_schema=baseline_schema,
        )
        current_suite_snapshot = snapshot_validation_suite(current_suite)
        suite_summary = build_existing_suite_summary(current_suite_snapshot)

        input_refs = [
            self._make_input_ref(
                kind="schema_summary",
                ref=f"schema-summary:{source_key}",
                payload=schema_summary,
                metadata={
                    "column_count": schema_summary["column_count"],
                    "observed_count": schema_summary["observed_count"],
                },
            ),
            self._make_input_ref(
                kind="suite_summary",
                ref=f"suite-summary:{source_key}",
                payload=suite_summary,
                metadata={
                    "check_count": suite_summary["check_count"],
                },
            ),
        ]
        if baseline_summary is not None:
            input_refs.append(
                self._make_input_ref(
                    kind="baseline_summary",
                    ref=f"baseline-summary:{source_key}",
                    payload=baseline_summary,
                    metadata={
                        "column_count": baseline_summary["column_count"],
                    },
                )
            )
        if history_summary["run_count"] > 0:
            input_refs.append(
                self._make_input_ref(
                    kind="history_summary",
                    ref=f"history-summary:{source_key}",
                    payload=history_summary,
                    metadata={
                        "run_count": history_summary["run_count"],
                        "failure_count": history_summary["failure_count"],
                    },
                )
            )

        return ContextBundle(
            source_key=source_key,
            source_ref=self._safe_source_ref(source_key),
            workspace_root=str(active_context.root_dir),
            input_refs=tuple(input_refs),
            current_suite=current_suite,
            current_suite_snapshot=current_suite_snapshot,
            schema_summary=schema_summary,
            baseline_summary=baseline_summary,
            history_summary=history_summary,
            existing_suite_summary=suite_summary,
            prompt_constraints=(
                "compile reviewable proposals only",
                "no write actions",
                "no apply actions",
                "no baseline mutations",
                f"use only intents {', '.join(SUPPORTED_INTENT_NAMES)}",
            ),
            summary_budget=self._summary_budget,
        )

    def _build_schema_summary(
        self,
        observed_df: pl.DataFrame,
        observed_count: int,
    ) -> dict[str, Any]:
        columns: list[dict[str, Any]] = []
        for name in observed_df.columns:
            series = observed_df.get_column(name)
            non_null_count = int(len(series) - series.null_count())
            unique_ratio = 0.0
            if non_null_count > 0:
                unique_ratio = round(float(series.n_unique()) / float(non_null_count), 4)
            summary = {
                "name": name,
                "dtype": str(series.dtype),
                "null_ratio": round(float(series.null_count()) / float(observed_count or 1), 4),
                "unique_ratio": unique_ratio,
                "observed_non_null_count": non_null_count,
            }
            if self._is_numeric_dtype(str(series.dtype)) and non_null_count > 0:
                numeric_series = series.drop_nulls()
                if len(numeric_series) > 0:
                    summary["min_value"] = self._normalize_scalar(numeric_series.min())
                    summary["max_value"] = self._normalize_scalar(numeric_series.max())
            elif self._is_string_dtype(str(series.dtype)) and non_null_count > 0:
                length_series = series.drop_nulls().cast(pl.String).str.len_chars()
                if len(length_series) > 0:
                    summary["min_length"] = int(length_series.min())
                    summary["max_length"] = int(length_series.max())
            columns.append(summary)
        return {
            "observed_count": observed_count,
            "column_count": len(columns),
            "columns": columns,
        }

    def _load_baseline_schema(
        self,
        context: TruthoundContext,
        source_key: str,
    ) -> Schema | None:
        index_path = context.baseline_index_path
        if not index_path.exists():
            return None

        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        entry = payload.get(source_key)
        if not isinstance(entry, dict):
            return None

        schema_file = entry.get("schema_file")
        if not isinstance(schema_file, str):
            return None

        schema_path = context.baselines_dir / schema_file
        if not schema_path.exists():
            return None

        try:
            return Schema.load(schema_path)
        except Exception:
            return None

    def _build_baseline_summary(self, schema: Schema | None) -> dict[str, Any] | None:
        if schema is None:
            return None

        columns: list[dict[str, Any]] = []
        for name, column in schema.columns.items():
            summary = {
                "name": name,
                "dtype": column.dtype,
                "nullable": column.nullable,
                "unique": column.unique,
            }
            if column.min_value is not None:
                summary["min_value"] = column.min_value
            if column.max_value is not None:
                summary["max_value"] = column.max_value
            if column.min_length is not None:
                summary["min_length"] = column.min_length
            if column.max_length is not None:
                summary["max_length"] = column.max_length
            columns.append(summary)
        return {
            "column_count": len(columns),
            "columns": columns,
        }

    def _load_history_summary(
        self,
        context: TruthoundContext,
        source_key: str,
    ) -> dict[str, Any]:
        history_path = context.baselines_dir / "metric-history.json"
        if not history_path.exists():
            return {
                "run_count": 0,
                "failure_count": 0,
                "recent_statuses": [],
                "latest_run_id": None,
            }

        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        entries = payload.get(source_key, []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            entries = []
        recent = [entry for entry in entries if isinstance(entry, dict)][-5:]
        statuses = [str(item.get("status", "unknown")) for item in recent]
        latest = recent[-1] if recent else {}
        return {
            "run_count": len(entries),
            "failure_count": sum(1 for item in entries if isinstance(item, dict) and item.get("status") == "failure"),
            "recent_statuses": statuses,
            "latest_run_id": latest.get("run_id"),
        }

    def _make_input_ref(
        self,
        *,
        kind: str,
        ref: str,
        payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> InputRef:
        return InputRef(
            kind=kind,
            ref=ref,
            hash=self._hash_payload(payload),
            redacted=True,
            metadata=metadata,
        )

    def _hash_payload(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _safe_source_ref(self, source_key: str) -> str:
        return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]

    def _normalize_scalar(self, value: Any) -> Any:
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _is_numeric_dtype(self, dtype: str) -> bool:
        return any(marker in dtype for marker in NUMERIC_DTYPE_MARKERS)

    def _is_string_dtype(self, dtype: str) -> bool:
        return any(marker in dtype for marker in STRING_DTYPE_MARKERS)


__all__ = [
    "ContextBundle",
    "ContextBundleBuilder",
    "SUPPORTED_INTENT_NAMES",
]
