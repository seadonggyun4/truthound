"""Phase 2 run-to-analysis pipeline for ``truthound.ai``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from truthound.ai.models import (
    AIProvider,
    InputRef,
    ProviderConfig,
    RunAnalysisArtifact,
    RunAnalysisLLMResponse,
    StructuredProviderRequest,
)
from truthound.ai.providers import ProviderResponseError, resolve_model_name, resolve_provider
from truthound.ai.store import AIArtifactStore
from truthound.context import TruthoundContext, get_context
from truthound.core.results import ValidationRunResult
from truthound.reporters.presentation import build_run_presentation
from truthound.schema import Schema

HISTORY_WINDOW_SIZE = 10
_SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


@dataclass(frozen=True)
class ResolvedRunInput:
    context: TruthoundContext
    run_result: ValidationRunResult
    run_path: Path
    docs_path: Path | None
    source_key: str
    history_key: str
    source_fingerprint: str | None


@dataclass(frozen=True)
class RunAnalysisContextBundle:
    run_result: ValidationRunResult
    source_key: str
    history_key: str
    workspace_root: str
    input_refs: tuple[InputRef, ...]
    run_overview: dict[str, Any]
    failed_checks: list[str]
    failed_check_details: list[dict[str, Any]]
    top_columns: list[str]
    top_column_details: list[dict[str, Any]]
    baseline_summary: dict[str, Any] | None
    history_window: dict[str, Any]
    available_evidence_refs: tuple[str, ...]
    docs_ref: str | None

    def build_system_prompt(self) -> str:
        return (
            "You produce operational run analyses for Truthound. "
            "Return one JSON object with keys summary, recommended_next_actions, evidence_refs. "
            "Evidence refs must be chosen only from the provided available_evidence_refs list. "
            "Use only aggregate run, baseline, docs-ref, and history summaries. "
            "Do not include raw rows, sample values, row excerpts, HTML content, or mutation instructions."
        )

    def build_user_prompt(self) -> str:
        sections = [
            f"run_id: {self.run_result.run_id}",
            f"source_key: {self.source_key}",
            f"run_overview: {self._render_run_overview()}",
            f"failed_checks: {self._render_failed_checks()}",
            f"top_columns: {self._render_top_columns()}",
            f"baseline_summary: {self._render_baseline_summary()}",
            f"history_window: {self._render_history_window()}",
            f"docs_ref: {self.docs_ref or 'none'}",
            f"available_evidence_refs: {', '.join(self.available_evidence_refs)}",
            "constraints: analysis only, no apply flow, no baseline mutation, no suite mutation",
        ]
        return "; ".join(section for section in sections if section)

    def _render_run_overview(self) -> str:
        severity_counts = self.run_overview.get("severity_counts", {})
        severity_parts = ",".join(
            f"{name}:{severity_counts.get(name, 0)}"
            for name in ("critical", "high", "medium", "low")
        )
        return (
            f"status {self.run_overview.get('status', 'unknown')} "
            f"suite_name {self.run_overview.get('suite_name', 'unknown')} "
            f"row_count {self.run_overview.get('row_count', 0)} "
            f"column_count {self.run_overview.get('column_count', 0)} "
            f"total_checks {self.run_overview.get('total_checks', 0)} "
            f"failed_check_count {self.run_overview.get('failed_check_count', 0)} "
            f"execution_issue_count {self.run_overview.get('execution_issue_count', 0)} "
            f"total_issues {self.run_overview.get('total_issues', 0)} "
            f"severity_counts {severity_parts or 'none'}"
        )

    def _render_failed_checks(self) -> str:
        if not self.failed_check_details:
            return "none"
        return " | ".join(
            (
                f"{item['name']} severity {item['highest_severity']} "
                f"issue_count {item['issue_count']} "
                f"execution_issue_count {item['execution_issue_count']}"
            )
            for item in self.failed_check_details
        )

    def _render_top_columns(self) -> str:
        if not self.top_column_details:
            return "none"
        return " | ".join(
            (
                f"{item['column']} issue_count {item['issue_count']} "
                f"highest_severity {item['highest_severity']}"
            )
            for item in self.top_column_details
        )

    def _render_baseline_summary(self) -> str:
        if not self.baseline_summary:
            return "none"
        columns = self.baseline_summary.get("columns", [])
        parts = [f"column_count {self.baseline_summary.get('column_count', len(columns))}"]
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
            if "min_length" in column:
                column_parts.append(f"min_length {column['min_length']}")
            if "max_length" in column:
                column_parts.append(f"max_length {column['max_length']}")
            parts.append(" ".join(column_parts))
        return " | ".join(parts)

    def _render_history_window(self) -> str:
        statuses = ",".join(self.history_window.get("recent_statuses", []))
        return (
            f"included {self.history_window.get('included', False)} "
            f"history_key {self.history_window.get('history_key', self.history_key)} "
            f"window_size {self.history_window.get('window_size', HISTORY_WINDOW_SIZE)} "
            f"run_count {self.history_window.get('run_count', 0)} "
            f"failure_count {self.history_window.get('failure_count', 0)} "
            f"success_count {self.history_window.get('success_count', 0)} "
            f"latest_run_id {self.history_window.get('latest_run_id', 'none') or 'none'} "
            f"recent_statuses {statuses or 'none'}"
        )


class RunInputResolver:
    """Resolve canonical run analysis inputs without mutating core workspace state."""

    def resolve(
        self,
        *,
        run: ValidationRunResult | None = None,
        run_id: str | None = None,
        context: TruthoundContext | None = None,
    ) -> ResolvedRunInput:
        if run is None and run_id is None:
            raise ValueError("explain_run requires either run or run_id")

        active_context = context or self._context_from_run(run) or get_context()
        loaded_run = self._load_run(run_id, active_context) if run_id else None
        if run is not None and loaded_run is not None and run.run_id != loaded_run.run_id:
            raise ValueError("run and run_id must reference the same persisted run")

        run_result = run or loaded_run
        if run_result is None:
            raise ValueError("failed to resolve run input")

        run_path = self._resolve_run_path(run_result, active_context)
        if not run_path.exists():
            raise FileNotFoundError(f"Persisted run artifact not found for {run_result.run_id}")

        docs_path = self._resolve_docs_path(run_result, active_context)
        metadata = dict(run_result.metadata)
        source_key = str(
            metadata.get("context_source_key")
            or active_context.resolve_source_key(run_result.source)
        )
        history_key = str(
            metadata.get("context_history_key")
            or metadata.get("context_source_key")
            or active_context.resolve_source_key(run_result.source)
        )
        source_fingerprint = metadata.get("context_source_fingerprint")
        if source_fingerprint is not None:
            source_fingerprint = str(source_fingerprint)

        return ResolvedRunInput(
            context=active_context,
            run_result=run_result,
            run_path=run_path,
            docs_path=docs_path,
            source_key=source_key,
            history_key=history_key,
            source_fingerprint=source_fingerprint,
        )

    def _context_from_run(self, run: ValidationRunResult | None) -> TruthoundContext | None:
        if run is None:
            return None
        root = run.metadata.get("context_root")
        if isinstance(root, str) and root:
            root_path = Path(root)
            if root_path.exists():
                return TruthoundContext(root_path)
        return None

    def _load_run(self, run_id: str, context: TruthoundContext) -> ValidationRunResult:
        run_path = context.runs_dir / f"{run_id}.json"
        if not run_path.exists():
            raise FileNotFoundError(f"Persisted run artifact not found for {run_id}")
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Persisted run artifact is not a JSON object: {run_id}")
        return ValidationRunResult.from_dict(payload)

    def _resolve_run_path(self, run_result: ValidationRunResult, context: TruthoundContext) -> Path:
        metadata_path = run_result.metadata.get("context_run_artifact")
        if isinstance(metadata_path, str) and metadata_path:
            candidate = Path(metadata_path)
            if candidate.exists():
                return candidate
        return context.runs_dir / f"{run_result.run_id}.json"

    def _resolve_docs_path(
        self,
        run_result: ValidationRunResult,
        context: TruthoundContext,
    ) -> Path | None:
        metadata_path = run_result.metadata.get("context_docs_artifact")
        if isinstance(metadata_path, str) and metadata_path:
            candidate = Path(metadata_path)
            if candidate.exists():
                return candidate
        fallback = context.docs_dir / f"{run_result.run_id}.html"
        return fallback if fallback.exists() else None


class RunEvidenceExtractor:
    """Build deterministic run analysis evidence without provider involvement."""

    def build_run_overview(self, run_result: ValidationRunResult) -> dict[str, Any]:
        presentation = build_run_presentation(run_result)
        return {
            "run_id": run_result.run_id,
            "status": presentation.status,
            "suite_name": run_result.suite_name,
            "row_count": run_result.row_count,
            "column_count": run_result.column_count,
            "total_checks": presentation.summary.total_checks,
            "failed_check_count": presentation.summary.failed_checks,
            "execution_issue_count": presentation.summary.total_execution_issues,
            "total_issues": presentation.summary.total_issues,
            "severity_counts": dict(presentation.issue_counts_by_severity),
        }

    def extract_failed_checks(
        self,
        run_result: ValidationRunResult,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        failures: dict[str, dict[str, Any]] = {}
        for check in run_result.checks:
            if check.success:
                continue
            highest_rank = 0
            highest_severity = "critical"
            if check.issues:
                severity_values = [
                    str(issue.severity.value if hasattr(issue.severity, "value") else issue.severity)
                    for issue in check.issues
                ]
                highest_severity = min(
                    severity_values,
                    key=lambda item: _SEVERITY_ORDER.get(item, 99),
                )
                highest_rank = _SEVERITY_ORDER.get(highest_severity, 99)
            failures[check.name] = {
                "name": check.name,
                "issue_count": int(check.issue_count),
                "execution_issue_count": 0,
                "highest_severity": highest_severity,
                "highest_severity_rank": highest_rank,
            }

        for execution_issue in run_result.execution_issues:
            name = execution_issue.check_name or "unknown"
            item = failures.setdefault(
                name,
                {
                    "name": name,
                    "issue_count": 0,
                    "execution_issue_count": 0,
                    "highest_severity": "critical",
                    "highest_severity_rank": 0,
                },
            )
            item["execution_issue_count"] += 1
            item["highest_severity"] = "critical"
            item["highest_severity_rank"] = 0

        ordered = sorted(
            failures.values(),
            key=lambda item: (
                item["highest_severity_rank"],
                -(item["issue_count"] + item["execution_issue_count"]),
                item["name"],
            ),
        )
        return [item["name"] for item in ordered], ordered

    def extract_top_columns(
        self,
        run_result: ValidationRunResult,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        presentation = build_run_presentation(run_result)
        counts = dict(presentation.issue_counts_by_column)
        highest_severity_by_column: dict[str, str] = {}
        for issue in run_result.issues:
            column = issue.column or "_table_"
            severity = str(issue.severity.value if hasattr(issue.severity, "value") else issue.severity)
            previous = highest_severity_by_column.get(column)
            if previous is None or _SEVERITY_ORDER.get(severity, 99) < _SEVERITY_ORDER.get(previous, 99):
                highest_severity_by_column[column] = severity

        if any(column != "_table_" for column in counts):
            counts.pop("_table_", None)

        ordered = sorted(
            (
                {
                    "column": column,
                    "issue_count": count,
                    "highest_severity": highest_severity_by_column.get(column, "low"),
                    "highest_severity_rank": _SEVERITY_ORDER.get(
                        highest_severity_by_column.get(column, "low"),
                        99,
                    ),
                }
                for column, count in counts.items()
            ),
            key=lambda item: (
                -item["issue_count"],
                item["highest_severity_rank"],
                item["column"],
            ),
        )[:5]
        return [item["column"] for item in ordered], ordered

    def load_baseline_summary(
        self,
        resolved: ResolvedRunInput,
    ) -> tuple[dict[str, Any] | None, str | None]:
        index_path = resolved.context.baseline_index_path
        if not index_path.exists():
            return None, None
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return None, None
        if not isinstance(payload, dict):
            return None, None

        baseline_key = self._resolve_baseline_key(resolved, payload)
        if baseline_key is None:
            return None, None

        entry = payload.get(baseline_key)
        if not isinstance(entry, dict):
            return None, None
        schema_file = entry.get("schema_file")
        if not isinstance(schema_file, str):
            return None, None
        schema_path = resolved.context.baselines_dir / schema_file
        if not schema_path.exists():
            return None, None

        try:
            schema = Schema.load(schema_path)
        except Exception:
            return None, None
        return self._build_baseline_summary(schema), baseline_key

    def build_history_window(
        self,
        *,
        context: TruthoundContext,
        history_key: str,
        include_history: bool,
        window_size: int = HISTORY_WINDOW_SIZE,
    ) -> dict[str, Any]:
        if not include_history:
            return self._empty_history_window(history_key=history_key, window_size=window_size, included=False)

        history_path = context.baselines_dir / "metric-history.json"
        if not history_path.exists():
            return self._empty_history_window(history_key=history_key, window_size=window_size, included=True)

        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        entries = payload.get(history_key, []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            entries = []
        recent = [item for item in entries if isinstance(item, dict)][-window_size:]
        failure_count = sum(1 for item in recent if item.get("status") == "failure")
        run_count = len(recent)
        return {
            "included": True,
            "history_key": history_key,
            "window_size": window_size,
            "run_count": run_count,
            "failure_count": failure_count,
            "success_count": max(run_count - failure_count, 0),
            "latest_run_id": recent[-1].get("run_id") if recent else None,
            "recent_statuses": [str(item.get("status", "unknown")) for item in recent],
        }

    def _resolve_baseline_key(
        self,
        resolved: ResolvedRunInput,
        baseline_index: dict[str, Any],
    ) -> str | None:
        if resolved.source_key in baseline_index:
            return resolved.source_key

        if len(baseline_index) == 1:
            return next(iter(baseline_index.keys()))

        source_prefix = f"{resolved.run_result.source}:"
        candidates = [
            key for key in baseline_index
            if isinstance(key, str) and key.startswith(source_prefix)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _build_baseline_summary(self, schema: Schema) -> dict[str, Any]:
        columns: list[dict[str, Any]] = []
        for name, column in schema.columns.items():
            summary = {
                "name": name,
                "dtype": column.dtype,
                "nullable": column.nullable,
                "unique": column.unique,
            }
            if column.min_value is not None and not isinstance(column.min_value, str):
                summary["min_value"] = column.min_value
            if column.max_value is not None and not isinstance(column.max_value, str):
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

    def _empty_history_window(
        self,
        *,
        history_key: str,
        window_size: int,
        included: bool,
    ) -> dict[str, Any]:
        return {
            "included": included,
            "history_key": history_key,
            "window_size": window_size,
            "run_count": 0,
            "failure_count": 0,
            "success_count": 0,
            "latest_run_id": None,
            "recent_statuses": [],
        }


class RunAnalysisContextBundleBuilder:
    """Build the read-only evidence bundle for canonical run analyses."""

    def __init__(self, *, history_window_size: int = HISTORY_WINDOW_SIZE) -> None:
        self._history_window_size = max(1, int(history_window_size))
        self._resolver = RunInputResolver()
        self._extractor = RunEvidenceExtractor()

    def build(
        self,
        *,
        run: ValidationRunResult | None = None,
        run_id: str | None = None,
        context: TruthoundContext | None = None,
        include_history: bool = True,
    ) -> RunAnalysisContextBundle:
        resolved = self._resolver.resolve(run=run, run_id=run_id, context=context)
        run_overview = self._extractor.build_run_overview(resolved.run_result)
        failed_checks, failed_check_details = self._extractor.extract_failed_checks(resolved.run_result)
        top_columns, top_column_details = self._extractor.extract_top_columns(resolved.run_result)
        baseline_summary, baseline_key = self._extractor.load_baseline_summary(resolved)
        history_window = self._extractor.build_history_window(
            context=resolved.context,
            history_key=resolved.history_key,
            include_history=include_history,
            window_size=self._history_window_size,
        )

        input_refs = [
            self._make_input_ref(
                kind="run_result",
                ref=f"runs:{resolved.run_result.run_id}",
                payload={
                    "run_id": resolved.run_result.run_id,
                    "status": run_overview["status"],
                    "issue_count": len(resolved.run_result.issues),
                    "failed_check_count": len(failed_checks),
                },
                metadata={
                    "status": run_overview["status"],
                    "issue_count": len(resolved.run_result.issues),
                    "failed_check_count": len(failed_checks),
                },
            )
        ]
        docs_ref: str | None = None
        if resolved.docs_path is not None:
            docs_ref = f"docs:{resolved.run_result.run_id}"
            input_refs.append(
                self._make_input_ref(
                    kind="docs_artifact",
                    ref=docs_ref,
                    payload={
                        "run_id": resolved.run_result.run_id,
                        "available": True,
                    },
                    metadata={"available": True},
                )
            )
        if baseline_summary is not None and baseline_key is not None:
            input_refs.append(
                self._make_input_ref(
                    kind="baseline_summary",
                    ref=f"baseline-summary:{baseline_key}",
                    payload=baseline_summary,
                    metadata={"column_count": baseline_summary["column_count"]},
                )
            )
        input_refs.append(
            self._make_input_ref(
                kind="history_window",
                ref=f"history-window:{resolved.history_key}",
                payload=history_window,
                metadata={
                    "included": history_window["included"],
                    "run_count": history_window["run_count"],
                    "failure_count": history_window["failure_count"],
                },
            )
        )

        return RunAnalysisContextBundle(
            run_result=resolved.run_result,
            source_key=resolved.source_key,
            history_key=resolved.history_key,
            workspace_root=str(resolved.context.root_dir),
            input_refs=tuple(input_refs),
            run_overview=run_overview,
            failed_checks=failed_checks,
            failed_check_details=failed_check_details,
            top_columns=top_columns,
            top_column_details=top_column_details,
            baseline_summary=baseline_summary,
            history_window=history_window,
            available_evidence_refs=tuple(item.ref for item in input_refs),
            docs_ref=docs_ref,
        )

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


def explain_run(
    run: ValidationRunResult | None = None,
    run_id: str | None = None,
    context: TruthoundContext | None = None,
    include_history: bool = True,
    provider: AIProvider | ProviderConfig | None = None,
    model: str | None = None,
    redact: str = "summary_only",
) -> RunAnalysisArtifact:
    """Produce a canonical persisted run analysis artifact."""

    if redact != "summary_only":
        raise ValueError("Phase 2 supports only redact='summary_only'")

    bundle = RunAnalysisContextBundleBuilder().build(
        run=run,
        run_id=run_id,
        context=context,
        include_history=include_history,
    )
    resolved_provider = resolve_provider(provider)
    resolved_model_name = resolve_model_name(
        model=model,
        provider=provider,
        resolved_provider=resolved_provider,
    )

    system_prompt = bundle.build_system_prompt()
    user_prompt = bundle.build_user_prompt()
    provider_request = StructuredProviderRequest(
        provider_name=resolved_provider.provider_name,
        model_name=resolved_model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format_name="run_analysis",
        response_model=RunAnalysisLLMResponse,
        input_refs=list(bundle.input_refs),
        metadata={
            "artifact_type": "run_analysis",
            "run_id": bundle.run_result.run_id,
            "source_key": bundle.source_key,
            "include_history": include_history,
        },
    )

    response = resolved_provider.generate_structured(provider_request)
    try:
        if isinstance(response.parsed_output, dict):
            llm_response = RunAnalysisLLMResponse.model_validate(response.parsed_output)
        else:
            llm_response = RunAnalysisLLMResponse.model_validate_json(response.output_text)
        artifact = RunAnalysisArtifact(
            artifact_id=f"run-analysis-{bundle.run_result.run_id}",
            source_key=bundle.source_key,
            input_refs=list(bundle.input_refs),
            model_provider=response.provider_name,
            model_name=response.model_name,
            prompt_hash=_hash_prompt(system_prompt, user_prompt),
            created_by="truthound.ai.explain_run",
            workspace_root=bundle.workspace_root,
            run_id=bundle.run_result.run_id,
            summary=llm_response.summary,
            evidence_refs=llm_response.evidence_refs,
            failed_checks=list(bundle.failed_checks),
            top_columns=list(bundle.top_columns),
            recommended_next_actions=list(llm_response.recommended_next_actions),
            history_window=dict(bundle.history_window),
        )
    except Exception as exc:
        raise ProviderResponseError(
            f"Run analysis response validation failed: {exc}"
        ) from exc

    active_context = context or RunInputResolver().resolve(
        run=bundle.run_result,
        context=context,
    ).context
    AIArtifactStore(active_context).write_analysis(artifact)
    return artifact


def _hash_prompt(system_prompt: str, user_prompt: str) -> str:
    digest = hashlib.sha256()
    digest.update(system_prompt.encode("utf-8"))
    digest.update(b"\n")
    digest.update(user_prompt.encode("utf-8"))
    return digest.hexdigest()[:16]


__all__ = [
    "RunAnalysisContextBundle",
    "RunAnalysisContextBundleBuilder",
    "RunEvidenceExtractor",
    "RunInputResolver",
    "explain_run",
]
