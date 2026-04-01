"""Deterministic compiler for Phase 1 suite proposal artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from truthound._ai_redaction import SummaryOnlyRedactor
from truthound.ai.context import ContextBundle
from truthound.ai.models import (
    CompileStatus,
    CompiledProposalCheck,
    ProposedCheckIntent,
    RejectedProposalItem,
    SuiteProposalArtifact,
    SuiteProposalLLMResponse,
    ValidationSuiteDiffCounts,
    ValidationSuiteDiffPreview,
)
from truthound.ai.suite_diff import build_formal_suite_diff
from truthound.validators import get_validator
from truthound.validators.base import RegexSafetyChecker

MAX_IN_SET_VALUES = 50
SUPPORTED_FORMAT_KINDS = {
    "email": "email",
    "url": "url",
    "phone": "phone",
    "uuid": "uuid",
    "ip_address": "ip_address",
    "ipv6_address": "ipv6_address",
}
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
class CompiledIntentResult:
    check: CompiledProposalCheck | None = None
    rejected_item: RejectedProposalItem | None = None


class ProposalCompiler:
    """Compile structured LLM output into a persisted suite proposal artifact."""

    def __init__(self) -> None:
        self._redactor = SummaryOnlyRedactor()

    def compile_artifact(
        self,
        *,
        response: SuiteProposalLLMResponse,
        context_bundle: ContextBundle,
        model_provider: str,
        model_name: str,
        prompt_hash: str,
        created_by: str = "truthound.ai.suggest_suite",
    ) -> SuiteProposalArtifact:
        compiled_checks: list[CompiledProposalCheck] = []
        rejected_items: list[RejectedProposalItem] = [
            RejectedProposalItem(
                source="model",
                intent="model_rejected_request",
                reason=reason,
            )
            for reason in response.rejected_requests
        ]
        compiler_errors: list[str] = []

        for intent in response.proposed_checks:
            result = self._compile_intent(intent, context_bundle=context_bundle)
            if result.check is not None:
                compiled_checks.append(result.check)
            if result.rejected_item is not None:
                rejected_items.append(result.rejected_item)

        diff_result = build_formal_suite_diff(
            current_suite=context_bundle.current_suite,
            compiled_checks=compiled_checks,
            rejected_items=rejected_items,
        )
        compiled_checks = diff_result.compiled_checks
        rejected_items = diff_result.rejected_items
        diff_preview = diff_result.diff_preview
        compile_status = self._resolve_compile_status(
            compiled_count=len(compiled_checks),
            rejected_count=len(rejected_items),
            compiler_errors=compiler_errors,
        )

        return SuiteProposalArtifact(
            source_key=context_bundle.source_key,
            input_refs=list(context_bundle.input_refs),
            model_provider=model_provider,
            model_name=model_name,
            prompt_hash=prompt_hash,
            created_by=created_by,
            workspace_root=context_bundle.workspace_root,
            summary=response.summary,
            rationale=response.rationale,
            checks=compiled_checks,
            risks=response.risks,
            compile_status=compile_status,
            diff_preview=diff_preview,
            rejected_items=rejected_items,
            compiled_check_count=len(compiled_checks),
            rejected_check_count=len(rejected_items),
            compiler_errors=compiler_errors,
        )

    def build_rejected_artifact(
        self,
        *,
        context_bundle: ContextBundle,
        model_provider: str,
        model_name: str,
        prompt_hash: str,
        error_code: str,
        created_by: str = "truthound.ai.suggest_suite",
    ) -> SuiteProposalArtifact:
        diff_preview = ValidationSuiteDiffPreview(
            current_suite=context_bundle.current_suite_snapshot,
            proposed_suite=context_bundle.current_suite_snapshot,
            counts=ValidationSuiteDiffCounts(),
        )
        return SuiteProposalArtifact(
            source_key=context_bundle.source_key,
            input_refs=list(context_bundle.input_refs),
            model_provider=model_provider,
            model_name=model_name,
            prompt_hash=prompt_hash,
            created_by=created_by,
            workspace_root=context_bundle.workspace_root,
            summary="AI proposal could not be compiled.",
            rationale="The provider returned a malformed or unsupported proposal payload.",
            checks=[],
            risks=[],
            compile_status=CompileStatus.REJECTED,
            diff_preview=diff_preview,
            rejected_items=[],
            compiled_check_count=0,
            rejected_check_count=0,
            compiler_errors=[error_code],
        )

    def _compile_intent(
        self,
        intent: ProposedCheckIntent,
        *,
        context_bundle: ContextBundle,
    ) -> CompiledIntentResult:
        try:
            compiled = self._compile_supported_intent(intent, context_bundle=context_bundle)
            return CompiledIntentResult(check=compiled)
        except ValueError as exc:
            return CompiledIntentResult(
                rejected_item=RejectedProposalItem(
                    source="compiler",
                    intent=intent.intent,
                    columns=list(intent.columns),
                    params=dict(intent.params),
                    reason=str(exc),
                    rationale=intent.rationale or None,
                )
            )

    def _compile_supported_intent(
        self,
        intent: ProposedCheckIntent,
        *,
        context_bundle: ContextBundle,
    ) -> CompiledProposalCheck:
        normalized_intent = intent.intent.strip().lower()
        columns = self._normalize_columns(intent.columns, context_bundle=context_bundle)
        params = dict(intent.params)

        if normalized_intent in {"null", "not_null"}:
            self._ensure_only_allowed_params(params, allowed=())
            return self._build_check(
                validator_name=normalized_intent,
                columns=columns,
                params={},
                rationale=intent.rationale,
            )

        if normalized_intent == "completeness_ratio":
            self._ensure_columns_required(columns, normalized_intent)
            self._ensure_string_or_numeric_columns(columns, context_bundle=context_bundle, intent=normalized_intent, allow_any=True)
            min_ratio = self._coerce_ratio(params.pop("min_ratio", None), required=True, label="min_ratio")
            self._ensure_only_allowed_params(params, allowed=())
            return self._build_check(
                validator_name="completeness_ratio",
                columns=columns,
                params={"min_ratio": min_ratio},
                rationale=intent.rationale,
            )

        if normalized_intent == "unique":
            self._ensure_columns_required(columns, normalized_intent)
            self._ensure_only_allowed_params(params, allowed=())
            return self._build_check(
                validator_name="unique",
                columns=columns,
                params={},
                rationale=intent.rationale,
            )

        if normalized_intent == "unique_ratio":
            self._ensure_columns_required(columns, normalized_intent)
            min_ratio = self._coerce_ratio(params.pop("min_ratio", None), required=False, label="min_ratio")
            max_ratio = self._coerce_ratio(params.pop("max_ratio", None), required=False, label="max_ratio")
            if min_ratio is None and max_ratio is None:
                raise ValueError("unique_ratio requires min_ratio or max_ratio")
            if min_ratio is not None and max_ratio is not None and min_ratio > max_ratio:
                raise ValueError("unique_ratio min_ratio cannot exceed max_ratio")
            self._ensure_only_allowed_params(params, allowed=())
            normalized_params: dict[str, Any] = {}
            if min_ratio is not None:
                normalized_params["min_ratio"] = min_ratio
            if max_ratio is not None:
                normalized_params["max_ratio"] = max_ratio
            return self._build_check(
                validator_name="unique_ratio",
                columns=columns,
                params=normalized_params,
                rationale=intent.rationale,
            )

        if normalized_intent == "between":
            self._ensure_columns_required(columns, normalized_intent)
            self._ensure_numeric_columns(columns, context_bundle=context_bundle, intent=normalized_intent)
            min_value = self._coerce_number(params.pop("min_value", None), label="min_value")
            max_value = self._coerce_number(params.pop("max_value", None), label="max_value")
            inclusive = params.pop("inclusive", True)
            if min_value is None and max_value is None:
                raise ValueError("between requires min_value or max_value")
            if min_value is not None and max_value is not None and min_value > max_value:
                raise ValueError("between min_value cannot exceed max_value")
            if not isinstance(inclusive, bool):
                raise ValueError("between inclusive must be a boolean")
            self._ensure_only_allowed_params(params, allowed=())
            normalized_params = {"inclusive": inclusive}
            if min_value is not None:
                normalized_params["min_value"] = min_value
            if max_value is not None:
                normalized_params["max_value"] = max_value
            return self._build_check(
                validator_name="between",
                columns=columns,
                params=normalized_params,
                rationale=intent.rationale,
            )

        if normalized_intent == "in_set":
            self._ensure_columns_required(columns, normalized_intent)
            allowed_values = params.pop("allowed_values", None)
            if not isinstance(allowed_values, list) or not allowed_values:
                raise ValueError("in_set requires a non-empty allowed_values list")
            if len(allowed_values) > MAX_IN_SET_VALUES:
                raise ValueError(f"in_set allowed_values cannot exceed {MAX_IN_SET_VALUES} entries")
            for value in allowed_values:
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    raise ValueError("in_set allowed_values must contain only scalar JSON values")
            self._redactor.assert_safe({"allowed_values": allowed_values}, label="in_set allowed_values")
            self._ensure_only_allowed_params(params, allowed=())
            return self._build_check(
                validator_name="in_set",
                columns=columns,
                params={"allowed_values": allowed_values},
                rationale=intent.rationale,
            )

        if normalized_intent == "length":
            self._ensure_columns_required(columns, normalized_intent)
            self._ensure_string_columns(columns, context_bundle=context_bundle, intent=normalized_intent)
            min_length = self._coerce_int(params.pop("min_length", None), label="min_length")
            max_length = self._coerce_int(params.pop("max_length", None), label="max_length")
            exact_length = self._coerce_int(params.pop("exact_length", None), label="exact_length")
            if min_length is None and max_length is None and exact_length is None:
                raise ValueError("length requires min_length, max_length, or exact_length")
            if exact_length is not None and (min_length is not None or max_length is not None):
                raise ValueError("length exact_length cannot be combined with min_length or max_length")
            if min_length is not None and max_length is not None and min_length > max_length:
                raise ValueError("length min_length cannot exceed max_length")
            self._ensure_only_allowed_params(params, allowed=())
            normalized_params: dict[str, Any] = {}
            if min_length is not None:
                normalized_params["min_length"] = min_length
            if max_length is not None:
                normalized_params["max_length"] = max_length
            if exact_length is not None:
                normalized_params["exact_length"] = exact_length
            return self._build_check(
                validator_name="length",
                columns=columns,
                params=normalized_params,
                rationale=intent.rationale,
            )

        if normalized_intent == "format":
            self._ensure_columns_required(columns, normalized_intent)
            self._ensure_string_columns(columns, context_bundle=context_bundle, intent=normalized_intent)
            format_kind = params.pop("format", params.pop("kind", None))
            if not isinstance(format_kind, str):
                raise ValueError("format requires a string format or kind parameter")
            validator_name = SUPPORTED_FORMAT_KINDS.get(format_kind.strip().lower())
            if validator_name is None:
                supported = ", ".join(sorted(SUPPORTED_FORMAT_KINDS))
                raise ValueError(f"format kind must be one of {supported}")
            self._ensure_only_allowed_params(params, allowed=())
            return self._build_check(
                validator_name=validator_name,
                columns=columns,
                params={},
                rationale=intent.rationale,
            )

        if normalized_intent == "regex":
            self._ensure_columns_required(columns, normalized_intent)
            self._ensure_string_columns(columns, context_bundle=context_bundle, intent=normalized_intent)
            pattern = params.pop("pattern", None)
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError("regex requires a non-empty pattern string")
            safe, error = RegexSafetyChecker.check_pattern(pattern)
            if not safe:
                raise ValueError(f"regex pattern is not allowed: {error}")
            if self._looks_like_nested_quantifier_pattern(pattern):
                raise ValueError("regex pattern is not allowed: nested quantifiers are unsafe")
            match_full = params.pop("match_full", True)
            dotall = params.pop("dotall", True)
            case_insensitive = params.pop("case_insensitive", False)
            if not all(isinstance(item, bool) for item in (match_full, dotall, case_insensitive)):
                raise ValueError("regex match_full, dotall, and case_insensitive must be booleans")
            self._redactor.assert_safe({"pattern": pattern, "rationale": intent.rationale}, label="regex proposal")
            self._ensure_only_allowed_params(params, allowed=())
            return self._build_check(
                validator_name="regex",
                columns=columns,
                params={
                    "pattern": pattern,
                    "match_full": match_full,
                    "dotall": dotall,
                    "case_insensitive": case_insensitive,
                },
                rationale=intent.rationale,
            )

        if normalized_intent == "mean_between":
            self._ensure_columns_required(columns, normalized_intent)
            self._ensure_numeric_columns(columns, context_bundle=context_bundle, intent=normalized_intent)
            min_value = self._coerce_number(params.pop("min_value", None), label="min_value")
            max_value = self._coerce_number(params.pop("max_value", None), label="max_value")
            if min_value is None and max_value is None:
                raise ValueError("mean_between requires min_value or max_value")
            if min_value is not None and max_value is not None and min_value > max_value:
                raise ValueError("mean_between min_value cannot exceed max_value")
            self._ensure_only_allowed_params(params, allowed=())
            normalized_params = {}
            if min_value is not None:
                normalized_params["min_value"] = min_value
            if max_value is not None:
                normalized_params["max_value"] = max_value
            return self._build_check(
                validator_name="mean_between",
                columns=columns,
                params=normalized_params,
                rationale=intent.rationale,
            )

        if normalized_intent == "sum_between":
            self._ensure_columns_required(columns, normalized_intent)
            self._ensure_numeric_columns(columns, context_bundle=context_bundle, intent=normalized_intent)
            min_value = self._coerce_number(params.pop("min_value", None), label="min_value")
            max_value = self._coerce_number(params.pop("max_value", None), label="max_value")
            if min_value is None and max_value is None:
                raise ValueError("sum_between requires min_value or max_value")
            if min_value is not None and max_value is not None and min_value > max_value:
                raise ValueError("sum_between min_value cannot exceed max_value")
            self._ensure_only_allowed_params(params, allowed=())
            normalized_params = {}
            if min_value is not None:
                normalized_params["min_value"] = min_value
            if max_value is not None:
                normalized_params["max_value"] = max_value
            return self._build_check(
                validator_name="sum_between",
                columns=columns,
                params=normalized_params,
                rationale=intent.rationale,
            )

        raise ValueError(f"unsupported intent {normalized_intent!r}")

    def _build_check(
        self,
        *,
        validator_name: str,
        columns: tuple[str, ...],
        params: dict[str, Any],
        rationale: str,
    ) -> CompiledProposalCheck:
        validator_cls = get_validator(validator_name)
        category = getattr(validator_cls, "category", "general")
        self._redactor.assert_safe(
            {
                "validator_name": validator_name,
                "columns": list(columns),
                "params": params,
                "rationale": rationale,
            },
            label="compiled proposal check",
        )
        return CompiledProposalCheck(
            check_key=self._check_key(
                validator_name=validator_name,
                columns=columns,
                params=params,
            ),
            validator_name=validator_name,
            category=str(category),
            columns=list(columns),
            params=params,
            rationale=rationale,
        )

    def _check_key(
        self,
        *,
        validator_name: str,
        columns: tuple[str, ...],
        params: dict[str, Any],
    ) -> str:
        normalized_params = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        return f"{validator_name}|{','.join(sorted(columns))}|{normalized_params}"

    def _normalize_columns(
        self,
        columns: list[str],
        *,
        context_bundle: ContextBundle,
    ) -> tuple[str, ...]:
        known_columns = {
            str(item.get("name"))
            for item in context_bundle.schema_summary.get("columns", [])
            if isinstance(item, dict) and item.get("name")
        }
        normalized: list[str] = []
        for column in columns:
            column_name = str(column).strip()
            if not column_name:
                continue
            if column_name not in known_columns:
                raise ValueError(f"unknown column {column_name!r}")
            if column_name not in normalized:
                normalized.append(column_name)
        return tuple(normalized)

    def _ensure_columns_required(self, columns: tuple[str, ...], intent: str) -> None:
        if not columns:
            raise ValueError(f"{intent} requires at least one known column")

    def _ensure_numeric_columns(
        self,
        columns: tuple[str, ...],
        *,
        context_bundle: ContextBundle,
        intent: str,
    ) -> None:
        self._ensure_columns_by_dtype(
            columns,
            context_bundle=context_bundle,
            intent=intent,
            predicate=lambda dtype: any(marker in dtype for marker in NUMERIC_DTYPE_MARKERS),
            label="numeric",
        )

    def _ensure_string_columns(
        self,
        columns: tuple[str, ...],
        *,
        context_bundle: ContextBundle,
        intent: str,
    ) -> None:
        self._ensure_columns_by_dtype(
            columns,
            context_bundle=context_bundle,
            intent=intent,
            predicate=lambda dtype: any(marker in dtype for marker in STRING_DTYPE_MARKERS),
            label="string",
        )

    def _ensure_string_or_numeric_columns(
        self,
        columns: tuple[str, ...],
        *,
        context_bundle: ContextBundle,
        intent: str,
        allow_any: bool = False,
    ) -> None:
        if allow_any and columns:
            return
        self._ensure_columns_by_dtype(
            columns,
            context_bundle=context_bundle,
            intent=intent,
            predicate=lambda dtype: any(marker in dtype for marker in NUMERIC_DTYPE_MARKERS + STRING_DTYPE_MARKERS),
            label="string or numeric",
        )

    def _ensure_columns_by_dtype(
        self,
        columns: tuple[str, ...],
        *,
        context_bundle: ContextBundle,
        intent: str,
        predicate: Any,
        label: str,
    ) -> None:
        summaries = {
            str(item.get("name")): str(item.get("dtype"))
            for item in context_bundle.schema_summary.get("columns", [])
            if isinstance(item, dict) and item.get("name")
        }
        for column in columns:
            dtype = summaries.get(column, "")
            if not predicate(dtype):
                raise ValueError(f"{intent} requires {label} columns, but {column!r} is {dtype!r}")

    def _ensure_only_allowed_params(self, params: dict[str, Any], *, allowed: tuple[str, ...]) -> None:
        unexpected = sorted(set(params) - set(allowed))
        if unexpected:
            raise ValueError(f"unsupported params: {', '.join(unexpected)}")

    def _coerce_ratio(self, value: Any, *, required: bool, label: str) -> float | None:
        if value is None:
            if required:
                raise ValueError(f"{label} is required")
            return None
        try:
            ratio = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{label} must be between 0.0 and 1.0")
        return round(ratio, 6)

    def _coerce_number(self, value: Any, *, label: str) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc

    def _coerce_int(self, value: Any, *, label: str) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer") from exc
        if parsed < 0:
            raise ValueError(f"{label} must be non-negative")
        return parsed

    def _resolve_compile_status(
        self,
        *,
        compiled_count: int,
        rejected_count: int,
        compiler_errors: list[str],
    ) -> CompileStatus:
        if compiled_count == 0:
            return CompileStatus.REJECTED
        if rejected_count > 0 or compiler_errors:
            return CompileStatus.PARTIAL
        return CompileStatus.READY

    def _looks_like_nested_quantifier_pattern(self, pattern: str) -> bool:
        return bool(re.search(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)(?:[+*]|\{\d+,?\d*\})", pattern))


__all__ = [
    "MAX_IN_SET_VALUES",
    "ProposalCompiler",
    "SUPPORTED_FORMAT_KINDS",
]
