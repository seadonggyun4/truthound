"""Process-local observability counters for AI prompt normalization.

The counters in this module are intentionally small and redacted.  They record
rollout modes, reason codes, and aggregate counts only; raw prompts, provider
outputs, sample rows, and API keys must never be stored here.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Any

from truthound.ai.models import CompileStatus

_METRICS_LOCK = threading.Lock()
_COUNT_KEYS = (
    "ai_prompt_normalization_requests_total",
    "ai_prompt_normalization_actionable_total",
    "ai_prompt_normalization_clarification_total",
    "ai_prompt_normalization_unicode_warning_total",
    "ai_prompt_normalization_candidate_total",
    "ai_prompt_normalization_unresolved_total",
    "ai_prompt_normalization_invalid_mode_total",
)
_MAP_KEYS = (
    "ai_prompt_normalization_modes",
    "ai_prompt_normalization_languages",
    "ai_prompt_candidate_intents",
    "ai_prompt_clarification_reasons",
    "ai_prompt_unresolved_reasons",
    "ai_prompt_unicode_warning_reasons",
    "ai_proposal_compile_statuses",
    "ai_proposal_rejection_sources",
    "ai_proposal_rejection_reasons",
    "ai_prompt_normalization_invalid_modes",
)
_counts: Counter[str] = Counter({key: 0 for key in _COUNT_KEYS})
_maps: dict[str, Counter[str]] = {key: Counter() for key in _MAP_KEYS}


def get_ai_prompt_metrics_snapshot() -> dict[str, Any]:
    """Return a redacted process-local snapshot of AI prompt metrics."""

    with _METRICS_LOCK:
        snapshot: dict[str, Any] = {key: int(_counts.get(key, 0)) for key in _COUNT_KEYS}
        snapshot.update({key: dict(sorted(counter.items())) for key, counter in _maps.items()})
    snapshot["ai_prompt_normalization_mode"] = _resolve_mode_for_reporting()
    return snapshot


def reset_ai_prompt_metrics() -> None:
    """Reset process-local AI prompt metrics for tests and smoke runs."""

    with _METRICS_LOCK:
        _counts.clear()
        _counts.update({key: 0 for key in _COUNT_KEYS})
        for counter in _maps.values():
            counter.clear()


def record_prompt_normalization_result(normalized_prompt: Any) -> None:
    """Record a normalized prompt result without storing raw prompt content."""

    mode = _enum_or_string(getattr(normalized_prompt, "mode", "unknown")) or "unknown"
    language = _safe_label(getattr(normalized_prompt, "language", None), default="unknown")
    candidates = list(getattr(normalized_prompt, "candidates", []) or [])
    unresolved_terms = list(getattr(normalized_prompt, "unresolved_terms", []) or [])
    unicode_warnings = list(getattr(normalized_prompt, "unicode_warnings", []) or [])
    clarification = getattr(normalized_prompt, "clarification", None)
    actionable = bool(getattr(normalized_prompt, "actionable", False))

    with _METRICS_LOCK:
        _counts["ai_prompt_normalization_requests_total"] += 1
        _counts["ai_prompt_normalization_candidate_total"] += len(candidates)
        _counts["ai_prompt_normalization_unresolved_total"] += len(unresolved_terms)
        if actionable:
            _counts["ai_prompt_normalization_actionable_total"] += 1
        if clarification is not None:
            _counts["ai_prompt_normalization_clarification_total"] += 1
            _maps["ai_prompt_clarification_reasons"][
                _safe_label(getattr(clarification, "reason", None), default="unknown")
            ] += 1
        if unicode_warnings:
            _counts["ai_prompt_normalization_unicode_warning_total"] += 1

        _maps["ai_prompt_normalization_modes"][mode] += 1
        _maps["ai_prompt_normalization_languages"][language] += 1
        for candidate in candidates:
            _maps["ai_prompt_candidate_intents"][
                _safe_label(getattr(candidate, "intent", None), default="unknown")
            ] += 1
        for term in unresolved_terms:
            _maps["ai_prompt_unresolved_reasons"][
                _safe_label(getattr(term, "reason", None), default="unknown")
            ] += 1
        for warning in unicode_warnings:
            _maps["ai_prompt_unicode_warning_reasons"][_safe_label(warning, default="unknown")] += 1


def record_proposal_compilation(artifact: Any) -> None:
    """Record proposal compile status and rejection reason codes."""

    status = _enum_or_string(getattr(artifact, "compile_status", CompileStatus.REJECTED))
    rejected_items = list(getattr(artifact, "rejected_items", []) or [])
    compiler_errors = list(getattr(artifact, "compiler_errors", []) or [])

    with _METRICS_LOCK:
        _maps["ai_proposal_compile_statuses"][status] += 1
        for item in rejected_items:
            _maps["ai_proposal_rejection_sources"][
                _safe_label(getattr(item, "source", None), default="unknown")
            ] += 1
            _maps["ai_proposal_rejection_reasons"][
                _safe_label(getattr(item, "reason", None), default="unknown")
            ] += 1
        for error in compiler_errors:
            _maps["ai_proposal_rejection_sources"]["compiler"] += 1
            _maps["ai_proposal_rejection_reasons"][_safe_label(error, default="unknown")] += 1


def record_prompt_normalization_invalid_mode(raw_value: str) -> None:
    """Record invalid rollout mode values without changing fallback behavior."""

    label = _safe_label(raw_value, default="empty")
    with _METRICS_LOCK:
        _counts["ai_prompt_normalization_invalid_mode_total"] += 1
        _maps["ai_prompt_normalization_invalid_modes"][label] += 1


def _resolve_mode_for_reporting() -> str:
    raw = os.getenv("TRUTHOUND_AI_PROMPT_NORMALIZATION", "enforce")
    value = str(raw).strip().lower()
    if value in {"off", "shadow", "enforce"}:
        return value
    return "enforce"


def _enum_or_string(value: Any) -> str:
    return _safe_label(getattr(value, "value", value), default="unknown")


def _safe_label(value: Any, *, default: str) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return default
    return text[:120]


__all__ = [
    "get_ai_prompt_metrics_snapshot",
    "record_prompt_normalization_invalid_mode",
    "record_prompt_normalization_result",
    "record_proposal_compilation",
    "reset_ai_prompt_metrics",
]
