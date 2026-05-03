from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from truthound.ai.normalization import PromptNormalizer

EVAL_RESULT_PATH_ENV = "TRUTHOUND_AI_EVAL_RESULT_PATH"
REQUIRED_CASE_FIELDS = {
    "id",
    "prompt",
    "columns",
    "task_label",
    "intent_label",
    "entity_labels",
    "threshold_labels",
    "ambiguity_label",
    "unicode_labels",
    "expected_outcome",
    "expected_candidates",
    "tags",
}


@dataclass(frozen=True)
class PromptEvaluationCase:
    id: str
    prompt: str
    columns: list[str]
    task_label: str
    intent_label: str
    entity_labels: list[str]
    threshold_labels: dict[str, Any]
    ambiguity_label: str
    unicode_labels: list[str]
    expected_outcome: str
    expected_candidates: list[dict[str, Any]]
    tags: list[str]
    expected_reason: str | None = None

    @property
    def split(self) -> str:
        if "mixed" in self.tags:
            return "mixed"
        if "ambiguous" in self.tags:
            return "ambiguous"
        if "unsupported" in self.tags:
            return "unsupported"
        return "golden"

    def as_contract_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "columns": self.columns,
            "task_label": self.task_label,
            "intent_label": self.intent_label,
            "entity_labels": self.entity_labels,
            "threshold_labels": self.threshold_labels,
            "ambiguity_label": self.ambiguity_label,
            "unicode_labels": self.unicode_labels,
            "expected_outcome": self.expected_outcome,
            "expected_candidates": self.expected_candidates,
            "tags": self.tags,
        }


@dataclass(frozen=True)
class PromptEvaluationFailure:
    case_id: str
    split: str
    expected_outcome: str
    actual_outcome: str
    reason_code: str
    tags: list[str]


@dataclass
class PromptAcceptanceSummary:
    total_count: int
    golden_count: int = 0
    mixed_count: int = 0
    ambiguous_count: int = 0
    unsupported_count: int = 0
    golden_ready_or_partial_count: int = 0
    mixed_ready_or_partial_count: int = 0
    ambiguous_clarification_count: int = 0
    crash_count: int = 0
    tag_counts: dict[str, int] = field(default_factory=dict)
    failures: list[PromptEvaluationFailure] = field(default_factory=list)

    @property
    def golden_ready_or_partial_rate(self) -> float:
        return _ratio(self.golden_ready_or_partial_count, self.golden_count)

    @property
    def mixed_ready_or_partial_rate(self) -> float:
        return _ratio(self.mixed_ready_or_partial_count, self.mixed_count)

    @property
    def ambiguous_clarification_rate(self) -> float:
        return _ratio(self.ambiguous_clarification_count, self.ambiguous_count)

    def model_dump(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "golden_count": self.golden_count,
            "mixed_count": self.mixed_count,
            "ambiguous_count": self.ambiguous_count,
            "unsupported_count": self.unsupported_count,
            "golden_ready_or_partial_count": self.golden_ready_or_partial_count,
            "mixed_ready_or_partial_count": self.mixed_ready_or_partial_count,
            "ambiguous_clarification_count": self.ambiguous_clarification_count,
            "golden_ready_or_partial_rate": self.golden_ready_or_partial_rate,
            "mixed_ready_or_partial_rate": self.mixed_ready_or_partial_rate,
            "ambiguous_clarification_rate": self.ambiguous_clarification_rate,
            "crash_count": self.crash_count,
            "tag_counts": dict(sorted(self.tag_counts.items())),
            "failures": [
                {
                    "case_id": item.case_id,
                    "split": item.split,
                    "expected_outcome": item.expected_outcome,
                    "actual_outcome": item.actual_outcome,
                    "reason_code": item.reason_code,
                    "tags": item.tags,
                }
                for item in self.failures[:25]
            ],
        }


def load_prompt_acceptance_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expand_prompt_acceptance_cases(payload: dict[str, Any]) -> list[PromptEvaluationCase]:
    common_columns = [str(item) for item in payload["common_columns"]]
    cases: list[PromptEvaluationCase] = []
    cases.extend(_expand_candidate_groups(payload.get("candidate_groups", []), common_columns))
    cases.extend(_expand_ambiguous_groups(payload.get("ambiguous_groups", []), common_columns))
    cases.extend(_explicit_cases(payload.get("unsupported_cases", [])))
    _assert_unique_ids(cases)
    _assert_case_contract(cases)
    return cases


def evaluate_prompt_acceptance(cases: list[PromptEvaluationCase]) -> PromptAcceptanceSummary:
    summary = PromptAcceptanceSummary(total_count=len(cases))
    tag_counts: Counter[str] = Counter()

    for case in cases:
        tag_counts.update(case.tags)
        if case.split == "golden":
            summary.golden_count += 1
        elif case.split == "mixed":
            summary.mixed_count += 1
        elif case.split == "ambiguous":
            summary.ambiguous_count += 1
        elif case.split == "unsupported":
            summary.unsupported_count += 1

        try:
            normalized = PromptNormalizer().normalize(case.prompt, columns=case.columns)
            actual_candidates = [
                {
                    "intent": item.intent,
                    "columns": list(item.columns),
                    "params": dict(item.params),
                }
                for item in normalized.candidates
            ]
            actual_outcome, reason_code = _classify_outcome(case, actual_candidates, normalized)
        except Exception as exc:  # pragma: no cover - asserted via crash_count in tests
            summary.crash_count += 1
            actual_outcome = "crash"
            reason_code = exc.__class__.__name__

        if case.split == "golden" and actual_outcome in {"ready", "partial"}:
            summary.golden_ready_or_partial_count += 1
        elif case.split == "mixed" and actual_outcome in {"ready", "partial"}:
            summary.mixed_ready_or_partial_count += 1
        elif case.split == "ambiguous" and actual_outcome == "clarification":
            summary.ambiguous_clarification_count += 1

        if not _case_passed(case, actual_outcome):
            summary.failures.append(
                PromptEvaluationFailure(
                    case_id=case.id,
                    split=case.split,
                    expected_outcome=case.expected_outcome,
                    actual_outcome=actual_outcome,
                    reason_code=reason_code,
                    tags=list(case.tags),
                )
            )

    summary.tag_counts = dict(tag_counts)
    _write_optional_summary(summary)
    return summary


def _expand_candidate_groups(
    groups: list[dict[str, Any]],
    common_columns: list[str],
) -> list[PromptEvaluationCase]:
    cases: list[PromptEvaluationCase] = []
    for group in groups:
        for entity_index, entity in enumerate(group["entities"], start=1):
            for term_index, term in enumerate(entity["terms"], start=1):
                for template_index, template in enumerate(group["prompt_templates"], start=1):
                    case_id = f"{group['id_prefix']}_{entity_index:02d}_{term_index:02d}_{template_index:02d}"
                    column = str(entity["column"])
                    params = dict(group.get("params", {}))
                    cases.append(
                        PromptEvaluationCase(
                            id=case_id,
                            prompt=str(template).format(term=term),
                            columns=common_columns,
                            task_label="validation_intent_normalization",
                            intent_label=str(group["intent"]),
                            entity_labels=[column],
                            threshold_labels=dict(group.get("threshold_labels", params)),
                            ambiguity_label="clear",
                            unicode_labels=["plain"],
                            expected_outcome="ready",
                            expected_candidates=[
                                {
                                    "intent": str(group["intent"]),
                                    "columns": [column],
                                    "params": params,
                                }
                            ],
                            tags=list(group.get("tags", [])),
                        )
                    )
    return cases


def _expand_ambiguous_groups(
    groups: list[dict[str, Any]],
    common_columns: list[str],
) -> list[PromptEvaluationCase]:
    cases: list[PromptEvaluationCase] = []
    for group in groups:
        for domain_index, domain in enumerate(group["domains"], start=1):
            for template_index, template in enumerate(group["prompt_templates"], start=1):
                case_id = f"{group['id_prefix']}_{domain_index:02d}_{template_index:02d}"
                cases.append(
                    PromptEvaluationCase(
                        id=case_id,
                        prompt=str(template).format(domain=domain),
                        columns=common_columns,
                        task_label="ambiguous_prompt_normalization",
                        intent_label="unknown",
                        entity_labels=[],
                        threshold_labels={},
                        ambiguity_label="ambiguous",
                        unicode_labels=["plain"],
                        expected_outcome="clarification",
                        expected_candidates=[],
                        expected_reason=str(group.get("expected_reason", "prompt_too_ambiguous")),
                        tags=list(group.get("tags", [])),
                    )
                )
    return cases


def _explicit_cases(items: list[dict[str, Any]]) -> list[PromptEvaluationCase]:
    cases: list[PromptEvaluationCase] = []
    for item in items:
        cases.append(
            PromptEvaluationCase(
                id=str(item["id"]),
                prompt=str(item["prompt"]),
                columns=[str(column) for column in item["columns"]],
                task_label=str(item["task_label"]),
                intent_label=str(item["intent_label"]),
                entity_labels=[str(label) for label in item["entity_labels"]],
                threshold_labels=dict(item["threshold_labels"]),
                ambiguity_label=str(item["ambiguity_label"]),
                unicode_labels=[str(label) for label in item["unicode_labels"]],
                expected_outcome=str(item["expected_outcome"]),
                expected_candidates=list(item.get("expected_candidates", [])),
                tags=[str(tag) for tag in item["tags"]],
                expected_reason=item.get("expected_reason"),
            )
        )
    return cases


def _classify_outcome(
    case: PromptEvaluationCase,
    actual_candidates: list[dict[str, Any]],
    normalized: Any,
) -> tuple[str, str]:
    if normalized.clarification is not None and not actual_candidates:
        return "clarification", str(normalized.clarification.reason)

    expected = case.expected_candidates
    matched = sum(1 for candidate in expected if candidate in actual_candidates)
    if expected and matched == len(expected):
        return "ready", "expected_candidates_matched"
    if expected and matched > 0:
        return "partial", "some_expected_candidates_matched"
    if actual_candidates:
        return "partial", "unexpected_actionable_candidates"
    return "rejected", "no_actionable_candidates"


def _case_passed(case: PromptEvaluationCase, actual_outcome: str) -> bool:
    if case.expected_outcome == "ready":
        return actual_outcome in {"ready", "partial"}
    if case.expected_outcome == "clarification":
        return actual_outcome == "clarification"
    return actual_outcome == case.expected_outcome


def _assert_unique_ids(cases: list[PromptEvaluationCase]) -> None:
    ids = [case.id for case in cases]
    duplicates = [case_id for case_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise AssertionError(f"duplicate prompt evaluation case ids: {duplicates[:5]}")


def _assert_case_contract(cases: list[PromptEvaluationCase]) -> None:
    for case in cases:
        payload = case.as_contract_payload()
        missing = sorted(REQUIRED_CASE_FIELDS - set(payload))
        if missing:
            raise AssertionError(f"{case.id} missing required fields: {missing}")
        if case.ambiguity_label not in {"clear", "ambiguous", "false_positive", "unicode_risk"}:
            raise AssertionError(f"{case.id} has invalid ambiguity_label={case.ambiguity_label}")


def _write_optional_summary(summary: PromptAcceptanceSummary) -> None:
    configured = os.getenv(EVAL_RESULT_PATH_ENV)
    if not configured:
        return
    path = Path(configured).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def cases_by_split(cases: list[PromptEvaluationCase]) -> dict[str, list[PromptEvaluationCase]]:
    grouped: dict[str, list[PromptEvaluationCase]] = defaultdict(list)
    for case in cases:
        grouped[case.split].append(case)
    return dict(grouped)
