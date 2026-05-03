from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from truthound.context import TruthoundContext

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract

FIXTURE_DIR = Path(__file__).parent / "fixtures/ai/normalization"
REQUIRED_CORPUS_LABELS = {
    "task_label",
    "intent_label",
    "entity_labels",
    "threshold_labels",
    "ambiguity_label",
    "unicode_labels",
}
CORPUS_FIXTURES = (
    "korean_golden_prompts.json",
    "ambiguous_prompts.json",
    "false_positive_prompts.json",
    "column_alias_cases.json",
    "unicode_edge_prompts.json",
)


class KoreanIntentProvider:
    provider_name = "fake-openai"
    default_model_name = "gpt-fake"
    supports_structured_outputs = True

    def generate_structured(self, request):
        from truthound.ai import StructuredProviderResponse

        payload = {
            "summary": "한글 intent를 포함한 제안",
            "rationale": "provider가 한국어 intent label을 반환해도 compiler 앞단에서 canonicalize되어야 합니다.",
            "proposed_checks": [
                {
                    "intent": "필수값",
                    "columns": ["email"],
                    "params": {},
                    "rationale": "이메일은 비어 있으면 안 됩니다.",
                }
            ],
            "risks": [],
            "rejected_requests": [],
        }
        return StructuredProviderResponse(
            provider_name=self.provider_name,
            model_name=request.model_name,
            output_text=json.dumps(payload, ensure_ascii=False),
            parsed_output=payload,
        )


def _load_fixture(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture_name", CORPUS_FIXTURES)
def test_normalization_corpus_uses_klue_style_labels(fixture_name: str) -> None:
    for case in _load_fixture(fixture_name):
        assert REQUIRED_CORPUS_LABELS.issubset(case), case["name"]
        assert isinstance(case["entity_labels"], list)
        assert isinstance(case["threshold_labels"], dict)
        assert isinstance(case["unicode_labels"], list)
        assert case["ambiguity_label"] in {"clear", "ambiguous", "false_positive", "unicode_risk"}


@pytest.mark.parametrize("case", _load_fixture("korean_golden_prompts.json"), ids=lambda item: item["name"])
def test_korean_golden_prompts_normalize_to_exact_candidates(case: dict[str, Any]) -> None:
    from truthound.ai.normalization import PromptNormalizer

    normalized = PromptNormalizer().normalize(
        case["prompt"],
        columns=case["columns"],
    )

    candidates = [
        {
            "intent": candidate.intent,
            "columns": candidate.columns,
            "params": candidate.params,
        }
        for candidate in normalized.candidates
    ]
    assert candidates == case["expected_candidates"]
    assert normalized.actionable is case["actionable"]
    assert normalized.lexicon_version
    assert normalized.lexicon_hash


@pytest.mark.parametrize("case", _load_fixture("unicode_edge_prompts.json"), ids=lambda item: item["name"])
def test_unicode_edge_prompts_are_audited_and_safely_normalized(case: dict[str, Any]) -> None:
    from truthound.ai.normalization import PromptNormalizer, normalize_prompt_text_with_audit

    audit = normalize_prompt_text_with_audit(case["prompt"])
    normalized = PromptNormalizer().normalize(case["prompt"], columns=case["columns"])

    assert "original_text" not in audit.model_dump()
    candidates = [
        {
            "intent": candidate.intent,
            "columns": candidate.columns,
            "params": candidate.params,
        }
        for candidate in normalized.candidates
    ]
    assert audit.raw_prompt_hash.startswith("sha256:")
    assert audit.normalized_text_hash.startswith("sha256:")
    assert normalized.raw_prompt_hash == audit.raw_prompt_hash
    assert normalized.normalized_text_hash == audit.normalized_text_hash
    assert normalized.unicode_warnings == case["expected_unicode_warnings"]
    assert candidates == case["expected_candidates"]
    assert normalized.actionable is case["actionable"]

    if "expected_normalized_text" in case:
        assert normalized.normalized_text == case["expected_normalized_text"]
    if "expected_clarification_reason" in case:
        assert normalized.clarification is not None
        assert normalized.clarification.reason == case["expected_clarification_reason"]
        assert [item.reason for item in normalized.unresolved_terms] == case["expected_unresolved_reasons"]


@pytest.mark.parametrize("case", _load_fixture("ambiguous_prompts.json"), ids=lambda item: item["name"])
def test_ambiguous_korean_prompt_requires_clarification(case: dict[str, Any]) -> None:
    from truthound.ai.normalization import PromptNormalizer

    normalized = PromptNormalizer().normalize(case["prompt"], columns=case["columns"])

    assert normalized.candidates == []
    assert normalized.clarification is not None
    assert normalized.clarification.reason == case["expected_reason"]


@pytest.mark.parametrize("case", _load_fixture("false_positive_prompts.json"), ids=lambda item: item["name"])
def test_false_positive_guard_blocks_broad_operational_phrasing(case: dict[str, Any]) -> None:
    from truthound.ai.normalization import PromptNormalizer

    normalized = PromptNormalizer().normalize(case["prompt"], columns=case["columns"])

    assert normalized.candidates == []
    assert normalized.clarification is not None
    assert normalized.clarification.reason == case["expected_reason"]
    assert [item.reason for item in normalized.unresolved_terms] == [case["expected_reason"]]


@pytest.mark.parametrize("case", _load_fixture("column_alias_cases.json"), ids=lambda item: item["name"])
def test_column_alias_resolution_is_unique_or_clarifies(case: dict[str, Any]) -> None:
    from truthound.ai.normalization import PromptNormalizer

    normalized = PromptNormalizer().normalize(case["prompt"], columns=case["columns"])

    columns = normalized.candidates[0].columns if normalized.candidates else []
    assert columns == case["expected_columns"]
    assert [item.reason for item in normalized.unresolved_terms] == case["expected_unresolved_reasons"]


def test_korean_provider_intent_is_canonicalized_before_compile(tmp_path: Path) -> None:
    from truthound.ai import suggest_suite

    artifact = suggest_suite(
        prompt="이메일 컬럼은 비어 있으면 안 됩니다",
        data=pl.DataFrame({"email": ["a@example.com", None]}),
        context=TruthoundContext(tmp_path),
        provider=KoreanIntentProvider(),
    )

    assert str(artifact.compile_status) == "ready"
    assert artifact.compiled_check_count == 1
    assert artifact.checks[0].validator_name == "not_null"
    assert artifact.rejected_check_count == 0
    normalization_refs = [item for item in artifact.input_refs if item.kind == "prompt_normalization"]
    assert len(normalization_refs) == 1
    assert normalization_refs[0].metadata["lexicon_version"]
    assert normalization_refs[0].metadata["lexicon_hash"]
    assert normalization_refs[0].metadata["raw_prompt_hash"].startswith("sha256:")
    assert normalization_refs[0].metadata["normalized_text_hash"].startswith("sha256:")
    assert "original_prompt" not in normalization_refs[0].metadata
