from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract

FIXTURE_PATH = Path(__file__).parent / "fixtures/ai/evaluation/prompt_acceptance.v1.json"


def _load_payload() -> dict:
    from tests.ai_prompt_evaluation import load_prompt_acceptance_payload

    return load_prompt_acceptance_payload(FIXTURE_PATH)


def _load_cases():
    from tests.ai_prompt_evaluation import expand_prompt_acceptance_cases

    return expand_prompt_acceptance_cases(_load_payload())


def test_prompt_evaluation_corpus_expands_to_phase7_minimum_counts() -> None:
    from tests.ai_prompt_evaluation import cases_by_split

    payload = _load_payload()
    cases = _load_cases()
    grouped = cases_by_split(cases)
    thresholds = payload["thresholds"]

    assert len(grouped["golden"]) >= thresholds["min_golden_cases"]
    assert len(grouped["mixed"]) >= thresholds["min_mixed_cases"]
    assert len(grouped["ambiguous"]) >= thresholds["min_ambiguous_cases"]
    assert thresholds["min_golden_ready_or_partial_rate"] == 0.9
    assert thresholds["min_mixed_ready_or_partial_rate"] == 0.9
    assert thresholds["min_ambiguous_clarification_rate"] == 0.95
    assert thresholds["max_crash_count"] == 0


def test_prompt_evaluation_corpus_uses_klue_style_case_contract() -> None:
    from tests.ai_prompt_evaluation import REQUIRED_CASE_FIELDS
    from truthound.ai.context import SUPPORTED_INTENT_NAMES
    from truthound.ai.dsl_coverage import get_default_validation_dsl_coverage

    supported = set(SUPPORTED_INTENT_NAMES)
    matrix = get_default_validation_dsl_coverage()
    assert set(matrix.intents) == supported

    for case in _load_cases():
        payload = case.as_contract_payload()
        assert set(payload) >= REQUIRED_CASE_FIELDS, case.id
        assert isinstance(case.entity_labels, list), case.id
        assert isinstance(case.threshold_labels, dict), case.id
        assert isinstance(case.unicode_labels, list), case.id
        if case.expected_candidates:
            for candidate in case.expected_candidates:
                assert candidate["intent"] in supported, case.id
                assert set(candidate["params"]) <= set(matrix.intents[candidate["intent"]].allowed_params)


def test_prompt_acceptance_metrics_meet_phase7_production_ready_gate() -> None:
    from tests.ai_prompt_evaluation import evaluate_prompt_acceptance

    payload = _load_payload()
    thresholds = payload["thresholds"]
    summary = evaluate_prompt_acceptance(_load_cases())

    assert summary.crash_count == thresholds["max_crash_count"]
    assert summary.golden_ready_or_partial_rate >= thresholds["min_golden_ready_or_partial_rate"]
    assert summary.mixed_ready_or_partial_rate >= thresholds["min_mixed_ready_or_partial_rate"]
    assert summary.ambiguous_clarification_rate >= thresholds["min_ambiguous_clarification_rate"]
    assert summary.failures == []
    assert summary.tag_counts["not_null"] > 0
    assert summary.tag_counts["between"] > 0
    assert summary.tag_counts["unsupported"] > 0


def test_prompt_acceptance_summary_can_be_persisted_without_raw_prompt_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from tests.ai_prompt_evaluation import EVAL_RESULT_PATH_ENV, evaluate_prompt_acceptance

    result_path = tmp_path / "prompt-acceptance-summary.json"
    monkeypatch.setenv(EVAL_RESULT_PATH_ENV, str(result_path))

    evaluate_prompt_acceptance(_load_cases())

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["golden_ready_or_partial_rate"] >= 0.9
    assert payload["ambiguous_clarification_rate"] >= 0.95
    serialized = result_path.read_text(encoding="utf-8")
    assert "prompt" not in payload["failures"][0] if payload["failures"] else True
    assert "이메일은 비어 있으면 안 됩니다" not in serialized


def test_phase7_manual_live_canary_contract_is_documented_and_cost_controlled() -> None:
    payload = _load_payload()
    canary = payload["manual_live_canary"]

    assert canary["golden_case_count"] == 3
    assert canary["ambiguous_case_count"] == 2
    assert canary["mixed_case_count"] == 2
    assert canary["provider_failure_fixture_count"] >= 1
    assert canary["env"] == [
        "TRUTHOUND_AI_RUN_LIVE_SMOKE",
        "TRUTHOUND_AI_SMOKE_MODEL_MATRIX",
        "TRUTHOUND_AI_SMOKE_RESULT_PATH",
    ]
    for fixture in payload["provider_failure_fixtures"]:
        assert (Path(__file__).parents[1] / fixture).exists()
