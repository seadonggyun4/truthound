from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).parents[1]
COVERAGE_PATH = REPO_ROOT / "src/truthound/ai/data/validation_dsl_coverage.v1.json"
PHASE4_FIXTURE = REPO_ROOT / "tests/fixtures/ai/phase4/happy_path_proposal_response.json"
NORMALIZATION_FIXTURE = REPO_ROOT / "tests/fixtures/ai/normalization/korean_golden_prompts.json"


def test_validation_dsl_coverage_matrix_contract_is_valid() -> None:
    from truthound.ai.context import SUPPORTED_INTENT_NAMES
    from truthound.ai.dsl_coverage import get_default_validation_dsl_coverage

    matrix = get_default_validation_dsl_coverage()

    assert COVERAGE_PATH.exists()
    assert matrix.schema_version == "1"
    assert matrix.dsl_version == "validation-dsl-v1"
    assert set(matrix.intents) == set(SUPPORTED_INTENT_NAMES)
    assert all(item.automatic_approximation == "forbidden" for item in matrix.intents.values())


def test_validation_dsl_coverage_matches_compiler_fixture_params() -> None:
    from truthound.ai.context import SUPPORTED_INTENT_NAMES
    from truthound.ai.dsl_coverage import get_default_validation_dsl_coverage

    matrix = get_default_validation_dsl_coverage()
    payload = json.loads(PHASE4_FIXTURE.read_text(encoding="utf-8"))
    proposed_checks = payload["proposed_checks"]

    assert {item["intent"] for item in proposed_checks} == set(SUPPORTED_INTENT_NAMES)
    for check in proposed_checks:
        coverage = matrix.intents[check["intent"]]
        param_keys = set(check.get("params", {}))
        assert param_keys <= set(coverage.allowed_params), check["intent"]
        if coverage.required_any_params:
            assert param_keys & set(coverage.required_any_params), check["intent"]


def test_deterministic_coverage_has_korean_golden_prompt_examples() -> None:
    from truthound.ai.dsl_coverage import get_default_validation_dsl_coverage

    matrix = get_default_validation_dsl_coverage()
    golden_cases = json.loads(NORMALIZATION_FIXTURE.read_text(encoding="utf-8"))
    golden_intents = {case["intent_label"] for case in golden_cases}

    missing = sorted(set(matrix.deterministic_intents) - golden_intents)
    assert missing == []


def test_validation_dsl_coverage_rejects_intent_drift() -> None:
    from truthound.ai.dsl_coverage import load_validation_dsl_coverage_from_text

    payload = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    payload["intents"].pop("between")

    with pytest.raises(ValueError, match="intents mismatch"):
        load_validation_dsl_coverage_from_text(json.dumps(payload, ensure_ascii=False))


def test_validation_dsl_coverage_rejects_required_param_drift() -> None:
    from truthound.ai.dsl_coverage import load_validation_dsl_coverage_from_text

    payload = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    payload["intents"]["between"]["required_any_params"].append("exclusive")

    with pytest.raises(ValueError, match="required_any_params"):
        load_validation_dsl_coverage_from_text(json.dumps(payload, ensure_ascii=False))
