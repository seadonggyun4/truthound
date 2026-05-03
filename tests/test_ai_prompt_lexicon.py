from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract

LEXICON_PATH = Path(__file__).parents[1] / "src/truthound/ai/data/prompt_lexicon.ko.json"


def test_prompt_lexicon_manifest_contract_is_valid() -> None:
    from truthound.ai.context import SUPPORTED_INTENT_NAMES
    from truthound.ai.prompt_lexicon import SUPPORTED_FORMAT_KINDS, get_default_prompt_lexicon

    lexicon = get_default_prompt_lexicon()

    assert LEXICON_PATH.exists()
    assert lexicon.schema_version == "1"
    assert lexicon.locale == "ko-KR"
    assert lexicon.lexicon_version
    assert set(lexicon.intent_synonyms).issubset(set(SUPPORTED_INTENT_NAMES))
    assert set(lexicon.format_synonyms).issubset(set(SUPPORTED_FORMAT_KINDS))
    assert lexicon.ambiguous_markers
    assert lexicon.false_positive_guards
    assert lexicon.content_hash


def test_prompt_lexicon_manifest_is_included_in_release_artifacts() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    wheel_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    sdist_include = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["force-include"]

    expected_data_files = {
        "src/truthound/ai/data/prompt_lexicon.ko.json": (
            "truthound/ai/data/prompt_lexicon.ko.json"
        ),
        "src/truthound/ai/data/validation_dsl_coverage.v1.json": (
            "truthound/ai/data/validation_dsl_coverage.v1.json"
        ),
    }

    for manifest_source, wheel_target in expected_data_files.items():
        assert wheel_include[manifest_source] == wheel_target
        assert sdist_include[manifest_source] == manifest_source


def test_prompt_lexicon_rejects_synonym_collisions() -> None:
    from truthound.ai.prompt_lexicon import load_prompt_lexicon_from_text

    payload = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    payload["intent_synonyms"]["not_null"].append("중복 없어")

    with pytest.raises(ValueError, match="maps to both"):
        load_prompt_lexicon_from_text(json.dumps(payload, ensure_ascii=False))


def test_prompt_lexicon_rejects_unsupported_intent_and_format() -> None:
    from truthound.ai.prompt_lexicon import load_prompt_lexicon_from_text

    payload = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    payload["intent_synonyms"]["unknown_intent"] = ["알 수 없음"]

    with pytest.raises(ValueError, match="unsupported prompt lexicon intents"):
        load_prompt_lexicon_from_text(json.dumps(payload, ensure_ascii=False))

    payload = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    payload["format_synonyms"]["postal_code"] = ["우편번호"]

    with pytest.raises(ValueError, match="unsupported prompt lexicon formats"):
        load_prompt_lexicon_from_text(json.dumps(payload, ensure_ascii=False))


def test_prompt_lexicon_rejects_empty_or_duplicate_values() -> None:
    from truthound.ai.prompt_lexicon import load_prompt_lexicon_from_text

    payload = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    payload["semantic_column_aliases"]["email"].append("  이메일  ")

    with pytest.raises(ValueError, match="duplicate value"):
        load_prompt_lexicon_from_text(json.dumps(payload, ensure_ascii=False))

    payload = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    payload["ambiguous_markers"].append(" ")

    with pytest.raises(ValueError, match="empty value"):
        load_prompt_lexicon_from_text(json.dumps(payload, ensure_ascii=False))
