"""Versioned prompt lexicon contract for AI prompt normalization.

The lexicon is intentionally repo-tracked only.  Runtime/operator override is
out of scope because synonym changes can silently increase false positives.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from importlib import resources
from typing import Any

from pydantic import Field, model_validator

from truthound.ai.context import SUPPORTED_INTENT_NAMES
from truthound.ai.models import BaseStrictModel

SUPPORTED_FORMAT_KINDS = (
    "email",
    "url",
    "phone",
    "uuid",
    "ip_address",
    "ipv6_address",
)

_WHITESPACE_RE = re.compile(r"\s+")
_DEFAULT_LEXICON_RESOURCE = "data/prompt_lexicon.ko.json"


class PromptLexicon(BaseStrictModel):
    schema_version: str
    locale: str
    lexicon_version: str
    intent_synonyms: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    format_synonyms: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    semantic_column_aliases: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    ambiguous_markers: tuple[str, ...] = ()
    false_positive_guards: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_contract(self) -> PromptLexicon:
        if self.schema_version != "1":
            raise ValueError("prompt lexicon schema_version must be '1'")
        if self.locale != "ko-KR":
            raise ValueError("phase 2 prompt lexicon locale must be 'ko-KR'")
        if not self.lexicon_version.strip():
            raise ValueError("prompt lexicon lexicon_version must be set")

        unsupported_intents = sorted(set(self.intent_synonyms) - set(SUPPORTED_INTENT_NAMES))
        if unsupported_intents:
            raise ValueError(f"unsupported prompt lexicon intents: {unsupported_intents}")

        unsupported_formats = sorted(set(self.format_synonyms) - set(SUPPORTED_FORMAT_KINDS))
        if unsupported_formats:
            raise ValueError(f"unsupported prompt lexicon formats: {unsupported_formats}")

        self._validate_nonempty_mapping(self.intent_synonyms, label="intent_synonyms")
        self._validate_nonempty_mapping(self.format_synonyms, label="format_synonyms")
        self._validate_nonempty_mapping(self.semantic_column_aliases, label="semantic_column_aliases")
        self._validate_nonempty_sequence(self.ambiguous_markers, label="ambiguous_markers")
        self._validate_nonempty_sequence(self.false_positive_guards, label="false_positive_guards")
        self._validate_synonym_collisions()
        return self

    @property
    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    @staticmethod
    def compact(value: str) -> str:
        normalized = normalize_lexicon_text(value).lower()
        return re.sub(r"[\s_\-.]+", "", normalized)

    @staticmethod
    def _validate_nonempty_mapping(mapping: dict[str, tuple[str, ...]], *, label: str) -> None:
        for key, values in mapping.items():
            if not normalize_lexicon_text(key):
                raise ValueError(f"{label} contains an empty key")
            PromptLexicon._validate_nonempty_sequence(values, label=f"{label}.{key}")

    @staticmethod
    def _validate_nonempty_sequence(values: tuple[str, ...], *, label: str) -> None:
        seen: set[str] = set()
        for value in values:
            normalized = normalize_lexicon_text(value)
            if not normalized:
                raise ValueError(f"{label} contains an empty value")
            compact = PromptLexicon.compact(normalized)
            if compact in seen:
                raise ValueError(f"{label} contains duplicate value {value!r}")
            seen.add(compact)

    def _validate_synonym_collisions(self) -> None:
        seen: dict[str, str] = {}
        for canonical, synonyms in self.intent_synonyms.items():
            keys = (canonical, *synonyms)
            for synonym in keys:
                compact = self.compact(synonym)
                existing = seen.get(compact)
                if existing is not None and existing != canonical:
                    raise ValueError(
                        f"prompt synonym {synonym!r} maps to both {existing!r} and {canonical!r}"
                    )
                seen[compact] = canonical


def normalize_lexicon_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def load_prompt_lexicon_from_text(text: str) -> PromptLexicon:
    payload: Any = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("prompt lexicon must be a JSON object")
    return PromptLexicon.model_validate(payload)


@lru_cache(maxsize=1)
def get_default_prompt_lexicon() -> PromptLexicon:
    text = (
        resources.files("truthound.ai")
        .joinpath(_DEFAULT_LEXICON_RESOURCE)
        .read_text(encoding="utf-8")
    )
    return load_prompt_lexicon_from_text(text)


__all__ = [
    "PromptLexicon",
    "SUPPORTED_FORMAT_KINDS",
    "get_default_prompt_lexicon",
    "load_prompt_lexicon_from_text",
    "normalize_lexicon_text",
]
