"""Prompt normalization helpers for AI suite proposals.

This module turns multilingual operator phrasing into Truthound's canonical
validation intent DSL before the proposal compiler runs.  It deliberately uses
stdlib Unicode normalization, deterministic regexes, and schema-aware lookup
tables so the dashboard can support Korean prompts without adding a tokenizer
runtime dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import Field

from truthound.ai.context import (
    SUPPORTED_INTENT_NAMES,
    ContextBundle,
    ContextBundleBuilder,
)
from truthound.ai.models import (
    BaseStrictModel,
    InputRef,
    ProposedCheckIntent,
    SuiteProposalLLMResponse,
)
from truthound.ai.prompt_lexicon import (
    PromptLexicon,
    get_default_prompt_lexicon,
)
from truthound.context import get_context

if TYPE_CHECKING:
    from truthound.context import TruthoundContext

PROMPT_NORMALIZATION_ENV = "TRUTHOUND_AI_PROMPT_NORMALIZATION"

_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = r"-?\d+(?:\.\d+)?"
_PERCENT_RE = re.compile(rf"(?P<value>{_NUMBER_RE})\s*%")
_RANGE_RE = re.compile(
    rf"(?P<min>{_NUMBER_RE})\s*(?:~|부터|에서)\s*(?P<max>{_NUMBER_RE})\s*(?:까지|사이)?"
)
_LOWER_UPPER_RE = re.compile(
    rf"(?P<min>{_NUMBER_RE})\s*(?:이상|>=)\s*(?P<max>{_NUMBER_RE})\s*(?:이하|<=)"
)
_LOWER_ONLY_RE = re.compile(rf"(?P<min>{_NUMBER_RE})\s*(?:이상|>=|보다\s*크거나\s*같)")
_UPPER_ONLY_RE = re.compile(rf"(?P<max>{_NUMBER_RE})\s*(?:이하|<=|보다\s*작거나\s*같)")
_SYMBOL_LOWER_ONLY_RE = re.compile(rf">=\s*(?P<min>{_NUMBER_RE})")
_SYMBOL_UPPER_ONLY_RE = re.compile(rf"<=\s*(?P<max>{_NUMBER_RE})")
_EXCLUSIVE_RE = re.compile(rf"{_NUMBER_RE}\s*(?:초과|미만|>|<)")
_LENGTH_EXACT_RE = re.compile(rf"(?P<value>{_NUMBER_RE})\s*(?:자|자리|글자)")
_LENGTH_RANGE_RE = re.compile(
    rf"(?P<min>{_NUMBER_RE})\s*(?:자|자리|글자)?\s*(?:~|부터|에서)\s*(?P<max>{_NUMBER_RE})\s*(?:자|자리|글자)?"
)
_QUOTED_TERM_RE = re.compile(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'")
_HASH_BYTES_PREFIX = "sha256:"
_SAFE_FORMAT_CONTROL_CHARS = {
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space / byte order mark
}
_SAFE_WIDTH_OR_SPACING_CATEGORIES = {
    "w",
    "na",
    "f",
    "h",
}
_SAFE_COMPATIBILITY_EXCEPTIONS = {
    "\u00a0",  # no-break space
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u202f",
    "\u205f",
    "\u3000",  # ideographic space
    "％",
    "～",
}

DEFAULT_PROMPT_LEXICON = get_default_prompt_lexicon()

# Backward-compatible aliases for tests and downstream code that inspected the
# old inline tables. New normalization logic reads from PromptLexicon directly.
INTENT_SYNONYMS = DEFAULT_PROMPT_LEXICON.intent_synonyms
FORMAT_SYNONYMS = DEFAULT_PROMPT_LEXICON.format_synonyms
SEMANTIC_COLUMN_ALIASES = DEFAULT_PROMPT_LEXICON.semantic_column_aliases
AMBIGUOUS_PROMPT_MARKERS = DEFAULT_PROMPT_LEXICON.ambiguous_markers
FALSE_POSITIVE_GUARDS = DEFAULT_PROMPT_LEXICON.false_positive_guards


class PromptNormalizationMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class UnresolvedPromptTerm(BaseStrictModel):
    term: str
    reason: str


class ClarificationRequest(BaseStrictModel):
    reason: str
    suggestions: list[str] = Field(default_factory=list)


class PromptNormalizationEvent(BaseStrictModel):
    kind: str
    before: str
    after: str
    count: int = 1


class PromptTextNormalizationResult(BaseStrictModel):
    normalized_text: str
    raw_prompt_hash: str
    normalized_text_hash: str
    normalization_events: list[PromptNormalizationEvent] = Field(default_factory=list)
    unicode_warnings: list[str] = Field(default_factory=list)
    blocking_warning: str | None = None


class NormalizedIntentCandidate(BaseStrictModel):
    intent: str
    columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source: str = "deterministic"
    rationale: str = ""

    def to_proposed_check(self) -> ProposedCheckIntent:
        return ProposedCheckIntent(
            intent=self.intent,
            columns=list(self.columns),
            params=dict(self.params),
            rationale=self.rationale,
        )


class NormalizedPrompt(BaseStrictModel):
    original_prompt: str
    normalized_text: str
    language: str
    mode: PromptNormalizationMode = PromptNormalizationMode.ENFORCE
    lexicon_version: str = ""
    lexicon_hash: str = ""
    raw_prompt_hash: str = ""
    normalized_text_hash: str = ""
    unicode_warnings: list[str] = Field(default_factory=list)
    candidates: list[NormalizedIntentCandidate] = Field(default_factory=list)
    unresolved_terms: list[UnresolvedPromptTerm] = Field(default_factory=list)
    clarification: ClarificationRequest | None = None

    @property
    def actionable(self) -> bool:
        return bool(self.candidates) and self.clarification is None

    def to_provider_guidance(self) -> str:
        mode_value = self.mode.value if isinstance(self.mode, PromptNormalizationMode) else str(self.mode)
        payload = {
            "mode": mode_value,
            "language": self.language,
            "lexicon_version": self.lexicon_version,
            "lexicon_hash": self.lexicon_hash,
            "unicode_warnings": self.unicode_warnings,
            "candidates": [
                {
                    "intent": item.intent,
                    "columns": item.columns,
                    "params": item.params,
                    "confidence": item.confidence,
                }
                for item in self.candidates
            ],
            "unresolved_terms": [
                {"term": item.term, "reason": item.reason}
                for item in self.unresolved_terms
            ],
            "clarification_required": self.clarification is not None,
        }
        return (
            "normalized_prompt_guidance: "
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}. "
            "Prefer these canonical candidates when they match the operator request. "
            "Use only allowed canonical intent names and known columns."
        )

    def to_input_ref(self) -> InputRef:
        mode_value = self.mode.value if isinstance(self.mode, PromptNormalizationMode) else str(self.mode)
        metadata = {
            "mode": mode_value,
            "language": self.language,
            "lexicon_version": self.lexicon_version,
            "lexicon_hash": self.lexicon_hash,
            "raw_prompt_hash": self.raw_prompt_hash,
            "normalized_text_hash": self.normalized_text_hash,
            "unicode_warnings": self.unicode_warnings,
            "candidate_count": len(self.candidates),
            "unresolved_count": len(self.unresolved_terms),
            "clarification_required": self.clarification is not None,
            "candidates": [
                {
                    "intent": item.intent,
                    "columns": item.columns,
                    "params": item.params,
                    "confidence": item.confidence,
                }
                for item in self.candidates
            ],
        }
        digest = hashlib.sha256(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return InputRef(
            kind="prompt_normalization",
            ref=f"prompt-normalization:{digest}",
            hash=digest,
            redacted=True,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ColumnResolution:
    columns: tuple[str, ...]
    unresolved_terms: tuple[UnresolvedPromptTerm, ...] = ()


class IntentCanonicalizer:
    """Map localized or synonymous intent labels to canonical Truthound intents."""

    def __init__(self, *, lexicon: PromptLexicon | None = None) -> None:
        self.lexicon = lexicon or DEFAULT_PROMPT_LEXICON

    def canonicalize(self, intent: str) -> str:
        normalized = normalize_prompt_text(intent).lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        if normalized in SUPPORTED_INTENT_NAMES:
            return normalized

        compact = normalized.replace("_", "")
        for canonical, synonyms in self.lexicon.intent_synonyms.items():
            if compact == canonical.replace("_", ""):
                return canonical
            if any(compact == normalize_prompt_text(item).lower().replace(" ", "").replace("_", "") for item in synonyms):
                return canonical
        return normalized

    def canonicalize_check(self, intent: ProposedCheckIntent) -> ProposedCheckIntent:
        canonical = self.canonicalize(intent.intent)
        return ProposedCheckIntent(
            intent=canonical,
            columns=list(intent.columns),
            params=dict(intent.params),
            rationale=intent.rationale,
        )

    def canonicalize_response(self, response: SuiteProposalLLMResponse) -> SuiteProposalLLMResponse:
        return SuiteProposalLLMResponse(
            summary=response.summary,
            rationale=response.rationale,
            proposed_checks=[self.canonicalize_check(item) for item in response.proposed_checks],
            risks=list(response.risks),
            rejected_requests=list(response.rejected_requests),
        )


class ColumnResolver:
    """Resolve prompt terms to known schema columns using exact and alias matches."""

    def __init__(
        self,
        columns: list[str] | tuple[str, ...],
        *,
        lexicon: PromptLexicon | None = None,
    ) -> None:
        self.lexicon = lexicon or DEFAULT_PROMPT_LEXICON
        self.columns = tuple(str(item) for item in columns if str(item))
        self._aliases = {column: self._aliases_for_column(column) for column in self.columns}

    @classmethod
    def from_context_bundle(
        cls,
        context_bundle: ContextBundle,
        *,
        lexicon: PromptLexicon | None = None,
    ) -> ColumnResolver:
        columns = [
            str(column.get("name"))
            for column in context_bundle.schema_summary.get("columns", [])
            if column.get("name")
        ]
        return cls(columns, lexicon=lexicon)

    def resolve(self, text: str, *, preferred_terms: list[str] | None = None) -> ColumnResolution:
        normalized_text = normalize_prompt_text(text).lower()
        terms = list(preferred_terms or [])
        quoted_terms = self._quoted_terms(text)
        terms.extend(quoted_terms)

        quoted_matches = self._resolve_exact_terms(quoted_terms)
        if quoted_matches:
            return ColumnResolution(columns=tuple(quoted_matches))

        exact_matches = self._resolve_exact_text(normalized_text)
        if exact_matches:
            return ColumnResolution(columns=tuple(exact_matches))

        alias_matches, ambiguous_aliases = self._resolve_alias_text(normalized_text)
        if len(alias_matches) == 1:
            return ColumnResolution(columns=tuple(alias_matches))
        if len(alias_matches) > 1 or ambiguous_aliases:
            term = ", ".join(_dedupe(ambiguous_aliases)) or "column"
            return ColumnResolution(
                columns=(),
                unresolved_terms=(UnresolvedPromptTerm(term=term, reason="column_alias_ambiguous"),),
            )

        unresolved = tuple(
            UnresolvedPromptTerm(term=term, reason="column_not_found")
            for term in _dedupe(terms)
            if term
        )
        return ColumnResolution(columns=(), unresolved_terms=unresolved)

    def resolve_required(self, text: str, *, preferred_terms: list[str] | None = None) -> ColumnResolution:
        resolution = self.resolve(text, preferred_terms=preferred_terms)
        if resolution.columns:
            return resolution
        if resolution.unresolved_terms:
            return resolution
        return ColumnResolution(
            columns=(),
            unresolved_terms=(UnresolvedPromptTerm(term="column", reason="column_ambiguous_or_missing"),),
        )

    def _resolve_exact_terms(self, terms: list[str]) -> list[str]:
        matches: list[str] = []
        for term in terms:
            normalized_term = _normalize_identifier(term)
            for column in self.columns:
                if normalized_term and normalized_term == _normalize_identifier(column):
                    matches.append(column)
        return _dedupe(matches)

    def _resolve_exact_text(self, normalized_text: str) -> list[str]:
        return _dedupe(
            [
                column
                for column in self.columns
                if self._text_contains_identifier(normalized_text, column)
            ]
        )

    def _resolve_alias_text(self, normalized_text: str) -> tuple[list[str], list[str]]:
        matches: list[str] = []
        ambiguous_aliases: list[str] = []
        compact_text = _compact(normalized_text)
        alias_to_columns: dict[str, list[str]] = {}
        alias_labels: dict[str, str] = {}
        for column, aliases in self._aliases.items():
            for alias in aliases:
                compact_alias = _compact(alias.lower())
                if not compact_alias:
                    continue
                alias_to_columns.setdefault(compact_alias, []).append(column)
                alias_labels.setdefault(compact_alias, alias)

        for compact_alias, columns in alias_to_columns.items():
            if compact_alias not in compact_text:
                continue
            unique_columns = _dedupe(columns)
            if len(unique_columns) == 1:
                matches.append(unique_columns[0])
            else:
                ambiguous_aliases.append(alias_labels[compact_alias])

        return _dedupe(matches), ambiguous_aliases

    def _text_contains_identifier(self, normalized_text: str, column: str) -> bool:
        normalized_column = _normalize_identifier(column)
        if not normalized_column:
            return False
        if len(normalized_column) < 3:
            pattern = rf"(?<![a-z0-9가-힣]){re.escape(normalized_column)}(?![a-z0-9가-힣])"
            return re.search(pattern, normalized_text.lower()) is not None
        return normalized_column in _compact(normalized_text)

    def _aliases_for_column(self, column: str) -> tuple[str, ...]:
        parts = [part for part in re.split(r"[_\-\s.]+", column.lower()) if part]
        aliases: set[str] = {column, column.replace("_", " "), column.replace("_", "")}
        for part in parts:
            for alias in self.lexicon.semantic_column_aliases.get(part, ()):
                aliases.add(alias)
        if "customer" in parts and "id" in parts:
            aliases.update({"고객id", "고객 id", "고객 아이디", "고객 식별자"})
        if "user" in parts and "id" in parts:
            aliases.update({"사용자id", "사용자 id", "유저 id", "사용자 아이디"})
        if "refund" in parts and ("rate" in parts or "ratio" in parts):
            aliases.update({"환불률", "환불 비율"})
        return tuple(sorted(aliases, key=len, reverse=True))

    def _quoted_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        for match in _QUOTED_TERM_RE.finditer(text):
            terms.extend(item for item in match.groups() if item)
        return terms


class PromptNormalizer:
    """Deterministically extract validation intent candidates from operator prompts."""

    def __init__(
        self,
        *,
        mode: PromptNormalizationMode | None = None,
        lexicon: PromptLexicon | None = None,
    ) -> None:
        self.mode = mode or get_prompt_normalization_mode()
        self.lexicon = lexicon or DEFAULT_PROMPT_LEXICON

    def normalize(
        self,
        prompt: str,
        *,
        context_bundle: ContextBundle | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> NormalizedPrompt:
        text_audit = normalize_prompt_text_with_audit(prompt)
        normalized_text = text_audit.normalized_text
        resolver = (
            ColumnResolver.from_context_bundle(context_bundle, lexicon=self.lexicon)
            if context_bundle is not None
            else ColumnResolver(columns or (), lexicon=self.lexicon)
        )
        candidates: list[NormalizedIntentCandidate] = []
        unresolved_terms: list[UnresolvedPromptTerm] = []

        if text_audit.blocking_warning is not None:
            return NormalizedPrompt(
                original_prompt=prompt,
                normalized_text=normalized_text,
                language="ko" if _contains_hangul(normalized_text) else "en",
                mode=self.mode,
                lexicon_version=self.lexicon.lexicon_version,
                lexicon_hash=self.lexicon.content_hash,
                raw_prompt_hash=text_audit.raw_prompt_hash,
                normalized_text_hash=text_audit.normalized_text_hash,
                unicode_warnings=list(text_audit.unicode_warnings),
                candidates=[],
                unresolved_terms=[
                    UnresolvedPromptTerm(
                        term=text_audit.blocking_warning,
                        reason=text_audit.blocking_warning,
                    )
                ],
                clarification=ClarificationRequest(
                    reason="unicode_normalization_risk",
                    suggestions=[
                        "한글 자모나 특수 기호가 분리되어 보입니다. 완성형 한글과 일반 숫자/기호로 다시 입력해주세요.",
                    ],
                ),
            )

        false_positive_guard = self._matching_false_positive_guard(normalized_text)
        if false_positive_guard is not None:
            return NormalizedPrompt(
                original_prompt=prompt,
                normalized_text=normalized_text,
                language="ko" if _contains_hangul(normalized_text) else "en",
                mode=self.mode,
                lexicon_version=self.lexicon.lexicon_version,
                lexicon_hash=self.lexicon.content_hash,
                raw_prompt_hash=text_audit.raw_prompt_hash,
                normalized_text_hash=text_audit.normalized_text_hash,
                unicode_warnings=list(text_audit.unicode_warnings),
                candidates=[],
                unresolved_terms=[
                    UnresolvedPromptTerm(term=false_positive_guard, reason="false_positive_guard")
                ],
                clarification=ClarificationRequest(
                    reason="false_positive_guard",
                    suggestions=["검증할 컬럼명과 기대 조건을 더 구체적으로 적어주세요."],
                ),
            )

        for candidate, unresolved in (
            self._extract_ratio_candidate(normalized_text, resolver),
            self._extract_enum_candidate(normalized_text, resolver),
            self._extract_format_candidate(normalized_text, resolver),
            self._extract_length_candidate(normalized_text, resolver),
            self._extract_numeric_candidate(normalized_text, resolver),
            self._extract_unique_candidate(normalized_text, resolver),
            self._extract_nullability_candidate(normalized_text, resolver),
            self._extract_aggregate_candidate(normalized_text, resolver),
        ):
            if candidate is not None:
                candidates.append(candidate)
            unresolved_terms.extend(unresolved)

        candidates = _dedupe_candidates(candidates)
        unresolved_terms = _dedupe_unresolved(unresolved_terms)
        clarification = self._build_clarification(normalized_text, candidates, unresolved_terms)
        return NormalizedPrompt(
            original_prompt=prompt,
            normalized_text=normalized_text,
            language="ko" if _contains_hangul(normalized_text) else "en",
            mode=self.mode,
            lexicon_version=self.lexicon.lexicon_version,
            lexicon_hash=self.lexicon.content_hash,
            raw_prompt_hash=text_audit.raw_prompt_hash,
            normalized_text_hash=text_audit.normalized_text_hash,
            unicode_warnings=list(text_audit.unicode_warnings),
            candidates=candidates,
            unresolved_terms=unresolved_terms,
            clarification=clarification,
        )

    def _matching_false_positive_guard(self, text: str) -> str | None:
        lowered = text.lower()
        for guard in self.lexicon.false_positive_guards:
            if _contains_any(lowered, (guard,)):
                return guard
        return None

    def _extract_nullability_candidate(
        self,
        text: str,
        resolver: ColumnResolver,
    ) -> tuple[NormalizedIntentCandidate | None, list[UnresolvedPromptTerm]]:
        lowered = text.lower()
        if "결측률" in lowered or "누락률" in lowered:
            return None, []
        if _contains_any(lowered, self.lexicon.intent_synonyms["not_null"]):
            resolution = resolver.resolve(text)
            return (
                NormalizedIntentCandidate(
                    intent="not_null",
                    columns=list(resolution.columns),
                    params={},
                    confidence=0.92,
                    rationale="Canonicalized from required/not-null phrasing.",
                ),
                list(resolution.unresolved_terms),
            )
        if _contains_any(lowered, self.lexicon.intent_synonyms["null"]):
            resolution = resolver.resolve(text)
            return (
                NormalizedIntentCandidate(
                    intent="null",
                    columns=list(resolution.columns),
                    params={},
                    confidence=0.86,
                    rationale="Canonicalized from null-required phrasing.",
                ),
                list(resolution.unresolved_terms),
            )
        return None, []

    def _extract_unique_candidate(
        self,
        text: str,
        resolver: ColumnResolver,
    ) -> tuple[NormalizedIntentCandidate | None, list[UnresolvedPromptTerm]]:
        lowered = text.lower()
        if "비율" in lowered or "ratio" in lowered or "률" in lowered:
            return None, []
        if not _contains_any(lowered, self.lexicon.intent_synonyms["unique"]):
            return None, []
        resolution = resolver.resolve_required(text)
        return (
            NormalizedIntentCandidate(
                intent="unique",
                columns=list(resolution.columns),
                params={},
                confidence=0.9,
                rationale="Canonicalized from uniqueness phrasing.",
            ),
            list(resolution.unresolved_terms),
        )

    def _extract_numeric_candidate(
        self,
        text: str,
        resolver: ColumnResolver,
    ) -> tuple[NormalizedIntentCandidate | None, list[UnresolvedPromptTerm]]:
        lowered = text.lower()
        if _contains_any(lowered, self.lexicon.intent_synonyms["mean_between"]) or _contains_any(
            lowered,
            self.lexicon.intent_synonyms["sum_between"],
        ):
            return None, []
        if _EXCLUSIVE_RE.search(text):
            return None, [UnresolvedPromptTerm(term="exclusive_range", reason="exclusive_bounds_not_supported")]
        params = _extract_numeric_bounds(text)
        if not params:
            return None, []
        resolution = resolver.resolve_required(text)
        return (
            NormalizedIntentCandidate(
                intent="between",
                columns=list(resolution.columns),
                params=params,
                confidence=0.88,
                rationale="Canonicalized from numeric bound phrasing.",
            ),
            list(resolution.unresolved_terms),
        )

    def _extract_aggregate_candidate(
        self,
        text: str,
        resolver: ColumnResolver,
    ) -> tuple[NormalizedIntentCandidate | None, list[UnresolvedPromptTerm]]:
        lowered = text.lower()
        intent = None
        if _contains_any(lowered, self.lexicon.intent_synonyms["mean_between"]):
            intent = "mean_between"
        elif _contains_any(lowered, self.lexicon.intent_synonyms["sum_between"]):
            intent = "sum_between"
        if intent is None:
            return None, []
        if _EXCLUSIVE_RE.search(text):
            return None, [UnresolvedPromptTerm(term="exclusive_range", reason="exclusive_bounds_not_supported")]
        params = _extract_numeric_bounds(text)
        if not params:
            return None, [UnresolvedPromptTerm(term=intent, reason="range_missing")]
        resolution = resolver.resolve_required(text)
        return (
            NormalizedIntentCandidate(
                intent=intent,
                columns=list(resolution.columns),
                params=params,
                confidence=0.84,
                rationale="Canonicalized from aggregate bound phrasing.",
            ),
            list(resolution.unresolved_terms),
        )

    def _extract_ratio_candidate(
        self,
        text: str,
        resolver: ColumnResolver,
    ) -> tuple[NormalizedIntentCandidate | None, list[UnresolvedPromptTerm]]:
        lowered = text.lower()
        percent = _PERCENT_RE.search(text)
        if percent is None:
            return None, []
        value = float(percent.group("value")) / 100.0
        if _contains_any(lowered, self.lexicon.intent_synonyms["completeness_ratio"]):
            resolution = resolver.resolve_required(text)
            return (
                NormalizedIntentCandidate(
                    intent="completeness_ratio",
                    columns=list(resolution.columns),
                    params={"min_ratio": round(max(0.0, min(1.0, 1.0 - value)), 6)},
                    confidence=0.86,
                    rationale="Canonicalized from missing-rate phrasing.",
                ),
                list(resolution.unresolved_terms),
            )
        if _contains_any(lowered, self.lexicon.intent_synonyms["unique_ratio"]):
            resolution = resolver.resolve_required(text)
            return (
                NormalizedIntentCandidate(
                    intent="unique_ratio",
                    columns=list(resolution.columns),
                    params={"min_ratio": round(max(0.0, min(1.0, value)), 6)},
                    confidence=0.86,
                    rationale="Canonicalized from unique-ratio phrasing.",
                ),
                list(resolution.unresolved_terms),
            )
        return None, []

    def _extract_format_candidate(
        self,
        text: str,
        resolver: ColumnResolver,
    ) -> tuple[NormalizedIntentCandidate | None, list[UnresolvedPromptTerm]]:
        lowered = text.lower()
        format_kind = None
        for canonical, synonyms in self.lexicon.format_synonyms.items():
            if _contains_any(lowered, synonyms):
                format_kind = canonical
                break
        if format_kind is None or not _contains_any(lowered, self.lexicon.intent_synonyms["format"]):
            return None, []
        resolution = resolver.resolve_required(text, preferred_terms=list(self.lexicon.format_synonyms[format_kind]))
        return (
            NormalizedIntentCandidate(
                intent="format",
                columns=list(resolution.columns),
                params={"format": format_kind},
                confidence=0.87,
                rationale=f"Canonicalized from {format_kind} format phrasing.",
            ),
            list(resolution.unresolved_terms),
        )

    def _extract_enum_candidate(
        self,
        text: str,
        resolver: ColumnResolver,
    ) -> tuple[NormalizedIntentCandidate | None, list[UnresolvedPromptTerm]]:
        lowered = text.lower()
        if not _contains_any(lowered, self.lexicon.intent_synonyms["in_set"]):
            return None, []
        values = _extract_enum_values(text)
        if not values:
            return None, [UnresolvedPromptTerm(term="allowed_values", reason="enum_values_missing")]
        resolution = resolver.resolve_required(text)
        return (
            NormalizedIntentCandidate(
                intent="in_set",
                columns=list(resolution.columns),
                params={"allowed_values": values},
                confidence=0.83,
                rationale="Canonicalized from allowed-value phrasing.",
            ),
            list(resolution.unresolved_terms),
        )

    def _extract_length_candidate(
        self,
        text: str,
        resolver: ColumnResolver,
    ) -> tuple[NormalizedIntentCandidate | None, list[UnresolvedPromptTerm]]:
        lowered = text.lower()
        if not _contains_any(lowered, self.lexicon.intent_synonyms["length"]):
            return None, []
        params: dict[str, Any] = {}
        range_match = _LENGTH_RANGE_RE.search(text)
        if range_match:
            params = {
                "min_length": int(float(range_match.group("min"))),
                "max_length": int(float(range_match.group("max"))),
            }
        else:
            exact_match = _LENGTH_EXACT_RE.search(text)
            if exact_match:
                value = int(float(exact_match.group("value")))
                params = {"min_length": value, "max_length": value}
        if not params:
            return None, [UnresolvedPromptTerm(term="length", reason="length_bounds_missing")]
        resolution = resolver.resolve_required(text)
        return (
            NormalizedIntentCandidate(
                intent="length",
                columns=list(resolution.columns),
                params=params,
                confidence=0.82,
                rationale="Canonicalized from length phrasing.",
            ),
            list(resolution.unresolved_terms),
        )

    def _build_clarification(
        self,
        text: str,
        candidates: list[NormalizedIntentCandidate],
        unresolved_terms: list[UnresolvedPromptTerm],
    ) -> ClarificationRequest | None:
        lowered = text.lower()
        if candidates and not unresolved_terms:
            return None
        if _contains_any(lowered, self.lexicon.ambiguous_markers) or (not candidates and _contains_hangul(text)):
            return ClarificationRequest(
                reason="prompt_too_ambiguous",
                suggestions=[
                    "컬럼명과 기대 조건을 함께 적어주세요. 예: 이메일은 비어 있으면 안 됩니다.",
                    "숫자 범위는 기준값을 포함해 적어주세요. 예: score는 0 이상 100 이하.",
                ],
            )
        if unresolved_terms:
            return ClarificationRequest(
                reason="unresolved_terms",
                suggestions=["컬럼명 또는 허용값을 더 명확히 적어주세요."],
            )
        return None


def normalize_prompt_text(text: str) -> str:
    return normalize_prompt_text_with_audit(text).normalized_text


def normalize_prompt_text_with_audit(text: str) -> PromptTextNormalizationResult:
    original = str(text)
    nfkc_text = unicodedata.normalize("NFKC", original)
    normalized = _WHITESPACE_RE.sub(" ", nfkc_text).strip()
    events = _build_normalization_events(original, nfkc_text, normalized)
    warnings = _detect_unicode_warnings(original, normalized)
    blocking_warning = _select_blocking_unicode_warning(warnings)
    return PromptTextNormalizationResult(
        normalized_text=normalized,
        raw_prompt_hash=_hash_prompt_text(original),
        normalized_text_hash=_hash_prompt_text(normalized),
        normalization_events=events,
        unicode_warnings=warnings,
        blocking_warning=blocking_warning,
    )


def get_prompt_normalization_mode() -> PromptNormalizationMode:
    raw = os.getenv(PROMPT_NORMALIZATION_ENV, PromptNormalizationMode.ENFORCE.value)
    try:
        return PromptNormalizationMode(str(raw).strip().lower())
    except ValueError:
        from truthound.ai.prompt_metrics import record_prompt_normalization_invalid_mode

        record_prompt_normalization_invalid_mode(str(raw))
        return PromptNormalizationMode.ENFORCE


def normalize_suite_prompt(
    prompt: str,
    *,
    data: Any = None,
    source: Any = None,
    context: TruthoundContext | None = None,
    sample_size: int = 1000,
    columns: list[str] | tuple[str, ...] | None = None,
) -> NormalizedPrompt:
    from truthound.ai.prompt_metrics import record_prompt_normalization_result

    if data is not None or source is not None:
        active_context = context or get_context()
        bundle = ContextBundleBuilder(summary_budget=sample_size).build(
            data=data,
            source=source,
            context=active_context,
        )
        normalized = PromptNormalizer().normalize(prompt, context_bundle=bundle)
        record_prompt_normalization_result(normalized)
        return normalized
    normalized = PromptNormalizer().normalize(prompt, columns=columns or ())
    record_prompt_normalization_result(normalized)
    return normalized


def _extract_numeric_bounds(text: str) -> dict[str, Any]:
    match = _LOWER_UPPER_RE.search(text) or _RANGE_RE.search(text)
    if match:
        return {
            "min_value": _coerce_number(match.group("min")),
            "max_value": _coerce_number(match.group("max")),
            "inclusive": True,
        }
    lower_match = _LOWER_ONLY_RE.search(text) or _SYMBOL_LOWER_ONLY_RE.search(text)
    if lower_match:
        return {"min_value": _coerce_number(lower_match.group("min")), "inclusive": True}
    upper_match = _UPPER_ONLY_RE.search(text) or _SYMBOL_UPPER_ONLY_RE.search(text)
    if upper_match:
        return {"max_value": _coerce_number(upper_match.group("max")), "inclusive": True}
    return {}


def _build_normalization_events(
    original: str,
    nfkc_text: str,
    normalized: str,
) -> list[PromptNormalizationEvent]:
    events: list[PromptNormalizationEvent] = []
    category_counts: dict[tuple[str, str, str], int] = {}
    for char in original:
        nfkc_char = unicodedata.normalize("NFKC", char)
        if nfkc_char == char:
            continue
        kind = _normalization_event_kind(char, nfkc_char)
        key = (kind, char, nfkc_char)
        category_counts[key] = category_counts.get(key, 0) + 1
    for (kind, before, after), count in sorted(category_counts.items()):
        events.append(PromptNormalizationEvent(kind=kind, before=before, after=after, count=count))
    if nfkc_text != normalized:
        events.append(
            PromptNormalizationEvent(
                kind="whitespace_collapse",
                before="nfkc_text",
                after="normalized_text",
                count=1,
            )
        )
    return events


def _normalization_event_kind(char: str, normalized: str) -> str:
    if char.isspace() and normalized.isspace():
        return "space_compatibility"
    if _is_hangul_jamo(char) and _contains_hangul(normalized):
        return "hangul_jamo_composed"
    if unicodedata.east_asian_width(char).lower() in _SAFE_WIDTH_OR_SPACING_CATEGORIES:
        return "width_compatibility"
    if char in _SAFE_COMPATIBILITY_EXCEPTIONS:
        return "symbol_compatibility"
    return "compatibility_mapping"


def _detect_unicode_warnings(original: str, normalized: str) -> list[str]:
    warnings: list[str] = []
    if any(_is_residual_hangul_jamo(char) for char in normalized):
        warnings.append("hangul_jamo_residual")
    if any(unicodedata.combining(char) for char in normalized):
        warnings.append("combining_mark_residual")
    unsupported_format_controls = [
        char
        for char in original
        if unicodedata.category(char) == "Cf" and char not in _SAFE_FORMAT_CONTROL_CHARS
    ]
    if unsupported_format_controls:
        warnings.append("unsupported_format_control_residue")
    if _has_unsafe_compatibility_change(original):
        warnings.append("compatibility_symbol_changed")
    return _dedupe(warnings)


def _select_blocking_unicode_warning(warnings: list[str]) -> str | None:
    for warning in (
        "hangul_jamo_residual",
        "combining_mark_residual",
        "unsupported_format_control_residue",
        "compatibility_symbol_changed",
    ):
        if warning in warnings:
            return warning
    return None


def _has_unsafe_compatibility_change(text: str) -> bool:
    for char in text:
        nfkc_char = unicodedata.normalize("NFKC", char)
        if nfkc_char == char:
            continue
        if char in _SAFE_COMPATIBILITY_EXCEPTIONS:
            continue
        if char.isspace() and nfkc_char.isspace():
            continue
        if _is_hangul_jamo(char):
            continue
        category = unicodedata.east_asian_width(char).lower()
        if category in _SAFE_WIDTH_OR_SPACING_CATEGORIES:
            continue
        if unicodedata.category(char) == "Cf" and char in _SAFE_FORMAT_CONTROL_CHARS:
            continue
        return True
    return False


def _hash_prompt_text(text: str) -> str:
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return f"{_HASH_BYTES_PREFIX}{digest[:16]}"


def _extract_enum_values(text: str) -> list[str]:
    head = re.split(r"중\s*하나|중에서|만\s*허용|허용", text, maxsplit=1)[0]
    if "는" in head:
        head = head.rsplit("는", 1)[1]
    elif "은" in head:
        head = head.rsplit("은", 1)[1]
    raw_values = re.split(r"[/,|]| 또는 | 혹은 | 그리고 ", head)
    values = [item.strip(" `\"'[]{}()") for item in raw_values]
    return [item for item in _dedupe(values) if item and not re.fullmatch(r"\w+컬럼|\w+필드", item)]


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    compact_text = _compact(text.lower())
    return any(_compact(marker.lower()) in compact_text for marker in markers)


def _contains_hangul(text: str) -> bool:
    return any("가" <= char <= "힣" for char in text)


def _is_hangul_jamo(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xD7B0 <= codepoint <= 0xD7FF
    )


def _is_residual_hangul_jamo(char: str) -> bool:
    return _is_hangul_jamo(char)


def _normalize_identifier(value: str) -> str:
    normalized = normalize_prompt_text(value).lower()
    normalized = re.sub(r"(컬럼|필드|열)$", "", normalized)
    return _compact(normalized)


def _compact(value: str) -> str:
    return re.sub(r"[\s_\-.]+", "", value)


def _coerce_number(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _dedupe_candidates(candidates: list[NormalizedIntentCandidate]) -> list[NormalizedIntentCandidate]:
    seen: set[str] = set()
    result: list[NormalizedIntentCandidate] = []
    for candidate in candidates:
        key = json.dumps(
            {
                "intent": candidate.intent,
                "columns": candidate.columns,
                "params": candidate.params,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _dedupe_unresolved(items: list[UnresolvedPromptTerm]) -> list[UnresolvedPromptTerm]:
    seen: set[tuple[str, str]] = set()
    result: list[UnresolvedPromptTerm] = []
    for item in items:
        key = (item.term, item.reason)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


__all__ = [
    "ClarificationRequest",
    "ColumnResolver",
    "IntentCanonicalizer",
    "NormalizedIntentCandidate",
    "NormalizedPrompt",
    "PROMPT_NORMALIZATION_ENV",
    "PromptNormalizationEvent",
    "PromptNormalizationMode",
    "PromptTextNormalizationResult",
    "PromptNormalizer",
    "UnresolvedPromptTerm",
    "get_prompt_normalization_mode",
    "normalize_suite_prompt",
    "normalize_prompt_text",
    "normalize_prompt_text_with_audit",
]
