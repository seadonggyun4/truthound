"""Exact standards-to-design audit. No I/O, fuzzy approvals or row inspection."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Literal

from .._contract import Contract, opaque, require
from ..limits import MAX_AUDIT_CHECKS, MAX_AUDIT_OBJECTS
from ..models import (
    Attribute,
    Column,
    DesignRevision,
    SourceLocation,
    StandardRevision,
    State,
    Table,
    Term,
    Word,
    known,
    unknown,
)
from ..normalization import (
    POLICY_VERSION,
    normalize_identifier,
    normalize_logical_name,
    normalize_type,
)
from ..results import Verdict
from ..serialization import semantic_digest

RULE_VERSION = "standards-v1"
RULE_IDS = (
    "name.syntax",
    "name.logical",
    "name.abbreviation",
    "term.active",
    "domain.assignment",
    "domain.type",
    "structure.unique",
    "structure.references",
)
FACETS = (
    "type_schema",
    "length",
    "length_unit",
    "precision",
    "scale",
    "timezone",
    "array_rank",
    "domain_ref",
    "enum_values_ref",
    "collation",
)
RULE_PACK_DIGEST = hashlib.sha256(
    json.dumps(
        {
            "version": RULE_VERSION,
            "rules": RULE_IDS,
            "facets": FACETS,
            "normalization": POLICY_VERSION,
            "empty": "INDETERMINATE",
            "failure_priority": True,
        },
        sort_keys=True,
    ).encode()
).hexdigest()
MAX_OBJECTS = MAX_AUDIT_OBJECTS


@dataclass(frozen=True, kw_only=True)
class TermBinding(Contract):
    """Trusted caller's approved mapping, bound to both semantic input digests.

    This value does not authorize itself. HTTP callers cannot supply bindings in
    the Depot v1 runner; a future mapping repository must verify approval first.
    """

    object_id: str
    term_id: str
    standard_digest: str
    design_digest: str
    approval_digest: str

    def _validate(self) -> None:
        opaque(self.object_id, "/object_id")
        opaque(self.term_id, "/term_id")
        for name in ("standard_digest", "design_digest", "approval_digest"):
            require(bool(re.fullmatch(r"[0-9a-f]{64}", getattr(self, name))), "invalid_digest")


@dataclass(frozen=True, kw_only=True)
class AuditCheck(Contract):
    fingerprint: str
    rule_id: str
    rule_version: str
    object_id: str
    verdict: Verdict
    code: str
    expected: Attribute
    observed: Attribute
    sources: tuple[SourceLocation, ...]
    standard_sources: tuple[SourceLocation, ...]
    standard_ids: tuple[str, ...]
    suggestion: str | None


@dataclass(frozen=True, kw_only=True)
class StandardsReport(Contract):
    contract_version: Literal["truthound.standards-audit/1"]
    rule_pack_digest: str
    standard_digest: str
    design_digest: str
    standard_id: str
    standard_revision: int
    design_id: str
    design_revision: int
    mapping_digest: str
    target_count: int
    checks: tuple[AuditCheck, ...] = field(metadata={"max_items": MAX_AUDIT_CHECKS})

    @property
    def verdict(self) -> str:
        counts = Counter(check.verdict for check in self.checks)
        if counts[Verdict.FAIL]:
            return "NONCONFORMANT"
        if not self.target_count or counts[Verdict.UNKNOWN] or not counts[Verdict.PASS]:
            return "INDETERMINATE"
        return "CONFORMANT"

    def as_dict(self) -> dict:
        result = asdict(self)
        counts = Counter(check.verdict.value for check in self.checks)
        result["verdict"] = self.verdict
        result["counts"] = {v.value: counts[v.value] for v in Verdict}
        coverage = defaultdict(Counter)
        for check in self.checks:
            coverage[check.rule_id][check.verdict.value] += 1
        result["coverage"] = {
            rule: {
                "targets": sum(values.values()),
                "evaluated": values["PASS"] + values["FAIL"],
                **{v.value: values[v.value] for v in Verdict},
            }
            for rule, values in sorted(coverage.items())
        }
        return result


def _same(expected: Attribute, observed: Attribute) -> Verdict:
    # Missing applicable information never becomes equality by comparing nulls.
    if any(v.state in (State.UNKNOWN, State.NOT_APPLICABLE) for v in (expected, observed)):
        return Verdict.UNKNOWN
    return Verdict.PASS if expected == observed else Verdict.FAIL


def _physical(words: tuple[Word, ...], standard: StandardRevision) -> str | None:
    length = sum(len(word.abbreviation) for word in words) + max(0, len(words) - 1) * len(
        standard.naming_policy.separator
    )
    if length > 4096:
        return None
    value = standard.naming_policy.separator.join(word.abbreviation for word in words)
    policy = standard.naming_policy.case
    if policy == "LOWER_ASCII":
        value = value.translate(
            str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
        )
    elif policy == "UPPER_ASCII":
        value = value.translate(
            str.maketrans("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        )
    return value


def audit_standards(
    standard: StandardRevision,
    design: DesignRevision,
    *,
    bindings: tuple[TermBinding, ...] = (),
) -> StandardsReport:
    """Run the complete v1 pack; invalid models/mappings raise ModelError.

    All applicable checks are retained, including PASS and UNKNOWN. Table names
    resolve exact logical words or terms; columns require exact logical terms.
    Domain assignment follows the resolved term, not a DB user-defined type.
    """
    require(type(standard) is StandardRevision and type(design) is DesignRevision, "invalid_model")
    require(
        type(bindings) is tuple and all(type(b) is TermBinding for b in bindings),
        "invalid_bindings",
    )
    objects = [(t, t) for t in design.tables]
    objects += [(t, c) for t in design.tables for c in t.columns]
    require(len(objects) <= MAX_OBJECTS, "audit_object_limit")
    sd, dd = semantic_digest(standard), semantic_digest(design)
    terms = {term.id: term for term in standard.terms}
    words = {word.id: word for word in standard.words}
    domains = {domain.id: domain for domain in standard.domains}
    term_names, word_names, abbreviations = defaultdict(list), defaultdict(list), defaultdict(list)
    for term in standard.terms:
        term_names[normalize_logical_name(term.logical_name)].append(term)
    for word in standard.words:
        word_names[normalize_logical_name(word.logical_name)].append(word)
        abbreviations[_physical((word,), standard)].append(word.id)
    bound = {}
    object_ids = {obj.id for _, obj in objects}
    for binding in bindings:
        require(binding.object_id in object_ids and binding.term_id in terms, "broken_mapping")
        require(binding.object_id not in bound, "ambiguous_mapping")
        require((binding.standard_digest, binding.design_digest) == (sd, dd), "stale_mapping")
        bound[binding.object_id] = binding
    mapping_digest = hashlib.sha256(
        json.dumps(
            [asdict(bound[key]) for key in sorted(bound)], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    source_index = defaultdict(list)
    for source in design.sources:
        source_index[source.object_id].append(source)
    standard_sources = defaultdict(list)
    for source in standard.sources:
        standard_sources[source.object_id].append(source)
    require(
        all(
            len(values) <= 32
            for index in (source_index, standard_sources)
            for values in index.values()
        ),
        "audit_source_limit",
    )

    @lru_cache(maxsize=2000)
    def resolved_word_info(identifier: str, is_term: bool):
        entry = terms[identifier] if is_term else words[identifier]
        members = tuple(words[word] for word in entry.word_ids) if is_term else (entry,)
        active = not entry.deprecated and all(not word.deprecated for word in members)
        ambiguous = any(len(abbreviations[_physical((word,), standard)]) != 1 for word in members)
        return active, _physical(members, standard), ambiguous

    checks = []

    def emit(
        rule: str,
        obj: Table | Column,
        verdict: Verdict,
        code: str,
        expected: Attribute,
        observed: Attribute,
        refs: tuple[str, ...] = (),
        suggestion: str | None = None,
    ) -> None:
        # Stable across input ordering; input digests bind the enclosing report.
        identity = [
            RULE_VERSION,
            rule,
            obj.id,
            verdict.value,
            code,
            asdict(expected),
            asdict(observed),
            sorted(refs),
            mapping_digest,
        ]
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        checks.append(
            AuditCheck(
                fingerprint=fingerprint,
                rule_id=rule,
                rule_version=RULE_VERSION,
                object_id=obj.id,
                verdict=verdict,
                code=code,
                expected=expected,
                observed=observed,
                sources=tuple(
                    sorted(source_index[obj.id], key=lambda s: (s.input_digest, s.pointer))
                ),
                standard_sources=tuple(
                    sorted(
                        (source for ref in refs for source in standard_sources[ref]),
                        key=lambda s: (s.object_id, s.input_digest, s.pointer),
                    )
                ),
                standard_ids=tuple(sorted(refs)),
                suggestion=suggestion,
            )
        )

    table_keys = Counter()
    column_keys = defaultdict(Counter)
    for table in design.tables:
        ns = normalize_identifier(table.namespace, design.dialect)
        name = normalize_identifier(table.physical_name, design.dialect)
        if ns.state is State.VALUE and name.state is State.VALUE:
            table_keys[(ns.value, name.value)] += 1
        for column in table.columns:
            key = normalize_identifier(column.physical_name, design.dialect)
            if key.state is State.VALUE:
                column_keys[table.id][key.value] += 1

    for table, obj in sorted(objects, key=lambda pair: pair[1].id):
        physical = obj.physical_name.text
        ascii_name = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*", physical))
        case = standard.naming_policy.case
        case_ok = case == "PRESERVE" or physical == (
            physical.lower() if case == "LOWER_ASCII" else physical.upper()
        )
        syntax_ok = (
            ascii_name and case_ok and len(physical.encode()) <= standard.naming_policy.max_bytes
        )
        if not standard.naming_policy.separator:
            syntax_ok = syntax_ok and "_" not in physical
        emit(
            "name.syntax",
            obj,
            Verdict.PASS if syntax_ok else Verdict.FAIL,
            "name_policy" if not syntax_ok else "matched",
            known(True),
            known(syntax_ok),
        )
        normalized = normalize_identifier(obj.physical_name, design.dialect)
        ns = normalize_identifier(table.namespace, design.dialect)
        if normalized.state is not State.VALUE or (
            isinstance(obj, Table) and ns.state is not State.VALUE
        ):
            emit(
                "structure.unique",
                obj,
                Verdict.UNKNOWN,
                "identifier_unsupported",
                known(1),
                unknown(),
            )
        else:
            count = (
                table_keys[(ns.value, normalized.value)]
                if isinstance(obj, Table)
                else column_keys[table.id][normalized.value]
            )
            emit(
                "structure.unique",
                obj,
                Verdict.PASS if count == 1 else Verdict.FAIL,
                "matched" if count == 1 else "duplicate_physical_name",
                known(1),
                known(count),
            )
        logical = obj.logical_name.name
        resolved: Term | Word | None = None
        explicit = bound.get(obj.id)
        if explicit:
            resolved = terms[explicit.term_id]
            emit(
                "name.logical",
                obj,
                Verdict.PASS,
                "approved_mapping",
                known(resolved.id),
                known(resolved.id),
                (resolved.id,),
            )
        elif logical.state in (State.UNKNOWN, State.NOT_APPLICABLE):
            emit(
                "name.logical",
                obj,
                Verdict.UNKNOWN,
                "logical_name_unknown",
                known("exact_dictionary_entry"),
                logical,
            )
        elif logical.state is State.ABSENT:
            emit(
                "name.logical",
                obj,
                Verdict.FAIL,
                "logical_name_absent",
                known("exact_dictionary_entry"),
                logical,
            )
        else:
            key = normalize_logical_name(logical.value)
            candidates = list(term_names[key])
            if isinstance(obj, Table):
                candidates += word_names[key]
            if len(candidates) == 1:
                resolved = candidates[0]
                emit(
                    "name.logical",
                    obj,
                    Verdict.PASS,
                    "matched",
                    known(resolved.logical_name),
                    logical,
                    (resolved.id,),
                )
            else:
                emit(
                    "name.logical",
                    obj,
                    Verdict.UNKNOWN if candidates else Verdict.FAIL,
                    "ambiguous_term" if candidates else "nonstandard_term",
                    known("exact_dictionary_entry"),
                    logical,
                )
        if resolved is None:
            emit(
                "name.abbreviation",
                obj,
                Verdict.UNKNOWN,
                "term_unresolved",
                unknown(),
                known(physical),
            )
            emit("term.active", obj, Verdict.UNKNOWN, "term_unresolved", known(True), unknown())
        else:
            active, expected_name, ambiguous = resolved_word_info(
                resolved.id, isinstance(resolved, Term)
            )
            emit(
                "term.active",
                obj,
                Verdict.PASS if active else Verdict.FAIL,
                "matched" if active else "deprecated_term_or_word",
                known(True),
                known(active),
                (resolved.id,),
            )
            if ambiguous or expected_name is None:
                emit(
                    "name.abbreviation",
                    obj,
                    Verdict.UNKNOWN,
                    "ambiguous_abbreviation" if ambiguous else "name_combination_limit",
                    unknown(),
                    known(physical),
                    (resolved.id,),
                )
            else:
                matched = physical == expected_name
                emit(
                    "name.abbreviation",
                    obj,
                    Verdict.PASS if matched else Verdict.FAIL,
                    "matched" if matched else "abbreviation_mismatch",
                    known(expected_name),
                    known(physical),
                    (resolved.id,),
                    expected_name if not matched else None,
                )
        if isinstance(obj, Table):
            # Core model construction already validates all ordered FK references.
            emit(
                "structure.references",
                obj,
                Verdict.PASS,
                "validated_graph",
                known(True),
                known(True),
            )
            continue
        if not isinstance(resolved, Term):
            emit(
                "domain.assignment",
                obj,
                Verdict.UNKNOWN,
                "domain_unresolved",
                known("term_domain"),
                unknown(),
            )
            emit(
                "domain.type",
                obj,
                Verdict.UNKNOWN,
                "domain_unresolved",
                unknown(),
                known(obj.type.native_type),
            )
            for facet in FACETS:
                emit(
                    "domain." + facet,
                    obj,
                    Verdict.UNKNOWN,
                    "domain_unresolved",
                    unknown(),
                    getattr(obj.type, facet),
                )
            continue
        domain = domains[resolved.domain_id]
        emit(
            "domain.assignment",
            obj,
            Verdict.PASS,
            "term_domain",
            known(domain.id),
            known(resolved.domain_id),
            (domain.id,),
        )
        expected_type = normalize_type(domain.type, standard.dialect)
        observed_type = normalize_type(obj.type, design.dialect)
        verdict = (
            _same(expected_type, observed_type)
            if standard.dialect == design.dialect
            else Verdict.UNKNOWN
        )
        emit(
            "domain.type",
            obj,
            verdict,
            "type_unresolved" if verdict is Verdict.UNKNOWN else "type_comparison",
            expected_type,
            observed_type,
            (domain.id,),
        )
        for facet in FACETS:
            expected, observed = getattr(domain.type, facet), getattr(obj.type, facet)
            verdict = (
                _same(expected, observed) if standard.dialect == design.dialect else Verdict.UNKNOWN
            )
            emit(
                "domain." + facet,
                obj,
                verdict,
                "facet_unresolved" if verdict is Verdict.UNKNOWN else "facet_comparison",
                expected,
                observed,
                (domain.id,),
            )
    return StandardsReport(
        contract_version="truthound.standards-audit/1",
        rule_pack_digest=RULE_PACK_DIGEST,
        standard_digest=sd,
        design_digest=dd,
        standard_id=standard.id,
        standard_revision=standard.revision,
        design_id=design.id,
        design_revision=design.revision,
        mapping_digest=mapping_digest,
        target_count=len(objects),
        checks=tuple(sorted(checks, key=lambda c: (c.object_id, c.rule_id, c.fingerprint))),
    )
