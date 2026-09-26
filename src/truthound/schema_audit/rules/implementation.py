"""Exact design/catalog comparison; never infer renames or evaluate SQL."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Literal

from .._contract import Contract, require
from ..limits import MAX_AUDIT_CHECKS, MAX_AUDIT_OBJECTS
from ..models import (
    Attribute,
    CatalogSnapshot,
    DesignRevision,
    SourceLocation,
    State,
    absent,
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
from .standards import FACETS

RULE_VERSION = "implementation-v2"
RULE_PACK_DIGEST = hashlib.sha256(
    (
        RULE_VERSION
        + POLICY_VERSION
        + repr(FACETS)
        + "exact-name;ordered-keys;comment-nfc;literal-defaults;coverage-required"
    ).encode()
).hexdigest()
MAX_OBJECTS = MAX_AUDIT_OBJECTS


@dataclass(frozen=True, kw_only=True)
class ImplementationCheck(Contract):
    fingerprint: str
    rule_id: str
    object_id: str
    verdict: Verdict
    expected: Attribute
    observed: Attribute
    design_sources: tuple[SourceLocation, ...]
    catalog_sources: tuple[SourceLocation, ...]


@dataclass(frozen=True, kw_only=True)
class ImplementationReport(Contract):
    contract_version: Literal["truthound.implementation-audit/1"]
    rule_version: str
    rule_pack_digest: str
    design_digest: str
    catalog_digest: str
    design_id: str
    catalog_id: str
    target_count: int
    checks: tuple[ImplementationCheck, ...] = field(metadata={"max_items": MAX_AUDIT_CHECKS})

    @property
    def verdict(self) -> str:
        counts = Counter(c.verdict for c in self.checks)
        if counts[Verdict.FAIL]:
            return "NONCONFORMANT"
        if not self.target_count or counts[Verdict.UNKNOWN] or not counts[Verdict.PASS]:
            return "INDETERMINATE"
        return "CONFORMANT"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["verdict"] = self.verdict
        counts = Counter(c.verdict.value for c in self.checks)
        result["counts"] = {v.value: counts[v.value] for v in Verdict}
        coverage = defaultdict(Counter)
        for c in self.checks:
            coverage[c.rule_id][c.verdict.value] += 1
        result["coverage"] = {k: dict(v) for k, v in sorted(coverage.items())}
        return result


def audit_implementation(design: DesignRevision, catalog: CatalogSnapshot) -> ImplementationReport:
    """Compare a declared design to one immutable snapshot, with COMMENT logical policy.

    Missing objects are failures only when the corresponding collection coverage
    is complete. Differences between unsupported expressions are UNKNOWN, not
    asserted semantic inequality. Constraint names are part of the exact contract.
    """
    require(type(design) is DesignRevision and type(catalog) is CatalogSnapshot, "invalid_model")
    for doc in (design, catalog):
        require(
            sum(1 + len(t.columns) + len(t.constraints) for t in doc.tables) <= MAX_OBJECTS,
            "audit_object_limit",
        )
    ds, cs = defaultdict(list), defaultdict(list)
    for doc, index in ((design, ds), (catalog, cs)):
        for source in doc.sources:
            index[source.object_id].append(source)
            require(len(index[source.object_id]) <= 32, "audit_source_limit")
    checks = []
    complete = {c.facet: c.state == "COMPLETE" for c in catalog.coverage}

    def add(rule, obj, expected, observed, other="", force=None):
        verdict = force
        if verdict is None:
            verdict = (
                Verdict.UNKNOWN
                if any(
                    a.state in (State.UNKNOWN, State.NOT_APPLICABLE) for a in (expected, observed)
                )
                else Verdict.PASS
                if expected == observed
                else Verdict.FAIL
            )
        fingerprint = hashlib.sha256(
            json.dumps(
                [RULE_VERSION, rule, obj, other, verdict, asdict(expected), asdict(observed)],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        checks.append(
            ImplementationCheck(
                fingerprint=fingerprint,
                rule_id=rule,
                object_id=obj,
                verdict=verdict,
                expected=expected,
                observed=observed,
                design_sources=tuple(sorted(ds[obj], key=lambda s: (s.input_digest, s.pointer))),
                catalog_sources=tuple(sorted(cs[other], key=lambda s: (s.input_digest, s.pointer))),
            )
        )

    for facet, value in sorted(complete.items()):
        add(
            "coverage." + facet.lower(), design.id, known(True), known(True) if value else unknown()
        )
    dialect_ok = (
        design.dialect == catalog.dialect
        and design.dialect.name == "postgresql"
        and design.dialect.major in (14, 17)
    )
    add(
        "dialect",
        design.id,
        known(f"{design.dialect.name}:{design.dialect.major}"),
        known(f"{catalog.dialect.name}:{catalog.dialect.major}") if dialect_ok else unknown(),
    )

    def name(identifier, dialect):
        value = normalize_identifier(identifier, dialect)
        return value.value if value.state is State.VALUE else None

    def index(items, key, side):
        result = defaultdict(list)
        for item in items:
            k = key(item)
            if k is None or (isinstance(k, tuple) and None in k):
                add("name.unsupported", item.id, unknown(), unknown())
            else:
                result[k].append(item)
        for values in result.values():
            if len(values) > 1:
                add("name.ambiguous." + side, values[0].id, unknown(), unknown())
        return result

    def pairs(left, right, key_left, key_right, facet, rule):
        li, ri = index(left, key_left, "design"), index(right, key_right, "catalog")
        names_complete = sum(map(len, li.values())) == len(left) and sum(
            map(len, ri.values())
        ) == len(right)
        for k in sorted(set(li) | set(ri)):
            a, b = li.get(k, []), ri.get(k, [])
            if len(a) > 1 or len(b) > 1:
                continue
            if not a or not b:
                obj = (a or b)[0]
                add(
                    rule + ".presence",
                    obj.id,
                    known(True) if a else absent(),
                    (known(True) if b else absent())
                    if complete[facet] and dialect_ok and names_complete
                    else unknown(),
                    obj.id if b else "",
                )
            else:
                add(rule + ".presence", a[0].id, known(True), known(True), b[0].id)
                yield a[0], b[0]

    def logical(a, b):
        exp, obs = a.logical_name, b.logical_name
        add(
            "logical.comment",
            a.id,
            known(normalize_logical_name(exp.name.value))
            if exp.name.state is State.VALUE
            else unknown(),
            known(normalize_logical_name(obs.name.value))
            if obs.source == "COMMENT" and obs.name.state is State.VALUE
            else unknown(),
            b.id,
        )

    def expression(rule, a, b, obj, other):
        if a.state is State.VALUE and b.state is State.VALUE:
            av, bv = a.value.strip(), b.value.strip()
            literal = r"(?:[+-]?[0-9]+(?:\.[0-9]+)?|true|false|null|'(?:[^']|'')*')"
            force = (
                None
                if av == bv or (re.fullmatch(literal, av) and re.fullmatch(literal, bv))
                else Verdict.UNKNOWN
            )
            add(rule, obj, known(av), known(bv), other, force)
        else:
            add(rule, obj, a, b, other)

    def table_key(t, dialect):
        return name(t.namespace, dialect), name(t.physical_name, dialect)

    scope = {n.text for n in catalog.scope}
    scoped = []
    for table in design.tables:
        namespace = name(table.namespace, design.dialect)
        if namespace is None:
            add("scope", table.id, known(True), unknown())
            scoped.append(table)
        elif namespace not in scope:
            add("scope", table.id, known(True), unknown())
        else:
            scoped.append(table)
    design_tables, catalog_tables = (
        {t.id: t for t in design.tables},
        {t.id: t for t in catalog.tables},
    )

    def references(key, table, tables, dialect):
        columns = {c.id: c for c in table.columns}
        values = [name(columns[c].physical_name, dialect) for c in key.column_ids]
        target = None
        if key.referenced_table_id:
            rt = tables[key.referenced_table_id]
            rc = {c.id: c for c in rt.columns}
            target = [
                table_key(rt, dialect),
                [name(rc[c].physical_name, dialect) for c in key.referenced_column_ids],
            ]
        raw = json.dumps([values, target, key.on_update, key.on_delete], ensure_ascii=False)
        if (
            len(raw) > 4096
            or None in values
            or (target and (None in target[0] or None in target[1]))
        ):
            return unknown()
        return known(raw)

    if dialect_ok:
        for left, right in pairs(
            scoped,
            catalog.tables,
            lambda t: table_key(t, design.dialect),
            lambda t: table_key(t, catalog.dialect),
            "TABLES",
            "table",
        ):
            logical(left, right)
            for a, b in pairs(
                left.columns,
                right.columns,
                lambda c: name(c.physical_name, design.dialect),
                lambda c: name(c.physical_name, catalog.dialect),
                "COLUMNS",
                "column",
            ):
                logical(a, b)
                add(
                    "column.type",
                    a.id,
                    normalize_type(a.type, design.dialect),
                    normalize_type(b.type, catalog.dialect),
                    b.id,
                )
                for facet in FACETS:
                    add(
                        "column.type." + facet,
                        a.id,
                        getattr(a.type, facet),
                        getattr(b.type, facet),
                        b.id,
                    )
                add("column.nullable", a.id, a.nullable, b.nullable, b.id)
                add("column.ordinal", a.id, known(a.ordinal), known(b.ordinal), b.id)
                expression("column.default", a.default, b.default, a.id, b.id)
            for a, b in pairs(
                left.constraints,
                right.constraints,
                lambda k: name(k.name, design.dialect),
                lambda k: name(k.name, catalog.dialect),
                "CONSTRAINTS",
                "constraint",
            ):
                add("constraint.kind", a.id, known(a.kind.value), known(b.kind.value), b.id)
                add(
                    "constraint.references",
                    a.id,
                    references(a, left, design_tables, design.dialect),
                    references(b, right, catalog_tables, catalog.dialect),
                    b.id,
                )
                add("constraint.deferrable", a.id, a.deferrable, b.deferrable, b.id)
                expression("constraint.expression", a.expression, b.expression, a.id, b.id)
    return ImplementationReport(
        contract_version="truthound.implementation-audit/1",
        rule_version=RULE_VERSION,
        rule_pack_digest=RULE_PACK_DIGEST,
        design_digest=semantic_digest(design),
        catalog_digest=semantic_digest(catalog),
        design_id=design.id,
        catalog_id=catalog.id,
        target_count=len(design.tables),
        checks=tuple(sorted(checks, key=lambda c: (c.rule_id, c.object_id, c.fingerprint))),
    )
