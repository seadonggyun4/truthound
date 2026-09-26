"""Immutable metadata values. No network, database, clock or approval authority."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ._contract import Contract, opaque, require, unique_ids


class State(str, Enum):
    VALUE = "VALUE"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Reason(str, Enum):
    NONE = "NONE"
    NOT_COLLECTED = "NOT_COLLECTED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    UNSUPPORTED = "UNSUPPORTED"
    INCOMPLETE = "INCOMPLETE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True, kw_only=True)
class Attribute(Contract):
    state: State
    value: str | int | bool | None
    reason: Reason

    def _validate(self) -> None:
        if self.state is State.VALUE:
            require(self.value is not None and self.reason is Reason.NONE, "invalid_observation")
        else:
            require(self.value is None, "unexpected_value", "/value")
            if self.state is State.ABSENT:
                require(self.reason is Reason.NONE, "unexpected_reason", "/reason")
            elif self.state is State.UNKNOWN:
                require(
                    self.reason not in (Reason.NONE, Reason.OUT_OF_SCOPE),
                    "unknown_reason_required",
                    "/reason",
                )
            else:
                require(self.reason is Reason.OUT_OF_SCOPE, "scope_reason_required", "/reason")


def known(value: str | int | bool) -> Attribute:
    return Attribute(state=State.VALUE, value=value, reason=Reason.NONE)


def absent() -> Attribute:
    return Attribute(state=State.ABSENT, value=None, reason=Reason.NONE)


def unknown(reason: Reason = Reason.NOT_COLLECTED) -> Attribute:
    return Attribute(state=State.UNKNOWN, value=None, reason=reason)


def _facet(value: Attribute, cls: type, path: str, minimum=None, maximum=None) -> None:
    if value.state is State.VALUE:
        require(type(value.value) is cls, "invalid_facet", path)
        if minimum is not None:
            require(value.value >= minimum, "facet_range", path)
        if maximum is not None:
            require(value.value <= maximum, "facet_range", path)


class NameOrigin(str, Enum):
    UNQUOTED = "UNQUOTED"
    QUOTED = "QUOTED"
    CATALOG = "CATALOG"


@dataclass(frozen=True, kw_only=True)
class Identifier(Contract):
    text: str
    origin: NameOrigin

    def _validate(self) -> None:
        require(bool(self.text), "empty_identifier", "/text")


@dataclass(frozen=True, kw_only=True)
class Dialect(Contract):
    name: str
    major: int

    def _validate(self) -> None:
        opaque(self.name, "/name")
        require(self.major > 0, "invalid_version", "/major")


@dataclass(frozen=True, kw_only=True)
class TypeDescriptor(Contract):
    native_type: str
    type_schema: Attribute
    length: Attribute
    length_unit: Attribute
    precision: Attribute
    scale: Attribute
    timezone: Attribute
    array_rank: Attribute
    domain_ref: Attribute
    enum_values_ref: Attribute
    collation: Attribute

    def _validate(self) -> None:
        require(bool(self.native_type), "empty_type", "/native_type")
        for key in ("type_schema", "length_unit", "domain_ref", "enum_values_ref", "collation"):
            _facet(getattr(self, key), str, "/" + key)
        _facet(self.length, int, "/length", 1)
        _facet(self.precision, int, "/precision", 1)
        _facet(self.scale, int, "/scale")
        _facet(self.array_rank, int, "/array_rank", 0, 32)
        _facet(self.timezone, bool, "/timezone")
        if self.length_unit.state is State.VALUE:
            require(
                self.length_unit.value in ("CHARACTERS", "OCTETS", "BITS"),
                "invalid_length_unit",
                "/length_unit",
            )


@dataclass(frozen=True, kw_only=True)
class LogicalName(Contract):
    name: Attribute
    source: Literal["DESIGN", "COMMENT", "EXTERNAL_MAPPING", "NONE"]

    def _validate(self) -> None:
        _facet(self.name, str, "/name")
        require(
            self.source != "NONE" or self.name.state is not State.VALUE,
            "logical_source_required",
            "/source",
        )


@dataclass(frozen=True, kw_only=True)
class SourceLocation(Contract):
    object_id: str
    input_digest: str
    pointer: str

    def _validate(self) -> None:
        opaque(self.object_id, "/object_id")
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", self.input_digest)),
            "invalid_digest",
            "/input_digest",
        )
        require(self.pointer == "" or self.pointer.startswith("/"), "invalid_pointer", "/pointer")
        require(not re.search(r"~(?:[^01]|$)", self.pointer), "invalid_pointer", "/pointer")


@dataclass(frozen=True, kw_only=True)
class Column(Contract):
    id: str
    physical_name: Identifier
    logical_name: LogicalName
    type: TypeDescriptor
    nullable: Attribute
    default: Attribute
    ordinal: int

    def _validate(self) -> None:
        opaque(self.id, "/id")
        _facet(self.nullable, bool, "/nullable")
        _facet(self.default, str, "/default")
        require(self.ordinal >= 1, "invalid_ordinal", "/ordinal")


class ConstraintKind(str, Enum):
    PRIMARY_KEY = "PRIMARY_KEY"
    UNIQUE = "UNIQUE"
    FOREIGN_KEY = "FOREIGN_KEY"
    CHECK = "CHECK"


Action = Literal["NO_ACTION", "RESTRICT", "CASCADE", "SET_NULL", "SET_DEFAULT"]


@dataclass(frozen=True, kw_only=True)
class Constraint(Contract):
    id: str
    name: Identifier
    kind: ConstraintKind
    column_ids: tuple[str, ...]
    referenced_table_id: str | None
    referenced_column_ids: tuple[str, ...]
    on_update: Action | None
    on_delete: Action | None
    deferrable: Attribute
    expression: Attribute

    def _validate(self) -> None:
        opaque(self.id, "/id")
        _facet(self.deferrable, bool, "/deferrable")
        _facet(self.expression, str, "/expression")
        for key in ("column_ids", "referenced_column_ids"):
            values = getattr(self, key)
            require(len(set(values)) == len(values), "duplicate_reference", "/" + key)
            for i, ref in enumerate(values):
                opaque(ref, f"/{key}/{i}")
        if self.kind is ConstraintKind.FOREIGN_KEY:
            require(self.referenced_table_id is not None, "missing_target", "/referenced_table_id")
            opaque(self.referenced_table_id, "/referenced_table_id")
            require(
                bool(self.column_ids) and len(self.column_ids) == len(self.referenced_column_ids),
                "foreign_key_arity",
                "/referenced_column_ids",
            )
            require(self.on_update is not None and self.on_delete is not None, "missing_action")
        else:
            require(
                self.referenced_table_id is None
                and not self.referenced_column_ids
                and self.on_update is None
                and self.on_delete is None,
                "unexpected_target",
            )
        if self.kind is not ConstraintKind.CHECK:
            require(bool(self.column_ids), "empty_key", "/column_ids")
            require(self.expression.state is State.ABSENT, "unexpected_expression", "/expression")
        else:
            require(
                self.expression.state in (State.VALUE, State.UNKNOWN),
                "check_expression_required",
                "/expression",
            )


@dataclass(frozen=True, kw_only=True)
class Table(Contract):
    id: str
    namespace: Identifier
    physical_name: Identifier
    logical_name: LogicalName
    columns: tuple[Column, ...]
    constraints: tuple[Constraint, ...]

    def _validate(self) -> None:
        opaque(self.id, "/id")
        columns = unique_ids(self.columns, "/columns")
        unique_ids(self.constraints, "/constraints")
        require(
            len({c.ordinal for c in self.columns}) == len(columns), "duplicate_ordinal", "/columns"
        )
        require(
            sum(c.kind is ConstraintKind.PRIMARY_KEY for c in self.constraints) <= 1,
            "multiple_primary_keys",
            "/constraints",
        )
        for i, constraint in enumerate(self.constraints):
            require(
                set(constraint.column_ids) <= columns.keys(),
                "broken_reference",
                f"/constraints/{i}/column_ids",
            )


@dataclass(frozen=True, kw_only=True)
class Word(Contract):
    id: str
    logical_name: str
    abbreviation: str
    deprecated: bool

    def _validate(self) -> None:
        opaque(self.id, "/id")
        require(bool(self.logical_name) and bool(self.abbreviation), "empty_word")


@dataclass(frozen=True, kw_only=True)
class Domain(Contract):
    id: str
    logical_name: str
    type: TypeDescriptor

    def _validate(self) -> None:
        opaque(self.id, "/id")
        require(bool(self.logical_name), "empty_name", "/logical_name")


@dataclass(frozen=True, kw_only=True)
class Term(Contract):
    id: str
    logical_name: str
    word_ids: tuple[str, ...]
    domain_id: str
    deprecated: bool

    def _validate(self) -> None:
        opaque(self.id, "/id")
        opaque(self.domain_id, "/domain_id")
        require(bool(self.word_ids) and bool(self.logical_name), "empty_term")
        for i, word in enumerate(self.word_ids):
            opaque(word, f"/word_ids/{i}")


@dataclass(frozen=True, kw_only=True)
class NamingPolicy(Contract):
    separator: str
    case: Literal["PRESERVE", "LOWER_ASCII", "UPPER_ASCII"]
    max_bytes: int

    def _validate(self) -> None:
        require(self.separator in ("", "_"), "unsupported_separator", "/separator")
        require(1 <= self.max_bytes <= 4096, "invalid_limit", "/max_bytes")


@dataclass(frozen=True, kw_only=True)
class Coverage(Contract):
    facet: Literal["TABLES", "COLUMNS", "CONSTRAINTS", "LOGICAL_NAMES", "TYPES", "DEFAULTS"]
    state: Literal["COMPLETE", "PARTIAL", "UNKNOWN", "NOT_APPLICABLE"]
    reason: Reason

    def _validate(self) -> None:
        if self.state == "COMPLETE":
            require(self.reason is Reason.NONE, "unexpected_reason", "/reason")
        elif self.state == "NOT_APPLICABLE":
            require(self.reason is Reason.OUT_OF_SCOPE, "scope_reason_required", "/reason")
        else:
            require(
                self.reason not in (Reason.NONE, Reason.OUT_OF_SCOPE),
                "coverage_reason_required",
                "/reason",
            )


def _revision(model) -> None:
    opaque(model.id, "/id")
    require(model.revision >= 1, "invalid_revision", "/revision")


def _sources(sources: tuple[SourceLocation, ...], ids: set[str]) -> None:
    for i, source in enumerate(sources):
        require(source.object_id in ids, "broken_reference", f"/sources/{i}/object_id")
    require(
        len({(s.object_id, s.input_digest, s.pointer) for s in sources}) == len(sources),
        "duplicate_source",
        "/sources",
    )


def _graph(tables: tuple[Table, ...], sources: tuple[SourceLocation, ...]) -> None:
    require(len(tables) <= 1000, "table_limit", "/tables")
    index = unique_ids(tables, "/tables")
    all_ids = set(index)
    count = 0
    for i, table in enumerate(tables):
        count += len(table.columns)
        for kind in ("columns", "constraints"):
            for j, item in enumerate(getattr(table, kind)):
                require(item.id not in all_ids, "duplicate_id", f"/tables/{i}/{kind}/{j}/id")
                all_ids.add(item.id)
    require(count <= 50000, "column_limit", "/tables")
    column_sets = {t.id: {c.id for c in t.columns} for t in tables}
    for i, table in enumerate(tables):
        for j, key in enumerate(table.constraints):
            if key.kind is ConstraintKind.FOREIGN_KEY:
                require(
                    key.referenced_table_id in index,
                    "broken_reference",
                    f"/tables/{i}/constraints/{j}/referenced_table_id",
                )
                require(
                    set(key.referenced_column_ids) <= column_sets[key.referenced_table_id],
                    "broken_reference",
                    f"/tables/{i}/constraints/{j}/referenced_column_ids",
                )
    _sources(sources, all_ids)


@dataclass(frozen=True, kw_only=True)
class StandardRevision(Contract):
    contract_version: Literal["truthound.schema-audit/1"]
    kind: Literal["STANDARD"]
    id: str
    revision: int
    dialect: Dialect
    words: tuple[Word, ...]
    terms: tuple[Term, ...]
    domains: tuple[Domain, ...]
    naming_policy: NamingPolicy
    sources: tuple[SourceLocation, ...]

    def _validate(self) -> None:
        _revision(self)
        words = unique_ids(self.words, "/words")
        domains = unique_ids(self.domains, "/domains")
        terms = unique_ids(self.terms, "/terms")
        require(
            len(set(words) | set(domains) | set(terms)) == len(words) + len(domains) + len(terms),
            "duplicate_id",
        )
        for i, term in enumerate(self.terms):
            require(set(term.word_ids) <= words.keys(), "broken_reference", f"/terms/{i}/word_ids")
            require(term.domain_id in domains, "broken_reference", f"/terms/{i}/domain_id")
        _sources(self.sources, set(words) | set(domains) | set(terms))


@dataclass(frozen=True, kw_only=True)
class DesignRevision(Contract):
    contract_version: Literal["truthound.schema-audit/1"]
    kind: Literal["DESIGN"]
    id: str
    revision: int
    dialect: Dialect
    tables: tuple[Table, ...]
    sources: tuple[SourceLocation, ...]

    def _validate(self) -> None:
        _revision(self)
        _graph(self.tables, self.sources)


@dataclass(frozen=True, kw_only=True)
class CatalogSnapshot(Contract):
    contract_version: Literal["truthound.schema-audit/1"]
    kind: Literal["CATALOG"]
    id: str
    revision: int
    dialect: Dialect
    binding_ref: str
    adapter_version: str
    captured_at: str
    scope: tuple[Identifier, ...]
    coverage: tuple[Coverage, ...]
    tables: tuple[Table, ...]
    sources: tuple[SourceLocation, ...]

    def _validate(self) -> None:
        from datetime import datetime

        _revision(self)
        opaque(self.binding_ref, "/binding_ref")
        opaque(self.adapter_version, "/adapter_version")
        require(
            bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", self.captured_at)),
            "invalid_timestamp",
            "/captured_at",
        )
        try:
            datetime.fromisoformat(self.captured_at.replace("Z", "+00:00"))
        except ValueError:
            require(False, "invalid_timestamp", "/captured_at")
        require(bool(self.scope), "empty_scope", "/scope")
        require(
            all(s.origin is NameOrigin.CATALOG for s in self.scope),
            "catalog_origin_required",
            "/scope",
        )
        names = {s.text for s in self.scope}
        require(len(names) == len(self.scope), "duplicate_scope", "/scope")
        required = {"TABLES", "COLUMNS", "CONSTRAINTS", "LOGICAL_NAMES", "TYPES", "DEFAULTS"}
        require(
            {c.facet for c in self.coverage} == required and len(self.coverage) == len(required),
            "coverage_facets_required",
            "/coverage",
        )
        _graph(self.tables, self.sources)
        for i, table in enumerate(self.tables):
            require(table.namespace.text in names, "outside_scope", f"/tables/{i}/namespace")
            identifiers = [table.namespace, table.physical_name]
            identifiers.extend(c.physical_name for c in table.columns)
            identifiers.extend(c.name for c in table.constraints)
            require(
                all(n.origin is NameOrigin.CATALOG for n in identifiers),
                "catalog_origin_required",
                f"/tables/{i}",
            )


Document = StandardRevision | DesignRevision | CatalogSnapshot
