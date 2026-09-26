"""Portable contract suite also run against built distributions on the server."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError, replace
from importlib.resources import files

from truthound.schema_audit import (
    VERSION,
    Attribute,
    Column,
    Constraint,
    ConstraintKind,
    DesignRevision,
    Dialect,
    Domain,
    Finding,
    FindingSet,
    Identifier,
    LogicalName,
    ModelError,
    NameOrigin,
    NamingPolicy,
    NormalizationRegistry,
    Reason,
    SourceLocation,
    StandardRevision,
    State,
    Table,
    Term,
    TypeDescriptor,
    Verdict,
    Word,
    absent,
    dump_document,
    json_schema,
    known,
    load_document,
    normalize_identifier,
    normalize_logical_name,
    normalize_type,
    raw_digest,
    semantic_bytes,
    semantic_digest,
    to_dict,
    unknown,
)

PG = Dialect(name="postgresql", major=17)


def ident(name, origin=NameOrigin.UNQUOTED):
    return Identifier(text=name, origin=origin)


def typ(name="integer", **kwargs):
    values = dict(
        native_type=name,
        type_schema=known("pg_catalog"),
        length=absent(),
        length_unit=absent(),
        precision=absent(),
        scale=absent(),
        timezone=absent(),
        array_rank=known(0),
        domain_ref=absent(),
        enum_values_ref=absent(),
        collation=absent(),
    )
    values.update(kwargs)
    return TypeDescriptor(**values)


def logical(label="Customer", source="DESIGN"):
    return LogicalName(name=known(label), source=source)


def column(cid="c1", ordinal=1):
    return Column(
        id=cid,
        physical_name=ident(cid),
        logical_name=logical(),
        type=typ(),
        nullable=known(False),
        default=absent(),
        ordinal=ordinal,
    )


def key(kid="pk", cols=("c1", "c2")):
    return Constraint(
        id=kid,
        name=ident(kid),
        kind=ConstraintKind.PRIMARY_KEY,
        column_ids=cols,
        referenced_table_id=None,
        referenced_column_ids=(),
        on_update=None,
        on_delete=None,
        deferrable=known(False),
        expression=absent(),
    )


def design():
    table = Table(
        id="t1",
        namespace=ident("public"),
        physical_name=ident("CUSTOMER"),
        logical_name=logical(),
        columns=(column(), column("c2", 2)),
        constraints=(key(),),
    )
    return DesignRevision(
        contract_version=VERSION,
        kind="DESIGN",
        id="d1",
        revision=1,
        dialect=PG,
        tables=(table,),
        sources=(),
    )


def standard():
    return StandardRevision(
        contract_version=VERSION,
        kind="STANDARD",
        id="s1",
        revision=1,
        dialect=PG,
        words=(
            Word(id="w1", logical_name="Customer", abbreviation="CUST", deprecated=False),
            Word(id="w2", logical_name="Number", abbreviation="NO", deprecated=False),
        ),
        domains=(Domain(id="n1", logical_name="Number", type=typ()),),
        terms=(
            Term(
                id="term1",
                logical_name="Customer number",
                word_ids=("w1", "w2"),
                domain_id="n1",
                deprecated=False,
            ),
        ),
        naming_policy=NamingPolicy(separator="_", case="UPPER_ASCII", max_bytes=63),
        sources=(),
    )


def catalog():
    doc = to_dict(design())
    for table in doc["tables"]:
        for name in (table["namespace"], table["physical_name"]):
            name["origin"] = "CATALOG"
        for col in table["columns"]:
            col["physical_name"]["origin"] = "CATALOG"
        for constraint in table["constraints"]:
            constraint["name"]["origin"] = "CATALOG"
    doc.update(
        kind="CATALOG",
        binding_ref="binding1",
        adapter_version="pg17-v1",
        captured_at="2026-09-26T00:00:00Z",
        scope=[dict(text="public", origin="CATALOG")],
        coverage=[
            dict(facet=f, state="COMPLETE", reason="NONE")
            for f in ("TABLES", "COLUMNS", "CONSTRAINTS", "LOGICAL_NAMES", "TYPES", "DEFAULTS")
        ],
    )
    return load_document(json.dumps(doc).encode())


class ContractTests(unittest.TestCase):
    def assert_bad(self, fn, code, pointer=None):
        with self.assertRaises(ModelError) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)
        if pointer is not None:
            self.assertEqual(caught.exception.pointer, pointer)

    def test_all_document_kinds_roundtrip_and_are_immutable(self):
        for model in (standard(), design(), catalog()):
            with self.subTest(kind=model.kind):
                self.assertEqual(load_document(dump_document(model)), model)
                with self.assertRaises(FrozenInstanceError):
                    model.revision = 2
        self.assert_bad(lambda: replace(design(), tables=[]), "immutable_array_required", "/tables")

    def test_unknown_absent_null_and_false_are_distinct(self):
        self.assertNotEqual(absent(), unknown())
        self.assertEqual(known(False).value, False)
        self.assert_bad(lambda: known(None), "invalid_observation")
        self.assert_bad(
            lambda: Attribute(state=State.UNKNOWN, value=None, reason=Reason.NONE),
            "unknown_reason_required",
        )
        self.assert_bad(
            lambda: Attribute(state=State.ABSENT, value="private", reason=Reason.NONE),
            "unexpected_value",
        )
        self.assert_bad(lambda: replace(column(), nullable=known("false")), "invalid_facet")

    def test_strict_constructors_no_bool_integer_or_mutable_children(self):
        for revision in (True, 1.0, "1", 0):
            with self.subTest(revision=revision), self.assertRaises(ModelError):
                replace(design(), revision=revision)
        self.assert_bad(lambda: replace(typ(), length=known(True)), "invalid_facet")
        self.assert_bad(lambda: Identifier(text="a", origin="CATALOG"), "invalid_enum")

    def test_standard_dangling_and_duplicate_references(self):
        base = standard()
        self.assert_bad(lambda: replace(base, words=base.words + base.words[:1]), "duplicate_id")
        for changed in (
            replace(base.terms[0], domain_id="missing"),
            replace(base.terms[0], word_ids=("missing",)),
        ):
            self.assert_bad(lambda changed=changed: replace(base, terms=(changed,)), "broken_reference")
        self.assert_bad(
            lambda: replace(base, domains=(replace(base.domains[0], id="w1"),)), "duplicate_id"
        )

    def test_graph_rejects_broken_local_and_foreign_keys(self):
        table = design().tables[0]
        self.assert_bad(
            lambda: replace(table, constraints=(key(cols=("missing",)),)), "broken_reference"
        )
        fk = replace(
            key("fk"),
            kind=ConstraintKind.FOREIGN_KEY,
            referenced_table_id="missing",
            referenced_column_ids=("c1", "c2"),
            on_update="NO_ACTION",
            on_delete="CASCADE",
        )
        self.assert_bad(
            lambda: replace(design(), tables=(replace(table, constraints=(fk,)),)),
            "broken_reference",
            "/tables/0/constraints/0/referenced_table_id",
        )
        self.assert_bad(lambda: replace(fk, referenced_column_ids=("c1",)), "foreign_key_arity")
        self.assert_bad(
            lambda: replace(fk, referenced_column_ids=("c1", "c1")), "duplicate_reference"
        )
        invalid_target = replace(
            fk, referenced_table_id="t1", referenced_column_ids=("c1", "missing")
        )
        self.assert_bad(
            lambda: replace(design(), tables=(replace(table, constraints=(invalid_target,)),)),
            "broken_reference",
        )

    def test_graph_cycles_allowed_and_global_ids_unique(self):
        table = design().tables[0]
        fk = replace(
            key("fk"),
            kind=ConstraintKind.FOREIGN_KEY,
            referenced_table_id="t1",
            referenced_column_ids=("c1", "c2"),
            on_update="NO_ACTION",
            on_delete="CASCADE",
        )
        self.assertEqual(
            replace(design(), tables=(replace(table, constraints=(fk,)),)).tables[0].constraints[0],
            fk,
        )
        self.assert_bad(
            lambda: replace(design(), tables=(table, replace(table, id="t2"))), "duplicate_id"
        )
        self.assert_bad(
            lambda: replace(table, columns=(column(), column("c3"))), "duplicate_ordinal"
        )
        self.assert_bad(
            lambda: replace(table, constraints=(key(), key("pk2"))), "multiple_primary_keys"
        )

    def test_catalog_requires_explicit_coverage_scope_and_stored_names(self):
        base = catalog()
        self.assert_bad(lambda: replace(base, coverage=()), "coverage_facets_required")
        self.assert_bad(
            lambda: replace(base, scope=(ident("elsewhere", NameOrigin.CATALOG),)), "outside_scope"
        )
        self.assert_bad(lambda: replace(base, tables=design().tables), "catalog_origin_required")
        self.assert_bad(
            lambda: replace(base, captured_at="2026-02-31T00:00:00Z"), "invalid_timestamp"
        )
        self.assert_bad(lambda: replace(base, captured_at="2026-09-26"), "invalid_timestamp")
        partial = replace(base.coverage[0], state="PARTIAL", reason=Reason.PERMISSION_DENIED)
        self.assertNotEqual(
            semantic_digest(base),
            semantic_digest(replace(base, coverage=(partial,) + base.coverage[1:])),
        )

    def test_source_references_and_redacted_failures(self):
        source = SourceLocation(object_id="t1", input_digest="a" * 64, pointer="/tables/0")
        self.assertEqual(replace(design(), sources=(source,)).sources, (source,))
        self.assert_bad(
            lambda: replace(design(), sources=(replace(source, object_id="missing"),)),
            "broken_reference",
        )
        self.assert_bad(lambda: replace(source, input_digest="secret-value"), "invalid_digest")
        self.assert_bad(lambda: replace(source, pointer="/bad~x"), "invalid_pointer")

    def test_json_unknown_fields_and_version_rejected(self):
        for mutate, code, path in (
            (lambda d: d.update(secret_password="canary"), "unknown_field", ""),
            (lambda d: d.update(contract_version="future"), "invalid_literal", "/contract_version"),
            (lambda d: d.pop("tables"), "missing_field", ""),
        ):
            doc = to_dict(design())
            mutate(doc)
            self.assert_bad(lambda doc=doc: load_document(json.dumps(doc).encode()), code, path)

    def test_json_nested_duplicate_keys_report_no_private_key(self):
        for raw in (b'{"secret":1,"secret":2}', b'{"secret":{"x":1,"x":2}}'):
            with self.assertRaises(ModelError) as caught:
                load_document(raw)
            self.assertEqual(caught.exception.code, "duplicate_key")
            self.assertNotIn("secret", str(caught.exception))

    def test_json_preserves_error_location_for_known_fields(self):
        doc = to_dict(design())
        doc["tables"][0]["columns"][0]["ordinal"] = True
        self.assert_bad(
            lambda: load_document(json.dumps(doc).encode()),
            "invalid_type",
            "/tables/0/columns/0/ordinal",
        )
        doc["tables"][0]["columns"][0]["ordinal"] = -1
        self.assert_bad(
            lambda: load_document(json.dumps(doc).encode()),
            "invalid_ordinal",
            "/tables/0/columns/0/ordinal",
        )

    def test_json_number_and_encoding_abuse(self):
        for payload in (b"NaN", b"Infinity", b"-Infinity", b"1.0", b"1e200"):
            self.assert_bad(lambda payload=payload: load_document(payload), "non_integer_number")
        for payload in (b"\xff", b"{bad", b'"\\ud800"', b'"\\u0000"', b"{} trailing"):
            with self.subTest(payload=payload), self.assertRaises(ModelError):
                load_document(payload)
        self.assert_bad(lambda: load_document("{}"), "bytes_required")

    def test_json_size_depth_and_string_budgets(self):
        self.assert_bad(lambda: load_document(b" " * (20 * 1024 * 1024 + 1)), "byte_limit")
        self.assert_bad(lambda: load_document(b"[" * 33 + b"]" * 33), "depth_limit")
        self.assert_bad(lambda: load_document(json.dumps("x" * 4097).encode()), "string_limit")
        self.assert_bad(lambda: load_document(b"[" + b"0," * 50000 + b"0]"), "item_limit")
        doc = to_dict(design())
        doc["tables"][0]["columns"][0]["default"] = to_dict(design())["tables"][0]["columns"][0][
            "nullable"
        ]
        with self.assertRaises(ModelError):
            load_document(json.dumps(doc).encode())

    def test_braces_inside_strings_do_not_trigger_depth_limit(self):
        table = design().tables[0]
        modified = replace(table.columns[0], default=known("'" + "[" * 64 + "'"))
        model = replace(design(), tables=(replace(table, columns=(modified, table.columns[1])),))
        self.assertEqual(load_document(dump_document(model)), model)

    def test_integer_range_and_unexpected_nulls(self):
        doc = to_dict(design())
        for value in (9007199254740992, None, -1):
            doc["revision"] = value
            with self.subTest(value=value), self.assertRaises(ModelError):
                load_document(json.dumps(doc).encode())

    def test_digest_fixed_projection_vector(self):
        model = replace(design(), tables=())
        expected = (
            b'{"model":{"contract_version":"truthound.schema-audit/1",'
            b'"dialect":{"major":17,"name":"postgresql"},"kind":"DESIGN","tables":[]},'
            b'"normalization":"pg17-ascii-nfc32-logical-v1","profile":"truthound.schema-audit.semantic/1"}'
        )
        self.assertEqual(semantic_bytes(model), expected)
        self.assertEqual(semantic_digest(model), hashlib.sha256(expected).hexdigest())
        self.assertEqual(
            raw_digest(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_digest_ignores_wire_spacing_and_provenance_not_raw_bytes(self):
        model = design()
        compact = dump_document(model)
        pretty = json.dumps(to_dict(model), indent=2).encode()
        self.assertNotEqual(raw_digest(compact), raw_digest(pretty))
        self.assertEqual(
            semantic_digest(load_document(compact)), semantic_digest(load_document(pretty))
        )
        changed = replace(
            model,
            id="different",
            revision=99,
            sources=(SourceLocation(object_id="t1", input_digest="f" * 64, pointer="/elsewhere"),),
        )
        self.assertEqual(semantic_digest(model), semantic_digest(changed))

    def test_digest_unordered_collections_but_ordered_keys_and_words(self):
        model = design()
        table = model.tables[0]
        reordered = replace(table, columns=tuple(reversed(table.columns)))
        self.assertEqual(
            semantic_digest(model), semantic_digest(replace(model, tables=(reordered,)))
        )
        reordered_key = replace(
            table, constraints=(replace(table.constraints[0], column_ids=("c2", "c1")),)
        )
        self.assertNotEqual(
            semantic_digest(model), semantic_digest(replace(model, tables=(reordered_key,)))
        )
        base = standard()
        self.assertEqual(
            semantic_digest(base), semantic_digest(replace(base, words=tuple(reversed(base.words))))
        )
        changed = replace(base, terms=(replace(base.terms[0], word_ids=("w2", "w1")),))
        self.assertNotEqual(semantic_digest(base), semantic_digest(changed))

    def test_unicode_logical_nfc_not_physical_or_defaults(self):
        self.assertEqual(normalize_logical_name("e\u0301"), "\u00e9")
        table = design().tables[0]
        a = replace(design(), tables=(replace(table, logical_name=logical("e\u0301")),))
        b = replace(design(), tables=(replace(table, logical_name=logical("\u00e9")),))
        self.assertEqual(semantic_digest(a), semantic_digest(b))
        a = replace(
            design(), tables=(replace(table, physical_name=ident("e\u0301", NameOrigin.QUOTED)),)
        )
        b = replace(
            design(), tables=(replace(table, physical_name=ident("\u00e9", NameOrigin.QUOTED)),)
        )
        self.assertNotEqual(semantic_digest(a), semantic_digest(b))

    def test_pg_identifier_origin_and_unsupported_folding(self):
        self.assertEqual(normalize_identifier(ident("FOO"), PG), known("foo"))
        for origin in (NameOrigin.QUOTED, NameOrigin.CATALOG):
            self.assertEqual(normalize_identifier(ident("Foo", origin), PG), known("Foo"))
        for name in ("x" * 64, "\u0130", '"foo"', "a b"):
            self.assertEqual(normalize_identifier(ident(name), PG).state, State.UNKNOWN)
        self.assertEqual(
            normalize_identifier(ident("FOO"), Dialect(name="postgresql", major=18)).state,
            State.UNKNOWN,
        )

    def test_type_aliases_do_not_erase_facets_or_custom_types(self):
        for name in ("int", "int4", "integer"):
            self.assertEqual(normalize_type(typ(name), PG), known("integer"))
        self.assertNotEqual(normalize_type(typ("int4"), PG), normalize_type(typ("int8"), PG))
        for value in (
            typ("numeric(10,2)"),
            typ("integer", type_schema=known("customer")),
            typ("integer", domain_ref=known("domain1")),
            typ("integer", type_schema=unknown()),
        ):
            self.assertEqual(normalize_type(value, PG).state, State.UNKNOWN)
        value = typ("numeric", precision=known(2), scale=known(-3))
        self.assertEqual(normalize_type(value, PG), known("numeric"))
        self.assertEqual(value.scale.value, -3)
        self.assertEqual(
            normalize_type(typ("numeric", precision=known(2), scale=known(4)), PG), known("numeric")
        )
        self.assertEqual(
            normalize_type(typ("numeric", scale=known(-1001)), PG).state, State.UNKNOWN
        )

    def test_timezone_conflict_and_array_rank_preserved(self):
        self.assertEqual(
            normalize_type(typ("timestamptz", timezone=known(False)), PG).state, State.UNKNOWN
        )
        self.assertEqual(
            normalize_type(typ("timestamptz", timezone=known(True)), PG),
            known("timestamp with time zone"),
        )
        self.assertEqual(typ(array_rank=known(2)).array_rank.value, 2)
        with self.assertRaises(ModelError):
            typ(array_rank=known(-1))

    def test_semantic_type_and_default_differences_are_not_hidden(self):
        model = design()
        table = model.tables[0]

        def with_type(t):
            return replace(
                model,
                tables=(
                    replace(table, columns=(replace(table.columns[0], type=t), table.columns[1])),
                ),
            )

        self.assertEqual(semantic_digest(with_type(typ("int4"))), semantic_digest(model))
        for t in (
            typ("bigint"),
            typ("integer", array_rank=known(1)),
            typ("varchar", length=known(3)),
        ):
            self.assertNotEqual(semantic_digest(with_type(t)), semantic_digest(model))
        a = replace(table.columns[0], default=known("'a  b'"))
        b = replace(a, default=known("'a b'"))
        self.assertNotEqual(
            semantic_digest(
                replace(model, tables=(replace(table, columns=(a, table.columns[1])),))
            ),
            semantic_digest(
                replace(model, tables=(replace(table, columns=(b, table.columns[1])),))
            ),
        )

    def test_registry_duplicate_or_unapproved_version_rejected(self):
        base = NormalizationRegistry()
        self.assert_bad(
            lambda: NormalizationRegistry(entries=base.entries * 2), "duplicate_dialect"
        )
        self.assert_bad(
            lambda: base.resolve(Dialect(name="oracle", major=19)), "unsupported_dialect"
        )
        self.assert_bad(lambda: NormalizationRegistry(entries=[]), "immutable_array_required")

    def test_result_unknown_requires_reason_and_ids_remain_stable(self):
        f = Finding(
            id="f1",
            rule_id="r1",
            rule_version="v1",
            object_id="c1",
            verdict=Verdict.UNKNOWN,
            reason=Reason.NOT_COLLECTED,
        )
        self.assert_bad(lambda: replace(f, reason=Reason.NONE), "unknown_reason_required")
        self.assert_bad(lambda: FindingSet(findings=(f, f)), "duplicate_id")
        g = replace(f, id="f2", object_id="c2")
        self.assertEqual(FindingSet(findings=(g, f)).ordered, (f, g))

    def test_packaged_json_schema_matches_code(self):
        resource = files("truthound.schema_audit").joinpath("schemas/document-v1.json")
        self.assertEqual(json.loads(resource.read_text()), json_schema())
        self.assertFalse(json_schema()["$defs"]["DesignRevision"]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
