"""Design/catalog comparison oracles, independent of database driver."""

import json
import unittest
from dataclasses import replace

try:
    from .test_standards import fixture
except ImportError:
    from test_standards import fixture

from truthound.schema_audit import (
    VERSION,
    CatalogSnapshot,
    Constraint,
    ConstraintKind,
    Coverage,
    Identifier,
    LogicalName,
    NameOrigin,
    Reason,
    Verdict,
    absent,
    audit_implementation,
    known,
    unknown,
)


def snapshot(design):
    def ident(value):
        return replace(
            value,
            origin=NameOrigin.CATALOG,
            text=value.text.lower() if value.origin is NameOrigin.UNQUOTED else value.text,
        )

    return CatalogSnapshot(
        contract_version=VERSION,
        kind="CATALOG",
        id="snapshot",
        revision=1,
        dialect=design.dialect,
        binding_ref="test",
        adapter_version="test",
        captured_at="2026-09-26T00:00:00Z",
        scope=(Identifier(text="public", origin=NameOrigin.CATALOG),),
        coverage=tuple(
            Coverage(facet=f, state="COMPLETE", reason=Reason.NONE)
            for f in ("TABLES", "COLUMNS", "CONSTRAINTS", "LOGICAL_NAMES", "TYPES", "DEFAULTS")
        ),
        tables=tuple(
            replace(
                t,
                namespace=ident(t.namespace),
                physical_name=ident(t.physical_name),
                logical_name=replace(t.logical_name, source="COMMENT"),
                columns=tuple(
                    replace(
                        c,
                        physical_name=ident(c.physical_name),
                        logical_name=replace(c.logical_name, source="COMMENT"),
                    )
                    for c in t.columns
                ),
                constraints=tuple(replace(k, name=ident(k.name)) for k in t.constraints),
            )
            for t in design.tables
        ),
        sources=(),
    )


class ImplementationTests(unittest.TestCase):
    def setUp(self):
        _, self.design = fixture()
        self.catalog = snapshot(self.design)

    def report(self, catalog=None, design=None):
        return audit_implementation(design or self.design, catalog or self.catalog)

    def column(self, **kwargs):
        table = self.catalog.tables[0]
        return replace(
            self.catalog, tables=(replace(table, columns=(replace(table.columns[0], **kwargs),)),)
        )

    def test_exact_conformance_and_deterministic_output(self):
        self.assertEqual(self.report().verdict, "CONFORMANT")
        self.assertEqual(json.dumps(self.report().as_dict()), json.dumps(self.report().as_dict()))

    def test_missing_table_is_fail(self):
        self.assertEqual(self.report(replace(self.catalog, tables=())).verdict, "NONCONFORMANT")

    def test_permission_incomplete_is_not_missing(self):
        cat = replace(
            self.catalog,
            tables=(),
            coverage=tuple(
                replace(c, state="PARTIAL", reason=Reason.PERMISSION_DENIED)
                for c in self.catalog.coverage
            ),
        )
        report = self.report(cat)
        self.assertEqual(report.verdict, "INDETERMINATE")
        self.assertFalse(any(c.verdict is Verdict.FAIL for c in report.checks))

    def test_extra_column(self):
        t = self.catalog.tables[0]
        extra = replace(
            t.columns[0],
            id="extra",
            physical_name=replace(t.columns[0].physical_name, text="extra"),
            ordinal=2,
        )
        self.assertEqual(
            self.report(
                replace(self.catalog, tables=(replace(t, columns=t.columns + (extra,)),))
            ).verdict,
            "NONCONFORMANT",
        )

    def test_quoted_names_are_not_case_folded(self):
        c = self.catalog.tables[0].columns[0]
        self.assertEqual(
            self.report(
                self.column(physical_name=replace(c.physical_name, text="CUST_NO"))
            ).verdict,
            "NONCONFORMANT",
        )

    def test_type_alias_and_facets(self):
        ty = self.catalog.tables[0].columns[0].type
        self.assertEqual(
            self.report(self.column(type=replace(ty, native_type="int8"))).verdict, "CONFORMANT"
        )
        for field, value in [
            ("length", known(10)),
            ("scale", known(2)),
            ("collation", known("public.x")),
            ("array_rank", known(1)),
        ]:
            with self.subTest(field=field):
                self.assertEqual(
                    self.report(self.column(type=replace(ty, **{field: value}))).verdict,
                    "NONCONFORMANT",
                )

    def test_nullable_difference(self):
        self.assertEqual(self.report(self.column(nullable=known(True))).verdict, "NONCONFORMANT")

    def test_absent_comment_is_unknown(self):
        self.assertEqual(
            self.report(
                self.column(logical_name=LogicalName(name=absent(), source="COMMENT"))
            ).verdict,
            "INDETERMINATE",
        )

    def test_default_expression_not_executed_or_inferred(self):
        d = self.design
        t = d.tables[0]
        d = replace(
            d, tables=(replace(t, columns=(replace(t.columns[0], default=known("now()")),)),)
        )
        result = self.report(self.column(default=known("CURRENT_TIMESTAMP")), d)
        self.assertEqual(result.verdict, "INDETERMINATE")

    def test_default_absent_vs_present(self):
        self.assertEqual(self.report(self.column(default=known("1"))).verdict, "NONCONFORMANT")

    def test_empty_snapshot_is_not_success(self):
        self.assertEqual(
            self.report(replace(self.catalog, tables=()), replace(self.design, tables=())).verdict,
            "INDETERMINATE",
        )

    def test_outside_scope_unknown(self):
        cat = replace(
            self.catalog, scope=(Identifier(text="other", origin=NameOrigin.CATALOG),), tables=()
        )
        self.assertEqual(self.report(cat).verdict, "INDETERMINATE")

    def test_constraint_order_and_actions(self):
        t = self.design.tables[0]
        c2 = replace(
            t.columns[0],
            id="second",
            physical_name=replace(t.columns[0].physical_name, text="seq"),
            ordinal=2,
        )
        k = Constraint(
            id="pk",
            name=Identifier(text="cust_pk", origin=NameOrigin.UNQUOTED),
            kind=ConstraintKind.PRIMARY_KEY,
            column_ids=("column", "second"),
            referenced_table_id=None,
            referenced_column_ids=(),
            on_update=None,
            on_delete=None,
            deferrable=known(False),
            expression=absent(),
        )
        d = replace(self.design, tables=(replace(t, columns=t.columns + (c2,), constraints=(k,)),))
        cat = snapshot(d)
        ct = cat.tables[0]
        cat = replace(
            cat,
            tables=(
                replace(
                    ct, constraints=(replace(ct.constraints[0], column_ids=("second", "column")),)
                ),
            ),
        )
        self.assertEqual(self.report(cat, d).verdict, "NONCONFORMANT")

    def test_fail_and_unknown_both_preserved(self):
        report = self.report(self.column(nullable=known(True), default=unknown()))
        self.assertEqual(report.verdict, "NONCONFORMANT")
        self.assertGreater(report.as_dict()["counts"]["UNKNOWN"], 0)

    def test_external_mapping_is_not_comment(self):
        c = self.catalog.tables[0].columns[0]
        self.assertEqual(
            self.report(
                self.column(logical_name=replace(c.logical_name, source="EXTERNAL_MAPPING"))
            ).verdict,
            "INDETERMINATE",
        )

    def test_unsupported_dialect_no_false_missing(self):
        cat = replace(self.catalog, dialect=replace(self.catalog.dialect, major=18))
        self.assertEqual(self.report(cat).verdict, "INDETERMINATE")

    def test_unresolved_identifier_does_not_produce_false_extra(self):
        t = self.design.tables[0]
        d = replace(
            self.design,
            tables=(
                replace(
                    t,
                    columns=(
                        replace(
                            t.columns[0],
                            physical_name=Identifier(text="non ASCII", origin=NameOrigin.UNQUOTED),
                        ),
                    ),
                ),
            ),
        )
        report = self.report(design=d)
        self.assertEqual(report.verdict, "INDETERMINATE")
        self.assertEqual(report.as_dict()["counts"]["FAIL"], 0)
        d = replace(
            self.design,
            tables=(
                replace(t, namespace=Identifier(text="non ASCII", origin=NameOrigin.UNQUOTED)),
            ),
        )
        report = self.report(design=d)
        self.assertEqual(report.verdict, "INDETERMINATE")
        self.assertEqual(report.as_dict()["counts"]["FAIL"], 0)

    def test_foreign_key_action_difference(self):
        t = self.design.tables[0]
        key = Constraint(
            id="fk",
            name=Identifier(text="self_fk", origin=NameOrigin.UNQUOTED),
            kind=ConstraintKind.FOREIGN_KEY,
            column_ids=("column",),
            referenced_table_id=t.id,
            referenced_column_ids=("column",),
            on_update="NO_ACTION",
            on_delete="CASCADE",
            deferrable=known(False),
            expression=absent(),
        )
        d = replace(self.design, tables=(replace(t, constraints=(key,)),))
        cat = snapshot(d)
        cat = replace(
            cat,
            tables=(
                replace(
                    cat.tables[0],
                    constraints=(replace(cat.tables[0].constraints[0], on_delete="RESTRICT"),),
                ),
            ),
        )
        report = self.report(cat, d)
        self.assertEqual(report.verdict, "NONCONFORMANT")
        self.assertTrue(
            any(
                c.rule_id == "constraint.references" and c.verdict is Verdict.FAIL
                for c in report.checks
            )
        )
