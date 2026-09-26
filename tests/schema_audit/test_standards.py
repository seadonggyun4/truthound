"""Exact synthetic oracles for standards auditing, no database or I/O."""

import json
import unittest
from dataclasses import replace

from truthound.schema_audit import (
    VERSION,
    Column,
    DesignRevision,
    Dialect,
    Domain,
    Identifier,
    LogicalName,
    ModelError,
    NameOrigin,
    NamingPolicy,
    StandardRevision,
    Table,
    Term,
    TermBinding,
    TypeDescriptor,
    Verdict,
    Word,
    absent,
    audit_standards,
    known,
    semantic_digest,
    unknown,
)


def fixture():
    dialect = Dialect(name="postgresql", major=17)
    integer = TypeDescriptor(
        native_type="bigint",
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
    standard = StandardRevision(
        contract_version=VERSION,
        kind="STANDARD",
        id="standard",
        revision=1,
        dialect=dialect,
        words=(
            Word(id="customer", logical_name="고객", abbreviation="CUST", deprecated=False),
            Word(id="number", logical_name="번호", abbreviation="NO", deprecated=False),
        ),
        terms=(
            Term(
                id="customer-number",
                logical_name="고객번호",
                word_ids=("customer", "number"),
                domain_id="id64",
                deprecated=False,
            ),
        ),
        domains=(Domain(id="id64", logical_name="식별자", type=integer),),
        naming_policy=NamingPolicy(separator="_", case="LOWER_ASCII", max_bytes=63),
        sources=(),
    )
    column = Column(
        id="column",
        physical_name=Identifier(text="cust_no", origin=NameOrigin.UNQUOTED),
        logical_name=LogicalName(name=known("고객번호"), source="DESIGN"),
        type=integer,
        nullable=known(False),
        default=absent(),
        ordinal=1,
    )
    table = Table(
        id="table",
        namespace=Identifier(text="public", origin=NameOrigin.UNQUOTED),
        physical_name=Identifier(text="cust", origin=NameOrigin.UNQUOTED),
        logical_name=LogicalName(name=known("고객"), source="DESIGN"),
        columns=(column,),
        constraints=(),
    )
    design = DesignRevision(
        contract_version=VERSION,
        kind="DESIGN",
        id="design",
        revision=1,
        dialect=dialect,
        tables=(table,),
        sources=(),
    )
    return standard, design


def change_column(design, **changes):
    table = design.tables[0]
    return replace(
        design, tables=(replace(table, columns=(replace(table.columns[0], **changes),)),)
    )


class StandardsRules(unittest.TestCase):
    def test_large_word_combination_is_bounded(self):
        standard, design = fixture()
        standard = replace(
            standard, terms=(replace(standard.terms[0], word_ids=("customer",) * 2000),)
        )
        report = audit_standards(standard, design)
        self.assertEqual(self.check_for(report, "name.abbreviation").code, "name_combination_limit")
        self.assertEqual(report.verdict, "INDETERMINATE")

    def test_both_sides_have_provenance(self):
        from truthound.schema_audit import SourceLocation

        standard, design = fixture()
        standard = replace(
            standard,
            sources=(
                SourceLocation(
                    object_id="customer-number", input_digest="a" * 64, pointer="/terms/0"
                ),
            ),
        )
        design = replace(
            design,
            sources=(
                SourceLocation(
                    object_id="column", input_digest="b" * 64, pointer="/tables/0/columns/0"
                ),
            ),
        )
        check = self.check_for(audit_standards(standard, design), "name.logical")
        self.assertEqual(check.sources[0].pointer, "/tables/0/columns/0")
        self.assertEqual(check.standard_sources[0].pointer, "/terms/0")

    def check_for(self, report, rule, object_id="column"):
        return next(
            check
            for check in report.checks
            if check.rule_id == rule and check.object_id == object_id
        )

    def test_exact_valid_fixture(self):
        report = audit_standards(*fixture())
        self.assertEqual(report.verdict, "CONFORMANT")
        self.assertEqual(report.target_count, 2)
        self.assertTrue(all(c.verdict is Verdict.PASS for c in report.checks))
        self.assertEqual(json.loads(json.dumps(report.as_dict()))["counts"]["FAIL"], 0)

    def test_no_substring_pass(self):
        standard, design = fixture()
        design = change_column(
            design, logical_name=LogicalName(name=known("신규고객번호"), source="DESIGN")
        )
        report = audit_standards(standard, design)
        self.assertEqual(self.check_for(report, "name.logical").code, "nonstandard_term")
        self.assertEqual(self.check_for(report, "domain.assignment").verdict, Verdict.UNKNOWN)
        self.assertEqual(report.verdict, "NONCONFORMANT")

    def test_unknown_logical_never_passes(self):
        standard, design = fixture()
        report = audit_standards(
            standard, change_column(design, logical_name=LogicalName(name=unknown(), source="NONE"))
        )
        self.assertEqual(report.verdict, "INDETERMINATE")
        self.assertGreater(report.as_dict()["counts"]["UNKNOWN"], 0)

    def test_absent_logical_is_failure_not_unknown(self):
        standard, design = fixture()
        report = audit_standards(
            standard, change_column(design, logical_name=LogicalName(name=absent(), source="NONE"))
        )
        self.assertEqual(self.check_for(report, "name.logical").verdict, Verdict.FAIL)

    def test_ambiguous_dictionary(self):
        standard, design = fixture()
        standard = replace(
            standard, terms=standard.terms + (replace(standard.terms[0], id="ambiguous"),)
        )
        self.assertEqual(
            self.check_for(audit_standards(standard, design), "name.logical").verdict,
            Verdict.UNKNOWN,
        )

    def test_deprecated_word(self):
        standard, design = fixture()
        standard = replace(
            standard, words=(standard.words[0], replace(standard.words[1], deprecated=True))
        )
        self.assertEqual(
            self.check_for(audit_standards(standard, design), "term.active").verdict, Verdict.FAIL
        )

    def test_ambiguous_abbreviation(self):
        standard, design = fixture()
        standard = replace(
            standard,
            words=standard.words
            + (Word(id="other", logical_name="다른번호", abbreviation="NO", deprecated=False),),
        )
        self.assertEqual(
            self.check_for(audit_standards(standard, design), "name.abbreviation").verdict,
            Verdict.UNKNOWN,
        )

    def test_rename_suggestion_does_not_change_verdict(self):
        standard, design = fixture()
        design = change_column(
            design, physical_name=Identifier(text="cust_num", origin=NameOrigin.UNQUOTED)
        )
        check = self.check_for(audit_standards(standard, design), "name.abbreviation")
        self.assertEqual((check.verdict, check.suggestion), (Verdict.FAIL, "cust_no"))

    def test_domain_mismatch_and_unknown_coexist(self):
        standard, design = fixture()
        descriptor = replace(
            design.tables[0].columns[0].type, native_type="integer", length=unknown()
        )
        report = audit_standards(standard, change_column(design, type=descriptor))
        self.assertEqual(report.verdict, "NONCONFORMANT")
        self.assertEqual(self.check_for(report, "domain.type").verdict, Verdict.FAIL)
        self.assertEqual(self.check_for(report, "domain.length").verdict, Verdict.UNKNOWN)

    def test_each_domain_facet(self):
        standard, design = fixture()
        for facet, value in (
            ("length", 10),
            ("precision", 5),
            ("scale", 2),
            ("length_unit", "OCTETS"),
            ("array_rank", 1),
            ("collation", "C"),
        ):
            with self.subTest(facet=facet):
                descriptor = replace(design.tables[0].columns[0].type, **{facet: known(value)})
                self.assertEqual(
                    self.check_for(
                        audit_standards(standard, change_column(design, type=descriptor)),
                        "domain." + facet,
                    ).verdict,
                    Verdict.FAIL,
                )

    def test_type_alias_is_conservative(self):
        standard, design = fixture()
        descriptor = replace(design.tables[0].columns[0].type, native_type="int8")
        self.assertEqual(
            audit_standards(standard, change_column(design, type=descriptor)).verdict, "CONFORMANT"
        )
        descriptor = replace(descriptor, type_schema=known("custom"))
        self.assertEqual(
            self.check_for(
                audit_standards(standard, change_column(design, type=descriptor)), "domain.type"
            ).verdict,
            Verdict.UNKNOWN,
        )

    def test_empty_is_not_conformant(self):
        standard, design = fixture()
        self.assertEqual(
            audit_standards(standard, replace(design, tables=())).verdict, "INDETERMINATE"
        )

    def test_unknown_dialect(self):
        standard, design = fixture()
        design = replace(design, dialect=Dialect(name="oracle", major=19))
        self.assertEqual(audit_standards(standard, design).verdict, "INDETERMINATE")

    def test_mapping_pinned_and_explicit(self):
        standard, design = fixture()
        design = change_column(
            design, logical_name=LogicalName(name=known("별칭"), source="DESIGN")
        )
        binding = TermBinding(
            object_id="column",
            term_id="customer-number",
            standard_digest=semantic_digest(standard),
            design_digest=semantic_digest(design),
            approval_digest="a" * 64,
        )
        self.assertEqual(
            audit_standards(standard, design, bindings=(binding,)).verdict, "CONFORMANT"
        )
        with self.assertRaisesRegex(ModelError, "stale_mapping"):
            audit_standards(standard, design, bindings=(replace(binding, design_digest="b" * 64),))
        with self.assertRaisesRegex(ModelError, "ambiguous_mapping"):
            audit_standards(standard, design, bindings=(binding, binding))

    def test_determinism_under_dictionary_reordering(self):
        standard, design = fixture()
        left = audit_standards(standard, design)
        right = audit_standards(replace(standard, words=standard.words[::-1]), design)
        self.assertEqual(left.as_dict(), right.as_dict())

    def test_duplicate_physical_names(self):
        standard, design = fixture()
        table = design.tables[0]
        duplicate = replace(table.columns[0], id="other-column", ordinal=2)
        report = audit_standards(
            standard,
            replace(design, tables=(replace(table, columns=table.columns + (duplicate,)),)),
        )
        self.assertEqual(self.check_for(report, "structure.unique").verdict, Verdict.FAIL)

    def test_naming_policy(self):
        standard, design = fixture()
        for name in ("CUST_NO", "cust__no", "cust-no", "cust_no_extra"):
            with self.subTest(name=name):
                changed = change_column(
                    design, physical_name=Identifier(text=name, origin=NameOrigin.QUOTED)
                )
                self.assertEqual(audit_standards(standard, changed).verdict, "NONCONFORMANT")

    def test_limits_fail_closed(self):
        standard, design = fixture()
        from unittest.mock import patch

        with (
            patch("truthound.schema_audit.rules.standards.MAX_OBJECTS", 1),
            self.assertRaisesRegex(ModelError, "audit_object_limit"),
        ):
            audit_standards(standard, design)


if __name__ == "__main__":
    unittest.main()
