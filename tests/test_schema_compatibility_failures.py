"""Compatibility checks cannot hide failed rules or ignore strict decisions."""

import polars as pl
import pytest

from truthound.profiler.evolution.changes import ChangeType, CompatibilityLevel, SchemaChange
from truthound.profiler.evolution.compatibility import (
    CompatibilityRule,
    NullabilityCompatibilityChecker,
    SchemaCompatibilityAnalyzer,
)

pytestmark = pytest.mark.fault


@pytest.mark.parametrize("failure", ["exception", "untyped"])
def test_rule_failure_cannot_report_full_compatibility(failure):
    class Broken(CompatibilityRule):
        name = "synthetic-failed-rule"

        def check(self, old, new):
            if failure == "exception":
                raise RuntimeError("synthetic-private-schema-rule")
            return "False", []

    with pytest.raises(RuntimeError, match="compatibility") as error:
        SchemaCompatibilityAnalyzer([Broken()]).analyze({"x": pl.Int64}, {"x": pl.Int64})
    assert "synthetic-private-schema-rule" not in str(error.value)


def test_strict_mode_rejects_nonbreaking_issue_from_native_custom_rule():
    class WarningRule(CompatibilityRule):
        name = "synthetic-warning"

        def check(self, old, new):
            return True, [SchemaChange(ChangeType.CONSTRAINT_ADDED, "x", breaking=False)]

    ordinary = SchemaCompatibilityAnalyzer([WarningRule()]).analyze({"x": pl.Int64}, {"x": pl.Int64})
    strict = SchemaCompatibilityAnalyzer([WarningRule()], strict=True).analyze({"x": pl.Int64}, {"x": pl.Int64})
    assert ordinary.is_compatible
    assert not strict.is_compatible and strict.level == CompatibilityLevel.NONE


def test_declared_nullability_tightening_is_breaking_without_inferring_from_values():
    rule = NullabilityCompatibilityChecker(old_nullable={"x": True}, new_nullable={"x": False})
    result = SchemaCompatibilityAnalyzer([rule]).analyze({"x": pl.Int64}, {"x": pl.Int64})
    assert not result.is_compatible
    assert result.issues[0].change_type == ChangeType.NULLABLE_CHANGED
    assert result.issues[0].old_value is True and result.issues[0].new_value is False


def test_unpaired_or_untyped_nullability_metadata_is_rejected():
    with pytest.raises(ValueError):
        NullabilityCompatibilityChecker(old_nullable={"x": "False"}, new_nullable={"x": False})
    rule = NullabilityCompatibilityChecker(old_nullable={"x": True}, new_nullable={})
    with pytest.raises(RuntimeError, match="compatibility"):
        SchemaCompatibilityAnalyzer([rule]).analyze({"x": pl.Int64}, {"x": pl.Int64})


def test_execution_guard_revocation_prevents_next_native_rule():
    calls, checked = [], []

    class Rule(CompatibilityRule):
        name = "guarded"

        def check(self, old, new):
            checked.append(True)
            return True, []

    def guard():
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("synthetic revoked grant")

    analyzer = SchemaCompatibilityAnalyzer([Rule(), Rule()])
    analyzer.set_execution_guard(guard)
    with pytest.raises(RuntimeError, match="revoked"):
        analyzer.analyze({"x": pl.Int64}, {"x": pl.Int64})
    assert checked == [True]
