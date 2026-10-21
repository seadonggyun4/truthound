"""Failure and repeated-instance accounting at the actual kernel boundary."""

import polars as pl
import pytest

from truthound.core import ScanPlanner, ValidationRuntime, ValidationSuite, build_validation_asset
from truthound.core.suite import CheckSpec
from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class ConfiguredValidator(Validator):
    name = "configured"
    category = "general"

    def __init__(self, column, *, fail=False, **kwargs):
        super().__init__(**kwargs)
        self.column = column
        self.fail = fail

    def validate(self, lf):
        if self.fail:
            raise ValueError("intentional failure")
        return [
            ValidationIssue(
                column=self.column,
                issue_type="configured",
                count=1,
                severity=Severity.HIGH,
                validator_name=self.name,
            )
        ]


class DependentValidator(Validator):
    name = "dependent"
    category = "general"
    dependencies = {"configured"}

    def validate(self, lf):
        raise AssertionError("failed dependency must prevent execution")


def run(validators, *, parallel):
    suite = ValidationSuite(
        checks=tuple(
            CheckSpec(id=str(i), name=v.name, category=v.category, factory=lambda v=v: v)
            for i, v in enumerate(validators)
        )
    )
    asset = build_validation_asset(pl.DataFrame({"a": [1], "b": [2]}))
    plan = ScanPlanner().plan(suite=suite, asset=asset, parallel=parallel, max_workers=2)
    return ValidationRuntime().execute(asset=asset, plan=plan)


@pytest.mark.contract
def test_parallel_retains_distinct_configurations_of_one_validator():
    validators = [ConfiguredValidator("a"), ConfiguredValidator("b")]
    sequential = run(validators, parallel=False)
    parallel = run(validators, parallel=True)
    assert sorted(i.column for i in parallel.issues) == ["a", "b"]
    assert sorted((i.to_dict() for i in parallel.issues), key=str) == sorted(
        (i.to_dict() for i in sequential.issues), key=str
    )


@pytest.mark.fault
def test_parallel_failure_and_dependent_skip_are_not_success():
    result = run(
        [ConfiguredValidator("a", fail=True), ConfiguredValidator("b"), DependentValidator()],
        parallel=True,
    )
    assert not result.success
    assert {i.check_name for i in result.execution_issues} == {"configured", "dependent"}
    assert all(not c.success for c in result.checks)
    assert (
        next(i for i in result.execution_issues if i.check_name == "dependent").exception_type
        is None
    )
    assert "intentional failure" in next(
        i.message for i in result.execution_issues if i.check_name == "configured"
    )
    filtered = result.filter_by_severity(Severity.CRITICAL)
    assert not filtered.success
    # Only the second instance has a HIGH quality issue, not an execution
    # failure. Filtering that issue must not borrow the first instance's
    # same-named execution failure. The real failure and dependency skip stay
    # failed, with all three planned instances accounted for.
    assert [c.metadata["_truthound_check_id"] for c in filtered.checks] == ["0", "1", "2"]
    assert [c.success for c in filtered.checks] == [False, True, False]
    assert [c.metadata["_truthound_execution_incomplete"] for c in filtered.checks] == [
        True,
        False,
        True,
    ]
    assert filtered.execution_issues == result.execution_issues


@pytest.mark.fault
@pytest.mark.parametrize("count", [2, 4])
def test_parallel_obeys_explicit_no_catch(count):
    with pytest.raises(ValueError, match="intentional failure"):
        run(
            [ConfiguredValidator("a", fail=True, catch_exceptions=False)]
            + [ConfiguredValidator("b") for _ in range(count - 1)],
            parallel=True,
        )
