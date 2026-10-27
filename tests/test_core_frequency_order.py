"""Complete evidence preserves encounter order for equal failure frequencies."""

import polars as pl
import pytest

from truthound.types import ResultFormat, ResultFormatConfig
from truthound.validators.distribution.range import RangeValidator

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("values", [[-1, 200], [200, -1], [-1, 200, 200, -1]])
def test_equal_frequency_failures_have_stable_first_encounter_order(values):
    validator = RangeValidator(columns=["age"], result_format="complete")
    issue = validator.validate(pl.DataFrame({"age": values}).lazy())[0]
    expected = list(dict.fromkeys(values))
    assert issue.result.partial_unexpected_counts == [
        {"value": value, "count": values.count(value)} for value in expected
    ]
    assert issue.sample_values == values
    assert issue.count == len(values)
    assert issue.details.endswith(
        "top failures: " + ", ".join(f"{value}({values.count(value)})" for value in expected)
    )


def test_frequency_precedes_encounter_order_and_limit_does_not_drop_counts():
    values = [-1, 200, 200, -2, -2, -2, -3, -3]
    validator = RangeValidator(
        columns=["age"],
        result_format=ResultFormatConfig(format=ResultFormat.COMPLETE, partial_unexpected_count=2),
    )
    issue = validator.validate(pl.DataFrame({"age": values}).lazy())[0]
    assert issue.result.partial_unexpected_counts == [
        {"value": -2, "count": 3},
        {"value": 200, "count": 2},
    ]
    assert issue.count == len(values)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('mostly', [None, 0.6])
def test_passing_first_column_does_not_shift_failure_evidence(batched, mostly):
    from truthound.validators.base import ExpressionBatchExecutor
    from truthound.validators.distribution.range import BetweenValidator

    first = [1, 2, 3] if mostly is None else [-1, 2, 3]
    frame = pl.DataFrame({'first': first, 'second': [2, -7, -7]}).lazy()
    validator = BetweenValidator(
        columns=['first', 'second'], min_value=0, max_value=5,
        mostly=mostly,
        result_format=ResultFormatConfig(format=ResultFormat.COMPLETE, include_unexpected_rows=True),
    )
    if batched:
        executor = ExpressionBatchExecutor()
        executor.add_validator(validator)
        issues = executor.execute(frame)
    else:
        issues = validator.validate(frame)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.column == 'second' and issue.count == 2
    assert issue.sample_values == [-7, -7]
    assert issue.result.partial_unexpected_counts == [{'value': -7, 'count': 2}]
    assert issue.result.unexpected_index_list == [1, 2]
    assert issue.result.unexpected_list == [-7, -7]
    assert issue.result.unexpected_rows.to_dict(as_series=False) == {'second': [-7, -7]}
