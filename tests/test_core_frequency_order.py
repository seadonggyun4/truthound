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
