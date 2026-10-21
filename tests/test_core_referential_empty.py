"""Empty anti-join aggregates retain counts and bounded samples, not execution errors."""

import polars as pl
import pytest

from truthound.validators.schema.referential import (
    ReferentialIntegrityValidator,
    StreamingReferentialIntegrityValidator,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "validator_type", [ReferentialIntegrityValidator, StreamingReferentialIntegrityValidator]
)
def test_empty_orphan_set_and_repeated_violations_keep_native_meaning(validator_type):
    reference = pl.DataFrame({"x": [1, 2]})
    validator = validator_type(column="x", reference_data=reference, sample_size=1)
    assert validator.validate(pl.DataFrame({"x": [1, 2, 1, None]}).lazy()) == []
    assert (
        validator.validate(pl.DataFrame({"x": pl.Series([None, None], dtype=pl.Int64)}).lazy())
        == []
    )
    assert validator.validate(pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)}).lazy()) == []
    issues = validator.validate(pl.DataFrame({"x": [1, 99, 99, None]}).lazy())
    assert len(issues) == 1
    assert issues[0].count == 2  # row count, not one unique orphan
    assert issues[0].issue_type == "referential_integrity_violation"
    assert issues[0].sample_values == ["99"]
