"""Out-of-reference-support populations must not silently become drift-free."""

import polars as pl
import pytest

from truthound.validators import get_validator


@pytest.mark.parametrize("name", ["psi", "feature_drift", "csi"])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("constant", [False, True])
def test_numeric_histograms_account_for_both_reference_tails(name, direction, cached, constant):
    reference = pl.DataFrame({"x": [1.0] * 100 if constant else [float(i) for i in range(100)]})
    options = {"columns": ["x"]} if name == "feature_drift" else {"column": "x"}
    if name == "psi":
        options["cache_reference"] = cached
    validator = get_validator(name)(reference_data=reference, **options)
    assert not validator.validate(reference.lazy())
    shifted = pl.DataFrame({"x": [float(direction * 1000 + i) for i in range(100)]})
    assert validator.validate(shifted.lazy()), "Out-of-support data is not an empty population"


def test_constant_cached_summary_is_finite_and_does_not_hide_constructor_error():
    import json
    from dataclasses import asdict

    from truthound.validators.cache import NumericStatistics

    stats = NumericStatistics.from_series(pl.Series("x", [1.0] * 100))
    assert stats.count == 100
    assert sum(stats.histogram_counts) == 1.0
    assert len(stats.histogram_counts) + 1 == len(stats.histogram_edges)
    # A point mass remains serializable without nonstandard Infinity/NaN.
    json.dumps(asdict(stats), allow_nan=False)


def test_default_reference_cache_never_reuses_a_different_baseline_by_column_name():
    from truthound.validators.drift.psi import PSIValidator

    first = pl.DataFrame({"x": [float(i) for i in range(100)]})
    second = pl.DataFrame({"x": [float(i + 1000) for i in range(100)]})
    old = PSIValidator(column="x", reference_data=first)
    assert not old.validate(first.lazy())
    new = PSIValidator(column="x", reference_data=second)
    assert not new.validate(second.lazy())
    assert new.get_reference_statistics().mean == 1049.5
    assert old.get_cache_key() != new.get_cache_key()
