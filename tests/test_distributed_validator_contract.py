"""Distributed adapters preserve actual validator semantics and owned lifetimes."""

from threading import Barrier

import dask.dataframe as dd
import pandas as pd
import pytest

from truthound.execution.distributed.dask_engine import DaskExecutionEngine
from truthound.execution.distributed.validator_adapter import (
    AdapterConfig,
    DistributedValidatorAdapter,
    ExecutionStrategy,
)
from truthound.validators.base import Validator
from truthound.validators.completeness.null import NullValidator
from truthound.validators.distribution.range import RangeValidator

pytestmark = pytest.mark.contract


def engine(data):
    return DaskExecutionEngine(dd.from_pandas(pd.DataFrame(data), npartitions=2))


class NamedCustom(Validator):
    name = "null_custom"
    category = "completeness"

    def validate(self, lf):
        raise RuntimeError("custom validation executed")


def test_name_cannot_replace_custom_validator_and_forced_native_rejects_it():
    data = engine({"x": [1, 2]})
    with pytest.raises(RuntimeError, match="custom validation executed"):
        DistributedValidatorAdapter(data).validate(NamedCustom())
    with pytest.raises(ValueError, match="native"):
        DistributedValidatorAdapter(
            data, AdapterConfig(strategy=ExecutionStrategy.NATIVE)
        ).validate(NamedCustom())


@pytest.mark.parametrize(
    "validator", [NullValidator(columns=["x"]), NullValidator(mostly=0.5), RangeValidator()]
)
def test_native_or_fallback_reuses_real_validator_issue_contract(validator):
    data = engine({"x": pd.Series([1, None], dtype="Int64"), "age": [-1, 20]})
    expected = validator.validate(data.to_polars_lazyframe())
    actual = DistributedValidatorAdapter(data).validate(validator)
    assert [issue.to_dict() for issue in actual] == [issue.to_dict() for issue in expected]


def test_sample_failure_releases_only_owned_child_and_preserves_root(monkeypatch):
    data = engine({"x": [1, 2, 3, 4]})
    exits = []
    native_sample = data.sample

    class ObservedChild(DaskExecutionEngine):
        def __exit__(self, *exc):
            exits.append(self)
            super().__exit__(*exc)

    def sample(n, seed):
        child = native_sample(n=n, seed=seed)
        return ObservedChild(child.dask_dataframe)

    monkeypatch.setattr(data, "sample", sample)
    adapter = DistributedValidatorAdapter(
        data, AdapterConfig(strategy=ExecutionStrategy.SAMPLE_AND_LOCAL, sample_size=2)
    )
    with pytest.raises(RuntimeError, match="custom validation executed"):
        adapter.validate(NamedCustom())
    assert len(exits) == 1
    assert data.count_rows() == 4


def test_sampled_issues_keep_observed_counts_and_structured_details():
    data = engine({"x": pd.Series([None] * 6, dtype="Int64"), "y": [1] * 6})
    adapter = DistributedValidatorAdapter(
        data, AdapterConfig(strategy=ExecutionStrategy.SAMPLE_AND_LOCAL, sample_size=2)
    )
    issues = adapter.validate(NullValidator(), columns=["x"])
    assert len(issues) == 1
    issue = issues[0]
    assert issue.count == issue.result.unexpected_count == 2
    assert issue.result.element_count == 2
    assert issue.sampling.total_rows == 6
    assert issue.sampling.sample_rows == 2
    assert issue.to_dict()["sampling"]["sample_rows"] == 2


def test_parallel_validation_really_overlaps_and_preserves_input_order():
    rendezvous = Barrier(2, timeout=5)

    class Concurrent(Validator):
        name = "concurrent_contract"
        category = "custom"

        def validate(self, lf):
            rendezvous.wait()
            return NullValidator().validate(lf)

    data = engine({"x": pd.Series([1, None], dtype="Int64")})
    adapter = DistributedValidatorAdapter(
        data,
        AdapterConfig(
            strategy=ExecutionStrategy.FULL_CONVERSION, parallel_validators=True, max_parallel=2
        ),
    )
    assert [issue.count for issue in adapter.validate_all([Concurrent(), Concurrent()])] == [1, 1]
