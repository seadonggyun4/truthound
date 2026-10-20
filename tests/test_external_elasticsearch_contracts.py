"""Regression coverage for real-service fixture failures, without a live account."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tests.integration.external.backends.elasticsearch_backend import ElasticsearchBackend
from tests.integration.external.tests.test_logging_sinks import _log_timestamp


@pytest.mark.parametrize("interval", [1, 60])
def test_all_hundred_log_rows_have_valid_distinct_timestamps(interval):
    values = [_log_timestamp(i, interval) for i in range(100)]
    times = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in values]
    assert len(set(times)) == 100
    assert all(
        (b - a).total_seconds() == interval
        for a, b in zip(times[:-1], times[1:], strict=True)
    )
    assert (times[-1] - times[0]).total_seconds() == 99 * interval


@pytest.mark.parametrize(
    "resolved",
    [
        {},
        {"truthound_test_a": {}, "truthound_test_b": {}},
        {"production": {}},
        {"truthound_test_*": {}},
    ],
)
def test_cleanup_keeps_destructive_name_safety_and_rejects_foreign_names(resolved):
    backend = ElasticsearchBackend()
    indices = Mock()
    indices.get.return_value = resolved
    backend._client = SimpleNamespace(indices=indices)
    unsafe = any(not n.startswith("truthound_test_") or "*" in n for n in resolved)
    if unsafe:
        with pytest.raises(ValueError):
            backend.cleanup_test_indices()
        indices.delete.assert_not_called()
    else:
        assert backend.cleanup_test_indices() == len(resolved)
        assert indices.delete.call_args_list == [call(index=n, ignore=[404]) for n in resolved]
    for prefix in ("", "*", "truthound_*", "a,b", "a?"):
        with pytest.raises(ValueError):
            backend.cleanup_test_indices(prefix)
