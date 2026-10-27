"""Invalid pressure settings/observations cannot silently admit work."""

import math

import pytest

from truthound.stores.backpressure.base import BackpressureConfig
from truthound.stores.backpressure.strategies import TokenBucketBackpressure

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "field",
    [
        "latency_threshold_ms",
        "min_pause_ms",
        "max_pause_ms",
        "base_rate",
        "min_rate",
        "sampling_interval_ms",
    ],
)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_timing_and_rates_cannot_create_a_strategy(field, invalid):
    with pytest.raises(ValueError):
        TokenBucketBackpressure(BackpressureConfig(**{field: invalid}))


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("adaptive_window_size", 0),
        ("adaptive_window_size", True),
        ("queue_depth_threshold", True),
        ("enabled", 1),
    ],
)
def test_invalid_window_and_boolean_settings_are_rejected(field, invalid):
    with pytest.raises(ValueError):
        TokenBucketBackpressure(BackpressureConfig(**{field: invalid}))


@pytest.mark.parametrize(
    "observations",
    [
        {"memory_percent": math.nan},
        {"queue_depth": True},
        {"latency_ms": -1},
        {"memory_percent": 90, "queue_depth": -1},
    ],
)
def test_observation_batch_is_validated_before_mutating_pressure(observations):
    strategy = TokenBucketBackpressure()
    strategy.update_metrics(memory_percent=20, queue_depth=4, latency_ms=2)
    before = strategy.metrics.to_dict()
    with pytest.raises(ValueError):
        strategy.update_metrics(**observations)
    after = strategy.metrics.to_dict()
    assert after == before
