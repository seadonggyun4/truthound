"""Native reversible pipelines must reject truncated or mismatched execution plans."""

import copy

import pytest

from truthound.stores.encryption.base import DecryptionError
from truthound.stores.encryption.pipeline import EncryptionPipeline, PipelineHeader, StageType

pytestmark = pytest.mark.fault


@pytest.mark.parametrize("change", ["missing", "extra", "type", "algorithm", "version"])
def test_encryption_reverse_rejects_header_plan_mismatch_before_any_stage(change):
    pipeline = EncryptionPipeline().add_compression("gzip").add_checksum()
    result = pipeline.process(b"synthetic" * 100)
    header = copy.deepcopy(result.header)
    if change == "missing":
        header.stages.pop()
    elif change == "extra":
        header.stages.append(copy.deepcopy(header.stages[-1]))
    elif change == "type":
        header.stages[0]["type"] = StageType.ENCRYPTION.value
    elif change == "algorithm":
        header.stages[0]["config"]["algorithm"] = "zstd"
    else:
        header.version = 255
    with pytest.raises(DecryptionError):
        pipeline.reverse(result.data, header)


@pytest.mark.parametrize("data", [b"THEP", b"THEP\x01", b"THEP\x01\x00\x01\x00\x01\x00\xff{}"])
def test_truncated_pipeline_header_has_typed_failure(data):
    with pytest.raises(DecryptionError):
        PipelineHeader.from_bytes(data)


def test_pipeline_authorization_revocation_stops_next_stage():
    from truthound.stores.encryption.pipeline import ChecksumStage

    calls = []

    class First(ChecksumStage):
        def process(self, data):
            calls.append("first")
            return super().process(data)

    class Second(ChecksumStage):
        def process(self, data):
            calls.append("second")
            return super().process(data)

    pipeline = EncryptionPipeline().add_stage(First()).add_stage(Second())

    def guard():
        if calls:
            raise PermissionError("synthetic-private-revoked")

    pipeline.set_execution_guard(guard)
    with pytest.raises(PermissionError):
        pipeline.process(b"synthetic")
    assert calls == ["first"]


def test_composite_backpressure_does_not_admit_when_a_strategy_raises():
    import asyncio

    from truthound.stores.backpressure.strategies import (
        CompositeBackpressure,
        TokenBucketBackpressure,
    )

    class Failed(TokenBucketBackpressure):
        async def acquire(self):
            raise OSError("synthetic-private-pressure-provider")

    async def scenario():
        pressure = CompositeBackpressure([Failed()])
        with pytest.raises(ExceptionGroup):
            await pressure.acquire()

    asyncio.run(scenario())


def test_backpressure_cancelled_pause_resets_paused_state():
    import asyncio

    from truthound.stores.backpressure.base import BackpressureConfig
    from truthound.stores.backpressure.strategies import QueueDepthBackpressure

    async def scenario():
        pressure = QueueDepthBackpressure(BackpressureConfig(min_pause_ms=100, max_pause_ms=100))
        pressure.update_metrics(queue_depth=100000)
        task = asyncio.create_task(pressure.acquire())
        try:
            await asyncio.sleep(0)
            assert pressure.state.is_paused
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not pressure.state.is_paused
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_token_bucket_reset_restores_native_permits_instead_of_old_depleted_bucket():
    from truthound.stores.backpressure.base import BackpressureConfig
    from truthound.stores.backpressure.strategies import TokenBucketBackpressure

    pressure = TokenBucketBackpressure(BackpressureConfig(base_rate=1, min_rate=1), bucket_size=1)
    assert pressure.try_acquire()
    assert not pressure.try_acquire()
    pressure.reset()
    assert pressure.try_acquire()


@pytest.mark.parametrize("method", ["process", "reverse"])
def test_placeholder_compression_stage_cannot_claim_identity_transform_success(method):
    from truthound.stores.compression.pipeline import PipelineError, PipelineStage, StageType

    stage = PipelineStage("unsupported", StageType.ENCRYPT, processor=None)
    with pytest.raises(PipelineError):
        getattr(stage, method)(b"synthetic")
