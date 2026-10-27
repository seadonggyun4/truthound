"""Queue native producer batches without losing per-message routing or acks."""

import asyncio
from datetime import UTC, datetime

import pytest

from truthound.realtime.adapters.kafka import KafkaAdapter, KafkaAdapterConfig
from truthound.realtime.protocols import MessageHeader, StreamMessage

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("failed_ack", [None, 0])
def test_native_batch_preserves_routing_and_observes_all_acknowledgements(failed_ack):
    async def run():
        queued, acknowledged = [], []

        class Producer:
            async def send(self, **arguments):
                index = len(queued)
                queued.append(arguments)
                future = asyncio.get_running_loop().create_future()

                def acknowledge():
                    # All sends must be queued before waiting on acknowledgements.
                    assert len(queued) == 3
                    acknowledged.append(index)
                    if index == failed_ack:
                        future.set_exception(RuntimeError("synthetic delivery failure"))
                    else:
                        future.set_result(None)

                asyncio.get_running_loop().call_soon(acknowledge)
                return future

        adapter = KafkaAdapter(KafkaAdapterConfig(bootstrap_servers="127.0.0.1:1", topic="default"))
        adapter._producer = Producer()
        messages = [
            StreamMessage(
                key=f"key-{index}", value={"id": index}, partition=index - 1,
                offset=-1, timestamp=datetime.now(UTC),
                headers=(MessageHeader(key="trace", value=b"synthetic"),),
                topic="alternate" if index == 2 else "",
            ) for index in range(3)
        ]
        if failed_ack is None:
            await adapter._do_produce_batch(messages)
        else:
            with pytest.raises(RuntimeError, match="synthetic delivery failure"):
                await adapter._do_produce_batch(messages)
        assert acknowledged == [0, 1, 2]
        assert [item["topic"] for item in queued] == ["default", "default", "alternate"]
        assert [item["partition"] for item in queued] == [None, 0, 1]
        assert [item["key"] for item in queued] == ["key-0", "key-1", "key-2"]
        assert [item["value"] for item in queued] == [{"id": 0}, {"id": 1}, {"id": 2}]
        assert all(item["headers"] == [("trace", b"synthetic")] for item in queued)

    asyncio.run(run())
