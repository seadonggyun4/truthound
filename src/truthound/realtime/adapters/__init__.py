"""Stream adapters for various messaging platforms.

This module provides adapter implementations for different streaming
platforms following the Protocol-based abstractions.

Available Adapters:
    - KafkaAdapter: Apache Kafka (aiokafka)
    - KinesisAdapter: AWS Kinesis (aiobotocore)
    - PulsarAdapter: Apache Pulsar (pulsar-client)
    - MockAdapter: Testing adapter
"""

from truthound.realtime.adapters.base import (
    BaseStreamAdapter,
    AdapterConfig,
    AdapterState,
)
from truthound.realtime.adapters.mock import (
    MockAdapter,
    MockAdapterConfig,
)
from truthound.realtime.adapters.kafka import (
    KafkaAdapter,
    KafkaAdapterConfig,
)
from truthound.realtime.adapters.kinesis import (
    KinesisAdapter,
    KinesisAdapterConfig,
)

__all__ = [
    # Base
    "BaseStreamAdapter",
    "AdapterConfig",
    "AdapterState",
    # Implementations
    "MockAdapter",
    "MockAdapterConfig",
    "KafkaAdapter",
    "KafkaAdapterConfig",
    "KinesisAdapter",
    "KinesisAdapterConfig",
]
