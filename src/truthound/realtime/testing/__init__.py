"""Testing utilities for real-time streaming validation.

Provides Testcontainers integration for real integration testing
with actual Kafka and other streaming platforms.
"""

from truthound.realtime.testing.containers import (
    KafkaTestContainer,
    RedisTestContainer,
    LocalStackTestContainer,
    StreamTestEnvironment,
)

from truthound.realtime.testing.fixtures import (
    TestMessageGenerator,
    TestFixtures,
    create_test_messages,
    create_test_schema,
)

__all__ = [
    # Containers
    "KafkaTestContainer",
    "RedisTestContainer",
    "LocalStackTestContainer",
    "StreamTestEnvironment",
    # Fixtures
    "TestMessageGenerator",
    "TestFixtures",
    "create_test_messages",
    "create_test_schema",
]
