# Streaming Data Sources

This document covers streaming data sources in Truthound, including Apache Kafka integration.

## Overview

Streaming data sources allow you to validate data from message streaming platforms. These are **bounded** data sources designed for batch processing of a fixed number of messages, not real-time streaming.

> **Note**: For real-time streaming validation with windowing and continuous processing, see `truthound.realtime.streaming`.

| Platform | Driver | Installation |
|----------|--------|--------------|
| Kafka | `aiokafka` | `pip install aiokafka` |

### Key Characteristics

- **Bounded consumption**: Reads up to `max_messages` messages
- **Async-first**: All operations are async
- **Schema inference**: Automatically infers schema from message samples
- **Message deserialization**: Supports JSON, Avro, and string formats

## Apache Kafka

### Installation

```bash
pip install aiokafka
```

### Basic Usage

```python
from truthound.datasources.streaming import KafkaDataSource, KafkaDataSourceConfig

config = KafkaDataSourceConfig(
    bootstrap_servers="localhost:9092",
    topic="my-topic",
    max_messages=10000,
)

source = KafkaDataSource(config)

async with source:
    schema = await source.get_schema_async()
    lf = await source.to_polars_lazyframe_async()
```

### Connection String

```python
from truthound.datasources.streaming import KafkaDataSource

# From connection string
source = KafkaDataSource.from_connection_string(
    "kafka://localhost:9092/my-topic",
    max_messages=5000,
)

# Multiple brokers
source = KafkaDataSource.from_connection_string(
    "kafka://broker1:9092,broker2:9092/my-topic",
)
```

### Configuration

```python
from truthound.datasources.streaming import KafkaDataSourceConfig

config = KafkaDataSourceConfig(
    # Connection
    bootstrap_servers="localhost:9092",
    topic="user-events",
    group_id="truthound-datasource",

    # Consumption
    max_messages=10000,           # Maximum messages to consume (bounded)
    consume_timeout=30.0,         # Timeout waiting for messages
    auto_offset_reset="earliest", # Start from beginning

    # Security
    security_protocol="PLAINTEXT",  # PLAINTEXT, SSL, SASL_SSL
    sasl_mechanism=None,            # PLAIN, SCRAM-SHA-256, etc.
    sasl_username=None,
    sasl_password=None,
    ssl_cafile=None,
    ssl_certfile=None,
    ssl_keyfile=None,

    # Partition control
    partition=None,               # Specific partition (None for all)
    start_offset=None,            # Starting offset
    end_offset=None,              # Ending offset

    # Message handling
    key_deserializer="string",    # json, string, bytes
    include_key=True,             # Include message key
    include_headers=False,        # Include message headers
    deserializer_type="json",     # Message value format

    # Metadata
    include_metadata=False,       # Include Kafka metadata
    metadata_prefix="_kafka_",    # Prefix for metadata fields

    # Schema inference
    schema_sample_size=100,       # Messages to sample for schema
)
```

### Authentication

#### SASL/PLAIN

```python
config = KafkaDataSourceConfig(
    bootstrap_servers="kafka.example.com:9093",
    topic="secure-topic",
    security_protocol="SASL_SSL",
    sasl_mechanism="PLAIN",
    sasl_username="user",
    sasl_password="password",
)
source = KafkaDataSource(config)
```

#### SASL/SCRAM

```python
config = KafkaDataSourceConfig(
    bootstrap_servers="kafka.example.com:9093",
    topic="secure-topic",
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-256",
    sasl_username="user",
    sasl_password="password",
)
```

#### SSL/TLS

```python
config = KafkaDataSourceConfig(
    bootstrap_servers="kafka.example.com:9093",
    topic="secure-topic",
    security_protocol="SSL",
    ssl_cafile="/path/to/ca.pem",
    ssl_certfile="/path/to/client.pem",
    ssl_keyfile="/path/to/client.key",
)
```

### Confluent Cloud

```python
from truthound.datasources.streaming import KafkaDataSource

source = KafkaDataSource.from_confluent(
    bootstrap_servers="pkc-xxxxx.us-east-1.aws.confluent.cloud:9092",
    topic="my-topic",
    api_key="ABCDEFGHIJKLMNOP",
    api_secret="secret123",
    max_messages=10000,
)
```

### Message Deserialization

#### JSON (Default)

```python
config = KafkaDataSourceConfig(
    bootstrap_servers="localhost:9092",
    topic="json-topic",
    deserializer_type="json",  # Default
)
```

Message format:
```json
{"user_id": 123, "action": "login", "timestamp": "2024-01-01T00:00:00Z"}
```

#### Avro

```python
config = KafkaDataSourceConfig(
    bootstrap_servers="localhost:9092",
    topic="avro-topic",
    deserializer_type="avro",
)
```

> **Note**: Avro deserialization requires the `fastavro` library: `pip install fastavro`

### Including Metadata

Include Kafka message metadata in the output:

```python
config = KafkaDataSourceConfig(
    bootstrap_servers="localhost:9092",
    topic="my-topic",
    include_metadata=True,      # Include topic, partition, offset, timestamp
    include_key=True,           # Include message key
    include_headers=True,       # Include message headers
    metadata_prefix="_kafka_",  # Prefix for metadata fields
)

source = KafkaDataSource(config)

async with source:
    lf = await source.to_polars_lazyframe_async()
    df = lf.collect()
    print(df.columns)
    # ['user_id', 'action', '_kafka_key', '_kafka_topic', '_kafka_partition', '_kafka_offset', '_kafka_timestamp']
```

### Partition Control

Read from specific partitions or offset ranges:

```python
# Specific partition
config = KafkaDataSourceConfig(
    bootstrap_servers="localhost:9092",
    topic="my-topic",
    partition=0,  # Only partition 0
)

# Offset range
config = KafkaDataSourceConfig(
    bootstrap_servers="localhost:9092",
    topic="my-topic",
    start_offset=1000,
    end_offset=2000,
)
```

### Topic Information

```python
source = KafkaDataSource(config)

async with source:
    # Get topic metadata
    info = await source._get_topic_info()
    # {'topic': 'my-topic', 'partitions': [0, 1, 2], 'partition_count': 3}

    # Get partition offsets
    offsets = await source.get_topic_offsets_async()
    # {'offsets': {0: (0, 1000), 1: (0, 1500), 2: (0, 800)}}
    # Format: {partition: (beginning_offset, end_offset)}
```

### Consuming Messages

#### Batch Consumption

```python
source = KafkaDataSource(config)

async with source:
    # Consume all messages up to max_messages
    lf = await source.to_polars_lazyframe_async()
    df = lf.collect()
    print(f"Consumed {len(df)} messages")
```

#### Iterative Consumption

```python
source = KafkaDataSource(config)

async with source:
    async for batch in source.iter_messages_async(
        batch_size=100,
        max_messages=10000,
    ):
        # Process each batch
        for msg in batch:
            process(msg)
```

### Consumer Group Operations

```python
source = KafkaDataSource(config)

async with source:
    # Consume messages
    lf = await source.to_polars_lazyframe_async()

    # Commit offsets (save progress)
    await source.commit_offsets_async()

    # Get committed offsets
    committed = await source.get_committed_offsets_async()
    # {0: 1000, 1: 1500, 2: 800}
```

### Sampling

```python
source = KafkaDataSource(config)

async with source:
    # Create a sampled data source with fewer messages
    sampled = await source.sample_async(n=1000)

    async with sampled:
        lf = await sampled.to_polars_lazyframe_async()
```

> **Note**: Kafka doesn't support random sampling. Sampling returns a source configured to consume fewer messages from the beginning.

## Message Deserializers

### Built-in Deserializers

| Type | Class | Description |
|------|-------|-------------|
| `json` | `JSONDeserializer` | JSON messages |
| `avro` | `AvroDeserializer` | Apache Avro (requires `fastavro`) |
| `string` | `JSONDeserializer` | Treated as JSON |

### JSONDeserializer

```python
from truthound.datasources.streaming import JSONDeserializer

deserializer = JSONDeserializer(encoding="utf-8")

# Deserialize bytes
message = deserializer.deserialize(b'{"user": "alice", "action": "login"}')
# {'user': 'alice', 'action': 'login'}
```

### AvroDeserializer

```python
from truthound.datasources.streaming import AvroDeserializer

# With schema
schema = {
    "type": "record",
    "name": "UserEvent",
    "fields": [
        {"name": "user_id", "type": "int"},
        {"name": "action", "type": "string"},
    ],
}

deserializer = AvroDeserializer(schema=schema)
message = deserializer.deserialize(avro_bytes)
```

## Validation Example

Using Kafka source with the validation API:

```python
import truthound as th
from truthound.datasources.streaming import KafkaDataSource, KafkaDataSourceConfig

async def validate_kafka_topic():
    config = KafkaDataSourceConfig(
        bootstrap_servers="localhost:9092",
        topic="user-events",
        max_messages=10000,
    )

    source = KafkaDataSource(config)

    async with source:
        # Get LazyFrame
        lf = await source.to_polars_lazyframe_async()
        df = lf.collect()

        # Validate with truthound
        report = th.check(
            df,
            validators=["null", "type"],
            columns=["user_id", "action", "timestamp"],
        )

        # Or with rules
        report = th.check(
            df,
            rules={
                "user_id": ["not_null"],
                "action": [{"type": "allowed_values", "values": ["login", "logout", "purchase"]}],
                "timestamp": ["not_null", {"type": "datetime"}],
            },
        )

        print(f"Validated {len(df)} messages, found {len(report.issues)} issues")

# Run
import asyncio
asyncio.run(validate_kafka_topic())
```

## Error Handling

```python
from truthound.datasources.streaming import (
    KafkaDataSource,
    KafkaDataSourceConfig,
    KafkaDataSourceError,
    KafkaConnectionError,
    DeserializationError,
)

config = KafkaDataSourceConfig(
    bootstrap_servers="localhost:9092",
    topic="my-topic",
)

try:
    source = KafkaDataSource(config)
    async with source:
        lf = await source.to_polars_lazyframe_async()
except KafkaConnectionError as e:
    print(f"Connection failed: {e}")
    print(f"Bootstrap servers: {e.bootstrap_servers}")
except DeserializationError as e:
    print(f"Failed to deserialize message: {e}")
    print(f"Offset: {e.offset}")
except KafkaDataSourceError as e:
    print(f"Kafka error: {e}")
```

## Factory Functions

```python
from truthound.datasources import from_kafka

# Create Kafka source
source = await from_kafka(
    bootstrap_servers="localhost:9092",
    topic="my-topic",
    max_messages=10000,
)

async with source:
    lf = await source.to_polars_lazyframe_async()
```

## Base Classes

For implementing custom streaming sources:

```python
from truthound.datasources.streaming import (
    BaseStreamingDataSource,
    StreamingDataSourceConfig,
    MessageDeserializer,
    JSONDeserializer,
    AvroDeserializer,
    StreamingDataSourceError,
    DeserializationError,
)
```

## Best Practices

1. **Set appropriate `max_messages`** - Balance between data coverage and memory
2. **Use `consume_timeout`** - Prevent hanging on empty topics
3. **Configure `auto_offset_reset`** - Use "earliest" for full topic scan, "latest" for recent only
4. **Enable SSL/SASL in production** - Secure connections with authentication
5. **Sample for schema inference** - Smaller samples are faster; increase if schema is complex
6. **Handle deserialization errors** - Malformed messages are skipped by default
7. **Commit offsets after processing** - Track progress for subsequent runs

## Phase 5 vs Phase 10

| Feature | Phase 5 (DataSources) | Phase 10 (Realtime) |
|---------|----------------------|---------------------|
| Purpose | Batch validation | Continuous monitoring |
| Consumption | Bounded (max_messages) | Unbounded (continuous) |
| API | Async DataSource | Stream adapters |
| Use case | Data quality checks | Real-time alerting |
| Module | `truthound.datasources.streaming` | `truthound.realtime.streaming` |

Use Phase 5 streaming sources for periodic batch validation of recent messages. Use Phase 10 realtime streaming for continuous, event-driven validation with windowing and micro-batch processing.
