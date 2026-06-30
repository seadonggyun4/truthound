# Streaming Data Sources

실무 운영 가이드에서 Truthound, Apache, Kafka을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 Streaming을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

> 실무 운영 가이드에서 `truthound.realtime.streaming`, Note을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 플랫폼 | 실무 운영 가이드에서 Driver을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 설치 |
|----------|--------|--------------|
| 실무 운영 가이드에서 Kafka을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `aiokafka`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install aiokafka`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Key Characteristics

- 실무 운영 가이드에서 `max_messages`, Bounded, Reads을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Async-first을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Schema, Automatically을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 JSON, Message, Supports, Avro을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Apache Kafka

### 설치

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

### 설정

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

실무 운영 가이드에서 Message을(를) 다루는 항목입니다:
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

> 실무 운영 가이드에서 `fastavro`, `pip install fastavro`, Note, Avro을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Including Metadata

실무 운영 가이드에서 Include, Kafka을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

> 실무 운영 가이드에서 Note, Kafka, Sampling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Message Deserializers

### Built-in Deserializers

| 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Class을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------|-------------|
| 실무 운영 가이드에서 `json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, `JSONDeserializer`, JSONDeserializer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `avro`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `AvroDeserializer`, AvroDeserializer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `fastavro`, Apache, Avro을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `string`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, `JSONDeserializer`, JSONDeserializer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, Treated을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

## 검증 Example

실무 운영 가이드에서 API, Kafka을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

## 권장 방식

1. 실무 운영 가이드에서 `max_messages`, Set, Balance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 `consume_timeout`, Prevent을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 `auto_offset_reset`, Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 Enable, SSL/SASL, Secure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 Sample, Smaller을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 실무 운영 가이드에서 Handle, Malformed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
7. 실무 운영 가이드에서 Commit, Track을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Phase 5 vs Phase 10

| 실무 운영 가이드에서 Feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Phase, DataSources을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Phase, Realtime을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|----------------------|---------------------|
| 실무 운영 가이드에서 Purpose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Batch 검증 | Continuous 모니터링 |
| 실무 운영 가이드에서 Consumption을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Bounded을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Unbounded을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 API을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Async, DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Stream을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터 품질 checks | 실무 운영 가이드에서 Real-time을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Module을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `truthound.datasources.streaming`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `truthound.realtime.streaming`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
