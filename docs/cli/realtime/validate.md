# truthound realtime validate

Validate streaming data from real-time sources.

## Synopsis

```bash
truthound realtime validate <source> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `source` | Yes | Streaming source (mock, kafka:topic, kinesis:stream) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--validators` | `-v` | All | Validators to use (comma-separated) |
| `--batch-size` | `-b` | `1000` | Batch size for processing |
| `--max-batches` | | `10` | Maximum batches to process (0=unlimited) |
| `--output` | `-o` | None | Output file for validation results |

## Description

The `realtime validate` command validates streaming data in real-time:

1. **Connects** to the streaming source
2. **Batches** incoming records
3. **Validates** each batch with specified validators
4. **Saves** checkpoints for recovery
5. **Reports** validation results

## Supported Sources

| Source | Format | Description | Dependency |
|--------|--------|-------------|------------|
| Mock | `mock` | Test mock data source | Built-in |
| Kafka | `kafka:topic_name` | Apache Kafka topic | `aiokafka` |
| Kinesis | `kinesis:stream_name` | AWS Kinesis stream | `aiobotocore` |

## Examples

### Basic Validation with Mock Source

```bash
truthound realtime validate mock
```

Output:
```
Realtime Validation
===================
Source: mock
Batch Size: 1000
Max Batches: 10

Processing batches...
  Batch 1/10: 1000 records, 5 issues
  Batch 2/10: 1000 records, 3 issues
  ...
  Batch 10/10: 1000 records, 2 issues

Summary
───────────────────────────────────────────────────────────────────
Total Records: 10,000
Total Issues: 42
Pass Rate: 99.58%
Checkpoint: abc12345
───────────────────────────────────────────────────────────────────
```

### Specify Validators

```bash
truthound realtime validate mock --validators null,range
```

Only runs `null` and `range` validators on the streaming data.

### Custom Batch Size

```bash
# Smaller batches for lower latency
truthound realtime validate mock --batch-size 500

# Larger batches for higher throughput
truthound realtime validate mock --batch-size 2000
```

### Limit Batch Count

```bash
# Process only 5 batches
truthound realtime validate mock --max-batches 5

# Unlimited processing (run until stopped)
truthound realtime validate mock --max-batches 0
```

### Kafka Topic Validation

```bash
# Basic Kafka validation
truthound realtime validate kafka:my_topic

# With custom settings
truthound realtime validate kafka:orders \
  --validators null,range,unique \
  --batch-size 500 \
  --max-batches 100
```

!!! note "Kafka Dependency"
    Kafka support requires aiokafka:
    ```bash
    pip install truthound[kafka]
    ```

### Kinesis Stream Validation

```bash
# Basic Kinesis validation
truthound realtime validate kinesis:my_stream

# With custom settings
truthound realtime validate kinesis:events \
  --batch-size 2000 \
  --max-batches 50
```

!!! note "Kinesis Dependency"
    Kinesis support requires aiobotocore:
    ```bash
    pip install truthound[kinesis]
    ```

### Save Results to File

```bash
truthound realtime validate mock -o results.json
```

Output file (`results.json`):
```json
{
  "source": "mock",
  "batch_size": 1000,
  "max_batches": 10,
  "total_records": 10000,
  "total_issues": 42,
  "pass_rate": 0.9958,
  "checkpoint_id": "abc12345",
  "batches": [
    {
      "batch_number": 1,
      "records": 1000,
      "issues": 5,
      "validators": {
        "null": {"passed": 998, "failed": 2},
        "range": {"passed": 997, "failed": 3}
      }
    }
  ],
  "summary": {
    "by_validator": {
      "null": {"total_failed": 20},
      "range": {"total_failed": 22}
    }
  }
}
```

## Validators

Common validators for streaming data:

| Validator | Description | Use Case |
|-----------|-------------|----------|
| `null` | Check for null values | Required fields |
| `range` | Check numeric ranges | Value bounds |
| `unique` | Check uniqueness | Duplicate detection |
| `pattern` | Check string patterns | Format validation |
| `type` | Check data types | Schema compliance |

```bash
# Multiple validators
truthound realtime validate kafka:orders --validators null,range,unique,pattern
```

## Batch Size Guidelines

| Batch Size | Latency | Throughput | Memory |
|------------|---------|------------|--------|
| 100-500 | Low | Lower | Low |
| 500-1000 | Medium | Medium | Medium |
| 1000-5000 | Higher | Higher | Higher |

**Recommendations:**
- **Low latency**: Use 100-500 batch size
- **High throughput**: Use 1000-5000 batch size
- **Memory constrained**: Use smaller batches

## Use Cases

### 1. Development Testing

```bash
# Test validation logic with mock data
truthound realtime validate mock \
  --validators null,range \
  --batch-size 100 \
  --max-batches 5
```

### 2. Production Kafka Pipeline

```bash
# Validate production Kafka messages
truthound realtime validate kafka:orders \
  --validators null,range,unique \
  --batch-size 1000 \
  --max-batches 0 \
  -o /logs/validation_$(date +%Y%m%d).json
```

### 3. CI/CD Integration

```yaml
# GitHub Actions
- name: Validate Streaming Data
  run: |
    truthound realtime validate kafka:test_topic \
      --max-batches 10 \
      -o results.json

    # Check results
    python -c "
    import json
    with open('results.json') as f:
        data = json.load(f)
    if data['pass_rate'] < 0.99:
        print(f'Pass rate too low: {data[\"pass_rate\"]}')
        exit(1)
    "
```

### 4. Kinesis Analytics

```bash
# Validate Kinesis stream before analytics
truthound realtime validate kinesis:clickstream \
  --validators null,type \
  --batch-size 2000 \
  --max-batches 100 \
  -o kinesis_validation.json
```

## Checkpoints

Validation automatically creates checkpoints for recovery:

```bash
# Checkpoints are saved to ./checkpoints/ by default
./checkpoints/
└── abc12345.json

# View checkpoint
truthound realtime checkpoint show abc12345

# Resume from checkpoint (automatic on restart)
truthound realtime validate kafka:orders
# Resumes from last checkpoint
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success (all batches validated) |
| 1 | Validation errors detected |
| 2 | Invalid arguments or connection error |

## Related Commands

- [`realtime monitor`](monitor.md) - Monitor validation metrics
- [`realtime checkpoint`](checkpoint.md) - Manage checkpoints
- [`check`](../core/check.md) - Batch validation

## See Also

- [Realtime Overview](index.md)
- [CI/CD Integration](../../guides/ci-cd.md)
- [Kafka Integration](../../guides/datasources.md)
