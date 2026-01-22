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
| `--validators` | `-v` | None | Validators to use (comma-separated) |
| `--batch-size` | `-b` | `1000` | Batch size for processing |
| `--max-batches` | | `10` | Maximum batches to process (0=unlimited) |
| `--output` | `-o` | None | Output file for validation results |
| `--checkpoint-dir` | `-c` | `./checkpoints` | Directory to save checkpoints |
| `--checkpoint-interval` | | `0` | Save checkpoint every N batches (0=final only) |

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
Starting streaming validation...
  Source: mock
  Batch size: 1000
  Validators: all

Batch 1: 1000 records, 5 issues [ISSUES]
Batch 2: 1000 records, 3 issues [ISSUES]
...
Batch 10: 1000 records, 2 issues [ISSUES]

Summary
========================================
Batches processed: 10
Total records: 10000
Total issues: 42
Pass rate: 99.58%
========================================
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
  "batches": [
    {
      "batch_number": 1,
      "records": 1000,
      "issues": 5
    },
    {
      "batch_number": 2,
      "records": 1000,
      "issues": 3
    }
  ],
  "stats": {
    "total_batches": 10,
    "total_records": 10000,
    "total_issues": 42,
    "pass_rate": 0.9958
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

Checkpoints are automatically saved to `./checkpoints` directory.

### Save Checkpoints

```bash
# Default: saves to ./checkpoints
truthound realtime validate mock

# Custom checkpoint directory
truthound realtime validate mock -c ./my_checkpoints

# Save checkpoint every 5 batches
truthound realtime validate mock --checkpoint-interval 5
```

Output:
```
Starting streaming validation...
  Source: mock
  Batch size: 1000
  Validators: all
  Checkpoint dir: checkpoints
  Checkpoint interval: every 5 batches

Batch 1b95bd0e: 1000 records, 0 issues [OK]
Batch 70fe5925: 1000 records, 0 issues [OK]
Batch 4ccadf60: 1000 records, 0 issues [OK]
Batch 8a2e1f3b: 1000 records, 0 issues [OK]
Batch 9c4d2e5a: 1000 records, 0 issues [OK]
  [Checkpoint saved: a1b2c3d4]
...

Final checkpoint saved: e5f6g7h8
```

### Manage Checkpoints

```bash
# List checkpoints (default: ./checkpoints)
truthound realtime checkpoint list

# View checkpoint details
truthound realtime checkpoint show a1b2c3d4

# Delete checkpoint
truthound realtime checkpoint delete a1b2c3d4

# Use custom directory
truthound realtime checkpoint list --dir ./my_checkpoints
```

### Checkpoint File Structure

```
./checkpoints/
├── checkpoint_a1b2c3d4.json
├── checkpoint_e5f6g7h8.json
└── ...
```

Each checkpoint contains:

- `checkpoint_id`: Unique identifier
- `created_at`: Creation timestamp
- `batch_count`: Number of batches processed
- `total_records`: Total records validated
- `total_issues`: Total issues found
- `state_snapshot`: Validation state for recovery

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (invalid arguments, connection error, or other error) |

> **Note**: Validation issues are reported in the output, but do not affect the exit code. Use `--output` and parse the JSON file for CI/CD decisions.

## Related Commands

- [`realtime monitor`](monitor.md) - Monitor validation metrics
- [`realtime checkpoint`](checkpoint.md) - Manage checkpoints
- [`check`](../core/check.md) - Batch validation

## See Also

- [Realtime Overview](index.md)
- [CI/CD Integration](../../guides/ci-cd.md)
- [Kafka Integration](../../guides/datasources.md)
