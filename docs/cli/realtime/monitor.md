# truthound realtime monitor

Monitor streaming validation metrics continuously.

## Synopsis

```bash
truthound realtime monitor <source> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `source` | Yes | Streaming source (mock, kafka:topic, kinesis:stream) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--interval` | `-i` | `5` | Monitoring interval in seconds |
| `--duration` | `-d` | `60` | Monitoring duration in seconds (0=unlimited) |

## Description

The `realtime monitor` command provides continuous metric monitoring:

1. **Connects** to the streaming source
2. **Collects** validation metrics at intervals
3. **Displays** real-time statistics
4. **Tracks** trends over time

## Supported Sources

| Source | Format | Description | Dependency |
|--------|--------|-------------|------------|
| Mock | `mock` | Test mock data source | Built-in |
| Kafka | `kafka:topic_name` | Apache Kafka topic | `aiokafka` |
| Kinesis | `kinesis:stream_name` | AWS Kinesis stream | `aiobotocore` |

## Examples

### Basic Monitoring

```bash
truthound realtime monitor mock
```

Output:
```
Realtime Monitoring
===================
Source: mock
Interval: 5s
Duration: 60s

Time        Records/s    Issues/s    Pass Rate    Lag
─────────────────────────────────────────────────────────────
00:00:05    1,234        12          99.03%       0
00:00:10    1,456        8           99.45%       0
00:00:15    1,389        15          98.92%       0
00:00:20    1,502        6           99.60%       0
...

Summary (60s)
─────────────────────────────────────────────────────────────
Total Records: 84,234
Total Issues: 523
Average Pass Rate: 99.38%
Peak Throughput: 1,567 records/s
─────────────────────────────────────────────────────────────
```

### Custom Interval

```bash
# Monitor every 10 seconds
truthound realtime monitor mock --interval 10

# Monitor every 2 seconds for higher resolution
truthound realtime monitor mock --interval 2
```

### Custom Duration

```bash
# Monitor for 5 minutes
truthound realtime monitor mock --duration 300

# Monitor for 1 hour
truthound realtime monitor mock --duration 3600

# Monitor indefinitely (Ctrl+C to stop)
truthound realtime monitor mock --duration 0
```

### Kafka Topic Monitoring

```bash
# Basic Kafka monitoring
truthound realtime monitor kafka:my_topic

# Custom interval and duration
truthound realtime monitor kafka:orders --interval 10 --duration 300
```

Output with Kafka:
```
Realtime Monitoring
===================
Source: kafka:orders
Interval: 10s
Duration: 300s

Time        Records/s    Issues/s    Pass Rate    Lag       Partitions
──────────────────────────────────────────────────────────────────────────
00:00:10    2,456        24          99.02%       1,234     3/3
00:00:20    2,789        18          99.35%       987       3/3
00:00:30    2,634        31          98.82%       1,567     3/3
...
```

### Kinesis Stream Monitoring

```bash
# Basic Kinesis monitoring
truthound realtime monitor kinesis:my_stream

# Extended monitoring
truthound realtime monitor kinesis:events --interval 5 --duration 600
```

## Metrics Displayed

| Metric | Description |
|--------|-------------|
| `Records/s` | Records processed per second |
| `Issues/s` | Validation issues per second |
| `Pass Rate` | Percentage of records passing validation |
| `Lag` | Consumer lag (Kafka/Kinesis only) |
| `Partitions` | Active partitions (Kafka only) |
| `Shards` | Active shards (Kinesis only) |

## Interval Guidelines

| Interval | Use Case | Overhead |
|----------|----------|----------|
| 1-2s | High-resolution debugging | Higher |
| 5-10s | Normal monitoring | Medium |
| 30-60s | Long-term trending | Lower |

**Recommendations:**
- **Debugging**: 1-2 second intervals
- **Normal operations**: 5-10 second intervals
- **Long-term monitoring**: 30-60 second intervals

## Use Cases

### 1. Development Debugging

```bash
# High-resolution monitoring for debugging
truthound realtime monitor mock --interval 2 --duration 60
```

### 2. Production Monitoring

```bash
# Standard production monitoring
truthound realtime monitor kafka:orders --interval 10 --duration 0
```

### 3. Performance Testing

```bash
# Monitor during load test
truthound realtime monitor kafka:test_topic --interval 5 --duration 300
```

### 4. Alerting Integration

```bash
#!/bin/bash
# monitor_with_alerts.sh

truthound realtime monitor kafka:orders --interval 10 --duration 0 | while read line; do
  pass_rate=$(echo "$line" | awk '{print $4}' | tr -d '%')
  if [ ! -z "$pass_rate" ] && [ "$pass_rate" -lt 95 ]; then
    echo "ALERT: Pass rate dropped to ${pass_rate}%"
    # Send alert notification
  fi
done
```

### 5. CI/CD Health Check

```yaml
# GitHub Actions
- name: Monitor Streaming Health
  run: |
    timeout 60 truthound realtime monitor kafka:test_topic \
      --interval 5 \
      --duration 30 || true
```

## Display Modes

### Console Mode (Default)

Real-time updating display with colors and formatting.

### JSON Output (Future)

```bash
# JSON output for programmatic parsing (planned)
truthound realtime monitor mock --format json
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Stop monitoring |

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Monitoring completed or stopped gracefully |
| 1 | Monitoring error |
| 2 | Invalid arguments or connection error |

## Related Commands

- [`realtime validate`](validate.md) - Validate streaming data
- [`realtime checkpoint`](checkpoint/index.md) - Manage checkpoints

## See Also

- [Realtime Overview](index.md)
- [Prometheus Metrics](../../concepts/advanced.md)
- [CI/CD Integration](../../guides/ci-cd.md)
