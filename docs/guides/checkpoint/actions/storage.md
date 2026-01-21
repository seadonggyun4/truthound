# Storage Actions

Actions for storing results and generating documentation.

## StoreValidationResult

Stores validation results to file system, S3, GCS, and other storage backends.

### Basic Usage

```python
from truthound.checkpoint.actions import StoreValidationResult

action = StoreValidationResult(
    store_path="./results",
    format="json",
    partition_by="date",
)
```

### Configuration (StoreResultConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `store_path` | `str \| Path` | `"./truthound_results"` | Storage path (local, `s3://`, `gs://`) |
| `store_type` | `str` | `"file"` | Storage type: `file`, `s3`, `gcs` |
| `format` | `str` | `"json"` | Format: `json`, `yaml` |
| `partition_by` | `str` | `"date"` | Partition: `date`, `checkpoint`, `status`, `` |
| `retention_days` | `int` | `0` | Retention period (0 = unlimited) |
| `include_validation_details` | `bool` | `True` | Include detailed validation results |
| `compress` | `bool` | `False` | Enable gzip compression |
| `notify_on` | `str` | `"always"` | Execution condition |

### Storage Path Structure

Storage path based on `partition_by`:

```
# partition_by="date"
./results/2024/01/15/{run_id}.json

# partition_by="checkpoint"
./results/daily_data_validation/{run_id}.json

# partition_by="status"
./results/failure/{run_id}.json

# partition_by="" (none)
./results/{run_id}.json
```

### Local File System

```python
action = StoreValidationResult(
    store_path="./truthound_results",
    store_type="file",
    format="json",
    partition_by="date",
    compress=True,  # Saves as .json.gz
)
```

### AWS S3

```python
action = StoreValidationResult(
    store_path="s3://my-bucket/dq-results",
    store_type="s3",
    format="json",
    partition_by="date",
)

# AWS credentials use environment variables or AWS configuration files
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
```

Requirement: `pip install boto3`

### Google Cloud Storage

```python
action = StoreValidationResult(
    store_path="gs://my-bucket/dq-results",
    store_type="gcs",
    format="json",
    partition_by="checkpoint",
)

# GCP credentials use GOOGLE_APPLICATION_CREDENTIALS environment variable
```

Requirement: `pip install google-cloud-storage`

### Result Format

Stored JSON structure:

```json
{
  "run_id": "20240115_120000_abc123",
  "checkpoint_name": "daily_data_validation",
  "run_time": "2024-01-15T12:00:00",
  "status": "failure",
  "data_asset": "users.csv",
  "duration_ms": 1523.5,
  "validation_result": {
    "statistics": {
      "total_issues": 150,
      "critical_issues": 5,
      "high_issues": 25,
      "medium_issues": 70,
      "low_issues": 50,
      "pass_rate": 0.85,
      "total_rows": 100000,
      "total_columns": 15
    },
    "results": [...]  // When include_validation_details=True
  },
  "action_results": [...],
  "metadata": {...}
}
```

---

## UpdateDataDocs

Generates HTML format validation reports.

### Basic Usage

```python
from truthound.checkpoint.actions import UpdateDataDocs

action = UpdateDataDocs(
    site_path="./docs",
    format="html",
    include_history=True,
)
```

### Configuration

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `site_path` | `str \| Path` | `"./truthound_docs"` | Output directory |
| `format` | `str` | `"html"` | Output format: `html`, `markdown` |
| `include_history` | `bool` | `True` | Include history |
| `max_history_items` | `int` | `100` | Maximum history items |
| `template` | `str` | `"default"` | Template: `default`, `minimal`, `detailed` |
| `notify_on` | `str` | `"always"` | Execution condition |

### Generated File Structure

```
./docs/
├── index.html                 # Dashboard
├── checkpoints/
│   └── daily_data_validation/
│       ├── index.html         # Checkpoint overview
│       └── runs/
│           ├── 20240115_120000.html
│           └── 20240114_120000.html
├── history/
│   └── trend.json            # Trend data
└── assets/
    ├── style.css
    └── script.js
```

### Template Options

```python
# Default template - includes all information
action = UpdateDataDocs(template="default")

# Minimal template - essential information only
action = UpdateDataDocs(template="minimal")

# Detailed template - all issue details
action = UpdateDataDocs(template="detailed")
```

### YAML Configuration

```yaml
actions:
  - type: store_result
    store_path: ./truthound_results
    partition_by: date
    format: json
    compress: true

  - type: update_docs
    site_path: ./truthound_docs
    format: html
    include_history: true
    max_history_items: 50
    template: default
```
