# Installation

## Basic Installation

Install Truthound using pip:

```bash
pip install truthound
```

This installs the core package with all essential features including:

- Polars for high-performance data processing
- PyYAML for configuration files
- Rich for beautiful console output
- Typer for the CLI interface

## Optional Dependencies

Truthound provides optional extras for specific use cases:

### Reports (HTML)

```bash
pip install truthound[reports]
```

Enables HTML report generation with Jinja2.

### PDF Export

```bash
pip install truthound[pdf]
```

Enables PDF export with WeasyPrint.

!!! warning "System Libraries Required"
    WeasyPrint requires system libraries (Pango, Cairo, etc.) that pip cannot install.
    You must install them **before** running `pip install truthound[pdf]`:

    ```bash
    # macOS
    brew install pango cairo gdk-pixbuf libffi

    # Ubuntu/Debian
    sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 \
      libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

    # Fedora/RHEL
    sudo dnf install pango gdk-pixbuf2 libffi-devel
    ```

    Without these, you'll get: `Error: cannot load library 'libpango-1.0-0'`

    See [Data Docs Guide](/guides/datadocs/#pdf-export-system-dependencies) for full instructions including Docker and Windows.

### Drift Detection

```bash
pip install truthound[drift]
```

Enables statistical drift detection with SciPy.

### Anomaly Detection

```bash
pip install truthound[anomaly]
```

Enables ML-based anomaly detection with scikit-learn.

### Cloud Storage

```bash
# AWS S3
pip install truthound[s3]

# Google Cloud Storage
pip install truthound[gcs]

# Azure Blob Storage
pip install truthound[azure]

# All cloud providers + database
pip install truthound[stores]
```

### Database

```bash
pip install truthound[database]
```

Enables SQLAlchemy-based database storage for validation results.

### Streaming

```bash
# Kafka
pip install truthound[kafka]
# or equivalently:
pip install truthound[streaming]

# All async datasources (Kafka, MongoDB, Elasticsearch)
pip install truthound[async-datasources]
```

Enables Kafka streaming support with aiokafka.

!!! note "Kinesis Support"
    Kinesis adapter uses aiobotocore which is included in the core package via boto3.

### NoSQL Databases

```bash
# MongoDB
pip install truthound[mongodb]

# Elasticsearch
pip install truthound[elasticsearch]

# All NoSQL
pip install truthound[nosql]
```

### Performance Optimization

```bash
pip install truthound[perf]
```

Includes xxhash for faster cache fingerprint generation (~10x faster).

### Dashboard (Web UI)

For the web-based dashboard and API server, install the separate `truthound-dashboard` package:

```bash
pip install truthound-dashboard
```

Then start the server:

```bash
truthound serve
```

See the [truthound-dashboard repository](https://github.com/seadonggyun4/truthound-dashboard) for more details.

### Full Installation

Install all optional dependencies:

```bash
pip install truthound[all]
```

### Development

For development with testing tools:

```bash
pip install truthound[dev]
```

## Verification

Verify your installation:

```bash
truthound --version
```

## Troubleshooting

### Common Issues

#### `ModuleNotFoundError: No module named 'truthound'`

Ensure you've installed Truthound:
```bash
pip install truthound
```

#### Permission Errors

Use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install truthound
```

#### Polars Installation Issues

Polars requires a Rust-compatible system. On older systems:
```bash
pip install polars-lts-cpu
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [First Validation Tutorial](first-validation.md)
