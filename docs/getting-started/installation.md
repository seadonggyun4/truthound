# Installation

## Basic Installation

Install Truthound using pip:

```bash
pip install truthound
```

This installs the core package with all essential features.

## Optional Dependencies

Truthound provides optional extras for specific use cases:

### Reports (HTML/PDF)

```bash
pip install truthound[reports]
```

Enables HTML report generation with Jinja2.

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

# All cloud providers
pip install truthound[stores]
```

### Streaming

```bash
pip install truthound[kafka]
pip install truthound[streaming]
```

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

Expected output:
```
truthound version 1.0.0
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
