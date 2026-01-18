<div align="center">
  <img src="assets/truthound_banner.png" alt="Truthound Banner" style="width: 100%; max-width: 1200px;" />
</div>

# Truthound

<div align="center">
<h2>Zero-Configuration Data Quality Framework Powered by Polars</h2>
<p><strong>Enterprise-grade data validation with zero setup</strong></p>
</div>

---

## What is Truthound?

Truthound is a high-performance data quality framework built on [Polars](https://pola.rs/). It provides comprehensive data validation, profiling, and monitoring capabilities with a focus on ease of use and performance.

### Key Features

- **Zero Configuration**: Start validating data immediately without complex setup
- **289 Validators**: Coverage across 28 categories for data quality needs
- **High Performance**: Polars-native implementation for efficient validation
- **Schema Inference**: Automatically learn schemas from your data
- **PII Detection**: Built-in scanning for personally identifiable information
- **CI/CD Integration**: Seamlessly integrate with your deployment pipeline
- **Extensible**: Create custom validators with the SDK

## Quick Start

```bash
# Install
pip install truthound

# Learn schema from data
truthound learn data.csv

# Validate data
truthound check data.csv

# Scan for PII
truthound scan data.csv
```

## Python API

```python
import truthound as th

# Check data quality
report = th.check("data.csv")
print(report)

# Profile data
profile = th.profile("data.csv")
print(profile)

# Learn schema
schema = th.learn("data.csv")
schema.save("schema.yaml")
```

## Documentation Sections

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Installation, quick start guide, and your first validation

    [:octicons-arrow-right-24: Get started](getting-started/index.md)

-   :material-book-open-variant: **User Guide**

    ---

    Comprehensive guide to CLI commands, validators, and configuration

    [:octicons-arrow-right-24: Read the guide](user-guide/index.md)

-   :material-api: **API Reference**

    ---

    Complete API documentation with examples

    [:octicons-arrow-right-24: Browse the API](api-reference/index.md)

-   :material-school: **Tutorials**

    ---

    Step-by-step tutorials for common use cases

    [:octicons-arrow-right-24: Learn more](tutorials/index.md)

</div>

## Why Truthound?

### Performance

Built on Polars, Truthound leverages:

- Lazy evaluation for query optimization
- Columnar memory layout for cache efficiency
- SIMD vectorized operations
- Multi-threaded execution

Actual performance depends on hardware, data characteristics, and Polars version. Run your own benchmarks for accurate measurements.

### Comprehensive Validation

Validators across multiple categories including:

- Schema validation
- Format validation (email, phone, URL, etc.)
- Statistical validation
- PII detection
- Data drift detection
- Anomaly detection

### Enterprise Ready

- Security: ReDoS protection, SQL injection prevention
- i18n: English and Korean language support
- Storage: S3, GCS, Azure Blob, Database backends
- CI/CD: 12 platform integrations
- Notifications: 9 providers (Slack, Teams, PagerDuty, etc.)

## License

Truthound is open source under the Apache License 2.0.

## Support

- [GitHub Issues](https://github.com/seadonggyun4/Truthound/issues)
- [Documentation](https://truthound.netlify.app/)
