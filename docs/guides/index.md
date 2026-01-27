# Python API Guides

This section provides comprehensive guides for using Truthound through the Python API. Each guide includes practical examples, common patterns, and best practices for production environments.

> **Looking for CLI documentation?** See [CLI Reference](../cli/index.md) for command-line usage.
>
> **Looking for API Reference?** See [Python API Reference](../python-api/index.md) for function signatures and parameters.

---

## Quick Start

```python
import truthound as th

# Read data from various sources
df = th.read("data.csv")                                     # File path
df = th.read({"a": [1, 2, 3], "b": ["x", "y", "z"]})         # Dict data
df = th.read("large_data.parquet", sample_size=10000)        # With sampling

# Basic validation
report = th.check("data.csv")
print(f"Found {len(report.issues)} issues")

# With specific validators
report = th.check(df, validators=["null", "duplicate", "range"])

# Schema-based validation
schema = th.learn("baseline.csv")
report = th.check("new_data.csv", schema=schema)

# Database validation
from truthound.datasources import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
report = th.check(source=source)

# Data drift detection (14 methods available)
drift = th.compare("baseline.csv", "current.csv", method="auto")        # Auto-select
drift = th.compare("baseline.csv", "current.csv", method="ks")          # Kolmogorov-Smirnov
drift = th.compare("baseline.csv", "current.csv", method="wasserstein") # Earth Mover's Distance
drift = th.compare("baseline.csv", "current.csv", method="anderson")    # Anderson-Darling
drift = th.compare("baseline.csv", "current.csv", method="hellinger")   # Hellinger distance
drift = th.compare("baseline.csv", "current.csv", method="mmd")         # Maximum Mean Discrepancy
```

---

## Guide Categories

### Core Functionality

| Guide | Description | Key Topics |
|-------|-------------|------------|
| [Validators](validators/index.md) | Data validation patterns | 289 validators, custom validators, error handling |
| [Data Sources](datasources/index.md) | Database and file connections | SQL, Cloud DW, Spark, streaming |
| [Profiling](profiler/index.md) | Automatic data analysis | Schema inference, rule generation, scheduling |

### Output and Reporting

| Guide | Description | Key Topics |
|-------|-------------|------------|
| [Data Docs](datadocs/index.md) | HTML report generation | Themes, charts, PDF export, templates |
| [Reporters](reporters/index.md) | Output formats | JSON, Console, JUnit, custom reporters |
| [Storage](stores/index.md) | Result persistence | S3, GCS, Azure, versioning, caching |

### Operations

| Guide | Description | Key Topics |
|-------|-------------|------------|
| [Configuration](configuration/index.md) | Environment setup | Logging, metrics, encryption, resilience |
| [CI/CD](checkpoint/index.md) | Pipeline integration | Checkpoints, notifications, routing |
| [Performance](advanced/performance.md) | Optimization | Parallel execution, pushdown, memory |

### Enterprise Features

| Guide | Description | Key Topics |
|-------|-------------|------------|
| [Advanced](advanced/index.md) | Enterprise capabilities | ML anomaly, lineage, realtime streaming |

---

## Common Workflows

### Workflow 1: Basic Data Validation

```python
import truthound as th

# 1. Validate data
report = th.check("data.csv")

# 2. Filter critical issues
critical = [i for i in report.issues if i.severity == "critical"]

# 3. Generate report
if critical:
    from truthound.datadocs import generate_html_report
    html = generate_html_report(report)
    Path("report.html").write_text(html)
```

### Workflow 2: Schema-Based Validation

```python
import truthound as th

# 1. Learn schema from baseline data
schema = th.learn("baseline.csv")
schema.save("schema.yaml")

# 2. Validate new data against schema
report = th.check("new_data.csv", schema="schema.yaml")

# 3. Check for schema violations
schema_issues = [i for i in report.issues if i.validator == "schema"]
```

### Workflow 3: Database Validation with Pushdown

```python
import truthound as th
from truthound.datasources import PostgreSQLDataSource

# 1. Connect to database
source = PostgreSQLDataSource(
    table="transactions",
    host="db.example.com",
    database="analytics",
    user="readonly",
)

# 2. Validate with query pushdown (runs on database server)
report = th.check(source=source, pushdown=True)

# 3. Save results
from truthound.stores import S3Store
store = S3Store(bucket="validation-results", prefix="daily/")
store.save(report, key=f"transactions_{date.today()}")
```

### Workflow 4: Profiling and Rule Generation

```python
import truthound as th

# 1. Profile data
profile = th.profile("data.csv")

# 2. Generate validation suite from profile
from truthound.profiler import generate_suite
suite = generate_suite(profile)

# 3. Execute suite on new data
report = suite.execute(new_data)
```

---

## Document Structure

Each guide follows a consistent structure:

1. **Overview** - Purpose and scope
2. **Quick Start** - Minimal working example
3. **Core Concepts** - Key classes and patterns
4. **Practical Examples** - Real-world use cases
5. **Configuration Options** - Available settings
6. **Best Practices** - Production recommendations
7. **Troubleshooting** - Common issues and solutions

---

## See Also

- [Getting Started](../getting-started/index.md) - Installation and first steps
- [Tutorials](../tutorials/index.md) - Step-by-step learning paths
- [Python API Reference](../python-api/index.md) - Complete API documentation
- [CLI Reference](../cli/index.md) - Command-line interface
