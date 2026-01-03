# User Guide

Comprehensive guide to using Truthound for data quality validation.

## Sections

<div class="grid cards" markdown>

-   :material-console: **CLI Reference**

    ---

    Complete command-line interface reference

    [:octicons-arrow-right-24: CLI Reference](cli-reference.md)

-   :material-check-all: **Validators**

    ---

    Guide to 289 built-in validators across 28 categories

    [:octicons-arrow-right-24: Validators](validators.md)

-   :material-chart-box: **Profiling**

    ---

    Data profiling and rule generation

    [:octicons-arrow-right-24: Profiling](profiling.md)

-   :material-pipe: **CI/CD Integration**

    ---

    Integrate with your deployment pipeline

    [:octicons-arrow-right-24: CI/CD](ci-cd.md)

-   :material-cog: **Configuration**

    ---

    Configuration files and options

    [:octicons-arrow-right-24: Configuration](configuration.md)

</div>

## Quick Reference

### Common CLI Commands

```bash
# Core commands
truthound learn <file>       # Learn schema from data
truthound check <file>       # Validate data quality
truthound scan <file>        # Scan for PII
truthound mask <file>        # Mask sensitive data
truthound profile <file>     # Basic profiling

# Advanced profiling
truthound auto-profile <file>     # Comprehensive profiling
truthound quick-suite <file>      # Profile + generate rules
truthound generate-suite <file>   # Generate rules from profile

# Checkpoints (CI/CD)
truthound checkpoint init         # Create sample config
truthound checkpoint run <name>   # Run checkpoint
truthound checkpoint list         # List checkpoints

# Documentation
truthound docs generate <file>    # Generate HTML report
truthound dashboard               # Launch interactive dashboard
```

### Python API Overview

```python
import truthound as th

# Core functions
report = th.check("data.csv")      # Validate
profile = th.profile("data.csv")   # Profile
schema = th.learn("data.csv")      # Learn schema
pii = th.scan("data.csv")          # Scan PII
masked = th.mask("data.csv")       # Mask data
drift = th.compare(old, new)       # Detect drift
```
