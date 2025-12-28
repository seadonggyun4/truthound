# CLI Reference

Complete reference for the Truthound command-line interface.

## Global Options

These options are available for all commands:

| Option | Description |
|--------|-------------|
| `--help` | Show help message and exit |
| `--version` | Show version and exit |

## Core Commands

### `truthound learn`

Learn schema from a data file.

```bash
truthound learn <file> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `file` | Path to the data file (CSV, Parquet, JSON) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `schema.yaml` | Output schema file path |
| `--no-constraints` | `false` | Don't infer constraints from data |

**Examples:**

```bash
# Basic usage
truthound learn data.csv

# Custom output path
truthound learn data.parquet -o my_schema.yaml

# Without constraint inference
truthound learn data.csv --no-constraints
```

---

### `truthound check`

Validate data quality in a file.

```bash
truthound check <file> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `file` | Path to the data file |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-v, --validators` | All | Comma-separated list of validators |
| `-s, --min-severity` | None | Minimum severity (low, medium, high, critical) |
| `--schema` | None | Schema file for validation |
| `--auto-schema` | `false` | Auto-learn and cache schema |
| `-f, --format` | `console` | Output format (console, json, html) |
| `-o, --output` | None | Output file path |
| `--strict` | `false` | Exit with code 1 if issues found |

**Examples:**

```bash
# Basic validation
truthound check data.csv

# With specific validators
truthound check data.csv -v null,duplicate,range

# With schema validation
truthound check data.csv --schema schema.yaml

# Strict mode for CI/CD
truthound check data.csv --strict

# JSON output
truthound check data.csv --format json -o report.json
```

---

### `truthound scan`

Scan for personally identifiable information (PII).

```bash
truthound scan <file> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --format` | `console` | Output format (console, json) |
| `-o, --output` | None | Output file path |

**Examples:**

```bash
truthound scan customers.csv
truthound scan data.parquet --format json -o pii_report.json
```

---

### `truthound mask`

Mask sensitive data in a file.

```bash
truthound mask <file> -o <output> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | Required | Output file path |
| `-c, --columns` | Auto-detect | Columns to mask |
| `-s, --strategy` | `redact` | Masking strategy (redact, hash, fake) |

**Strategies:**

| Strategy | Description |
|----------|-------------|
| `redact` | Replace with asterisks (`****`) |
| `hash` | Replace with hashed values |
| `fake` | Replace with realistic fake data |

**Examples:**

```bash
# Mask with auto-detection
truthound mask data.csv -o masked.csv

# Mask specific columns
truthound mask data.csv -o masked.csv -c email,phone,ssn

# Use hashing strategy
truthound mask data.csv -o masked.csv --strategy hash
```

---

### `truthound profile`

Generate a statistical profile of the data.

```bash
truthound profile <file> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --format` | `console` | Output format (console, json) |
| `-o, --output` | None | Output file path |

---

### `truthound compare`

Compare two datasets and detect data drift.

```bash
truthound compare <baseline> <current> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `baseline` | Baseline (reference) data file |
| `current` | Current data file to compare |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --columns` | All | Columns to compare |
| `-m, --method` | `auto` | Detection method (auto, ks, psi, chi2, js) |
| `-t, --threshold` | Auto | Custom drift threshold |
| `-f, --format` | `console` | Output format |
| `-o, --output` | None | Output file path |
| `--strict` | `false` | Exit with code 1 if drift detected |

**Detection Methods:**

| Method | Use Case |
|--------|----------|
| `auto` | Automatically select best method |
| `ks` | Kolmogorov-Smirnov (numeric data) |
| `psi` | Population Stability Index |
| `chi2` | Chi-squared (categorical data) |
| `js` | Jensen-Shannon divergence |

---

## Auto-Profiling Commands

### `truthound auto-profile`

Profile data with auto-detection of types and patterns.

```bash
truthound auto-profile <file> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | None | Output file path for profile JSON |
| `-f, --format` | `console` | Output format (console, json, yaml) |
| `--patterns/--no-patterns` | `true` | Include pattern detection |
| `--correlations/--no-correlations` | `false` | Include correlation analysis |
| `-s, --sample` | All | Sample size for profiling |
| `--top-n` | `10` | Number of top/bottom values |

---

### `truthound generate-suite`

Generate validation rules from a profile.

```bash
truthound generate-suite <profile_file> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | None | Output file path |
| `-f, --format` | `yaml` | Output format (yaml, json, python, toml, checkpoint) |
| `-s, --strictness` | `medium` | Rule strictness (loose, medium, strict) |
| `-i, --include` | All | Include only these categories |
| `-e, --exclude` | None | Exclude these categories |
| `--min-confidence` | None | Minimum confidence (low, medium, high) |
| `-n, --name` | None | Suite name |
| `-p, --preset` | None | Configuration preset |

---

### `truthound quick-suite`

Profile data and generate rules in one step.

```bash
truthound quick-suite <file> [OPTIONS]
```

Same options as `auto-profile` + `generate-suite` combined.

---

## Checkpoint Commands

### `truthound checkpoint run`

Run a checkpoint validation pipeline.

```bash
truthound checkpoint run <name> [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-c, --config` | Configuration file (YAML/JSON) |
| `-d, --data` | Override data source path |
| `-v, --validators` | Override validators |
| `-o, --output` | Output file for results |
| `--strict` | Exit with code 1 if issues found |
| `--slack` | Slack webhook URL |
| `--webhook` | Webhook URL |
| `--github-summary` | Write GitHub Actions summary |

---

### `truthound checkpoint list`

List available checkpoints.

```bash
truthound checkpoint list [OPTIONS]
```

---

### `truthound checkpoint validate`

Validate a checkpoint configuration file.

```bash
truthound checkpoint validate <config_file>
```

---

### `truthound checkpoint init`

Initialize a sample checkpoint configuration.

```bash
truthound checkpoint init [OPTIONS]
```

---

## Documentation Commands

### `truthound docs generate`

Generate HTML report from profile data.

```bash
truthound docs generate <profile_file> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | Auto | Output file path |
| `-t, --title` | "Data Profile Report" | Report title |
| `--theme` | `professional` | Theme (light, dark, professional, minimal, modern) |
| `--charts` | `apexcharts` | Chart library |
| `-f, --format` | `html` | Output format (html, pdf) |

---

### `truthound docs themes`

List available report themes.

---

### `truthound dashboard`

Launch interactive dashboard.

```bash
truthound dashboard [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --profile` | None | Profile JSON file |
| `--port` | `8080` | Server port |
| `--host` | `localhost` | Server host |
| `--debug` | `false` | Enable debug mode |

---

## ML Commands

### `truthound ml anomaly`

Detect anomalies using ML methods.

```bash
truthound ml anomaly <file> [OPTIONS]
```

---

### `truthound ml drift`

Detect data drift between datasets.

```bash
truthound ml drift <baseline> <current> [OPTIONS]
```

---

### `truthound ml learn-rules`

Learn validation rules from data.

```bash
truthound ml learn-rules <file> [OPTIONS]
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error or validation failed (with `--strict`) |
| 2 | Usage error (invalid arguments) |
