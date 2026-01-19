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
| `-o, --output` | None | Output file path (required for html format) |
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

# HTML report (requires jinja2)
truthound check data.csv --format html -o report.html
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
| `-f, --format` | `console` | Output format (console, json, html) |
| `-o, --output` | None | Output file path (required for html format) |

**Examples:**

```bash
truthound scan customers.csv
truthound scan data.parquet --format json -o pii_report.json
truthound scan data.csv --format html -o pii_report.html
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

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --method` | `zscore` | Detection method (zscore, iqr, mad, isolation_forest) |
| `-c, --contamination` | `0.1` | Expected proportion of outliers (0.0 to 0.5) |
| `--columns` | All | Comma-separated columns to analyze |
| `-o, --output` | None | Output file path for results |
| `-f, --format` | `console` | Output format (console, json) |

**Examples:**

```bash
truthound ml anomaly data.csv
truthound ml anomaly data.csv --method isolation_forest --contamination 0.05
truthound ml anomaly data.csv --method iqr --columns "amount,price"
truthound ml anomaly data.csv --format json -o anomalies.json
```

---

### `truthound ml drift`

Detect data drift between datasets.

```bash
truthound ml drift <baseline> <current> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --method` | `feature` | Detection method (distribution, feature, multivariate) |
| `-t, --threshold` | `0.1` | Drift detection threshold |
| `--columns` | All | Comma-separated columns to analyze |
| `-o, --output` | None | Output file path |

**Examples:**

```bash
truthound ml drift baseline.csv current.csv
truthound ml drift ref.parquet new.parquet --method multivariate
truthound ml drift old.csv new.csv --threshold 0.2 --output drift_report.json
```

---

### `truthound ml learn-rules`

Learn validation rules from data.

```bash
truthound ml learn-rules <file> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `learned_rules.json` | Output file for learned rules |
| `-s, --strictness` | `medium` | Rule strictness (loose, medium, strict) |
| `--min-confidence` | `0.9` | Minimum rule confidence |
| `--max-rules` | `100` | Maximum number of rules to generate |

**Examples:**

```bash
truthound ml learn-rules data.csv
truthound ml learn-rules data.csv --strictness strict --min-confidence 0.95
truthound ml learn-rules data.parquet --output my_rules.json
```

---

## Lineage Commands

### `truthound lineage show`

Display lineage information from a lineage JSON file.

```bash
truthound lineage show <lineage_file> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --node` | None | Show lineage for specific node |
| `-d, --direction` | `both` | Direction (upstream, downstream, both) |
| `-f, --format` | `console` | Output format (console, json, dot) |

**Examples:**

```bash
truthound lineage show lineage.json
truthound lineage show lineage.json --node my_table --direction upstream
truthound lineage show lineage.json --format dot > lineage.dot
```

---

### `truthound lineage impact`

Analyze impact of changes to a data asset.

```bash
truthound lineage impact <lineage_file> <node> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--max-depth` | `-1` | Maximum depth for impact analysis (-1=unlimited) |
| `-o, --output` | None | Output file for results |

**Examples:**

```bash
truthound lineage impact lineage.json raw_data
truthound lineage impact lineage.json my_table --max-depth 3 --output impact.json
```

---

### `truthound lineage visualize`

Generate visual representation of lineage graph.

```bash
truthound lineage visualize <lineage_file> -o <output> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | Required | Output file path |
| `-r, --renderer` | `d3` | Renderer (d3, cytoscape, graphviz, mermaid) |
| `-t, --theme` | `light` | Theme (light, dark) |
| `-f, --focus` | None | Focus on specific node |

**Examples:**

```bash
truthound lineage visualize lineage.json -o graph.html
truthound lineage visualize lineage.json -o graph.html --renderer cytoscape --theme dark
truthound lineage visualize lineage.json -o graph.svg --renderer graphviz
```

---

## Realtime Commands

### `truthound realtime validate`

Validate streaming data in real-time.

```bash
truthound realtime validate <source> [OPTIONS]
```

**Source Formats:**

| Format | Description |
|--------|-------------|
| `mock` | Mock data source for testing |
| `kafka:topic_name` | Kafka topic (requires aiokafka) |
| `kinesis:stream_name` | Kinesis stream (requires aiobotocore) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-v, --validators` | All | Comma-separated validators |
| `-b, --batch-size` | `1000` | Batch size |
| `--max-batches` | `10` | Maximum batches to process (0=unlimited) |
| `-o, --output` | None | Output file for results |

**Examples:**

```bash
truthound realtime validate mock --max-batches 5
truthound realtime validate mock --validators null,range --batch-size 500
truthound realtime validate kafka:my_topic --max-batches 100
```

---

### `truthound realtime monitor`

Monitor streaming validation metrics.

```bash
truthound realtime monitor <source> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --interval` | `5` | Monitoring interval in seconds |
| `-d, --duration` | `60` | Total monitoring duration (0=indefinite) |

**Examples:**

```bash
truthound realtime monitor mock --interval 5 --duration 60
truthound realtime monitor kafka:my_topic --interval 10
```

---

### `truthound realtime checkpoint list`

List available streaming validation checkpoints.

```bash
truthound realtime checkpoint list [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --dir` | `./checkpoints` | Checkpoint directory |
| `-f, --format` | `console` | Output format (console, json) |

**Examples:**

```bash
truthound realtime checkpoint list
truthound realtime checkpoint list --dir ./my_checkpoints
truthound realtime checkpoint list --format json
```

---

### `truthound realtime checkpoint show`

Show details of a specific checkpoint.

```bash
truthound realtime checkpoint show <checkpoint_id> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --dir` | `./checkpoints` | Checkpoint directory |

**Examples:**

```bash
truthound realtime checkpoint show abc12345
truthound realtime checkpoint show abc12345 --dir ./my_checkpoints
```

---

### `truthound realtime checkpoint delete`

Delete a checkpoint.

```bash
truthound realtime checkpoint delete <checkpoint_id> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --dir` | `./checkpoints` | Checkpoint directory |
| `-f, --force` | `false` | Skip confirmation |

**Examples:**

```bash
truthound realtime checkpoint delete abc12345
truthound realtime checkpoint delete abc12345 --force
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error or validation failed (with `--strict`) |
| 2 | Usage error (invalid arguments) |
