# CLI Reference

Complete reference for the Truthound command-line interface.

## Installation

```bash
pip install truthound
```

After installation, the `truthound` command is available globally.

!!! note "CLI vs Python API"
    The CLI only supports file-based inputs. For SQL databases (PostgreSQL, MySQL, SQLite), Spark DataFrames, or Cloud Data Warehouses (BigQuery, Snowflake, Redshift, Databricks), use the [Python API](../python-api/index.md) with the `source=` parameter.

## Supported Input Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| CSV | `.csv` | Comma-separated values |
| JSON | `.json` | Standard JSON array |
| Parquet | `.parquet` | Columnar storage format |
| NDJSON | `.ndjson` | Newline-delimited JSON |
| JSONL | `.jsonl` | JSON Lines (same as NDJSON) |

## Quick Reference

### Core Commands Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `learn` | FILE (required) | `--output, -o` (schema.yaml), `--no-constraints` |
| `check` | FILE (required) | `--validators, -v`, `--min-severity, -s` (low/medium/high/critical), `--schema`, `--auto-schema`, `--format, -f` (console/json/html), `--output, -o`, `--strict` |
| `scan` | FILE (required) | `--format, -f` (console/json/html), `--output, -o` |
| `mask` | FILE (required) | `--output, -o` (required), `--columns, -c`, `--strategy, -s` (redact/hash/fake), `--strict` |
| `profile` | FILE (required) | `--format, -f` (console/json), `--output, -o` |
| `compare` | BASELINE CURRENT (required) | `--columns, -c`, `--method, -m` (auto/ks/psi/chi2/js), `--threshold, -t`, `--format, -f` (console/json), `--output, -o`, `--strict` |

### Profiler Commands Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `auto-profile` | FILE (required) | `--output, -o`, `--format, -f` (console/json/yaml), `--patterns/--no-patterns`, `--correlations/--no-correlations`, `--sample, -s`, `--top-n` (10) |
| `generate-suite` | PROFILE_FILE (required) | `--output, -o`, `--format, -f` (yaml/json/python/toml/checkpoint), `--strictness, -s` (loose/medium/strict), `--include, -i`, `--exclude, -e`, `--min-confidence` (low/medium/high), `--name, -n`, `--preset, -p`, `--config, -c`, `--group-by-category`, `--code-style` (functional/class_based/declarative) |
| `quick-suite` | FILE (required) | `--output, -o`, `--format, -f` (yaml/json/python/toml/checkpoint), `--strictness, -s` (loose/medium/strict), `--include, -i`, `--exclude, -e`, `--min-confidence`, `--name, -n`, `--preset, -p`, `--sample-size` |
| `list-formats` | - | - |
| `list-presets` | - | - |
| `list-categories` | - | - |

### Checkpoint Commands Summary

!!! warning "checkpoint vs realtime checkpoint"
    `truthound checkpoint` is for **CI/CD pipelines** (YAML configuration file based).

    For **streaming validation** state management, use [`truthound realtime checkpoint`](#realtime-commands-summary).

| Command | Arguments | Options |
|---------|-----------|---------|
| `checkpoint run` | NAME (required) | `--config, -c` (truthound.yaml), `--data, -d`, `--validators, -v`, `--output, -o`, `--format, -f` (console/json), `--strict`, `--store`, `--slack`, `--webhook`, `--github-summary` |
| `checkpoint list` | - | `--config, -c`, `--format, -f` (console/json) |
| `checkpoint validate` | CONFIG_FILE (required) | `--strict, -s` |
| `checkpoint init` | - | `--output, -o` (truthound.yaml), `--format, -f` (yaml/json) |

### ML Commands Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `ml anomaly` | FILE (required) | `--method, -m` (zscore/iqr/mad/isolation_forest), `--contamination, -c` (0.1), `--columns`, `--output, -o`, `--format, -f` (console/json) |
| `ml drift` | BASELINE CURRENT (required) | `--method, -m` (distribution/feature/multivariate), `--threshold, -t` (0.1), `--columns`, `--output, -o` |
| `ml learn-rules` | FILE (required) | `--output, -o` (learned_rules.json), `--strictness, -s` (loose/medium/strict), `--min-confidence` (0.9), `--max-rules` (100) |

### Docs Commands Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `docs generate` | PROFILE_FILE (required) | `--output, -o`, `--title, -t` ("Data Profile Report"), `--subtitle, -s`, `--theme` (light/dark/professional/minimal/modern), `--format, -f` (html/pdf) |
| `docs themes` | - | - |

### Dashboard Command Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `dashboard` | - | `--profile, -p`, `--port` (8080), `--host` (localhost), `--title, -t` ("Truthound Dashboard"), `--debug` |

### Realtime Commands Summary

!!! warning "realtime checkpoint vs checkpoint"
    `truthound realtime checkpoint` is for **streaming validation** state management (`--dir` option based).

    For **CI/CD pipelines**, use [`truthound checkpoint`](#checkpoint-commands-summary).

| Command | Arguments | Options |
|---------|-----------|---------|
| `realtime validate` | SOURCE (required) | `--validators, -v`, `--batch-size, -b` (1000), `--max-batches` (10, 0=unlimited), `--output, -o` |
| `realtime monitor` | SOURCE (required) | `--interval, -i` (5), `--duration, -d` (60, 0=unlimited) |
| `realtime checkpoint list` | - | `--dir, -d` (./checkpoints), `--format, -f` (console/json) |
| `realtime checkpoint show` | CHECKPOINT_ID (required) | `--dir, -d` |
| `realtime checkpoint delete` | CHECKPOINT_ID (required) | `--dir, -d`, `--force, -f` |

### Benchmark Commands Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `benchmark run` | BENCHMARK (optional) | `--suite, -s` (quick/ci/full/profiling/validation), `--size` (tiny/small/medium/large/xlarge), `--rows, -r`, `--iterations, -i` (5), `--warmup, -w` (2), `--output, -o`, `--format, -f` (console/json/html), `--save-baseline`, `--compare-baseline`, `--verbose, -v` |
| `benchmark list` | - | `--format, -f` (console/json) |
| `benchmark compare` | BASELINE CURRENT (required) | `--threshold, -t` (10.0%), `--format, -f` (console/json) |

### Scaffolding Commands Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `new validator` | NAME (required) | `--output, -o` (.), `--template, -t` (basic/column/pattern/range/comparison/composite/full), `--author, -a`, `--description, -d`, `--category, -c` (custom), `--tests/--no-tests` (--tests), `--docs/--no-docs` (--no-docs), `--severity, -s` (MEDIUM), `--pattern`, `--min`, `--max` |
| `new reporter` | NAME (required) | `--output, -o` (.), `--template, -t` (basic/full), `--author, -a`, `--description, -d`, `--tests/--no-tests` (--tests), `--docs/--no-docs` (--no-docs), `--extension, -e` (.txt), `--content-type` (text/plain) |
| `new plugin` | NAME (required) | `--output, -o` (.), `--type, -t` (validator/reporter/hook/datasource/action/full), `--author, -a`, `--description, -d`, `--tests/--no-tests` (--tests), `--min-version` (0.1.0), `--python` (3.10) |
| `new list` | - | `--verbose, -v` |
| `new templates` | SCAFFOLD_TYPE (required) | - |

### Plugin Commands Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `plugin list` | - | `--type, -t` (validator/reporter/hook/datasource/action/custom), `--state, -s` (discovered/loading/loaded/active/inactive/error/unloading), `--verbose, -v`, `--json` |
| `plugin info` | NAME (required) | `--json` |
| `plugin load` | NAME (required) | `--activate/--no-activate` (--activate) |
| `plugin unload` | NAME (required) | - |
| `plugin enable` | NAME (required) | - |
| `plugin disable` | NAME (required) | - |
| `plugin create` | NAME (required) | `--output, -o` (.), `--type, -t` (validator/reporter/hook/custom), `--author` |

## Global Options

These options are available for all commands:

| Option | Description |
|--------|-------------|
| `--help` | Show help message and exit |
| `--version` | Show version and exit |

## Command Groups

### [Core Commands](core/index.md)

Essential data quality operations:

| Command | Description |
|---------|-------------|
| [`learn`](core/learn.md) | Learn schema from data |
| [`check`](core/check.md) | Validate data quality |
| [`scan`](core/scan.md) | Scan for PII |
| [`mask`](core/mask.md) | Mask sensitive data |
| [`profile`](core/profile.md) | Generate data profile |
| [`compare`](core/compare.md) | Detect data drift |

### [Profiler Commands](profiler/index.md)

Advanced profiling and rule generation:

| Command | Description |
|---------|-------------|
| [`auto-profile`](profiler/auto-profile.md) | Profile with auto-detection |
| [`generate-suite`](profiler/generate-suite.md) | Generate validation rules from profile |
| [`quick-suite`](profiler/quick-suite.md) | Profile and generate rules in one step |
| `list-formats` | List supported output formats |
| `list-presets` | List available presets |
| `list-categories` | List rule categories |

### [Checkpoint Commands](checkpoint/index.md)

CI/CD pipeline integration:

| Command | Description |
|---------|-------------|
| [`checkpoint run`](checkpoint/run.md) | Run validation pipeline |
| [`checkpoint list`](checkpoint/list.md) | List available checkpoints |
| [`checkpoint validate`](checkpoint/validate.md) | Validate configuration |
| [`checkpoint init`](checkpoint/init.md) | Initialize sample config |

### [ML Commands](ml/index.md)

Machine learning-based detection:

| Command | Description |
|---------|-------------|
| [`ml anomaly`](ml/anomaly.md) | Detect anomalies |
| [`ml drift`](ml/drift.md) | Detect data drift |
| [`ml learn-rules`](ml/learn-rules.md) | Learn validation rules |

### Lineage Commands Summary

| Command | Arguments | Options |
|---------|-----------|---------|
| `lineage show` | LINEAGE_FILE (required) | `--node, -n`, `--direction, -d` (upstream/downstream/both), `--format, -f` (console/json/dot) |
| `lineage impact` | LINEAGE_FILE NODE (required) | `--max-depth` (-1), `--output, -o` |
| `lineage visualize` | LINEAGE_FILE (required) | `--output, -o` (required), `--renderer, -r` (d3/cytoscape/graphviz/mermaid), `--theme, -t` (light/dark), `--focus, -f` |

### [Lineage Commands](lineage/index.md)

Data lineage tracking:

| Command | Description |
|---------|-------------|
| [`lineage show`](lineage/show.md) | Display lineage information |
| [`lineage impact`](lineage/impact.md) | Analyze change impact |
| [`lineage visualize`](lineage/visualize.md) | Generate lineage visualization |

### [Docs Commands](docs/index.md)

Documentation generation:

| Command | Description |
|---------|-------------|
| [`docs generate`](docs/generate.md) | Generate HTML/PDF report |
| [`docs themes`](docs/themes.md) | List available themes |

### [Dashboard Command](dashboard.md)

Interactive data exploration:

| Command | Description |
|---------|-------------|
| [`dashboard`](dashboard.md) | Launch interactive dashboard |

### [Realtime Commands](realtime/index.md)

Streaming validation:

| Command | Description |
|---------|-------------|
| [`realtime validate`](realtime/validate.md) | Validate streaming data |
| [`realtime monitor`](realtime/monitor.md) | Monitor validation metrics |
| [`realtime checkpoint`](realtime/checkpoint/index.md) | Manage validation checkpoints |

### [Benchmark Commands](benchmark/index.md)

Performance testing:

| Command | Description |
|---------|-------------|
| [`benchmark run`](benchmark/run.md) | Run performance benchmarks |
| [`benchmark list`](benchmark/list.md) | List available benchmarks |
| [`benchmark compare`](benchmark/compare.md) | Compare benchmark results |

### [Scaffolding Commands](scaffolding/index.md)

Code generation:

| Command | Description |
|---------|-------------|
| [`new validator`](scaffolding/new-validator.md) | Create custom validator |
| [`new reporter`](scaffolding/new-reporter.md) | Create custom reporter |
| [`new plugin`](scaffolding/new-plugin.md) | Create plugin package |
| [`new list`](scaffolding/new-list.md) | List scaffold types |
| [`new templates`](scaffolding/new-templates.md) | List available templates |

### [Plugin Commands](plugin/index.md)

Plugin management:

| Command | Description |
|---------|-------------|
| [`plugin list`](plugin/list.md) | List discovered plugins |
| [`plugin info`](plugin/info.md) | Show plugin details |
| [`plugin load`](plugin/load.md) | Load a plugin |
| [`plugin unload`](plugin/unload.md) | Unload a plugin |
| [`plugin enable`](plugin/enable.md) | Enable a plugin |
| [`plugin disable`](plugin/disable.md) | Disable a plugin |
| [`plugin create`](plugin/create.md) | Create plugin template |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error or validation failed (with `--strict`) |
| 2 | Usage error (invalid arguments) |

## Quick Examples

```bash
# Learn schema from data
truthound learn data.csv -o schema.yaml

# Validate data quality
truthound check data.csv --strict

# Scan for PII
truthound scan customers.csv

# Mask sensitive data
truthound mask data.csv -o masked.csv --strategy hash

# Generate data profile
truthound profile data.csv --format json -o profile.json

# Compare datasets for drift
truthound compare baseline.csv current.csv --method psi
```
