# truthound read

Read and preview data from files or database connections. Supports row/column selection, multiple output formats, and schema inspection.

## Synopsis

```bash
truthound read [FILE] [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | No | Path to the data file (CSV, JSON, Parquet, NDJSON) |

## Data Source Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--connection` | `--conn` | None | Database connection string |
| `--table` | | None | Database table name |
| `--query` | | None | SQL query (alternative to `--table`) |
| `--source-config` | `--sc` | None | Path to data source config file (JSON/YAML) |
| `--source-name` | | None | Custom label for the data source |

## Selection Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--head` | `-n` | None | Show only the first N rows |
| `--sample` | `-s` | None | Random sample of N rows |
| `--columns` | `-c` | None | Columns to include (comma-separated) |

## Output Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--format` | `-f` | `table` | Output format (table, csv, json, parquet, ndjson) |
| `--output` | `-o` | None | Output file path |

## Inspection Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--schema-only` | | `false` | Show only column names and types |
| `--count-only` | | `false` | Show only the row count |

## Examples

### Basic Reading

```bash
truthound read data.csv
truthound read data.parquet --head 20
truthound read data.csv --columns id,name,age
```

### Database Reading

```bash
truthound read --connection "postgresql://user:pass@host/db" --table users
truthound read --connection "sqlite:///data.db" --table orders --head 10
truthound read --source-config db.yaml --sample 1000
```

### Schema Inspection

```bash
truthound read data.csv --schema-only
truthound read --connection "postgresql://host/db" --table users --schema-only
```

### Format Conversion

```bash
truthound read data.csv --format json -o output.json
truthound read data.csv --format parquet -o output.parquet
truthound read data.csv --format csv --head 100
```

### Row Count

```bash
truthound read data.csv --count-only
```

## Related Commands

- [`check`](check.md) - Validate data quality
- [`profile`](profile.md) - Generate data profile
- [`learn`](learn.md) - Learn schema from data

## See Also

- [Python API: th.read()](../../python-api/core-functions.md#thread)
- [Data Source Options](../../guides/datasources/cli-datasource-guide.md)
