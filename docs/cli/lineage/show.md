# truthound lineage show

Display data lineage information. This command shows the relationships between data assets, including upstream sources and downstream consumers.

## Synopsis

```bash
truthound lineage show <lineage_file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `lineage_file` | Yes | Path to the lineage file (JSON) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--node` | `-n` | None | Focus on a specific node |
| `--direction` | `-d` | `both` | Traversal direction (upstream, downstream, both) |
| `--format` | `-f` | `console` | Output format (currently only `console` is implemented) |

!!! warning "Format Option Limitation"
    The `--format` option is defined but **only console output is currently implemented**.
    For JSON or DOT export, use `lineage visualize` command instead.

## Description

The `lineage show` command displays lineage information:

1. **Shows** node relationships and dependencies
2. **Filters** by specific node and direction

## Examples

### Basic Usage

```bash
truthound lineage show lineage.json
```

Output:
```
Lineage Graph Summary
========================================
Nodes: 8
Edges: 10

Root nodes (2):
  raw_data (source)
  external_api (source)

Leaf nodes (2):
  analytics_table (table)
  data_warehouse (table)
```

### Focus on Specific Node

```bash
truthound lineage show lineage.json --node analytics_table
```

Output:
```
Lineage for: analytics_table
Type: table

Upstream (2 nodes):
  <- aggregated_data (transformation)
  <- api_processed (transformation)

Downstream (0 nodes):
```

### Upstream Only

Show only where data comes from:

```bash
truthound lineage show lineage.json --node analytics_table --direction upstream
```

Output:
```
Lineage for: analytics_table
Type: table

Upstream (2 nodes):
  <- aggregated_data (transformation)
  <- api_processed (transformation)
```

### Downstream Only

Show where data goes:

```bash
truthound lineage show lineage.json --node raw_data --direction downstream
```

Output:
```
Lineage for: raw_data
Type: source

Downstream (1 nodes):
  -> cleaned_data (transformation)
```

### Visual Export (Alternative)

For JSON, DOT (Graphviz), or other export formats, use the `lineage visualize` command:

```bash
# Generate interactive HTML visualization
truthound lineage visualize lineage.json -o graph.html

# Generate Graphviz DOT file
truthound lineage visualize lineage.json -o graph.dot --renderer graphviz

# Generate Mermaid diagram
truthound lineage visualize lineage.json -o graph.md --renderer mermaid
```

See [`lineage visualize`](visualize.md) for more details.

## Direction Options

| Direction | Description |
|-----------|-------------|
| `upstream` | Show only data sources (where data comes from) |
| `downstream` | Show only data consumers (where data goes) |
| `both` | Show both upstream and downstream (default) |

## Use Cases

### 1. Data Discovery

```bash
# What feeds into my table?
truthound lineage show lineage.json --node my_table --direction upstream
```

### 2. Debugging Data Issues

```bash
# Trace data flow both ways
truthound lineage show lineage.json --node problematic_table --direction both
```

### 3. Dependency Documentation

For documentation export, use `lineage visualize`:

```bash
# Generate DOT file for documentation
truthound lineage visualize lineage.json -o docs/lineage.dot --renderer graphviz
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (node not found, invalid file, or other error) |

## Related Commands

- [`lineage impact`](impact.md) - Analyze change impact
- [`lineage visualize`](visualize.md) - Generate visualization

## See Also

- [Lineage Overview](index.md)
- [Advanced Features](../../concepts/advanced.md)
