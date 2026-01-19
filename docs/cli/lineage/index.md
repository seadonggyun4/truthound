# Lineage Commands

Data lineage tracking and visualization commands.

## Overview

| Command | Description | Primary Use Case |
|---------|-------------|------------------|
| [`show`](show.md) | Display lineage information | Explore dependencies |
| [`impact`](impact.md) | Analyze change impact | Change management |
| [`visualize`](visualize.md) | Generate lineage visualization | Documentation |

## What is Data Lineage?

Data lineage tracks the flow of data through your systems:

- **Upstream**: Where data comes from (sources)
- **Downstream**: Where data goes (consumers)
- **Transformations**: How data is modified

## Lineage File Format

Lineage information is stored in JSON format:

```json
{
  "version": "1.0",
  "nodes": [
    {
      "id": "raw_data",
      "type": "source",
      "name": "Raw Data",
      "metadata": {
        "path": "s3://bucket/raw/",
        "format": "parquet"
      }
    },
    {
      "id": "cleaned_data",
      "type": "transformation",
      "name": "Cleaned Data",
      "metadata": {
        "path": "s3://bucket/cleaned/",
        "transformation": "data_cleaning_job"
      }
    },
    {
      "id": "analytics_table",
      "type": "sink",
      "name": "Analytics Table",
      "metadata": {
        "database": "analytics",
        "table": "user_metrics"
      }
    }
  ],
  "edges": [
    {
      "source": "raw_data",
      "target": "cleaned_data"
    },
    {
      "source": "cleaned_data",
      "target": "analytics_table"
    }
  ]
}
```

## Workflow

```mermaid
graph LR
    A[Lineage File] --> B{Task?}
    B -->|Explore| C[lineage show]
    B -->|Impact| D[lineage impact]
    B -->|Document| E[lineage visualize]
    C --> F[Console/JSON/DOT]
    D --> G[Impact Report]
    E --> H[HTML/SVG/Mermaid]
```

## Quick Examples

### Show Lineage

```bash
# Show all lineage
truthound lineage show lineage.json

# Show upstream of a specific node
truthound lineage show lineage.json --node analytics_table --direction upstream

# Export as DOT format
truthound lineage show lineage.json --format dot > lineage.dot
```

### Analyze Impact

```bash
# Analyze impact of changing raw_data
truthound lineage impact lineage.json raw_data

# Limit analysis depth
truthound lineage impact lineage.json raw_data --max-depth 2
```

### Visualize

```bash
# Generate interactive D3 visualization
truthound lineage visualize lineage.json -o graph.html

# Generate Graphviz diagram
truthound lineage visualize lineage.json -o graph.svg --renderer graphviz

# Generate Mermaid diagram
truthound lineage visualize lineage.json -o graph.md --renderer mermaid
```

## Visualization Renderers

| Renderer | Output | Features |
|----------|--------|----------|
| `d3` | HTML | Interactive, pan/zoom, tooltips |
| `cytoscape` | HTML | Interactive, layouts, search |
| `graphviz` | SVG/PNG | Static, publication quality |
| `mermaid` | Markdown | Embeddable in docs |

## Node Types

| Type | Description | Icon |
|------|-------------|------|
| `source` | Data source (files, APIs) | 📥 |
| `transformation` | Data transformation | ⚙️ |
| `sink` | Data destination | 📤 |
| `model` | ML model | 🤖 |
| `report` | Report/Dashboard | 📊 |

## Use Cases

### 1. Data Discovery

```bash
# Explore what feeds into a table
truthound lineage show lineage.json --node my_table --direction upstream
```

### 2. Change Impact Analysis

```bash
# Before changing a source, check what's affected
truthound lineage impact lineage.json source_table -o impact_report.json
```

### 3. Documentation

```bash
# Generate documentation for data catalog
truthound lineage visualize lineage.json -o docs/lineage.html --renderer d3 --theme light
```

### 4. Debugging

```bash
# Trace data flow to find issues
truthound lineage show lineage.json --node failed_table --direction both
```

## Integration with OpenLineage

Truthound supports the OpenLineage standard for interoperability:

```bash
# Import from OpenLineage events
truthound lineage import openlineage_events.json -o lineage.json

# Export to OpenLineage format
truthound lineage export lineage.json --format openlineage -o events.json
```

## Next Steps

- [show](show.md) - Display lineage information
- [impact](impact.md) - Analyze change impact
- [visualize](visualize.md) - Generate visualization

## See Also

- [Advanced Features](../../concepts/advanced.md)
- [OpenLineage Integration](../../concepts/openlineage.md)
