# truthound lineage impact

Analyze the impact of changes to a data asset. This command identifies all downstream dependencies that would be affected by modifying a specific node.

## Synopsis

```bash
truthound lineage impact <lineage_file> <node> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `lineage_file` | Yes | Path to the lineage file (JSON) |
| `node` | Yes | Node ID to analyze impact for |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--max-depth` | | `-1` | Maximum analysis depth (-1 for unlimited) |
| `--output` | `-o` | None | Output file path |

## Description

The `lineage impact` command performs impact analysis:

1. **Identifies** all downstream dependencies of a node
2. **Calculates** impact levels based on distance
3. **Reports** affected assets with details
4. **Helps** plan changes safely

## Examples

### Basic Impact Analysis

```bash
truthound lineage impact lineage.json raw_data
```

Output:
```
Impact Analysis: raw_data
=========================

Change to 'raw_data' affects 5 downstream nodes:

Impact Analysis for: Raw Data
Total affected nodes: 5
Maximum depth: 3
  critical: 1
  high: 1
  medium: 3

Affected nodes:
  [!!] cleaned_data (depth=1)
  [!] aggregated_data (depth=2)
  [!!] data_warehouse (depth=2)
  [!] analytics_table (depth=3)
  [!!!] dashboard_model (depth=3)
```

### Limited Depth Analysis

Analyze only immediate and second-level impacts:

```bash
truthound lineage impact lineage.json raw_data --max-depth 2
```

Output:
```
Impact Analysis for: Raw Data
Total affected nodes: 3
Maximum depth: 2
  high: 1
  medium: 2

Affected nodes:
  [!!] cleaned_data (depth=1)
  [!] aggregated_data (depth=2)
  [!!] data_warehouse (depth=2)
```

### Save to File

```bash
truthound lineage impact lineage.json raw_data -o impact_report.json
```

Output file (`impact_report.json`):
```json
{
  "source_node": "raw_data",
  "analysis_timestamp": "2024-01-15T10:30:00Z",
  "max_depth": -1,
  "summary": {
    "total_affected": 5,
    "critical_nodes": 3,
    "max_impact_level": 3
  },
  "impact_levels": {
    "1": [
      {
        "id": "cleaned_data",
        "type": "transformation",
        "name": "Cleaned Data",
        "critical": false,
        "path": ["raw_data", "cleaned_data"]
      }
    ],
    "2": [
      {
        "id": "aggregated_data",
        "type": "transformation",
        "name": "Aggregated Data",
        "critical": false,
        "path": ["raw_data", "cleaned_data", "aggregated_data"]
      },
      {
        "id": "data_warehouse",
        "type": "table",
        "name": "Data Warehouse",
        "critical": true,
        "path": ["raw_data", "cleaned_data", "data_warehouse"]
      }
    ],
    "3": [
      {
        "id": "analytics_table",
        "type": "table",
        "name": "Analytics Table",
        "critical": true,
        "path": ["raw_data", "cleaned_data", "aggregated_data", "analytics_table"]
      },
      {
        "id": "dashboard_report",
        "type": "report",
        "name": "Dashboard Report",
        "critical": true,
        "path": ["raw_data", "cleaned_data", "aggregated_data", "dashboard_report"]
      }
    ]
  },
  "recommendations": [
    "Review all 5 affected downstream nodes",
    "3 critical nodes require stakeholder notification",
    "Consider running validation after changes"
  ]
}
```

### Multiple Node Analysis

Analyze impact for multiple nodes by running sequentially:

```bash
# Analyze multiple sources
for node in raw_data external_api config_table; do
  truthound lineage impact lineage.json $node -o impact_${node}.json
done
```

## Impact Levels

| Level | Description | Risk |
|-------|-------------|------|
| 1 | Direct downstream (immediate consumers) | High |
| 2 | Second-level downstream | Medium |
| 3+ | Transitive downstream | Lower |

## Impact Level Classification

Impact levels are calculated based on node type and distance:

| Type | Default Level | Reason |
|------|---------------|--------|
| `model` | Critical | ML model dependency |
| `source` | High | Data source |
| `table` | High | Database table |
| `external` | High | External system integration |
| `report` | Medium | Output report |
| `transformation` | Medium | Intermediate processing |
| `validation` | Low | Validation checkpoint |

**Impact Level Markers:**

| Marker | Level | Description |
|--------|-------|-------------|
| `[!!!]` | Critical | Requires immediate attention |
| `[!!]` | High | Significant impact |
| `[!]` | Medium | Moderate impact |
| `[-]` | Low | Minor impact |
| `[ ]` | None | No impact |

## Use Cases

### 1. Pre-Change Analysis

Before modifying a data source:

```bash
# Check what will be affected
truthound lineage impact lineage.json source_to_change

# If impact is acceptable, proceed with changes
```

### 2. Incident Response

When a data source has issues:

```bash
# Quickly identify affected downstream
truthound lineage impact lineage.json broken_source -o affected.json

# Notify stakeholders of affected critical nodes
```

### 3. Migration Planning

When planning data migrations:

```bash
# Analyze impact of each table to migrate
truthound lineage impact lineage.json legacy_table --max-depth 3
```

### 4. CI/CD Integration

```yaml
# GitHub Actions
- name: Analyze Change Impact
  run: |
    # Get changed files and analyze impact
    truthound lineage impact lineage.json $CHANGED_NODE -o impact.json

    # Check if critical nodes are affected
    CRITICAL=$(jq '.summary.critical_nodes' impact.json)
    if [ "$CRITICAL" -gt 0 ]; then
      echo "⚠️ $CRITICAL critical nodes affected"
      echo "Requires manual approval"
      exit 1
    fi
```

### 5. Documentation

```bash
# Generate impact analysis for documentation
truthound lineage impact lineage.json core_data_source -o docs/impact_analysis.json
```

## Interpretation Guide

### Low Impact (0-2 nodes)
- Safe to proceed with standard review
- Minimal downstream effects

### Medium Impact (3-5 nodes)
- Requires careful review
- Consider notifying affected teams

### High Impact (6+ nodes or critical nodes)
- Requires stakeholder approval
- Plan rollback strategy
- Consider phased rollout

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (node not found, invalid file, or other error) |

## Related Commands

- [`lineage show`](show.md) - Display lineage information
- [`lineage visualize`](visualize.md) - Generate visualization

## See Also

- [Lineage Overview](index.md)
- [CI/CD Integration](../../guides/ci-cd.md)
