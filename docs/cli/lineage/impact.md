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

Impact Level 1 (Direct):
  ⚠️  cleaned_data (transformation)
      Path: raw_data → cleaned_data

Impact Level 2:
  ⚠️  aggregated_data (transformation)
      Path: raw_data → cleaned_data → aggregated_data

  ⚠️  data_warehouse (sink)
      Path: raw_data → cleaned_data → data_warehouse

Impact Level 3:
  ⚠️  analytics_table (sink)
      Path: raw_data → cleaned_data → aggregated_data → analytics_table

  ⚠️  dashboard_report (report)
      Path: raw_data → cleaned_data → aggregated_data → dashboard_report

Summary
───────────────────────────────────────────────
Impact Level    Nodes    Critical
───────────────────────────────────────────────
Level 1         1        0
Level 2         2        1
Level 3         2        2
───────────────────────────────────────────────
Total           5        3

Recommendation: HIGH IMPACT - Review all affected nodes before changes.
```

### Limited Depth Analysis

Analyze only immediate and second-level impacts:

```bash
truthound lineage impact lineage.json raw_data --max-depth 2
```

Output:
```
Impact Analysis: raw_data (max depth: 2)
========================================

Impact Level 1 (Direct):
  ⚠️  cleaned_data (transformation)

Impact Level 2:
  ⚠️  aggregated_data (transformation)
  ⚠️  data_warehouse (sink)

Total: 3 nodes within depth 2
Note: Additional nodes may exist beyond depth 2
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
        "type": "sink",
        "name": "Data Warehouse",
        "critical": true,
        "path": ["raw_data", "cleaned_data", "data_warehouse"]
      }
    ],
    "3": [
      {
        "id": "analytics_table",
        "type": "sink",
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

## Critical Node Classification

Nodes are marked as critical based on:

| Type | Critical | Reason |
|------|----------|--------|
| `sink` | Yes | End user-facing |
| `report` | Yes | Business critical |
| `model` | Yes | ML model dependency |
| `transformation` | No | Intermediate |
| `source` | No | Not downstream |

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
| 0 | Success (no critical impact) |
| 1 | Critical nodes affected |
| 2 | Node not found or invalid file |

## Related Commands

- [`lineage show`](show.md) - Display lineage information
- [`lineage visualize`](visualize.md) - Generate visualization

## See Also

- [Lineage Overview](index.md)
- [CI/CD Integration](../../guides/ci-cd.md)
