# Data Lineage

Truthound provides comprehensive data lineage tracking, impact analysis, and visualization capabilities.

---

## Graph traversal and persistence

Upstream/downstream depth limits use shortest reachable distance in the DAG,
with breadth-first insertion order. Depth must be `-1` (unlimited) or a nonnegative
integer. Traversal, cycle checks and tracker path queries use iterative walks,
so deep graphs do not depend on Python's recursion limit. Tracker path queries
return a shortest path. Graph saves serialize finite JSON before atomically
publishing a private sibling file; serialization or pre-replacement I/O failures
preserve the previous file. Directory synchronization failures after replacement
remain errors and require reopening to inspect the actual published state.
Graph/tracker constructors capture independent configuration. `max_history=0`
retains no tracking history; negative/noninteger history limits are rejected.
These controls do not add automatic external lineage collection.


## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Lineage Tracking](#lineage-tracking)
4. [Impact Analysis](#impact-analysis)
5. [Visualization](#visualization)
6. [OpenLineage Integration](#openlineage-integration)
7. [Configuration Reference](#configuration-reference)

---

## Overview

The lineage module (`truthound.lineage`) provides:

- **10 Node Types**: Source, Table, File, Stream, Transformation, Validation, Model, Report, External, Virtual
- **8 Edge Types**: Derived, Validated, Used, Transformed, Joined, Aggregated, Filtered, Depends
- **4 Visualization Renderers**: D3, Cytoscape, Graphviz, Mermaid
- **Impact Analysis**: Downstream impact assessment with severity scoring
- **OpenLineage Integration**: Industry-standard lineage events

**Location**: `src/truthound/lineage/`

```
lineage/
├── __init__.py              # Public API exports
├── base.py                  # Core data structures (854 lines)
├── tracker.py               # LineageTracker implementation (513 lines)
├── impact_analysis.py       # Impact analysis engine (475 lines)
├── visualization/
│   └── renderers/
│       ├── d3.py            # D3.js renderer
│       ├── cytoscape.py     # Cytoscape.js renderer
│       ├── graphviz.py      # Graphviz DOT renderer
│       └── mermaid.py       # Mermaid diagram renderer
└── integrations/
    └── openlineage.py       # OpenLineage integration (549 lines)
```

---

## Core Concepts

### Node Types

| NodeType | Description |
|----------|-------------|
| `SOURCE` | Raw data source (files, APIs) |
| `TABLE` | Database table |
| `FILE` | File-based data |
| `STREAM` | Streaming source |
| `TRANSFORMATION` | Data transformation |
| `VALIDATION` | Validation checkpoint |
| `MODEL` | ML model |
| `REPORT` | Output report |
| `EXTERNAL` | External system |
| `VIRTUAL` | Virtual/computed dataset |

### Edge Types

| EdgeType | Description |
|----------|-------------|
| `DERIVED_FROM` | Data derivation |
| `VALIDATED_BY` | Validation relationship |
| `USED_BY` | Usage relationship |
| `TRANSFORMED_TO` | Transformation |
| `JOINED_WITH` | Join operation |
| `AGGREGATED_TO` | Aggregation |
| `FILTERED_TO` | Filter operation |
| `DEPENDS_ON` | Generic dependency |

### Operation Types

| OperationType | Description |
|---------------|-------------|
| `READ` | Data read operation |
| `WRITE` | Data write operation |
| `TRANSFORM` | Data transformation |
| `FILTER` | Data filtering |
| `JOIN` | Join operation |
| `AGGREGATE` | Aggregation operation |
| `VALIDATE` | Validation operation |
| `PROFILE` | Profiling operation |
| `MASK` | Data masking |
| `EXPORT` | Data export |

---

## Lineage Tracking

### Basic Usage

```python
from truthound.lineage import LineageTracker, LineageConfig

# Create tracker
config = LineageConfig(
    track_column_level=True,   # Track column-level lineage
    track_row_level=False,     # Row-level (expensive)
    store_samples=False,       # Store data samples
    max_history=100,           # Max operations in history
    auto_track=True,           # Auto-track operations
    persist_path=None,         # Auto-save path
)

tracker = LineageTracker(config)

# Track a data source
source_id = tracker.track_source(
    name="raw_customers",
    source_type="file",        # file, table, stream, external
    location="/data/customers.csv",
    schema={"id": "Int64", "name": "String", "email": "String"},
    owner="data-team",
    tags=["pii", "customer"],
)

# Track another source
db_id = tracker.track_source(
    name="orders_table",
    source_type="table",
    location="mydb.public.orders",
    schema={"order_id": "Int64", "customer_id": "Int64", "amount": "Float64"},
)

# Track a transformation
transformed_id = tracker.track_transformation(
    name="customer_orders",
    sources=[source_id, db_id],
    operation="join",
    location="/data/customer_orders.parquet",
    schema={"id": "Int64", "name": "String", "order_count": "Int64", "total_amount": "Float64"},
    column_mapping={
        "id": [(source_id, "id")],
        "name": [(source_id, "name")],
        "order_count": [(db_id, "order_id")],
        "total_amount": [(db_id, "amount")],
    },
)

# Track validation
validation_id = tracker.track_validation(
    name="validated_orders",
    sources=[transformed_id],
    validators=["NullValidator", "RangeValidator"],
    result_summary={"total_issues": 5, "critical": 0},
)

# Track output
output_id = tracker.track_output(
    name="customer_report",
    sources=[validation_id],
    output_type="report",
    location="/reports/customers.html",
)
```

### Context Manager

```python
from truthound.lineage import LineageTracker, OperationType

tracker = LineageTracker()

# Use context manager for automatic tracking
with tracker.track("data_pipeline", OperationType.TRANSFORM) as ctx:
    ctx.sources.append(source_id)

    # Do transformation work...

    ctx.targets.append(output_id)
    ctx.metadata["rows_processed"] = 10000

# Context automatically recorded with timing
```

### Query Lineage

```python
# Get upstream lineage
upstream = tracker.graph.get_upstream("customer_report", depth=3)
print("Upstream nodes:", [n.name for n in upstream])

# Get downstream lineage
downstream = tracker.graph.get_downstream("raw_customers", depth=-1)  # -1 = unlimited
print("Downstream nodes:", [n.name for n in downstream])

# Get path between nodes
path = tracker.get_path("raw_customers", "customer_report")
if path:
    print("Path:", " -> ".join(n.name for n in path))

# Get root nodes (no upstream)
roots = tracker.graph.get_roots()

# Get leaf nodes (no downstream)
leaves = tracker.graph.get_leaves()

# Topological sort
ordered = tracker.graph.topological_sort()
```

### Persistence

```python
# Save lineage graph
tracker.save("/path/to/lineage.json")

# Load lineage graph
tracker.load("/path/to/lineage.json")

# Export to JSON string
json_str = tracker.export_to_json()

# Clear all lineage
tracker.clear()
```

### Global Tracker

```python
from truthound.lineage import LineageTracker

# Set global tracker
tracker = LineageTracker()
LineageTracker.set_current(tracker)

# Get current tracker (from anywhere)
current = LineageTracker.get_current()
```

---

## Impact Analysis

### ImpactAnalyzer

Analyze downstream impact of changes.

```python
from truthound.lineage import ImpactAnalyzer, ImpactLevel

# Create analyzer
analyzer = ImpactAnalyzer(
    tracker.graph,
    impact_rules={
        NodeType.REPORT: ImpactLevel.CRITICAL,
        NodeType.MODEL: ImpactLevel.HIGH,
        NodeType.TRANSFORMATION: ImpactLevel.MEDIUM,
        NodeType.VALIDATION: ImpactLevel.LOW,
    },
)

# Analyze impact of a node
result = analyzer.analyze_impact("raw_customers", max_depth=-1)

print(result.summary())
print(f"Total affected: {result.total_affected}")
print(f"Max depth: {result.max_depth}")
print(f"Analysis time: {result.analysis_time_ms:.2f}ms")

# Get nodes by impact level
critical = result.get_by_level(ImpactLevel.CRITICAL)
high = result.get_by_level(ImpactLevel.HIGH)

# Get critical nodes
for node in result.get_critical_nodes():
    print(f"Critical: {node.node.name} (distance: {node.distance})")
    print(f"  Path: {' -> '.join(node.path)}")
    print(f"  Reason: {node.impact_reason}")

# Get by node type
reports = result.get_by_type(NodeType.REPORT)
```

### What-If Analysis

```python
# Simulate deletion
deletion_impact = analyzer.what_if_delete("raw_customers")

print(f"Node to delete: {deletion_impact['node_to_delete']}")
print(f"Would affect: {deletion_impact['total_affected']} nodes")
print(f"Would be orphaned: {deletion_impact['would_be_orphaned']}")
print(f"Critical impacts: {deletion_impact['critical_impacts']}")
print(f"Recommendation: {deletion_impact['recommendation']}")
```

### Schema Change Analysis

```python
# Analyze schema change impact
new_schema = {
    "id": "Int64",
    "name": "String",
    # "email" column removed
    "phone": "String",  # new column added
}

schema_impact = analyzer.compare_schemas("raw_customers", new_schema)

print(f"Node ID: {schema_impact['node_id']}")
print(f"Added columns: {schema_impact['added_columns']}")
print(f"Removed columns: {schema_impact['removed_columns']}")
print(f"Type changes: {schema_impact['type_changes']}")
print(f"Affected by removal: {schema_impact['affected_by_removal']}")
print(f"Safe to apply: {schema_impact['safe']}")
```

### Dependency Chain

```python
# Find dependency chain between nodes
chain = analyzer.get_dependency_chain("raw_customers", "customer_report")

if chain:
    print("Dependency chain:")
    for node in chain:
        print(f"  -> {node.name} ({node.node_type})")
```

### Find Critical Paths

```python
# Find all paths to critical nodes
critical_paths = analyzer.find_critical_paths("raw_customers")

for path in critical_paths:
    print("Critical path:")
    print(" -> ".join(node.name for node in path))
```

---

## Visualization

### Render Formats

| Format | Description | Interactive |
|--------|-------------|-------------|
| `D3_JSON` | D3.js force-directed | Yes |
| `CYTOSCAPE_JSON` | Cytoscape.js | Yes |
| `DOT` | Graphviz DOT format | No |
| `MERMAID` | Mermaid diagram | No |
| `SVG` | SVG image | No |
| `PNG` | PNG image | No |

### RenderConfig

```python
from truthound.lineage.visualization import RenderConfig, RenderFormat

config = RenderConfig(
    format=RenderFormat.D3_JSON,
    include_metadata=True,
    max_depth=-1,              # Unlimited depth
    layout="force",            # force, hierarchical, circular, grid
    theme="light",             # "light" or "dark"

    # Custom colors (optional, overrides theme defaults)
    node_colors={
        "SOURCE": "#4CAF50",
        "TABLE": "#2196F3",
        "TRANSFORMATION": "#FF9800",
        "VALIDATION": "#9C27B0",
        "REPORT": "#F44336",
    },

    edge_colors={
        "DERIVED_FROM": "#666666",
        "VALIDATED_BY": "#9C27B0",
        "TRANSFORMED_TO": "#FF9800",
    },

    # Highlighting
    highlight_nodes=["raw_customers"],

    # Filtering
    filter_node_types=["SOURCE", "TRANSFORMATION", "REPORT"],
    filter_edge_types=None,

    # Layout
    orientation="TB",          # TB, BT, LR, RL
    width=1200,
    height=800,
)
```

### Theme Support

All renderers support light and dark themes:

```python
from truthound.lineage.visualization import get_renderer

# Factory function with theme
renderer = get_renderer("d3", theme="dark")
html = renderer.render_html(graph)

# Or renderer with theme
from truthound.lineage.visualization.renderers import D3Renderer
renderer = D3Renderer(theme="dark")
html = renderer.render_html(graph)
```

| Theme | Background | Text | Use Case |
|-------|------------|------|----------|
| `light` | White | Dark | Documents, presentations |
| `dark` | Navy (`#1a1a2e`) | Light | Dark mode UIs, dashboards |

!!! note "Mermaid Theme Limitation"
    For MermaidRenderer, the theme only applies to `render_html()` output.
    The `render()` and `render_markdown()` methods output raw Mermaid syntax,
    which is rendered by the viewing platform (GitHub, GitLab, etc.) with its own theme.

### D3 Renderer

Interactive force-directed visualization.

```python
from truthound.lineage.visualization.renderers import D3Renderer

# Light theme (default)
renderer = D3Renderer()

# Dark theme
renderer = D3Renderer(theme="dark")

# Get JSON for D3.js
json_data = renderer.render(tracker.graph, config)

# Get complete interactive HTML
html = renderer.render_html(tracker.graph, config)

# Save HTML file
with open("lineage.html", "w") as f:
    f.write(html)

# Render subgraph (downstream from node)
subgraph_json = renderer.render_subgraph(
    tracker.graph,
    root_node_id="raw_customers",
    direction="downstream",
    max_depth=3,
    config=config,
)
```

**D3 JSON Structure:**
```json
{
  "nodes": [
    {
      "id": "node_id",
      "name": "Node Name",
      "type": "SOURCE",
      "color": "#4CAF50",
      "size": 20,
      "highlighted": false,
      "metadata": {...}
    }
  ],
  "links": [
    {
      "source": 0,
      "target": 1,
      "type": "DERIVED_FROM",
      "color": "#666666",
      "metadata": {...}
    }
  ],
  "config": {
    "layout": "force",
    "width": 1200,
    "height": 800
  }
}
```

### Cytoscape Renderer

Multi-layout graph visualization.

```python
from truthound.lineage.visualization.renderers import CytoscapeRenderer

renderer = CytoscapeRenderer()

# Available layouts
print(CytoscapeRenderer.LAYOUTS)
# {'force': 'cose', 'hierarchical': 'dagre', 'circular': 'circle', 'grid': 'grid', ...}

# Get JSON for Cytoscape.js
json_data = renderer.render(tracker.graph, config)

# Get complete interactive HTML
html = renderer.render_html(tracker.graph, config)
```

**Features:**
- Layout selector dropdown
- Node info panel on click
- Multiple layout algorithms
- Dagre for hierarchical layout

### Graphviz Renderer

DOT format for Graphviz.

```python
from truthound.lineage.visualization.renderers import GraphvizRenderer

renderer = GraphvizRenderer()

# Node shapes for each type
print(GraphvizRenderer.NODE_SHAPES)
# {'SOURCE': 'cylinder', 'TABLE': 'box3d', 'TRANSFORMATION': 'component', ...}

# Get DOT format
dot = renderer.render(tracker.graph, config)

# Get SVG (requires graphviz library)
svg = renderer.render_svg(tracker.graph, config)

# Get PNG (requires graphviz library)
png_bytes = renderer.render_png(tracker.graph, config)
```

### Mermaid Renderer

Mermaid diagram syntax.

```python
from truthound.lineage.visualization.renderers import MermaidRenderer

renderer = MermaidRenderer()

# Get Mermaid syntax
mermaid = renderer.render(tracker.graph, config)

# Get Markdown with embedded diagram
markdown = renderer.render_markdown(
    tracker.graph,
    config,
    title="Data Lineage Diagram",
)

# Get HTML with auto-rendering
html = renderer.render_html(tracker.graph, config)
```

**Mermaid Node Shapes:**
```
SOURCE: [(name)]      # Cylinder
TABLE: [name]         # Rectangle
TRANSFORMATION: [[name]]  # Subroutine
VALIDATION: {name}    # Diamond
REPORT: [name]        # Rectangle
```

**Mermaid Edge Arrows:**
```
DERIVED_FROM: -->
VALIDATED_BY: -.->
TRANSFORMED_TO: ==>
JOINED_WITH: <-->
```

---

## OpenLineage Integration

### OpenLineageEmitter

Emit lineage events in OpenLineage format.

```python
from truthound.lineage.integrations.openlineage import (
    OpenLineageEmitter,
    OpenLineageConfig,
)

config = OpenLineageConfig(
    endpoint="http://lineage-server:5000/api/v1/lineage",
    api_key="your-api-key",
    namespace="truthound",
    producer="truthound",
    timeout_seconds=30,
)

emitter = OpenLineageEmitter(config)
```

### Delivery, failure and shutdown

The emitter captures its configuration and run inputs. Only an HTTP 2xx response
acknowledges a transition. Transport failures and non-2xx responses raise
`OpenLineageEmissionError` with `run_id`, `event_type` and `outcome_unknown=True`;
the prior acknowledged state remains unchanged. Inspect `unknown_runs` and use
`resolve_unknown(run_id, delivered=True|False)` only after checking the receiver.
Resolution records that determination without sending another event. An uncertain
run cannot transition again until resolved; there are no automatic retries,
redirects or environment-proxy routing. Receiver response bodies and credentials
are excluded from delivery errors.

`max_event_bytes` defaults to 1 MiB. Events must contain finite, acyclic JSON data
within this limit. Graph emission snapshots and validates every planned event
before the first request, and emits dataset names without copying raw samples.
Network failure can still leave a graph partially delivered: completed runs stay
completed and the failed transition remains uncertain. Do not replay the whole
graph to recover it. This state belongs to the live emitter; it is not a durable
journal and cannot establish delivery after process loss.

Inside an event loop, await `start_run_async`, `emit_running_async`,
`emit_complete_async`, `emit_fail_async`, `emit_abort_async`,
`emit_from_graph_async` and `resolve_unknown_async`. Cancellation waits for the
owned request to finish; it does not retract a request received by the server.
Use `await emitter.aclose()` (or `async with`) to join pending work. Synchronous
callers use `close()` or `with`. Closing does not automatically abort active runs.
The synchronous state properties acquire the delivery lock; read them after
awaiting pending work rather than while a request is in flight.

Scalar custom facets are wrapped in a `value` object with facet metadata. Producer
identifiers without a URI scheme receive a Truthound URI. Quality metrics use
`inputFacets`; output row counts, including zero, use `outputFacets`.

### Manual Event Emission

```python
# Start a run
run = emitter.start_run(
    job_name="data-pipeline",
    inputs=[
        emitter.build_input_dataset(
            "raw_customers",
            schema=[
                {"name": "id", "type": "Int64"},
                {"name": "name", "type": "String"},
            ],
        ),
    ],
    parent_run_id=None,        # For nested jobs
    facets={"custom": "data"},
)

# Emit progress
emitter.emit_running(run, facets={"progress": "50%"})

# Complete the run
emitter.emit_complete(
    run,
    outputs=[
        emitter.build_output_dataset(
            "processed_customers",
            schema=[
                {"name": "id", "type": "Int64"},
                {"name": "name", "type": "String"},
            ],
            row_count=10000,
        ),
    ],
)

# Or mark as failed
# emitter.emit_fail(run, error=Exception("Pipeline failed"))

# Or mark as aborted
# emitter.emit_abort(run, reason="User cancelled")
```

### Convert Graph to OpenLineage

```python
# Emit entire lineage graph as OpenLineage events
runs = emitter.emit_from_graph(
    tracker.graph,
    job_name="truthound-lineage",
)

print(f"Emitted {len(runs)} run events")
```

### DatasetFacets

```python
from truthound.lineage.integrations.openlineage import DatasetFacets

facets = DatasetFacets(
    schema_fields=[
        {"name": "id", "type": "Int64"},
        {"name": "name", "type": "String"},
    ],
    data_source={
        "uri": "postgresql://host:5432/db",
        "name": "customers",
    },
    lifecycle_state="production",
    ownership={
        "owners": [{"name": "data-team", "type": "team"}],
    },
    quality_metrics={
        "rowCount": 10000,
        "nullCount": {"id": 0, "name": 5},
    },
)
```

### Event Types

| EventType | Description |
|-----------|-------------|
| `START` | Run started |
| `RUNNING` | Run in progress |
| `COMPLETE` | Run completed successfully |
| `ABORT` | Run aborted |
| `FAIL` | Run failed |
| `OTHER` | Other event type |

---

## Configuration Reference

### LineageConfig

```python
from truthound.lineage import LineageConfig

config = LineageConfig(
    track_column_level=True,   # Track column-level lineage
    track_row_level=False,     # Track row-level (expensive)
    store_samples=False,       # Store data samples
    max_history=100,           # Max operations in history
    auto_track=True,           # Auto-track operations
    persist_path=None,         # Auto-save path
    metadata={},               # Custom metadata
)
```

### LineageNode

```python
from truthound.lineage.base import LineageNode, LineageMetadata

node = LineageNode(
    id="unique_id",
    name="Node Name",
    node_type=NodeType.SOURCE,
    location="/path/to/data",
    schema={"col1": "Int64", "col2": "String"},
    metadata=LineageMetadata(
        description="Data source description",
        owner="data-team",
        tags=("pii", "customer"),
        properties={"custom": "value"},
    ),
    column_lineage=(
        ColumnLineage(
            column="col1",
            source_columns=(("source_id", "src_col"),),
            transformation="direct",
            dtype="Int64",
        ),
    ),
)
```

### LineageEdge

```python
from truthound.lineage.base import LineageEdge

edge = LineageEdge(
    source="source_node_id",
    target="target_node_id",
    edge_type=EdgeType.DERIVED_FROM,
    operation=OperationType.TRANSFORM,
    metadata=LineageMetadata(
        description="Join operation",
        owner="pipeline",
        tags=("etl",),
    ),
)
```

---

## Thread Safety

- `LineageGraph` uses `threading.RLock()` for all operations
- `LineageTracker` uses `ContextVar` to isolate and restore nested thread/task contexts
- Global tracker access is thread-safe

---

## Error Handling

```python
from truthound.lineage.base import (
    LineageError,
    NodeNotFoundError,
    CyclicDependencyError,
)

try:
    tracker.graph.add_edge(edge)
except CyclicDependencyError as e:
    print(f"Cycle detected: {e}")

try:
    node = tracker.graph.get_node("invalid_id")
except NodeNotFoundError as e:
    print(f"Node not found: {e}")
```

---

## See Also

- [Impact Analysis](../ci-cd/index.md) - CI/CD integration
- [OpenLineage](https://openlineage.io/) - OpenLineage specification
- [Visualization](../datadocs/index.md) - Report visualization

### Tracker configuration and durable edits

`track_column_level=False` omits transformation column mappings. With `track_row_level=True`, `track_transformation(..., row_mapping={0: [("input", 4)]})` records explicit zero-based output/source row positions in node metadata `row_lineage`. Referenced nodes must be declared sources; positions must be nonnegative integers. This does not infer row provenance from arbitrary dataframe operations or retain row values. With row tracking disabled, row mappings are omitted. MCP accepts bounded `row_mapping` records containing `row` and `sources` (`node`, `row`), and rejects duplicate output positions.

`with tracker.track("operation") as ctx` makes the tracker current within that context. Use `ctx.add_source(node_id)` and `ctx.add_target(node_id)` for existing graph nodes. On successful exit, `auto_track=True` creates an operation node and its input/output edges. Nested contexts restore the previous tracker; failed bodies do not create an operation node. `auto_track=False` retains context history without creating those edges. This is explicit instrumentation, not interception of unrelated Python or Polars operations.

`persist_path` reopens an existing graph and publishes each successful tracker edit atomically. Invalid edits leave the previous graph and snapshot intact. Native file locks serialize cooperating writers; a stale snapshot cannot overwrite newer bytes. Snapshots are bounded to 16 MiB and symbolic links are rejected. Publication/acknowledgement failures fence further edits: reopen the tracker to inspect the actual committed file before continuing. Direct edits through `tracker.graph` and explicit `save()` remain separate graph operations; automatic persistence applies to tracker methods. Tracker edits publish a new graph snapshot, so obtain `tracker.graph` again after editing rather than mutating a previously retained graph reference. MCP does not accept caller-selected `persist_path`; its existing artifact lifecycle remains authoritative.



`store_samples` now controls explicitly supplied `samples=[{column: value, ...}]` metadata on source, transformation, validation and output tracking. It is off by default and drops that reserved field from new tracking calls. When enabled, the tracker captures independent finite JSON rows, bounded to 100 rows and 64 KiB per node. Invalid or oversized samples do not publish a graph mutation. No source is read automatically. Existing/imported graph metadata is not retroactively erased by changing this flag. In MCP tracking events the explicit input is `properties.samples`; this opt-in does not read an asset or bypass its disclosure policy. Keep sensitive data out of shared graph exports.

Native HTML rendering escapes graph labels as data, and DOT/Mermaid IDs remain distinct even when punctuation differs. Renderers capture configuration instead of changing caller-owned settings. Subgraph methods inherit `RenderConfig.max_depth` when their depth argument is omitted; explicit `-1` overrides it. D3 grid/circular/hierarchical layouts provide fixed coordinates; nodes blocked by cycles share the final hierarchical rank. Cytoscape uses selected node sizes. Graphviz SVG/PNG output fits within the configured width/height bounds with aspect ratio preserved; it requires the `lineage-visualization` extra and a system `dot`. Native Mermaid HTML pins Mermaid 12.2.0. Whole-graph HTML/Markdown/SVG/PNG is also available through [MCP](../mcp.md).

Native tracking history captures completed context snapshots. Explicit saves to the configured primary graph file use the same version fence as tracking edits; a stale owner must reload the primary file before writing. Confirmed primary reload also resets an unconfirmed publication fence. History is memory-only and graph files serialize graph state, not callbacks or context history. Approved [MCP tracker sessions](../mcp.md) preserve this distinction.
