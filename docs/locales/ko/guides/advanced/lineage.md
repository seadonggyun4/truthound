# Data Lineage

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 테이블 of Contents

1. [개요](#overview)
2. [Core 개념](#core-concepts)
3. 실무 운영 가이드에서 Lineage, Tracking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 Impact, Analysis을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 Visualization을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. [OpenLineage 통합](#openlineage-integration)
7. [설정 레퍼런스](#configuration-reference)

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 `truthound.lineage`을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Node, Types, Source, Table, File, Stream, Transformation, Validation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Edge, Types, Derived, Validated, Used, Transformed, Joined, Aggregated을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Visualization, Renderers, Cytoscape, Graphviz, Mermaid을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Impact, Analysis, Downstream을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 OpenLineage, Integration, Industry-standard을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 `src/truthound/lineage/`, Location을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Core 개념

### Node Types

| 실무 운영 가이드에서 NodeType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `SOURCE`, SOURCE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Raw data 소스 (파일, APIs) |
| 실무 운영 가이드에서 `TABLE`, TABLE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 테이블 |
| 실무 운영 가이드에서 `FILE`, FILE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 파일-based data |
| 실무 운영 가이드에서 `STREAM`, STREAM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Streaming 소스 |
| 실무 운영 가이드에서 `TRANSFORMATION`, TRANSFORMATION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `VALIDATION`, VALIDATION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증 체크포인트 |
| 실무 운영 가이드에서 `MODEL`, MODEL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `REPORT`, REPORT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Output 리포트 |
| 실무 운영 가이드에서 `EXTERNAL`, EXTERNAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 External을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `VIRTUAL`, VIRTUAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Virtual/computed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Edge Types

| 실무 운영 가이드에서 EdgeType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `DERIVED_FROM`, DERIVED_FROM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `VALIDATED_BY`, VALIDATED_BY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증 relationship |
| 실무 운영 가이드에서 `USED_BY`, USED_BY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Usage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TRANSFORMED_TO`, TRANSFORMED_TO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Transformation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `JOINED_WITH`, JOINED_WITH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Join을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AGGREGATED_TO`, AGGREGATED_TO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Aggregation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FILTERED_TO`, FILTERED_TO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DEPENDS_ON`, DEPENDS_ON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Generic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Operation Types

| 실무 운영 가이드에서 OperationType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------------|-------------|
| 실무 운영 가이드에서 `READ`, READ을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `WRITE`, WRITE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TRANSFORM`, TRANSFORM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FILTER`, FILTER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `JOIN`, JOIN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Join을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AGGREGATE`, AGGREGATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Aggregation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `VALIDATE`, VALIDATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증 operation |
| 실무 운영 가이드에서 `PROFILE`, PROFILE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일링 operation |
| 실무 운영 가이드에서 `MASK`, MASK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `EXPORT`, EXPORT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Impact Analysis

### ImpactAnalyzer

실무 운영 가이드에서 Analyze을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

### 스키마 Change Analysis

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Visualization

### Render Formats

| 실무 운영 가이드에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Interactive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|-------------|
| 실무 운영 가이드에서 JSON, `D3_JSON`, D3_JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 D3.js을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 JSON, `CYTOSCAPE_JSON`, CYTOSCAPE_JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cytoscape.js을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DOT`, DOT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Graphviz, DOT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MERMAID`, MERMAID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Mermaid을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SVG`, SVG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SVG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PNG`, PNG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PNG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

| 실무 운영 가이드에서 Theme을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Background을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Text을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|------------|------|----------|
| 실무 운영 가이드에서 `light`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 White을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Documents을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `dark`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `#1a1a2e`, Navy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Light을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dark, UIs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

!!! note "참고"
실무 운영 가이드에서 `render_html()`, MermaidRenderer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 `render()`, `render_markdown()`, Mermaid을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 GitHub, GitLab을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### D3 Renderer

실무 운영 가이드에서 Interactive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 JSON, Structure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
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

실무 운영 가이드에서 Multi-layout을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Layout을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Node을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Multiple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Dagre을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Graphviz Renderer

실무 운영 가이드에서 DOT, Graphviz을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Mermaid을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Mermaid, Node, Shapes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
```
SOURCE: [(name)]      # Cylinder
TABLE: [name]         # Rectangle
TRANSFORMATION: [[name]]  # Subroutine
VALIDATION: {name}    # Diamond
REPORT: [name]        # Rectangle
```

실무 운영 가이드에서 Mermaid, Edge, Arrows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
```
DERIVED_FROM: -->
VALIDATED_BY: -.->
TRANSFORMED_TO: ==>
JOINED_WITH: <-->
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## OpenLineage 통합

### OpenLineageEmitter

실무 운영 가이드에서 OpenLineage, Emit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

| 실무 운영 가이드에서 EventType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| 실무 운영 가이드에서 `START`, START을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `RUNNING`, RUNNING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COMPLETE`, COMPLETE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ABORT`, ABORT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FAIL`, FAIL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `OTHER`, OTHER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Other을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설정 레퍼런스

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Thread Safety

- 실무 운영 가이드에서 `LineageGraph`, `threading.RLock()`, LineageGraph, RLock을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `LineageTracker`, LineageTracker을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Global을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- 실무 운영 가이드에서 Impact, Analysis, CI/CD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 OpenLineage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Visualization, Report을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
