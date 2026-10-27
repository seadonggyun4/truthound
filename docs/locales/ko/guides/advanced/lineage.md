# Data Lineage

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 그래프 탐색과 저장

상·하류 깊이 제한은 DAG에서 도달 가능한 최단 거리를 사용하며 너비 우선 삽입
순서로 반환합니다. 깊이는 `-1`(무제한) 또는 비음수 정수여야 합니다. 탐색·순환
검사·tracker 경로 질의는 반복 방식으로 수행하므로 깊은 그래프가 Python 재귀
한도에 의존하지 않습니다. tracker는 최단 경로를 반환합니다. 그래프 저장은
유한한 JSON으로 직렬화한 뒤 비공개 임시 형제 파일을 원자적으로 게시합니다.
직렬화 또는 파일 교체 전 I/O 실패는 기존 파일을 보존합니다. 교체 후 디렉터리
동기화 실패도 오류이며 다시 열어 실제 게시 상태를 확인해야 합니다.
그래프·tracker 생성자는 설정을 독립 복사합니다. `max_history=0`은 추적 이력을
남기지 않으며 음수·정수가 아닌 이력 한도는 거부합니다. 이 기능이 외부 계보의
자동 수집을 추가하는 것은 아닙니다.


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

### 전송·실패·종료

Emitter는 설정과 실행 입력을 복사하며 HTTP 2xx 응답만 전환의 수신 확인으로
인정합니다. 전송 오류나 비-2xx 응답은 `run_id`, `event_type`,
`outcome_unknown=True`를 가진 `OpenLineageEmissionError`를 발생시키고 이전에
확인된 상태를 보존합니다. `unknown_runs`를 조회하고 수신기를 확인한 뒤에만
`resolve_unknown(run_id, delivered=True|False)`로 판정을 기록하세요.
이 작업은 이벤트를 재전송하지 않습니다. 결과 불명 실행은 판정 전까지 다음
전환을 거부합니다. 자동 재시도·리다이렉트·환경 프록시 경유는 없으며 전송
오류에 수신 응답 본문이나 인증 정보를 포함하지 않습니다.

`max_event_bytes` 기본값은 1 MiB입니다. 이벤트는 유한 수와 비순환 JSON으로
구성되어야 하며 크기 한도를 준수해야 합니다. 그래프 전송은 스냅샷의 모든
예정 이벤트를 첫 요청 전에 검증하고, 원시 표본을 복사하지 않고 데이터셋
이름을 전송합니다. 네트워크 실패 시 그래프 일부만 전송될 수 있습니다.
완료 실행은 완료 상태를 유지하고 실패 전환은 결과 불명으로 남습니다.
복구를 위해 전체 그래프를 재전송하지 마세요. 이 상태는 실행 중인 emitter에
속하며 영구 저널이 아니므로 프로세스 소실 후 수신 여부를 증명하지 못합니다.

이벤트 루프에서는 `start_run_async`, `emit_running_async`,
`emit_complete_async`, `emit_fail_async`, `emit_abort_async`,
`emit_from_graph_async`, `resolve_unknown_async`를 await하세요. 취소는 소유한
요청이 끝날 때까지 기다리며 서버가 받은 요청을 철회하지 않습니다.
`await emitter.aclose()` 또는 `async with`로 대기 작업을 회수합니다.
동기 호출에서는 `close()` 또는 `with`를 사용합니다. 종료가 활성 실행을 자동
중단하지는 않습니다. 동기 상태 속성은 전송 잠금을 사용하므로 진행 중 요청을
await한 뒤 조회하세요.

스칼라 사용자 facet은 메타데이터와 `value`를 가진 객체로 감쌉니다. URI scheme이
없는 producer에는 Truthound URI를 부여합니다. 품질 지표는 `inputFacets`,
0을 포함한 출력 행 수는 `outputFacets`에 기록합니다.

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
- `LineageTracker`는 `ContextVar`로 스레드·비동기 작업의 중첩 컨텍스트를 격리하고 복원합니다.
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

### 추적기 설정과 지속성 변경

`track_column_level=False`는 변환 열 매핑을 생략합니다. `track_row_level=True`에서 `track_transformation(..., row_mapping={0: [("input", 4)]})`는 0 기반 출력·입력 행 위치를 노드 메타데이터 `row_lineage`에 기록합니다. 참조 노드는 선언된 입력이어야 하고 위치는 음이 아닌 정수여야 합니다. 임의 데이터프레임 연산의 행 출처를 자동 추론하거나 행 값을 보관하지 않습니다. 행 추적이 비활성화되면 매핑을 생략합니다. MCP는 `row`와 `sources`(`node`, `row`)가 있는 제한된 `row_mapping` 레코드를 받으며 출력 위치 중복을 거부합니다.

`with tracker.track("operation") as ctx`는 컨텍스트 안에서 현재 추적기를 지정합니다. 기존 그래프 노드를 `ctx.add_source(node_id)`, `ctx.add_target(node_id)`로 선언하세요. 정상 종료 시 `auto_track=True`이면 연산 노드와 입출력 간선을 생성합니다. 중첩 컨텍스트는 이전 추적기를 복원하며 본문 실패 시 연산 노드를 생성하지 않습니다. `auto_track=False`는 컨텍스트 이력만 남기고 간선을 만들지 않습니다. 명시적으로 계측한 연산을 연결하며 무관한 Python·Polars 연산을 가로채지 않습니다.

`persist_path`는 기존 그래프를 다시 열고 성공한 추적기 변경마다 원자적으로 저장합니다. 잘못된 변경은 이전 그래프와 파일을 보존합니다. 네이티브 파일 잠금으로 협력 작성자를 직렬화하고 오래된 스냅샷의 덮어쓰기를 거부합니다. 스냅샷은 16 MiB로 제한하고 심볼릭 링크를 거부합니다. 게시·확인 응답 실패 후에는 추가 변경을 차단하므로 추적기를 다시 열어 실제 커밋된 파일을 확인해야 합니다. `tracker.graph` 직접 수정과 명시적 `save()`는 별도 그래프 연산이며 자동 저장은 추적기 메서드에 적용됩니다. 추적기 변경은 새 그래프 스냅샷을 반영하므로 이전 그래프 참조를 수정하지 말고 변경 뒤 `tracker.graph`를 다시 얻으세요. MCP는 호출자가 선택한 `persist_path`를 받지 않으며 기존 아티팩트 수명주기를 사용합니다.



`store_samples`는 source·transformation·validation·output 추적에 명시적으로 전달한 `samples=[{열: 값, ...}]` 메타데이터를 제어합니다. 기본값은 비활성이며 새 추적 호출에서 해당 예약 필드를 제외합니다. 활성화하면 노드당 100행·64 KiB 이내의 유한한 JSON 행을 독립적으로 복사합니다. 잘못되거나 너무 큰 sample은 그래프 변경을 게시하지 않습니다. 원본 데이터를 자동으로 읽지 않으며, 설정 변경이 기존·가져온 그래프 메타데이터를 소급 삭제하지도 않습니다. MCP 추적 이벤트에서는 `properties.samples`로 명시적 입력을 전달하며 자산을 읽거나 공개 정책을 우회하지 않습니다. 민감한 데이터는 공유 그래프 내보내기에 포함하지 마세요.

네이티브 HTML은 그래프 이름을 데이터로 인코딩하고, DOT·Mermaid는 문장부호가 다른 노드 ID도 구분합니다. 렌더러는 호출자의 설정을 변경하지 않고 복사합니다. 부분 그래프의 깊이 인자를 생략하면 `RenderConfig.max_depth`를 따르며 명시적 `-1`은 이를 재정의합니다. D3 격자·원형·계층 레이아웃은 고정 좌표를 제공하고 순환 때문에 순위를 정하지 못한 노드는 마지막 계층에 배치합니다. Cytoscape는 선택한 노드 크기를 적용합니다. Graphviz SVG·PNG는 종횡비를 유지하며 설정한 너비·높이 경계 안에 맞춥니다. `lineage-visualization` extra와 시스템 `dot`이 필요합니다. Mermaid HTML은 Mermaid 12.2.0을 고정합니다. 전체 그래프 HTML·Markdown·SVG·PNG도 [MCP](../mcp.md)에서 제공합니다.

네이티브 추적 이력은 완료된 문맥의 스냅샷입니다. 설정된 주 그래프 파일의 명시적 저장은 추적 수정과 같은 버전 확인을 적용하므로 오래된 소유자는 먼저 주 파일을 다시 읽어야 합니다. 확인된 주 파일 재열기는 불명확한 게시의 쓰기 차단도 해소합니다. 이력은 메모리에만 유지하며 그래프 파일은 콜백·문맥 이력이 아닌 그래프 상태를 저장합니다. 승인된 [MCP 트래커 세션](../mcp.md)도 이 구분을 유지합니다.
