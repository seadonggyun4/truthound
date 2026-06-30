# truthound lineage show

CLI 명령 실행에서 Display을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Synopsis

```bash
truthound lineage show <lineage_file> [OPTIONS]
```

## Arguments

| CLI 명령 실행에서 Argument을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-------------|
| CLI 명령 실행에서 `lineage_file`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 JSON, Path을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Options

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|---------|-------------|
| CLI 명령 실행에서 `--node`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-n`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Focus을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--direction`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-d`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `both`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Traversal을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-f`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `console`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `console`, Output을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

!!! warning "참고"
CLI 명령 실행에서 `--format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
CLI 명령 실행에서 JSON, `lineage visualize`, DOT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Description

CLI 명령 실행에서 `lineage show`을(를) 다루는 항목입니다:

1. CLI 명령 실행에서 Shows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. CLI 명령 실행에서 Filters을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 예시

### Basic Usage

```bash
truthound lineage show lineage.json
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
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

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Lineage for: analytics_table
Type: table

Upstream (2 nodes):
  <- aggregated_data (transformation)
  <- api_processed (transformation)

Downstream (0 nodes):
```

### Upstream Only

CLI 명령 실행에서 Show을(를) 다루는 항목입니다:

```bash
truthound lineage show lineage.json --node analytics_table --direction upstream
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Lineage for: analytics_table
Type: table

Upstream (2 nodes):
  <- aggregated_data (transformation)
  <- api_processed (transformation)
```

### Downstream Only

CLI 명령 실행에서 Show을(를) 다루는 항목입니다:

```bash
truthound lineage show lineage.json --node raw_data --direction downstream
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Lineage for: raw_data
Type: source

Downstream (1 nodes):
  -> cleaned_data (transformation)
```

### Visual Export (Alternative)

CLI 명령 실행에서 JSON, `lineage visualize`, DOT, Graphviz을(를) 다루는 항목입니다:

```bash
# Generate interactive HTML visualization
truthound lineage visualize lineage.json -o graph.html

# Generate Graphviz DOT file
truthound lineage visualize lineage.json -o graph.dot --renderer graphviz

# Generate Mermaid diagram
truthound lineage visualize lineage.json -o graph.md --renderer mermaid
```

CLI 명령 실행에서 `lineage visualize`, See을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Direction Options

| CLI 명령 실행에서 Direction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| CLI 명령 실행에서 `upstream`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Show을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `downstream`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Show을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `both`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Show을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

CLI 명령 실행에서 `lineage visualize`을(를) 다루는 항목입니다:

```bash
# Generate DOT file for documentation
truthound lineage visualize lineage.json -o docs/lineage.dot --renderer graphviz
```

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Related Commands

- CLI 명령 실행에서 `lineage impact`, Analyze을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 `lineage visualize`, Generate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- [Lineage 개요](index.md)
- CLI 명령 실행에서 Advanced, Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
