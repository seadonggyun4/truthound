---
title: Batch Processing
---

# Batch Processing

오케스트레이션 실행에서 Large을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Main Components

- 오케스트레이션 실행에서 `BatchExecutor`, BatchExecutor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `AsyncBatchExecutor`, AsyncBatchExecutor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `BatchConfig`, BatchConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Polars, `RowCountChunker`, `PolarsChunker`, `DatasetListChunker`, RowCountChunker, PolarsChunker, DatasetListChunker을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `LoggingBatchHook`, `MetricsBatchHook`, LoggingBatchHook, MetricsBatchHook을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Basic Usage

```python
from common.engines import BatchConfig, BatchExecutor, TruthoundEngine

engine = TruthoundEngine()
executor = BatchExecutor(engine, BatchConfig(batch_size=10000, max_workers=4))
result = executor.check_batch(large_dataframe, auto_schema=True)
```

## When To Use Batch Execution

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Operational Choices

오케스트레이션 실행에서 Choose을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Pages

- [Streaming 검증](streaming.md)
- 오케스트레이션 실행에서 Lifecycle, Management을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
