---
title: 시작하기
---

# 시작하기

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Prerequisites

- 오케스트레이션 실행에서 Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `truthound>=3.0,<4.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:
  - 오케스트레이션 실행에서 Airflow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
  - 오케스트레이션 실행에서 Dagster을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
  - 오케스트레이션 실행에서 Prefect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
  - 오케스트레이션 실행에서 dbt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
  - 오케스트레이션 실행에서 Mage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
  - 오케스트레이션 실행에서 Kestra을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Install Truthound 3.x With A 플랫폼 Package

```bash
pip install truthound-orchestration "truthound>=3.0,<4.0"
pip install truthound-orchestration[airflow] "truthound>=3.0,<4.0"
pip install truthound-orchestration[dagster] "truthound>=3.0,<4.0"
pip install truthound-orchestration[prefect] "truthound>=3.0,<4.0"
pip install truthound-orchestration[mage] "truthound>=3.0,<4.0"
pip install truthound-orchestration[kestra] "truthound>=3.0,<4.0"
```

오케스트레이션 실행에서 Truthound, `truthound-orchestration 3.x`, `Truthound 3.x`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

오케스트레이션 실행에서 Install을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Fastest First Run

오케스트레이션 실행에서 Truthound, Truthound-first을(를) 다루는 항목입니다:

```python
import polars as pl

from common.engines import TruthoundEngine

engine = TruthoundEngine()
data = pl.read_csv("data.csv")

result = engine.check(data)
print(result.status.name)
```

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Truthound, `check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Per-플랫폼 Zero-설정 Quickstarts

### Airflow

```python
from truthound_airflow.operators import DataQualityCheckOperator

check = DataQualityCheckOperator(
    task_id="check_users",
    data_path="/opt/airflow/data/users.parquet",
    rules=[{"column": "user_id", "type": "not_null"}],
)
```

오케스트레이션 실행에서 Airflow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Dagster

```python
from dagster import Definitions, asset
from truthound_dagster.resources import DataQualityResource

@asset
def validated_users(data_quality: DataQualityResource):
    return data_quality.check(load_users(), rules=[{"column": "id", "type": "not_null"}])

defs = Definitions(resources={"data_quality": DataQualityResource()})
```

오케스트레이션 실행에서 Dagster, `DataQualityResource()`, DataQualityResource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Prefect

```python
from prefect import flow
from truthound_prefect.tasks import data_quality_check_task

@flow(name="validate-users")
async def validate_users(data):
    return await data_quality_check_task(
        data=data,
        rules=[{"column": "id", "type": "not_null"}],
    )
```

오케스트레이션 실행에서 Truthound, Prefect, Truthound-backed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### dbt

```yaml
# packages.yml
packages:
  - package: truthound/truthound
    version: ">=3.0.0,<4.0.0"
```

```yaml
# models/schema.yml
version: 2

models:
  - name: stg_users
    tests:
      - truthound.truthound_check:
          rules:
            - column: user_id
              check: not_null
```

오케스트레이션 실행에서 dbt, `dbt deps`, `dbt test`, Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Mage

```python
from truthound_mage import CheckTransformer, CheckBlockConfig

transformer = CheckTransformer(
    config=CheckBlockConfig(
        auto_schema=True,
        rules=[{"column": "id", "type": "not_null"}],
    )
)
result = transformer.execute(dataframe)
```

### Kestra

```python
from truthound_kestra.scripts import check_quality_script

result = check_quality_script(
    input_uri="data/users.parquet",
    rules=[{"column": "id", "type": "not_null"}],
)
```

## What To Expect From Zero-설정

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- default 엔진: Truthound
- 오케스트레이션 실행에서 `safe_auto`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- default 런타임 context: ephemeral
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Where To Go Next

오케스트레이션 실행에서 Choose을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 Choose, Platform을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Architecture을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Zero-Config을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Compatibility, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Troubleshooting을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 오케스트레이션 실행 개요

오케스트레이션 실행에서 Move을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Prefect, Dagster을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Great Expectations, Truthound, Pandera, Great, Expectations, Truthound-first을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 dbt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
