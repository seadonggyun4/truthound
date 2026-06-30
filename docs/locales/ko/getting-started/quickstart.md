# 빠른 시작

초기 설정과 첫 실행에서 CLI, API, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## What You Will Build

- a zero-설정 검증 run
- 초기 설정과 첫 실행에서 `.truthound/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Python 워크플로우

### Step 1: Validate data with zero 설정

```python
import truthound as th

run = th.check(
    {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
)

print(run.execution_mode)
print(run.planned_execution_mode)
print([issue.issue_type for issue in run.issues])
print(run.metadata["context_root"])
```

초기 설정과 첫 실행에서 ValidationRunResult, Truthound, `th.check()`, `.truthound/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Step 2: Inspect the canonical 결과

```python
print(run.source)
print(run.execution_mode)
print(run.planned_execution_mode)
print([check.name for check in run.checks])
print([issue.issue_type for issue in run.issues])
print(run.metadata.get("context_run_artifact"))
```

초기 설정과 첫 실행에서 ValidationRunResult, `execution_mode`, `planned_execution_mode`, `planned_execution_mode="sequential"`, `execution_mode="threadpool"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 초기 설정과 첫 실행 개요

```python
import truthound as th

context = th.get_context()

print(context.workspace_dir)
print(context.baselines_dir)
print(context.docs_dir)
```

초기 설정과 첫 실행에서 TruthoundContext, Truthound을(를) 다루는 항목입니다:

- 초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## CLI 워크플로우

초기 설정과 첫 실행에서 Truthound, CLI, Python을(를) 다루는 항목입니다:

```bash
truthound check customers.csv
truthound profile customers.csv
truthound scan customers.csv
```

초기 설정과 첫 실행에서 CLI을(를) 다루는 항목입니다:

- 초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CI-friendly 검증 entry points
- 초기 설정과 첫 실행에서 Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

초기 설정과 첫 실행에서 CLI, Continue, Reference을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Move Beyond Zero-설정

### 초기 설정과 첫 실행 개요

```python
from truthound.core import ScanPlanner, ValidationRuntime, ValidationSuite, build_validation_asset
from truthound.context import TruthoundContext

data = {"id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]}
context = TruthoundContext.discover()
suite = ValidationSuite.from_legacy(context=context, validators=["null", "unique"], data=data)
asset = build_validation_asset(data)
plan = ScanPlanner().plan(suite=suite, asset=asset, parallel=True)
run = ValidationRuntime().execute(asset=asset, plan=plan)
```

초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Build docs and 리포트

```python
from truthound.datadocs import generate_validation_report
from truthound.reporters import get_reporter

html = generate_validation_report(run, title="Customer Quality Overview")
json_payload = get_reporter("json").render(run)
run.write("quality-report.json")
run.write("quality-report.html")
```

### Manage plugins

```bash
truthound plugins list
truthound plugins list --json
```

초기 설정과 첫 실행에서 `PluginManager`, `EnterprisePluginManager`, PluginManager, EnterprisePluginManager을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Common 문제 해결

### I do not see `.truthound/`

초기 설정과 첫 실행에서 `run.metadata`, Check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### I want stricter or more explicit 검증

초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- [첫 검증](first-validation.md)
- 초기 설정과 첫 실행에서 Validators, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Datasources, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [체크포인트 Guide](../guides/checkpoints.md)

### I am upgrading older 2.x code

초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```bash
truthound doctor . --migrate-2to3
truthound doctor . --workspace
```

초기 설정과 첫 실행에서 `truthound.compare`, `Report`, `report.validation_run`, `CheckpointResult.validation_result`, `.truthound/`, Report, CheckpointResult.validation_result을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Where To Go Next

- [첫 검증](first-validation.md)
- [튜토리얼](../tutorials/index.md)
- [아키텍처](../concepts/architecture.md)
- 초기 설정과 첫 실행에서 Zero-Config, Context을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [레퍼런스](../reference/index.md)
- [마이그레이션 to 3.0](../guides/migration-3.0.md)
