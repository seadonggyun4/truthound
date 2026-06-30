# 마이그레이션 to Truthound 2.0

## What Stays the Same

- 실무 운영 가이드에서 `th.check()`, `th.scan()`, `th.mask()`, `th.profile()`, `th.learn()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 CLI, `truthound check`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `Report`, Report을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## What Changed

- 실무 운영 가이드에서 `truthound.core`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 ValidationRunResult, `report.validation_run`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- `PluginManager` is the single lifecycle 런타임
- 실무 운영 가이드에서 `EnterprisePluginManager`, EnterprisePluginManager을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `use_engine`, `--use-engine`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 CLI, `plugins`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Recommended Upgrade Path

### 1. Keep the Existing Facade

```python
import truthound as th

report = th.check(data, validators=["null", "unique"])
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 2. Start Reading Structured 결과

```python
run = report.validation_run
print(run.planned_execution_mode)
print(run.execution_mode)
print(run.to_dict())
```

### 3. Move Reporting and 검증 Docs to the 결과 Model

```python
from truthound.datadocs import generate_validation_report
from truthound.reporters import get_reporter

run = report.validation_run
json_report = get_reporter("json").render(run)
validation_docs = generate_validation_report(run)
```

실무 운영 가이드에서 `generate_html_report(report)`, `Report`, Legacy, Report, DTOs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 실무 운영 가이드 개요

```python
from truthound.core import ScanPlanner, ValidationRuntime, ValidationSuite, build_validation_asset

suite = ValidationSuite.from_legacy(validators=["null", "unique"])
asset = build_validation_asset(data)
plan = ScanPlanner().plan(suite=suite, asset=asset, parallel=True)
run = ValidationRuntime().execute(asset=asset, plan=plan)
```

### 5. Update Plugin Integrations

실무 운영 가이드에서 Prefer을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 `register_check_factory()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `register_data_asset_provider()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `register_reporter()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `register_hook()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 ValidationRunResult, Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Breaking Changes

- 실무 운영 가이드에서 `use_engine`, `th.check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `--use-engine`, `truthound check`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `truthound.stores.results.ValidationResult`, ValidationResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Recommended Reading

- [아키텍처](../concepts/architecture.md)
- [Plugin 플랫폼](../concepts/plugins.md)
- [Truthound 2.0 릴리스 노트](../releases/truthound-2.0.md)
