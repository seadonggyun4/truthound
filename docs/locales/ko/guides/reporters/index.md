# 리포터 Guide

실무 운영 가이드에서 ValidationRunResult, Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 `th.check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
import truthound as th
from truthound.reporters import get_reporter

run_result = th.check("data.csv")

# Built-in factory reporters
json_output = get_reporter("json").render(run_result)
markdown_output = get_reporter("markdown").render(run_result)

# Write directly to disk
get_reporter("html").write(run_result, "reports/data-quality.html")
```

## Reporter Families

실무 운영 가이드에서 Truthound, They을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Family을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Primary을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Notes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------------|---------------|-------|
| Built-in 검증 reporters | 실무 운영 가이드에서 `get_reporter(...)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `console`, `json`, `markdown`, `html`, Thin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 CI-native을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `truthound.reporters.ci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Provider-specific을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `truthound.reporters.sdk`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, YAML, YAML/CSV/NDJSON/JUnit/table을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Quality을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `truthound.reporters.quality`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Separate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 `get_quality_reporter(...)`, `get_reporter(...)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
리포트.

## Choose A Path

### Use built-in 검증 reporters

실무 운영 가이드에서 `get_reporter(...)`을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 `console`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `markdown`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `html`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `ci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `github`, `gitlab`, `jenkins`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Use CI-native reporters

실무 운영 가이드에서 `truthound.reporters.ci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 JSON, JSON/Markdown을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 ValidationRunResult, Public을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 `LegacyValidationResultView`, LegacyValidationResultView을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Use SDK templates

실무 운영 가이드에서 SDK을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 YAML, `YAMLReporter`, YAMLReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `CSVReporter`, CSVReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 JSON, `NDJSONReporter`, NDJSONReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `JUnitXMLReporter`, JUnitXMLReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `TableReporter`, TableReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 `truthound.reporters.sdk.templates`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 `get_reporter(...)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Build a custom reporter

실무 운영 가이드에서 `ValidationReporter`, ValidationReporter, SDK을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 `RunPresentation`, `to_legacy_view()`, RunPresentation, Call을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Use quality reporters

실무 운영 가이드에서 `truthound.reporters.quality`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 CLI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Input Model

실무 운영 가이드에서 ValidationRunResult, Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 Reporters을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
import truthound as th

run_result = th.check("data.csv")

print(run_result.success)
print(len(run_result.checks))
print(len(run_result.issues))
print(run_result.source)
print(run_result.run_id)
```

실무 운영 가이드에서 Common을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 `issue.validator_name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `issue.issue_type`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `issue.column`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `issue.severity`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `issue.message`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `issue.count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Built-in Reporter Registry

### Standard output

```python
from truthound.reporters import get_reporter

reporter = get_reporter("console")
print(reporter.render(run_result))
```

### 실무 운영 가이드 개요

```python
from truthound.reporters import get_reporter

for fmt in ["json", "markdown", "html"]:
    reporter = get_reporter(fmt)
    reporter.write(run_result, f"reports/result{reporter.file_extension}")
```

### CI-native output

```python
from truthound.reporters import get_reporter

github_reporter = get_reporter("github")
exit_code = github_reporter.report_to_ci(run_result)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 `truthound.reporters.ci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SDK Template 리포터

실무 운영 가이드에서 SDK을(를) 다루는 항목입니다:

```python
from truthound.reporters.sdk.templates import (
    CSVReporter,
    JUnitXMLReporter,
    NDJSONReporter,
    TableReporter,
    YAMLReporter,
)

yaml_output = YAMLReporter().render(run_result)
xml_output = JUnitXMLReporter().render(run_result)
```

실무 운영 가이드에서 Recommended을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|
| 실무 운영 가이드에서 YAML, `YAMLReporter`, YAMLReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Human-readable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CSVReporter`, CSVReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Spreadsheet을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 JSON, `NDJSONReporter`, NDJSONReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Event/log 파이프라인 |
| 실무 운영 가이드에서 `JUnitXMLReporter`, JUnitXMLReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Test-리포트 ingestion |
| 실무 운영 가이드에서 `TableReporter`, TableReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Compact text 테이블 |

## Common 워크플로우s

### 실무 운영 가이드 개요

```python
from truthound.reporters import get_reporter

json_reporter = get_reporter("json")
markdown_reporter = get_reporter("markdown")

json_reporter.write(run_result, "artifacts/validation.json")
markdown_reporter.write(run_result, "artifacts/validation.md")
```

### HTML 리포트 with graceful dependency expectations

```python
from truthound.reporters import get_reporter

html_reporter = get_reporter("html", title="Customer Quality Report")
html_reporter.write(run_result, "site/customer-quality.html")
```

실무 운영 가이드에서 `html`, `jinja2`, Install을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 HTML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Custom reporter registration

```python
from truthound.reporters.base import ReporterConfig, ValidationReporter
from truthound.reporters.factory import register_reporter


@register_reporter("severity-summary")
class SeveritySummaryReporter(ValidationReporter[ReporterConfig]):
    name = "severity-summary"
    file_extension = ".txt"
    content_type = "text/plain"

    @classmethod
    def _default_config(cls) -> ReporterConfig:
        return ReporterConfig()

    def render(self, data) -> str:
        presentation = self.present(data)
        counts = presentation.issue_counts_by_severity
        return "\n".join(
            [
                f"status={presentation.status}",
                f"critical={counts.get('critical', 0)}",
                f"high={counts.get('high', 0)}",
                f"medium={counts.get('medium', 0)}",
                f"low={counts.get('low', 0)}",
            ]
        )
```

## Reporter Families

| 실무 운영 가이드에서 Family을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Primary을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------------|------------|
| 실무 운영 가이드에서 Console을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Console, Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 JSON, YAML, JSON/YAML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, YAML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 HTML, HTML/Markdown을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | docs, 아티팩트, static publishing | 실무 운영 가이드에서 HTML, Markdown을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 CI/CD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | [CI/CD 리포터](ci-reporters.md) |
| 실무 운영 가이드에서 SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Reporter, SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Quality을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Quality, Reporter, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 문제 해결

### 실무 운영 가이드 개요

실무 운영 가이드에서 SDK을(를) 다루는 항목입니다:

```python
from truthound.reporters.sdk.templates import YAMLReporter, JUnitXMLReporter
```

실무 운영 가이드에서 SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### I have a legacy 리포트-like object

실무 운영 가이드에서 ValidationRunResult, Current, Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
the run 결과 model.

### I want richer access than `run_result.issues`

실무 운영 가이드에서 Inside을(를) 다루는 항목입니다:

```python
presentation = self.present(run_result)
legacy_view = presentation.to_legacy_view()
```

실무 운영 가이드에서 `presentation`, `legacy_view`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
결과 rows.

## 함께 보기

- 실무 운영 가이드에서 API, Python, Reporters을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Console, Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 JSON, YAML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 HTML, Markdown을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [CI/CD 리포터](ci-reporters.md)
- 실무 운영 가이드에서 Reporter, SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Quality, Reporter, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
