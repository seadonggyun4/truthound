# 리포터

Python API 사용에서 ValidationRunResult, Output을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Canonical Contract

Python API 사용에서 ValidationRunResult, Truthound, API, APIs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

- Python API 사용에서 ValidationRunResult, `th.check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `truthound.reporters`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `RunPresentation`, RunPresentation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `truthound.stores.results.ValidationResult`, ValidationResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Python API 사용 개요

Python API 사용에서 Truthound, Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| Python API 사용에서 Family을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Entry을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Primary을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|---------------|----------------|
| Built-in 검증 reporters | Python API 사용에서 `truthound.reporters.get_reporter(...)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 HTML, JSON, Markdown을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 CI-native을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `truthound.reporters.ci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Platform-specific을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `truthound.reporters.sdk.templates`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 JSON, YAML, CSV, NDJSON, JUnit, XML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 Quality을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `truthound.reporters.quality`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Rule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

Python API 사용에서 `truthound.reporters.quality`, `get_reporter(...)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
import truthound as th
from truthound.reporters import get_reporter
from truthound.reporters.sdk.templates import YAMLReporter

run_result = th.check("data.csv")

json_output = get_reporter("json").render(run_result)
markdown_output = get_reporter("markdown").render(run_result)
yaml_output = YAMLReporter().render(run_result)

get_reporter("html").write(run_result, "report.html")
```

## Built-in `get_reporter(...)` Registry

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| Python API 사용에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Notes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|----------|-------|
| Python API 사용에서 `console`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `ConsoleReporter`, ConsoleReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Rich을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 JSON, `JSONReporter`, JSONReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Machine-readable 검증 payload |
| Python API 사용에서 `markdown`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `MarkdownReporter`, MarkdownReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `html`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 HTML, `HTMLReporter`, HTMLReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Browser-friendly 검증 리포트 pages |
| Python API 사용에서 `ci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Selects을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `github`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `GitHubActionsReporter`, GitHubActionsReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 GitHub을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `gitlab`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `GitLabCIReporter`, GitLabCIReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 GitLab, CI/code-quality을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `jenkins`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `JenkinsReporter`, JenkinsReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Jenkins-oriented을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `azure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `AzureDevOpsReporter`, AzureDevOpsReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Azure, DevOps을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `circleci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `CircleCIReporter`, CircleCIReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 CircleCI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `bitbucket`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `BitbucketPipelinesReporter`, BitbucketPipelinesReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Bitbucket 파이프라인 summaries |

Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- Python API 사용에서 `truthound.reporters.sdk.templates`, SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `truthound.reporters.sdk`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `truthound.reporters.quality`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Shared Input Model

Python API 사용에서 ValidationRunResult을(를) 다루는 항목입니다:

```python
import truthound as th
from truthound.reporters import get_reporter

run_result = th.check("data.csv")

print(run_result.run_id)
print(len(run_result.checks))
print(len(run_result.issues))

reporter = get_reporter("json")
print(reporter.render(run_result))
```

Python API 사용에서 `RunPresentation`, Under, RunPresentation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 HTML, JSON, Markdown을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Built-in 검증 리포터

### `ConsoleReporter`

```python
from truthound.reporters.console_reporter import ConsoleReporter

reporter = ConsoleReporter(color=True, compact=False)
print(reporter.render(run_result))
```

### `JSONReporter`

```python
from truthound.reporters.json_reporter import JSONReporter

reporter = JSONReporter(indent=2, include_null_values=True)
print(reporter.render(run_result))
```

Python API 사용에서 Representative을(를) 다루는 항목입니다:

```json
{
  "report_type": "validation",
  "generated_at": "2026-03-21 20:41:09",
  "title": "Truthound Validation Report",
  "result": {
    "run_id": "run_20260321_204109_2770ac12",
    "run_time": "2026-03-21T20:41:09.672591",
    "data_asset": "dict",
    "status": "failure",
    "success": false
  },
  "statistics": {
    "total_validators": 5,
    "passed_validators": 3,
    "failed_validators": 2
  },
  "issues": [
    {
      "validator": "unique",
      "column": "id",
      "issue_type": "unique_violation",
      "severity": "critical"
    }
  ]
}
```

### `MarkdownReporter`

```python
from truthound.reporters.markdown_reporter import MarkdownReporter

reporter = MarkdownReporter(include_toc=True, include_badges=True)
markdown = reporter.render(run_result)
```

### `HTMLReporter`

```python
from truthound.reporters.html_reporter import HTMLReporter

reporter = HTMLReporter(title="Customer Data Quality")
reporter.write(run_result, "report.html")
```

Python API 사용에서 HTML, `HTMLReporter`, `jinja2`, HTMLReporter, Install을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## CI-native 리포터

Python API 사용에서 `truthound.reporters.ci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.reporters.ci import GitHubActionsReporter

reporter = GitHubActionsReporter()
exit_code = reporter.report_to_ci(run_result)
```

Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 `LegacyValidationResultView`, LegacyValidationResultView을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SDK Templates And Custom 리포터

Python API 사용에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 `truthound.reporters.sdk.templates`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 `get_reporter(...)`을(를) 다루는 항목입니다:

| Python API 사용에서 Class을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Import을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Typical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|-------------|
| Python API 사용에서 YAML, `YAMLReporter`, YAMLReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `truthound.reporters.sdk.templates`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Human-readable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `CSVReporter`, CSVReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `truthound.reporters.sdk.templates`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Spreadsheet을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 JSON, `NDJSONReporter`, NDJSONReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `truthound.reporters.sdk.templates`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Log/event 파이프라인 |
| Python API 사용에서 `JUnitXMLReporter`, JUnitXMLReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `truthound.reporters.sdk.templates`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Test-리포트 통합 |
| Python API 사용에서 `TableReporter`, TableReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `truthound.reporters.sdk.templates`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Plain text or compact 테이블 |

```python
from truthound.reporters.sdk.templates import JUnitXMLReporter, YAMLReporter

yaml_output = YAMLReporter().render(run_result)
xml_output = JUnitXMLReporter().render(run_result)
```

Python API 사용에서 `ValidationReporter`, ValidationReporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
canonicalization and presentation 플로우.

```python
from truthound.core.results import ValidationRunResult
from truthound.reporters.base import ReporterConfig, ValidationReporter


class CSVReporter(ValidationReporter[ReporterConfig]):
    name = "csv"
    file_extension = ".csv"
    content_type = "text/csv"

    @classmethod
    def _default_config(cls) -> ReporterConfig:
        return ReporterConfig()

    def render(self, data: ValidationRunResult) -> str:
        presentation = self.present(data)
        lines = ["validator,column,issue_type,severity,count"]
        for issue in presentation.issues:
            lines.append(
                f"{issue.validator_name},{issue.column},{issue.issue_type},{issue.severity},{issue.count}"
            )
        return "\n".join(lines)
```

Python API 사용에서 Register을(를) 다루는 항목입니다:

```python
from truthound.reporters import register_reporter

register_reporter("csv", CSVReporter)
```

## Python API 사용 개요

Python API 사용에서 `truthound.reporters.quality`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 `get_quality_reporter(...)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.reporters.quality import get_quality_reporter

reporter = get_quality_reporter("console")
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 `th.check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 호환성 Notes

- Python API 사용에서 `truthound.stores.results.ValidationResult`, Passing, ValidationResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 ValidationRunResult, DTOs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `Report`, Legacy, Report을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `Report`, Report, Migration을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Reading

- Python API 사용에서 Core, Functions을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [가이드: 리포터](../guides/reporters/index.md)
- Python API 사용에서 Guides, Reporter, SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Guides, Quality, Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Guides, DataDocs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [마이그레이션 to 3.0](../guides/migration-3.0.md)
