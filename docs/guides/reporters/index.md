# Truthound Reporters

Truthound 리포터 시스템은 검증 결과를 다양한 형식으로 출력합니다.

## 문서 구조

| 문서 | 설명 |
|------|------|
| [Console Reporter](console.md) | 터미널 출력 (Rich 기반) |
| [JSON & YAML](json-yaml.md) | JSON, YAML, NDJSON 형식 |
| [HTML & Markdown](html-markdown.md) | HTML, Markdown, Table 형식 |
| [CI/CD Reporters](ci-reporters.md) | GitHub Actions, GitLab CI, Jenkins 등 |
| [Reporter SDK](custom-sdk.md) | 커스텀 리포터 개발 SDK |

---

## 개요

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Reporter Factory                         │
│              get_reporter(format, **config)                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    JSON     │    │   Console   │    │  Markdown   │
    │  Reporter   │    │  Reporter   │    │  Reporter   │
    └─────────────┘    └─────────────┘    └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │    HTML     │
                       │  Reporter   │
                       └─────────────┘
```

### 주요 기능

- **다양한 형식**: JSON, Console, Markdown, HTML, YAML, CSV, JUnit XML
- **통합 인터페이스**: 모든 리포터에서 동일한 API
- **커스터마이징**: 제목, 테마, 템플릿 설정
- **확장성**: 런타임에 커스텀 리포터 등록

---

## Quick Start

### 기본 사용법

```python
from truthound.reporters import get_reporter
import truthound as th

# 검증 실행
result = th.check("data.csv")

# 리포터 생성
reporter = get_reporter("json")

# 문자열로 렌더링
json_output = reporter.render(result)

# 파일로 쓰기
reporter.write(result, "report.json")
```

### 사용 가능한 형식

| 형식 | 의존성 | 사용 사례 |
|------|--------|----------|
| `json` | (내장) | API 통합, 프로그래밍 접근 |
| `console` | rich | 터미널 출력, 디버깅 |
| `markdown` | (내장) | 문서화, GitHub/GitLab |
| `html` | jinja2 | 웹 대시보드, 이메일 리포트 |

### 형식 확인

```python
from truthound.reporters.factory import list_available_formats, is_format_available

# 사용 가능한 모든 형식 나열
print(list_available_formats())
# ['console', 'json', 'markdown', 'html']

# 특정 형식 확인
if is_format_available("html"):
    reporter = get_reporter("html")
```

---

## 리포터 비교

### 출력 형식 비교

| 리포터 | 출력 형식 | 파일 확장자 | 주요 용도 |
|--------|-----------|-------------|-----------|
| ConsoleReporter | 텍스트 (Rich) | - | 터미널 출력 |
| JSONReporter | JSON | `.json` | API, 자동화 |
| YAMLReporter | YAML | `.yaml` | 설정, 가독성 |
| MarkdownReporter | Markdown | `.md` | 문서화 |
| HTMLReporter | HTML | `.html` | 웹 리포트 |
| TableReporter | ASCII/Markdown | `.txt` | 간단한 테이블 |
| JUnitXMLReporter | XML | `.xml` | CI/CD 통합 |

### CI/CD 플랫폼 비교

| 플랫폼 | 리포터 | 주요 기능 |
|--------|--------|----------|
| GitHub Actions | `GitHubActionsReporter` | Annotations, Step Summaries |
| GitLab CI | `GitLabCIReporter` | Code Quality JSON, JUnit XML |
| Jenkins | `JenkinsReporter` | JUnit XML, warnings-ng JSON |
| Azure DevOps | `AzureDevOpsReporter` | VSO Commands, Variables |
| CircleCI | `CircleCIReporter` | Test Metadata, Artifacts |
| Bitbucket Pipelines | `BitbucketPipelinesReporter` | Reports, Annotations |

### SDK 템플릿 비교

| 템플릿 | 용도 | 특징 |
|--------|------|------|
| CSVReporter | 데이터 내보내기 | 스프레드시트 호환 |
| YAMLReporter | 설정 파일 | 사람이 읽기 쉬움 |
| JUnitXMLReporter | CI/CD | 테스트 프레임워크 호환 |
| NDJSONReporter | 로그 수집 | ELK, Splunk 통합 |
| TableReporter | 터미널 | 4가지 스타일 지원 |

---

## 공통 인터페이스

모든 리포터는 동일한 인터페이스를 구현합니다:

```python
class BaseReporter(Generic[ConfigT, InputT], ABC):
    name: str                   # 리포터 이름
    file_extension: str         # 기본 파일 확장자
    content_type: str           # MIME 타입

    def render(self, data: InputT) -> str:
        """결과를 문자열로 렌더링."""
        ...

    def write(self, data: InputT, path: str | Path | None = None) -> Path:
        """결과를 파일로 쓰기. 작성된 경로 반환."""
        ...

    def report(self, data: InputT, path: str | Path | None = None) -> str:
        """렌더링하고 선택적으로 파일로 쓰기. 렌더링된 문자열 반환."""
        ...
```

> **참고**: `ConsoleReporter`는 터미널 직접 출력을 위한 `print(data)` 메서드가 추가로 있습니다.

---

## 커스텀 리포터 등록

### 데코레이터 사용

```python
from truthound.reporters import register_reporter
from truthound.reporters.base import ValidationReporter, ReporterConfig

@register_reporter("my_custom")
class MyCustomReporter(ValidationReporter[ReporterConfig]):
    name = "my_custom"
    file_extension = ".custom"

    def render(self, data):
        return f"Custom: {data.status.value}"
```

### SDK 데코레이터 사용

```python
from truthound.reporters.sdk import create_reporter

@create_reporter("simple", extension=".txt")
def render_simple(result, config):
    return f"Status: {result.status.value}"
```

자세한 내용은 [Reporter SDK](custom-sdk.md) 문서를 참조하세요.

---

## 통합 예시

### GitHub Actions

```yaml
name: Data Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install truthound

      - name: Run validation
        run: |
          python -c "
          import truthound as th
          from truthound.reporters.ci import GitHubActionsReporter

          result = th.check('data/*.csv')
          reporter = GitHubActionsReporter()
          exit_code = reporter.report_to_ci(result)
          exit(exit_code)
          "
```

### FastAPI 엔드포인트

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from truthound.reporters import get_reporter
import truthound as th

app = FastAPI()

@app.post("/validate")
async def validate_data(file_path: str, format: str = "json"):
    result = th.check(file_path)
    reporter = get_reporter(format)
    content = reporter.render(result)

    if format == "html":
        return HTMLResponse(content=content)
    return JSONResponse(content=json.loads(content))
```

### Airflow 태스크

```python
from airflow.decorators import task
from truthound.reporters import get_reporter

@task
def validate_and_report(data_path: str, report_path: str):
    import truthound as th

    result = th.check(data_path)

    # HTML 리포트 생성
    reporter = get_reporter("html", title="Daily Data Quality")
    reporter.write(result, report_path)

    # 검증 실패 시 태스크 실패
    if not result.success:
        raise ValueError(f"Data quality check failed for {data_path}")

    return report_path
```

---

## 요약

Truthound 리포터는 유연한 출력 포맷팅을 제공합니다:

- **4개 내장 형식**: JSON, Console, Markdown, HTML
- **6개 CI/CD 리포터**: GitHub Actions, GitLab CI, Jenkins, Azure DevOps, CircleCI, Bitbucket
- **5개 SDK 템플릿**: CSV, YAML, JUnit XML, NDJSON, Table
- **6개 SDK Mixin**: Formatting, Aggregation, Filtering, Serialization, Templating, Streaming
- **스키마 검증**: JSON, XML, CSV 스키마 및 검증 유틸리티
- **테스트 유틸리티**: Mock 데이터, Assertion, 벤치마킹
- **통합 인터페이스**: 모든 리포터에서 동일한 API
- **확장성**: `@register_reporter`로 커스텀 리포터 등록

자세한 내용은 각 문서를 참조하세요:

- [Console Reporter](console.md) - 터미널 출력
- [JSON & YAML](json-yaml.md) - 구조화된 데이터 형식
- [HTML & Markdown](html-markdown.md) - 문서 및 웹 리포트
- [CI/CD Reporters](ci-reporters.md) - CI/CD 플랫폼 통합
- [Reporter SDK](custom-sdk.md) - 커스텀 리포터 개발
