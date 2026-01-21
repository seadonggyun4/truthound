# HTML & Markdown Reporters

HTML과 Markdown 형식으로 검증 보고서를 생성하는 리포터입니다.

## HTML Reporter

### 기본 사용법

```python
from truthound.reporters import get_reporter

reporter = get_reporter("html")
html_output = reporter.render(validation_result)
```

### 설정 옵션

`HTMLReporterConfig`는 다음 옵션을 제공합니다:

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `template` | `str \| None` | `None` | 커스텀 Jinja2 템플릿 |
| `template_path` | `str \| None` | `None` | 템플릿 파일 경로 |
| `inline_css` | `bool` | `True` | CSS를 HTML에 인라인 |
| `include_js` | `bool` | `True` | JavaScript 포함 |
| `theme` | `str` | `"light"` | 테마 ("light", "dark") |
| `include_charts` | `bool` | `True` | 차트 포함 |
| `responsive` | `bool` | `True` | 반응형 디자인 |

### 사용 예시

#### 기본 HTML 보고서

```python
from truthound.reporters import get_reporter

reporter = get_reporter("html")
html_output = reporter.render(result)

# 파일로 저장
reporter.write(result, "validation_report.html")
```

#### 다크 테마

```python
reporter = get_reporter("html", theme="dark")
html_output = reporter.render(result)
```

#### 커스텀 템플릿

```python
custom_template = """
<!DOCTYPE html>
<html>
<head><title>{{ title }}</title></head>
<body>
  <h1>{{ result.data_asset }} Validation</h1>
  <p>Status: {{ result.status.value }}</p>
  <ul>
  {% for r in result.results if not r.success %}
    <li>{{ r.validator_name }}: {{ r.message }}</li>
  {% endfor %}
  </ul>
</body>
</html>
"""

reporter = get_reporter("html", template=custom_template)
html_output = reporter.render(result)
```

#### 템플릿 파일 사용

```python
reporter = get_reporter("html", template_path="templates/my_report.html")
html_output = reporter.render(result)
```

### 기본 템플릿 기능

기본 HTML 템플릿은 다음을 포함합니다:

1. **요약 섹션**: 전체 통계 및 상태
2. **심각도별 분석**: 도넛 차트 (ApexCharts)
3. **이슈 테이블**: 정렬/필터 가능한 테이블
4. **상세 정보**: 각 validator 결과의 세부 정보
5. **반응형 디자인**: 모바일 최적화

### 템플릿 컨텍스트 변수

커스텀 템플릿에서 사용 가능한 변수:

| 변수 | 타입 | 설명 |
|------|------|------|
| `result` | `ValidationResult` | 검증 결과 객체 |
| `title` | `str` | 보고서 제목 |
| `timestamp` | `str` | 생성 시간 |
| `config` | `HTMLReporterConfig` | 리포터 설정 |
| `statistics` | `dict` | 통계 정보 |

### 의존성

HTML Reporter는 Jinja2 템플릿 엔진을 사용합니다:

```bash
pip install jinja2
```

---

## Markdown Reporter

### 기본 사용법

```python
from truthound.reporters import get_reporter

reporter = get_reporter("markdown")
md_output = reporter.render(validation_result)
```

### 설정 옵션

`MarkdownReporterConfig`는 다음 옵션을 제공합니다:

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `include_toc` | `bool` | `True` | 목차 포함 |
| `heading_level` | `int` | `1` | 시작 헤딩 레벨 (1-6) |
| `include_badges` | `bool` | `True` | shields.io 배지 포함 |
| `table_style` | `str` | `"github"` | 테이블 스타일 |
| `include_details` | `bool` | `True` | 상세 섹션 포함 |
| `include_statistics` | `bool` | `True` | 통계 섹션 포함 |

### 사용 예시

#### 기본 Markdown

```python
from truthound.reporters import get_reporter

reporter = get_reporter("markdown")
md_output = reporter.render(result)
```

출력 예시:
```markdown
# Validation Report

![Status](https://img.shields.io/badge/Status-FAILED-red)
![Pass Rate](https://img.shields.io/badge/Pass%20Rate-80%25-yellow)

## Summary

| Metric | Value |
|--------|-------|
| Data Asset | customer_data.csv |
| Run ID | abc123-def456 |
| Status | FAILED |
| Total Validators | 10 |
| Passed | 8 |
| Failed | 2 |
| Pass Rate | 80.0% |

## Issues by Severity

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 1 |
| Medium | 0 |
| Low | 0 |

## Failed Validations

### NullValidator - email

- **Severity**: critical
- **Count**: 5
- **Message**: Found 5 null values

### RangeValidator - age

- **Severity**: high
- **Count**: 3
- **Message**: 3 values out of range
```

#### 배지 비활성화

```python
reporter = get_reporter("markdown", include_badges=False)
```

#### 목차 비활성화

```python
reporter = get_reporter("markdown", include_toc=False)
```

#### 헤딩 레벨 조정

문서에 포함할 때 헤딩 레벨을 조정할 수 있습니다:

```python
# h2부터 시작
reporter = get_reporter("markdown", heading_level=2)
```

### shields.io 배지

`include_badges=True`일 때 다음 배지가 생성됩니다:

- **Status 배지**: 통과(녹색) / 실패(빨간색)
- **Pass Rate 배지**: 비율에 따른 색상
- **Issues 배지**: 이슈 수 표시

```markdown
![Status](https://img.shields.io/badge/Status-PASSED-green)
![Pass Rate](https://img.shields.io/badge/Pass%20Rate-100%25-brightgreen)
![Issues](https://img.shields.io/badge/Issues-0-brightgreen)
```

---

## Table Reporter (SDK)

SDK에서 제공하는 ASCII/Unicode 테이블 리포터입니다.

### 기본 사용법

```python
from truthound.reporters.sdk import TableReporter

reporter = TableReporter()
table_output = reporter.render(validation_result)
```

### 설정 옵션

`TableReporterConfig`는 다음 옵션을 제공합니다:

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `style` | `str` | `"ascii"` | 테이블 스타일 |
| `include_passed` | `bool` | `False` | 통과한 validator 포함 |
| `max_column_width` | `int` | `50` | 최대 컬럼 너비 |
| `columns` | `list[str]` | `["validator", "column", "severity", "message"]` | 표시할 컬럼 |
| `sort_by` | `str \| None` | `"severity"` | 정렬 기준 |
| `sort_ascending` | `bool` | `False` | 오름차순 정렬 |

### 테이블 스타일

#### ASCII (기본)
```
+------------+--------+----------+---------------------------+
| validator  | column | severity | message                   |
+------------+--------+----------+---------------------------+
| NullValidator | email | critical | Found 5 null values    |
| RangeValidator | age | high     | 3 values out of range  |
+------------+--------+----------+---------------------------+
```

#### Markdown
```
| validator  | column | severity | message                   |
|------------|--------|----------|---------------------------|
| NullValidator | email | critical | Found 5 null values    |
| RangeValidator | age | high     | 3 values out of range  |
```

#### Grid (Unicode)
```
╔════════════╤════════╤══════════╤═══════════════════════════╗
║ validator  │ column │ severity │ message                   ║
╠════════════╪════════╪══════════╪═══════════════════════════╣
║ NullValidator │ email │ critical │ Found 5 null values   ║
║ RangeValidator │ age │ high     │ 3 values out of range ║
╚════════════╧════════╧══════════╧═══════════════════════════╝
```

#### Simple
```
validator       column   severity   message
-----------     ------   --------   ---------------------------
NullValidator   email    critical   Found 5 null values
RangeValidator  age      high       3 values out of range
```

---

## 파일 출력

```python
# HTML 파일 저장
html_reporter = get_reporter("html")
html_reporter.write(result, "report.html")

# Markdown 파일 저장
md_reporter = get_reporter("markdown")
md_reporter.write(result, "VALIDATION_REPORT.md")
```

## API 레퍼런스

### HTMLReporter

```python
class HTMLReporter(ValidationReporter[HTMLReporterConfig]):
    """HTML 형식 리포터 (Jinja2 기반)."""

    name = "html"
    file_extension = ".html"
    content_type = "text/html"

    def render(self, data: ValidationResult) -> str:
        """검증 결과를 HTML로 렌더링."""
        ...
```

### MarkdownReporter

```python
class MarkdownReporter(ValidationReporter[MarkdownReporterConfig]):
    """Markdown 형식 리포터."""

    name = "markdown"
    file_extension = ".md"
    content_type = "text/markdown"

    def render(self, data: ValidationResult) -> str:
        """검증 결과를 Markdown으로 렌더링."""
        ...
```

### TableReporter

```python
class TableReporter(ValidationReporter[TableReporterConfig]):
    """ASCII/Unicode 테이블 리포터."""

    name = "table"
    file_extension = ".txt"
    content_type = "text/plain"

    def render(self, data: ValidationResult) -> str:
        """검증 결과를 테이블로 렌더링."""
        ...
```
