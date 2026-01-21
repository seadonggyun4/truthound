# Reporter SDK

Truthound는 커스텀 리포터 개발을 위한 종합 SDK를 제공합니다.

## 개요

Reporter SDK는 다음 기능을 제공합니다:

- **Mixins**: 공통 기능 재사용 (포맷팅, 집계, 필터링 등)
- **Builder**: 데코레이터 및 빌더 패턴으로 리포터 생성
- **Templates**: 사전 정의된 리포터 템플릿 (CSV, YAML, JUnit 등)
- **Schema**: 출력 형식 검증
- **Testing**: 테스트 유틸리티 및 목 데이터 생성

---

## Quick Start

### 데코레이터로 간단한 리포터 생성

```python
from truthound.reporters.sdk import create_reporter

@create_reporter("my_format", extension=".myf")
def render_my_format(result, config):
    return f"Status: {result.status.value}"

# 사용
from truthound.reporters import get_reporter
reporter = get_reporter("my_format")
output = reporter.render(validation_result)
```

### Mixin을 활용한 전체 리포터

```python
from truthound.reporters.sdk import (
    FormattingMixin,
    AggregationMixin,
    FilteringMixin,
)
from truthound.reporters.base import ValidationReporter, ReporterConfig

class MyReporterConfig(ReporterConfig):
    custom_option: str = "default"

class MyReporter(FormattingMixin, AggregationMixin, FilteringMixin, ValidationReporter[MyReporterConfig]):
    name = "my_format"
    file_extension = ".myf"

    @classmethod
    def _default_config(cls):
        return MyReporterConfig()

    def render(self, data):
        # Mixin 메서드 사용
        issues = self.filter_by_severity(data, min_severity="medium")
        grouped = self.group_by_column(issues)
        return self.format_as_table(grouped)
```

---

## Mixins

SDK는 6개의 Mixin을 제공합니다.

### FormattingMixin

출력 포맷팅 유틸리티:

```python
from truthound.reporters.sdk import FormattingMixin

class MyReporter(FormattingMixin, ValidationReporter):
    def render(self, data):
        # 테이블 포맷팅 (ascii, markdown, grid, simple 스타일)
        rows = [{"name": r.column, "message": r.message} for r in data.results]
        table = self.format_as_table(rows, style="markdown")

        # 숫자 포맷팅
        rate = self.format_percentage(data.statistics.pass_rate)

        # 날짜 포맷팅
        date = self.format_datetime(data.run_time)

        # 바이트 크기 포맷팅
        size = self.format_bytes(1024000)  # "1000.0 KB"

        return f"{table}\nPass Rate: {rate}"
```

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `format_as_table(rows, columns, style)` | 데이터를 테이블 형식으로 포맷 (style: ascii/markdown/grid/simple) |
| `format_percentage(value, precision)` | 백분율 포맷 (예: "85.5%") |
| `format_number(value, precision)` | 숫자 포맷 (천 단위 구분) |
| `format_datetime(dt, format)` | 날짜/시간 포맷 |
| `format_duration(seconds)` | 실행 시간 포맷 (예: "2h 30m 15s") |
| `format_bytes(size)` | 바이트 크기 포맷 (예: "1.5 MB") |
| `format_relative_time(dt)` | 상대 시간 포맷 (예: "5 minutes ago") |
| `truncate(text, max_length, suffix)` | 텍스트 길이 제한 |
| `indent(text, prefix)` | 텍스트 들여쓰기 |
| `wrap(text, width)` | 텍스트 줄 바꿈 |

### AggregationMixin

데이터 집계 유틸리티:

```python
from truthound.reporters.sdk import AggregationMixin

class MyReporter(AggregationMixin, ValidationReporter):
    def render(self, data):
        # 컬럼별 그룹화
        by_column = self.group_by_column(data.results)

        # severity별 그룹화
        by_severity = self.group_by_severity(data.results)

        # validator별 그룹화
        by_validator = self.group_by_validator(data.results)

        # 통계 계산
        stats = self.get_summary_stats(data)

        return self.format_groups(by_severity)
```

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `group_by_column(results)` | 컬럼별 결과 그룹화 |
| `group_by_severity(results)` | severity별 결과 그룹화 |
| `group_by_validator(results)` | validator별 결과 그룹화 |
| `group_by(results, key)` | 커스텀 키 함수로 그룹화 |
| `get_summary_stats(result)` | 통계 계산 (pass_rate, counts 등) |
| `count_by_severity(results)` | severity별 카운트 |
| `count_by_column(results)` | 컬럼별 카운트 |

### FilteringMixin

데이터 필터링 유틸리티:

```python
from truthound.reporters.sdk import FilteringMixin

class MyReporter(FilteringMixin, ValidationReporter):
    def render(self, data):
        # severity로 필터링
        critical = self.filter_by_severity(data.results, min_severity="critical")

        # 실패한 항목만
        failed = self.filter_failed(data.results)

        # 특정 컬럼만
        email_issues = self.filter_by_column(data.results, include_columns=["email"])

        # 특정 validator만
        null_issues = self.filter_by_validator(data.results, include_validators=["NullValidator"])

        # severity로 정렬
        sorted_results = self.sort_by_severity(failed)

        return self.format_issues(sorted_results)
```

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `filter_by_severity(results, min_severity, max_severity)` | severity 범위로 필터링 |
| `filter_failed(results)` | 실패한 결과만 필터링 |
| `filter_passed(results)` | 통과한 결과만 필터링 |
| `filter_by_column(results, include_columns, exclude_columns)` | 특정 컬럼 결과 필터링 |
| `filter_by_validator(results, include_validators, exclude_validators)` | 특정 validator 결과 필터링 |
| `sort_by_severity(results, ascending)` | severity로 정렬 |
| `sort_by_column(results, ascending)` | 컬럼명으로 정렬 |
| `limit(results, count, offset)` | 결과 수 제한 |

### SerializationMixin

직렬화 유틸리티:

```python
from truthound.reporters.sdk import SerializationMixin

class MyReporter(SerializationMixin, ValidationReporter):
    def render(self, data):
        # JSON 문자열로
        as_json = self.to_json(data, indent=2)

        # CSV 문자열로
        rows = [{"name": "col1", "count": 10}]
        as_csv = self.to_csv(rows, columns=["name", "count"])

        # XML 요소 생성
        xml_elem = self.to_xml_element(
            "issue",
            value="message",
            attributes={"severity": "high"}
        )

        return as_json
```

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `to_json(data, indent, sort_keys)` | JSON 문자열로 직렬화 |
| `to_csv(rows, columns, delimiter)` | CSV 문자열로 직렬화 |
| `to_xml_element(tag, value, attributes)` | XML 요소 문자열 생성 |

### TemplatingMixin

템플릿 렌더링 유틸리티:

```python
from truthound.reporters.sdk import TemplatingMixin

class MyReporter(TemplatingMixin, ValidationReporter):
    template_string = """
    Report: {{ data.data_asset }}
    Status: {{ data.status.value }}
    {% for issue in data.issues %}
    - {{ issue.message }}
    {% endfor %}
    """

    def render(self, data):
        return self.render_template(self.template_string, data=data)
```

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `render_template(template, context)` | Jinja2 템플릿 렌더링 |
| `render_template_file(path, context)` | 파일 기반 템플릿 렌더링 |
| `interpolate(template, context)` | 단순 문자열 보간 (Jinja2 불필요) |

### StreamingMixin

스트리밍 출력 유틸리티:

```python
from truthound.reporters.sdk import StreamingMixin

class MyReporter(StreamingMixin, ValidationReporter):
    def render(self, data):
        # 청크 단위로 생성
        for chunk in self.stream_results(data.results, chunk_size=100):
            yield self.format_chunk(chunk)

    def render_lines(self, data):
        # 라인 단위 스트리밍
        formatter = lambda r: f"{r.validator_name}: {r.message}"
        return self.render_streaming(data.results, formatter)
```

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `stream_results(results, chunk_size)` | 청크 단위 이터레이터 |
| `stream_lines(results, formatter)` | 라인 단위 이터레이터 |
| `render_streaming(results, formatter)` | 스트리밍 결과를 문자열로 조합 |

---

## Builder

### @create_reporter 데코레이터

함수를 리포터로 변환:

```python
from truthound.reporters.sdk import create_reporter

@create_reporter(
    name="simple",
    extension=".txt",
    content_type="text/plain"
)
def render_simple(result, config):
    """간단한 텍스트 리포터."""
    lines = [
        f"Data Asset: {result.data_asset}",
        f"Status: {result.status.value}",
        f"Pass Rate: {result.pass_rate * 100:.1f}%",
    ]
    return "\n".join(lines)

# 자동으로 등록됨
from truthound.reporters import get_reporter
reporter = get_reporter("simple")
```

### @create_validation_reporter 데코레이터

ValidationReporter 전체 기능 포함:

```python
from truthound.reporters.sdk import create_validation_reporter
from truthound.reporters.base import ReporterConfig

class MyConfig(ReporterConfig):
    prefix: str = ">"
    include_timestamp: bool = True

@create_validation_reporter(
    name="prefixed",
    extension=".txt",
    config_class=MyConfig
)
def render_prefixed(result, config):
    lines = []
    if config.include_timestamp:
        lines.append(f"{config.prefix} Time: {result.run_time}")
    lines.append(f"{config.prefix} Status: {result.status.value}")
    return "\n".join(lines)
```

### ReporterBuilder

Fluent 빌더 패턴:

```python
from truthound.reporters.sdk import ReporterBuilder

# ReporterBuilder는 이름을 생성자에서 받음
reporter_class = (
    ReporterBuilder("custom")
    .with_extension(".custom")
    .with_content_type("text/plain")
    .with_mixin(FormattingMixin)
    .with_mixin(FilteringMixin)
    .with_renderer(lambda self, data: f"Status: {data.status.value}")
    .build()
)

# 인스턴스 생성
instance = reporter_class()
output = instance.render(validation_result)
```

**Builder 메서드:**

| 메서드 | 설명 |
|--------|------|
| `ReporterBuilder(name)` | 리포터 이름으로 빌더 생성 |
| `with_extension(ext)` | 파일 확장자 설정 |
| `with_content_type(type)` | MIME 타입 설정 |
| `with_mixin(mixin_class)` | Mixin 추가 |
| `with_mixins(*mixins)` | 여러 Mixin 추가 |
| `with_config_class(cls)` | 설정 클래스 지정 |
| `with_renderer(func)` | 렌더링 함수 지정 (self, data를 인자로 받음) |
| `with_post_processor(func)` | 후처리 함수 추가 |
| `with_attribute(name, value)` | 클래스 속성 추가 |
| `register_as(name)` | 팩토리 등록 이름 지정 |
| `build()` | 리포터 클래스 생성 |

---

## Templates

SDK는 사전 정의된 리포터 템플릿을 제공합니다.

### CSVReporter

```python
from truthound.reporters.sdk import CSVReporter

reporter = CSVReporter(
    delimiter=",",
    include_header=True,
    include_passed=False,
    quoting="minimal"  # minimal, all, none, nonnumeric
)
csv_output = reporter.render(result)
```

**CSVReporterConfig 옵션:**

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `delimiter` | `str` | `","` | 필드 구분자 |
| `include_header` | `bool` | `True` | 헤더 행 포함 |
| `include_passed` | `bool` | `False` | 통과 항목 포함 |
| `quoting` | `str` | `"minimal"` | 인용 방식 |
| `columns` | `list[str]` | `None` | 포함할 컬럼 (None=전체) |

### YAMLReporter

```python
from truthound.reporters.sdk import YAMLReporter

reporter = YAMLReporter(
    default_flow_style=False,
    indent=2,
    include_passed=False,
    sort_keys=False
)
yaml_output = reporter.render(result)
```

**YAMLReporterConfig 옵션:**

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `default_flow_style` | `bool` | `False` | 플로우 스타일 사용 |
| `indent` | `int` | `2` | 들여쓰기 크기 |
| `include_passed` | `bool` | `False` | 통과 항목 포함 |
| `sort_keys` | `bool` | `False` | 키 정렬 |

### JUnitXMLReporter

CI/CD 통합용 JUnit XML 형식:

```python
from truthound.reporters.sdk import JUnitXMLReporter

reporter = JUnitXMLReporter(
    testsuite_name="Truthound Validation",
    include_stdout=True,
    include_properties=True
)
xml_output = reporter.render(result)
```

**JUnitXMLReporterConfig 옵션:**

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `testsuite_name` | `str` | `"Truthound Validation"` | 테스트 스위트 이름 |
| `include_stdout` | `bool` | `True` | system-out 포함 |
| `include_properties` | `bool` | `True` | properties 포함 |
| `include_passed` | `bool` | `False` | 통과 테스트 포함 |

### NDJSONReporter

Newline Delimited JSON (로그 수집 시스템 통합용):

```python
from truthound.reporters.sdk import NDJSONReporter

reporter = NDJSONReporter(
    include_metadata=True,
    include_passed=False,
    compact=True
)
ndjson_output = reporter.render(result)
```

**NDJSONReporterConfig 옵션:**

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `include_metadata` | `bool` | `True` | 메타데이터 라인 포함 |
| `include_passed` | `bool` | `False` | 통과 항목 포함 |
| `compact` | `bool` | `True` | 컴팩트 JSON |

### TableReporter

텍스트 테이블 출력:

```python
from truthound.reporters.sdk import TableReporter

reporter = TableReporter(
    style="grid",  # ascii, markdown, grid, simple
    max_width=120,
    include_passed=False
)
table_output = reporter.render(result)
```

**TableReporterConfig 옵션:**

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `style` | `str` | `"ascii"` | 테이블 스타일 |
| `max_width` | `int` | `120` | 최대 너비 |
| `include_passed` | `bool` | `False` | 통과 항목 포함 |
| `show_index` | `bool` | `False` | 인덱스 표시 |

---

## Schema Validation

출력 형식 검증을 위한 스키마 시스템입니다.

### 기본 사용법

```python
from truthound.reporters.sdk import validate_output, JSONSchema

# 스키마 정의
schema = JSONSchema(
    required_fields=["status", "data_asset", "issues"],
    field_types={
        "status": str,
        "data_asset": str,
        "issues": list,
        "pass_rate": float,
    }
)

# 출력 검증
result = validate_output(json_output, schema)
if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error.message}")
```

### 스키마 타입

#### JSONSchema

```python
from truthound.reporters.sdk import JSONSchema

schema = JSONSchema(
    required_fields=["status", "data_asset"],
    field_types={"status": str, "issues": list},
    allow_extra_fields=True,
    max_depth=10
)
```

**옵션:**

| 옵션 | 타입 | 설명 |
|------|------|------|
| `required_fields` | `list[str]` | 필수 필드 목록 |
| `field_types` | `dict[str, type]` | 필드별 타입 |
| `allow_extra_fields` | `bool` | 추가 필드 허용 |
| `max_depth` | `int` | 최대 중첩 깊이 |

#### XMLSchema

```python
from truthound.reporters.sdk import XMLSchema

schema = XMLSchema(
    root_element="testsuites",
    required_elements=["testsuite", "testcase"],
    required_attributes={"testsuite": ["name", "tests"]},
    validate_dtd=False
)
```

**옵션:**

| 옵션 | 타입 | 설명 |
|------|------|------|
| `root_element` | `str` | 루트 요소 이름 |
| `required_elements` | `list[str]` | 필수 요소 목록 |
| `required_attributes` | `dict[str, list[str]]` | 요소별 필수 속성 |
| `validate_dtd` | `bool` | DTD 검증 여부 |

#### CSVSchema

```python
from truthound.reporters.sdk import CSVSchema

schema = CSVSchema(
    required_columns=["validator", "column", "severity", "message"],
    column_types={"severity": str, "count": int},
    allow_extra_columns=True,
    min_rows=0
)
```

**옵션:**

| 옵션 | 타입 | 설명 |
|------|------|------|
| `required_columns` | `list[str]` | 필수 컬럼 목록 |
| `column_types` | `dict[str, type]` | 컬럼별 타입 |
| `allow_extra_columns` | `bool` | 추가 컬럼 허용 |
| `min_rows` | `int` | 최소 행 수 |

#### TextSchema

```python
from truthound.reporters.sdk import TextSchema

schema = TextSchema(
    required_patterns=[r"Status:", r"Pass Rate:"],
    forbidden_patterns=[r"ERROR", r"EXCEPTION"],
    max_length=100000,
    encoding="utf-8"
)
```

**옵션:**

| 옵션 | 타입 | 설명 |
|------|------|------|
| `required_patterns` | `list[str]` | 필수 패턴 (정규식) |
| `forbidden_patterns` | `list[str]` | 금지 패턴 (정규식) |
| `max_length` | `int` | 최대 문자 수 |
| `encoding` | `str` | 인코딩 |

### 스키마 등록 및 관리

```python
from truthound.reporters.sdk import (
    register_schema,
    get_schema,
    unregister_schema,
    validate_reporter_output,
)

# 스키마 등록
register_schema("my_format", schema)

# 스키마 조회
my_schema = get_schema("my_format")

# 스키마 제거
unregister_schema("my_format")

# 리포터 출력 자동 검증
is_valid = validate_reporter_output("json", json_output)
```

### 스키마 추론 및 병합

```python
from truthound.reporters.sdk import infer_schema, merge_schemas

# 샘플 데이터에서 스키마 추론
inferred = infer_schema(sample_output, format="json")

# 여러 스키마 병합
merged = merge_schemas([schema1, schema2], strategy="union")
```

---

## Testing Utilities

리포터 테스트를 위한 유틸리티입니다.

### Mock 데이터 생성

#### create_mock_result

```python
from truthound.reporters.sdk import create_mock_result

# 기본 목 결과
result = create_mock_result()

# 커스텀 설정
result = create_mock_result(
    data_asset="test_data.csv",
    status="failure",
    pass_rate=0.75,
    issue_count=5,
    severity_distribution={"critical": 2, "high": 2, "medium": 1}
)
```

**파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `data_asset` | `str` | `"test_data.csv"` | 데이터 에셋 이름 |
| `status` | `str` | `"failure"` | 검증 상태 |
| `pass_rate` | `float` | `0.8` | 통과율 |
| `issue_count` | `int` | `3` | 이슈 개수 |
| `severity_distribution` | `dict` | `None` | severity별 이슈 수 |

#### MockResultBuilder

Fluent 빌더 패턴:

```python
from truthound.reporters.sdk import MockResultBuilder

result = (
    MockResultBuilder()
    .with_data_asset("orders.parquet")
    .with_status("failure")
    .with_pass_rate(0.65)
    .add_issue(
        validator="NullValidator",
        column="email",
        severity="critical",
        message="Found 10 null values"
    )
    .add_issue(
        validator="RangeValidator",
        column="age",
        severity="high",
        message="5 values out of range"
    )
    .with_run_time("2024-01-15T10:30:45")
    .build()
)
```

**Builder 메서드:**

| 메서드 | 설명 |
|--------|------|
| `with_data_asset(name)` | 데이터 에셋 이름 설정 |
| `with_status(status)` | 검증 상태 설정 |
| `with_pass_rate(rate)` | 통과율 설정 |
| `with_run_time(time)` | 실행 시간 설정 |
| `add_issue(...)` | 이슈 추가 |
| `add_issues(issues)` | 여러 이슈 추가 |
| `build()` | MockValidationResult 생성 |

#### create_mock_results

여러 결과 생성:

```python
from truthound.reporters.sdk import create_mock_results

# 5개의 랜덤 결과 생성
results = create_mock_results(count=5)

# 다양한 상태 분포
results = create_mock_results(
    count=10,
    status_distribution={"success": 0.7, "failure": 0.3}
)
```

### Assertion 함수

#### 범용 검증

```python
from truthound.reporters.sdk import assert_valid_output

# 출력 형식 자동 감지 및 검증
assert_valid_output(output, format="json")
assert_valid_output(output, format="xml")
assert_valid_output(output, format="csv")
```

#### JSON 검증

```python
from truthound.reporters.sdk import assert_json_valid

# 기본 JSON 검증
assert_json_valid(json_output)

# 스키마와 함께 검증
assert_json_valid(json_output, schema=my_schema)

# 필수 필드 검증
assert_json_valid(json_output, required_fields=["status", "issues"])
```

#### XML 검증

```python
from truthound.reporters.sdk import assert_xml_valid

# 기본 XML 검증
assert_xml_valid(xml_output)

# 루트 요소 검증
assert_xml_valid(xml_output, root_element="testsuites")

# XSD 스키마로 검증
assert_xml_valid(xml_output, xsd_path="schema.xsd")
```

#### CSV 검증

```python
from truthound.reporters.sdk import assert_csv_valid

# 기본 CSV 검증
assert_csv_valid(csv_output)

# 컬럼 검증
assert_csv_valid(
    csv_output,
    required_columns=["validator", "column", "severity"],
    min_rows=1
)
```

#### 패턴 매칭

```python
from truthound.reporters.sdk import assert_contains_patterns

assert_contains_patterns(
    output,
    patterns=[
        r"Status: (success|failure)",
        r"Pass Rate: \d+\.\d+%",
        r"Total Issues: \d+"
    ]
)
```

### ReporterTestCase

테스트 케이스 베이스 클래스:

```python
from truthound.reporters.sdk import ReporterTestCase
from truthound.reporters import get_reporter

class TestMyReporter(ReporterTestCase):
    reporter_name = "my_format"

    def test_basic_render(self):
        """기본 렌더링 테스트."""
        reporter = get_reporter(self.reporter_name)
        result = self.create_sample_result()

        output = reporter.render(result)

        self.assert_output_valid(output)
        self.assertIn("Status:", output)

    def test_empty_issues(self):
        """이슈 없는 경우 테스트."""
        result = self.create_result_with_no_issues()
        output = self.render(result)

        self.assert_output_valid(output)

    def test_edge_cases(self):
        """엣지 케이스 테스트."""
        for edge_case in self.get_edge_cases():
            with self.subTest(edge_case=edge_case.name):
                output = self.render(edge_case.data)
                self.assert_output_valid(output)
```

**제공 메서드:**

| 메서드 | 설명 |
|--------|------|
| `create_sample_result()` | 표준 샘플 결과 생성 |
| `create_result_with_no_issues()` | 이슈 없는 결과 생성 |
| `create_result_with_many_issues(n)` | n개 이슈 결과 생성 |
| `get_edge_cases()` | 엣지 케이스 목록 반환 |
| `render(result)` | 리포터로 렌더링 |
| `assert_output_valid(output)` | 출력 검증 |

### 테스트 데이터 생성

```python
from truthound.reporters.sdk import (
    create_sample_data,
    create_edge_case_data,
    create_stress_test_data,
)

# 표준 샘플 데이터
sample = create_sample_data()

# 엣지 케이스 데이터
edge_cases = create_edge_case_data()
# 반환: empty_result, single_issue, max_severity, unicode_content, ...

# 스트레스 테스트 데이터
stress = create_stress_test_data(
    issue_count=10000,
    validator_count=100
)
```

### 출력 캡처 및 벤치마크

#### capture_output

```python
from truthound.reporters.sdk import capture_output

# stdout/stderr 캡처
with capture_output() as captured:
    reporter.print(result)

print(f"Stdout: {captured.stdout}")
print(f"Stderr: {captured.stderr}")
```

#### benchmark_reporter

```python
from truthound.reporters.sdk import benchmark_reporter, BenchmarkResult

# 리포터 성능 벤치마크
result: BenchmarkResult = benchmark_reporter(
    reporter=get_reporter("json"),
    data=create_stress_test_data(issue_count=1000),
    iterations=100
)

print(f"Mean time: {result.mean_time:.4f}s")
print(f"Std dev: {result.std_dev:.4f}s")
print(f"Min time: {result.min_time:.4f}s")
print(f"Max time: {result.max_time:.4f}s")
print(f"Throughput: {result.throughput:.2f} ops/sec")
```

**BenchmarkResult 필드:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `mean_time` | `float` | 평균 실행 시간 (초) |
| `std_dev` | `float` | 표준 편차 |
| `min_time` | `float` | 최소 실행 시간 |
| `max_time` | `float` | 최대 실행 시간 |
| `throughput` | `float` | 초당 처리량 |
| `iterations` | `int` | 반복 횟수 |

---

## 커스텀 리포터 등록

### 데코레이터로 등록

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

### 수동 등록

```python
from truthound.reporters.factory import register_reporter

register_reporter("my_custom", MyCustomReporter)

# 사용
reporter = get_reporter("my_custom")
```

---

## API 레퍼런스

### SDK Exports

```python
from truthound.reporters.sdk import (
    # Mixins
    FormattingMixin,
    AggregationMixin,
    FilteringMixin,
    SerializationMixin,
    TemplatingMixin,
    StreamingMixin,

    # Builder
    ReporterBuilder,
    create_reporter,
    create_validation_reporter,

    # Templates
    CSVReporter,
    YAMLReporter,
    JUnitXMLReporter,
    NDJSONReporter,
    TableReporter,

    # Schema
    ReportSchema,
    JSONSchema,
    XMLSchema,
    CSVSchema,
    TextSchema,
    ValidationResult,
    ValidationError,
    SchemaError,
    validate_output,
    register_schema,
    get_schema,
    unregister_schema,
    validate_reporter_output,
    infer_schema,
    merge_schemas,

    # Testing
    ReporterTestCase,
    create_mock_result,
    create_mock_results,
    create_mock_validator_result,
    MockResultBuilder,
    MockValidationResult,
    MockValidatorResult,
    assert_valid_output,
    assert_json_valid,
    assert_xml_valid,
    assert_csv_valid,
    assert_contains_patterns,
    create_sample_data,
    create_edge_case_data,
    create_stress_test_data,
    capture_output,
    benchmark_reporter,
    BenchmarkResult,
)
```
