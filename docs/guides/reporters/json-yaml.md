# JSON & YAML Reporters

JSON과 YAML 형식으로 검증 결과를 출력하는 리포터입니다.

## JSON Reporter

### 기본 사용법

```python
from truthound.reporters import get_reporter

reporter = get_reporter("json")
json_output = reporter.render(validation_result)
```

### 설정 옵션

`JSONReporterConfig`는 다음 옵션을 제공합니다:

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `indent` | `int \| None` | `2` | JSON 들여쓰기 (None = 한 줄) |
| `sort_keys` | `bool` | `False` | 키 정렬 여부 |
| `ensure_ascii` | `bool` | `False` | ASCII 문자만 사용 |
| `include_null_values` | `bool` | `True` | null 값 포함 여부 |
| `date_format` | `str` | `"iso"` | 날짜 형식 ("iso" \| "timestamp") |

### 사용 예시

#### 기본 JSON 출력

```python
from truthound.reporters import get_reporter

reporter = get_reporter("json", indent=2)
output = reporter.render(result)
```

출력 예시:
```json
{
  "run_id": "abc123-def456",
  "data_asset": "customer_data.csv",
  "status": "failure",
  "run_time": "2024-01-15T10:30:45",
  "success": false,
  "statistics": {
    "total_validators": 10,
    "passed_validators": 8,
    "failed_validators": 2,
    "pass_rate": 0.8,
    "execution_time_ms": 150
  },
  "results": [
    {
      "validator_name": "NullValidator",
      "column": "email",
      "success": false,
      "severity": "critical",
      "issue_type": "null_values",
      "count": 5,
      "message": "Found 5 null values"
    }
  ]
}
```

#### 컴팩트 JSON (한 줄)

```python
reporter = get_reporter("json", indent=None)
compact_output = reporter.render(result)

# 또는 메서드 사용
reporter = get_reporter("json")
compact_output = reporter.render_compact(result)
```

#### NDJSON (Newline Delimited JSON)

로그 수집 시스템(ELK, Splunk)과의 통합에 유용합니다:

```python
reporter = get_reporter("json")
ndjson_output = reporter.render_lines(result)
```

출력:
```json
{"type":"metadata","run_id":"abc123","data_asset":"data.csv"}
{"type":"result","validator":"NullValidator","column":"email","success":false}
{"type":"result","validator":"RangeValidator","column":"age","success":false}
```

### 날짜 형식

```python
# ISO 8601 형식 (기본)
reporter = get_reporter("json", date_format="iso")
# 출력: "2024-01-15T10:30:45"

# Unix timestamp
reporter = get_reporter("json", date_format="timestamp")
# 출력: 1705314645
```

---

## YAML Reporter

### 기본 사용법

```python
from truthound.reporters import get_reporter

reporter = get_reporter("yaml")
yaml_output = reporter.render(validation_result)
```

### 설정 옵션

`YAMLReporterConfig`는 다음 옵션을 제공합니다:

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `default_flow_style` | `bool` | `False` | 플로우 스타일 사용 |
| `indent` | `int` | `2` | 들여쓰기 크기 |
| `include_passed` | `bool` | `False` | 통과한 validator 포함 |
| `sort_keys` | `bool` | `False` | 키 정렬 여부 |

### 사용 예시

```python
from truthound.reporters import get_reporter

reporter = get_reporter("yaml", include_passed=False)
yaml_output = reporter.render(result)
```

출력 예시:
```yaml
validation_result:
  run_id: abc123-def456
  data_asset: customer_data.csv
  status: failure
  run_time: '2024-01-15T10:30:45'
  summary:
    total_validators: 10
    passed: 8
    failed: 2
    pass_rate: '80.0%'
  severity_counts:
    critical: 1
    high: 1
  issues:
    - validator: NullValidator
      column: email
      severity: critical
      issue_type: null_values
      message: Found 5 null values
      count: 5
    - validator: RangeValidator
      column: age
      severity: high
      issue_type: out_of_range
      message: 3 values out of range
      count: 3
```

### 의존성

YAML Reporter는 PyYAML 라이브러리가 필요합니다:

```bash
pip install pyyaml
```

---

## NDJSON Reporter (SDK)

SDK에서 제공하는 전용 NDJSON 리포터입니다.

### 기본 사용법

```python
from truthound.reporters.sdk import NDJSONReporter

reporter = NDJSONReporter()
ndjson_output = reporter.render(validation_result)
```

### 설정 옵션

`NDJSONReporterConfig`는 다음 옵션을 제공합니다:

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `include_passed` | `bool` | `False` | 통과한 validator 포함 |
| `include_metadata` | `bool` | `True` | 메타데이터 라인 포함 |
| `compact` | `bool` | `True` | 컴팩트 JSON 사용 |

### 출력 형식

```json
{"type":"metadata","run_id":"abc123","data_asset":"data.csv","status":"failure","total_validators":10}
{"type":"result","validator":"NullValidator","column":"email","success":false,"severity":"critical","count":5,"message":"Found 5 null values"}
{"type":"result","validator":"RangeValidator","column":"age","success":false,"severity":"high","count":3,"message":"3 values out of range"}
```

---

## 파일 출력

모든 리포터에서 파일 출력을 지원합니다:

```python
# 방법 1: write() 메서드
reporter = get_reporter("json")
path = reporter.write(result, "report.json")

# 방법 2: report() 메서드 (렌더링 + 선택적 파일 출력)
output = reporter.report(result, path="report.json")

# 바이트로 렌더링
bytes_output = reporter.render_to_bytes(result)
```

## API 레퍼런스

### JSONReporter

```python
class JSONReporter(ValidationReporter[JSONReporterConfig]):
    """JSON 형식 리포터."""

    name = "json"
    file_extension = ".json"
    content_type = "application/json"

    def render(self, data: ValidationResult) -> str:
        """검증 결과를 JSON으로 렌더링."""
        ...

    def render_compact(self, data: ValidationResult) -> str:
        """컴팩트 JSON (한 줄) 렌더링."""
        ...

    def render_lines(self, data: ValidationResult) -> str:
        """NDJSON 형식 렌더링."""
        ...
```

### YAMLReporter

```python
class YAMLReporter(ValidationReporter[YAMLReporterConfig]):
    """YAML 형식 리포터."""

    name = "yaml"
    file_extension = ".yaml"
    content_type = "application/yaml"

    def render(self, data: ValidationResult) -> str:
        """검증 결과를 YAML로 렌더링."""
        ...
```

### NDJSONReporter

```python
class NDJSONReporter(ValidationReporter[NDJSONReporterConfig]):
    """NDJSON (Newline Delimited JSON) 리포터."""

    name = "ndjson"
    file_extension = ".ndjson"
    content_type = "application/x-ndjson"

    def render(self, data: ValidationResult) -> str:
        """검증 결과를 NDJSON으로 렌더링."""
        ...
```
