# Console Reporter

Console Reporter는 Rich 라이브러리를 사용하여 터미널에 색상이 있는 아름다운 검증 결과를 출력합니다.

## 기본 사용법

```python
from truthound.reporters import get_reporter

reporter = get_reporter("console")
output = reporter.render(validation_result)
print(output)

# 또는 직접 출력
reporter.print(validation_result)
```

## 설정 옵션

`ConsoleReporterConfig`는 다음 옵션을 제공합니다:

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `color` | `bool` | `True` | ANSI 색상 출력 활성화 |
| `width` | `int \| None` | `None` | 출력 너비 (None = 터미널 너비) |
| `show_header` | `bool` | `True` | 헤더 섹션 표시 |
| `show_summary` | `bool` | `True` | 요약 통계 표시 |
| `show_issues_table` | `bool` | `True` | 이슈 테이블 표시 |
| `compact` | `bool` | `False` | 컴팩트 모드 (간략한 출력) |
| `severity_colors` | `dict[str, str]` | 기본 색상 맵 | 심각도별 색상 설정 |

### 기본 심각도 색상

```python
DEFAULT_SEVERITY_COLORS = {
    "critical": "bold red",
    "high": "red",
    "medium": "yellow",
    "low": "green",
    "info": "blue",
}
```

## 사용 예시

### 기본 사용

```python
from truthound.reporters import get_reporter
from truthound.reporters.console_reporter import ConsoleReporterConfig

# 기본 설정
reporter = get_reporter("console")
output = reporter.render(result)

# 커스텀 설정
config = ConsoleReporterConfig(
    color=True,
    width=120,
    compact=False,
    show_issues_table=True,
)
reporter = get_reporter("console", **config.__dict__)
```

### 컴팩트 모드

```python
reporter = get_reporter("console", compact=True)
output = reporter.render(result)
```

컴팩트 모드는 한 줄 요약만 출력합니다:
```
✓ Validation PASSED: 10/10 validators passed (100.0%) in 0.05s
```

또는 실패 시:
```
✗ Validation FAILED: 8/10 validators passed (80.0%) - 2 critical, 0 high, 0 medium, 0 low issues
```

### 색상 비활성화

CI 환경이나 파일 출력 시 색상을 비활성화할 수 있습니다:

```python
reporter = get_reporter("console", color=False)
```

### 직접 터미널 출력

`print()` 메서드를 사용하면 Rich Console을 통해 직접 터미널에 출력합니다:

```python
reporter = get_reporter("console")
reporter.print(validation_result)  # 터미널에 직접 출력
```

## 출력 형식

### 전체 출력 (기본)

```
╭─────────────────────────────────────────────────────────────╮
│              Truthound Validation Report                    │
╰─────────────────────────────────────────────────────────────╯

Data Asset: customer_data.csv
Run ID: abc123-def456
Status: FAILED
Timestamp: 2024-01-15 10:30:45

┌──────────────────────────────────────────────────────────────┐
│                         Summary                              │
├──────────────────────────────────────────────────────────────┤
│ Total Validators: 10                                         │
│ Passed: 8                                                    │
│ Failed: 2                                                    │
│ Pass Rate: 80.0%                                             │
│ Execution Time: 0.05s                                        │
└──────────────────────────────────────────────────────────────┘

Issues by Severity:
  🔴 Critical: 1
  🟠 High: 1
  🟡 Medium: 0
  🟢 Low: 0

┌──────────────┬──────────┬───────────┬─────────────────────────┐
│ Validator    │ Column   │ Severity  │ Message                 │
├──────────────┼──────────┼───────────┼─────────────────────────┤
│ NullValidator│ email    │ critical  │ 5 null values found     │
│ RangeValidator│ age     │ high      │ 3 values out of range   │
└──────────────┴──────────┴───────────┴─────────────────────────┘
```

## 클래스 속성

```python
class ConsoleReporter:
    name = "console"
    file_extension = ".txt"
    content_type = "text/plain"
```

## API 레퍼런스

### ConsoleReporter

```python
class ConsoleReporter(ValidationReporter[ConsoleReporterConfig]):
    """Rich 기반 콘솔 리포터."""

    def render(self, data: ValidationResult) -> str:
        """검증 결과를 문자열로 렌더링."""
        ...

    def print(self, data: ValidationResult) -> None:
        """검증 결과를 터미널에 직접 출력."""
        ...

    def render_compact(self, data: ValidationResult) -> str:
        """컴팩트 한 줄 요약 렌더링."""
        ...
```

## 의존성

Console Reporter는 Rich 라이브러리를 사용합니다:

```bash
pip install rich
```

Rich는 Truthound의 기본 의존성에 포함되어 있습니다.
