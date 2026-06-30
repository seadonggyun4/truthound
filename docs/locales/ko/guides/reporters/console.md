# Console Reporter

실무 운영 가이드에서 Console, Reporter, Rich을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Basic Usage

```python
from truthound.reporters import get_reporter

reporter = get_reporter("console")
output = reporter.render(run_result)
print(output)

# Or print directly
reporter.print(run_result)
```

## 설정 Options

실무 운영 가이드에서 `ConsoleReporterConfig`, ConsoleReporterConfig을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|---------|-------------|
| 실무 운영 가이드에서 `color`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Enable, ANSI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `width`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Output, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `show_header`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Display을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `show_summary`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Display을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `show_issues_table`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Display issues 테이블 |
| 실무 운영 가이드에서 `compact`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Compact을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `severity_colors`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `dict[str, str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Colors을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Default Severity Colors

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

### Basic Usage

```python
from truthound.reporters import get_reporter
from truthound.reporters.console_reporter import ConsoleReporterConfig

# Default settings
reporter = get_reporter("console")
output = reporter.render(result)

# Custom settings
config = ConsoleReporterConfig(
    color=True,
    width=120,
    compact=False,
    show_issues_table=True,
)
reporter = get_reporter("console", **config.__dict__)
```

### Compact Mode

```python
reporter = get_reporter("console", compact=True)
output = reporter.render(result)
```

실무 운영 가이드에서 Compact을(를) 다루는 항목입니다:
```
✓ Validation PASSED: 10/10 validators passed (100.0%) in 0.05s
```

Or on 실패:
```
✗ Validation FAILED: 8/10 validators passed (80.0%) - 2 critical, 0 high, 0 medium, 0 low issues
```

### Disable Colors

실무 운영 가이드에서 Colors을(를) 다루는 항목입니다:

```python
reporter = get_reporter("console", color=False)
```

### Direct Terminal Output

실무 운영 가이드에서 `print()`, Rich, Console을(를) 다루는 항목입니다:

```python
reporter = get_reporter("console")
reporter.print(run_result)  # Direct terminal output
```

## Output Format

### Full Output (Default)

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

## Class Attributes

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
    """Rich-based console reporter."""

    def render(self, data: ValidationRunResult) -> str:
        """Render the canonical validation run result as string."""
        ...

    def print(self, data: ValidationRunResult) -> None:
        """Print the canonical validation run result directly to terminal."""
        ...

    def render_compact(self, data: ValidationRunResult) -> str:
        """Render compact single-line summary."""
        ...
```

## Dependencies

실무 운영 가이드에서 Console, Reporter, Rich을(를) 다루는 항목입니다:

```bash
pip install rich
```

실무 운영 가이드에서 Truthound, Rich을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
