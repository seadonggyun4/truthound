# Console Reporter

Console Reporter outputs beautifully colored validation results to the terminal using the Rich library.

## Basic Usage

```python
from truthound.reporters import get_reporter

reporter = get_reporter("console")
output = reporter.render(run_result)
print(output)

# Or print directly
reporter.print(run_result)
```

## Configuration Options

`ConsoleReporterConfig` provides the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `color` | `bool` | `True` | Enable ANSI color output |
| `width` | `int \| None` | `None` | Output width (None = terminal width) |
| `show_header` | `bool` | `True` | Display header section |
| `show_summary` | `bool` | `True` | Display summary statistics |
| `show_issues_table` | `bool` | `True` | Display issues table |
| `compact` | `bool` | `False` | Compact mode (brief output) |
| `severity_colors` | `dict[str, str]` | Default color map | Colors per severity |

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

## Usage Examples

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

Compact mode outputs a single-line summary:
```
✓ Validation PASSED: 10/10 validators passed (100.0%) in 0.05s
```

Or on failure:
```
✗ Validation FAILED: 8/10 validators passed (80.0%) - 2 critical, 0 high, 0 medium, 0 low issues
```

### Disable Colors

Colors can be disabled for CI environments or file output:

```python
reporter = get_reporter("console", color=False)
```

### Direct Terminal Output

The `print()` method outputs directly to the terminal via Rich Console:

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

## API Reference

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

Console Reporter uses the Rich library:

```bash
pip install rich
```

Rich is included in Truthound's default dependencies.
