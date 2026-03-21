# truthound new reporter

Create a reporter scaffold that targets the 3.0 reporter contract.

## Synopsis

```bash
truthound new reporter <name> [OPTIONS]
```

## What It Generates

The scaffold produces a package with:

- a `ValidationReporter` implementation
- a reporter config dataclass
- optional tests
- optional docs

The generated code targets `ValidationRunResult`, not the older pre-3.0 report
object surface.

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `.` | Output directory |
| `--template` | `-t` | `basic` | `basic` or `full` |
| `--author` | `-a` | None | Author name |
| `--description` | `-d` | None | Reporter description |
| `--tests/--no-tests` | | `--tests` | Generate tests |
| `--docs/--no-docs` | | `--no-docs` | Generate docs |
| `--extension` | `-e` | `.txt` | Output file extension |
| `--content-type` | | `text/plain` | MIME content type |

## Basic Example

```bash
truthound new reporter my_reporter
```

Generated structure:

```text
./my_reporter/
├── __init__.py
└── reporter.py
```

Representative generated code:

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

from truthound.reporters.base import ReporterConfig, ValidationReporter
from truthound.reporters.factory import register_reporter

if TYPE_CHECKING:
    from truthound.core import ValidationRunResult


@dataclass
class MyReporterReporterConfig(ReporterConfig):
    include_passed: bool = False
    include_samples: bool = True
    max_issues: int | None = None


@register_reporter("my_reporter")
class MyReporterReporter(ValidationReporter[MyReporterReporterConfig]):
    name = "my_reporter"
    file_extension = ".txt"
    content_type = "text/plain"

    @classmethod
    def _default_config(cls) -> MyReporterReporterConfig:
        return MyReporterReporterConfig()

    def render(self, data: "ValidationRunResult") -> str:
        presentation = self.present(data)
        legacy_view = presentation.to_legacy_view()
        results = [r for r in legacy_view.results if not r.success]

        lines = [
            f"Validation Report: {presentation.source}",
            f"Status: {presentation.status}",
            f"Run ID: {presentation.run_id}",
            "",
        ]

        for result in results:
            lines.append(f"{result.validator_name}: {result.message}")

        return "\\n".join(lines)
```

## Full Template

```bash
truthound new reporter detailed_report --template full --docs
```

The full template adds SDK mixins and more configuration knobs such as:

- sorting
- severity filtering
- richer serialization helpers
- documentation and test scaffolding

## Generated Reporter Contract

Scaffolded reporters are built on these current 3.0 concepts:

- `ValidationReporter`
- `ReporterConfig`
- `ValidationRunResult`
- `RunPresentation`
- compatibility rows via `presentation.to_legacy_view()`

When you need issue metadata in generated or custom code, use fields like:

- `result.validator_name`
- `result.column`
- `result.severity`
- `result.message`

## Typical Follow-Up

After generating the scaffold:

1. Adjust the config dataclass for format-specific options.
2. Use `presentation = self.present(data)` for high-level summary data.
3. Use `legacy_view = presentation.to_legacy_view()` when you want flattened
   compatibility rows for tables or grouping helpers.
4. Register the reporter name you want teammates to call via `get_reporter(...)`.

## See Also

- [Reporters Guide](../../guides/reporters/index.md)
- [Reporter SDK](../../guides/reporters/custom-sdk.md)
- [Python API: Reporters](../../python-api/reporters.md)
