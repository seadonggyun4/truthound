# Truthound Reporters

This document provides comprehensive documentation for Truthound's report generation system, which outputs validation results in various formats.

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Built-in Reporters](#3-built-in-reporters)
4. [Configuration Reference](#4-configuration-reference)
5. [Custom Reporters](#5-custom-reporters)
6. [Integration Examples](#6-integration-examples)

---

## 1. Overview

Truthound reporters transform validation results into human-readable or machine-readable formats. The system supports multiple output formats through a unified interface.

### Architecture

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

### Key Features

- **Multiple Formats**: JSON, Console, Markdown, HTML
- **Unified Interface**: Same API across all reporters
- **Customizable**: Titles, themes, templates
- **Extensible**: Register custom reporters at runtime

---

## 2. Quick Start

### Basic Usage

```python
from truthound.reporters import get_reporter
from truthound.stores import ValidationResult
import truthound as th

# Run validation
report = th.check("data.csv")
result = ValidationResult.from_report(report, "data.csv")

# Create reporter
reporter = get_reporter("json")

# Render to string
json_output = reporter.render(result)

# Write to file
reporter.write(result, "report.json")
```

### Available Formats

| Format | Package | Use Case |
|--------|---------|----------|
| `json` | (built-in) | API integration, programmatic access |
| `console` | rich | Terminal output, debugging |
| `markdown` | (built-in) | Documentation, GitHub/GitLab |
| `html` | jinja2 | Web dashboards, email reports |

### Check Available Formats

```python
from truthound.reporters.factory import list_available_formats, is_format_available

# List all available formats
print(list_available_formats())
# ['console', 'json', 'markdown', 'html']

# Check specific format
if is_format_available("html"):
    reporter = get_reporter("html")
```

---

## 3. Built-in Reporters

### 3.1 JSON Reporter

Machine-readable JSON output.

```python
from truthound.reporters import get_reporter

reporter = get_reporter(
    "json",
    indent=2,              # Pretty-print indentation
    include_metadata=True,  # Include run metadata
)

# Render to string
json_str = reporter.render(result)

# Write to file
reporter.write(result, "report.json")
```

**Output Example**:
```json
{
  "run_id": "run_2024_001",
  "run_time": "2024-01-15T10:30:00",
  "data_asset": "customers.csv",
  "status": "failure",
  "statistics": {
    "total_validators": 10,
    "passed_validators": 8,
    "failed_validators": 2,
    "total_rows": 10000,
    "total_issues": 2,
    "pass_rate": 0.8
  },
  "results": [
    {
      "validator_name": "null_check",
      "success": false,
      "column": "email",
      "issue_type": "null_values",
      "count": 150,
      "severity": "high",
      "message": "Found 150 null values in email column"
    }
  ]
}
```

**Configuration**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indent` | int | `2` | JSON indentation level |
| `include_metadata` | bool | `True` | Include run metadata |
| `timestamp_format` | str | `%Y-%m-%d %H:%M:%S` | Timestamp format |

### 3.2 Console Reporter

Rich terminal output with colors and formatting.

```python
from truthound.reporters import get_reporter

reporter = get_reporter(
    "console",
    color=True,           # Enable colors
    verbose=False,        # Detailed output
    show_passed=False,    # Show passed validators
)

# Print to console
reporter.report(result)
```

**Output Example**:
```
╭──────────────────────────────────────────────────────────────╮
│                    Validation Report                          │
│                    customers.csv                              │
╰──────────────────────────────────────────────────────────────╯

Status: ❌ FAILURE

Statistics:
  ├── Total Validators: 10
  ├── Passed: 8 (80.0%)
  ├── Failed: 2
  └── Total Issues: 2

Issues:
  ┌─────────────┬────────────┬───────┬──────────┬─────────────────────────┐
  │ Column      │ Issue Type │ Count │ Severity │ Message                 │
  ├─────────────┼────────────┼───────┼──────────┼─────────────────────────┤
  │ email       │ null_values│ 150   │ HIGH     │ Found 150 null values   │
  │ customer_id │ duplicates │ 25    │ MEDIUM   │ Found 25 duplicate vals │
  └─────────────┴────────────┴───────┴──────────┴─────────────────────────┘
```

**Configuration**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `color` | bool | `True` | Enable colored output |
| `verbose` | bool | `False` | Show detailed information |
| `show_passed` | bool | `False` | Include passed validators |
| `width` | int | `None` | Console width (auto-detect) |

### 3.3 Markdown Reporter

Markdown format for documentation and wikis.

```python
from truthound.reporters import get_reporter

reporter = get_reporter(
    "markdown",
    title="Data Quality Report",
    include_toc=True,       # Table of contents
    include_badges=True,    # Status badges
)

# Render to string
md_content = reporter.render(result)

# Write to file
reporter.write(result, "REPORT.md")
```

**Output Example**:
```markdown
# Data Quality Report

**Data Asset**: customers.csv
**Status**: ❌ Failure
**Run Time**: 2024-01-15 10:30:00

## Summary

| Metric | Value |
|--------|-------|
| Total Validators | 10 |
| Passed | 8 (80.0%) |
| Failed | 2 |
| Total Issues | 2 |

## Issues

### High Severity

| Column | Issue Type | Count | Message |
|--------|------------|-------|---------|
| email | null_values | 150 | Found 150 null values in email column |

### Medium Severity

| Column | Issue Type | Count | Message |
|--------|------------|-------|---------|
| customer_id | duplicates | 25 | Found 25 duplicate values |
```

**Configuration**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title` | str | `Validation Report` | Report title |
| `include_toc` | bool | `False` | Include table of contents |
| `include_badges` | bool | `True` | Include status badges |

### 3.4 HTML Reporter

Rich HTML output with embedded CSS.

**Installation**:
```bash
pip install truthound[all]
# or
pip install jinja2
```

**Usage**:
```python
from truthound.reporters import get_reporter

reporter = get_reporter(
    "html",
    title="Data Quality Report",
    theme="light",           # "light" or "dark"
    custom_css="",           # Additional CSS
    template_path=None,      # Custom Jinja2 template
)

# Render to string
html_content = reporter.render(result)

# Write to file
reporter.write(result, "report.html")
```

**Features**:
- Responsive design
- Status badges with colors
- Statistics dashboard
- Issue table with severity highlighting
- Custom themes and CSS
- Mobile-friendly

**Configuration**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title` | str | `Validation Report` | Page title |
| `theme` | str | `light` | Color theme |
| `custom_css` | str | `""` | Additional CSS styles |
| `template_path` | str | None | Path to custom template |
| `inline_css` | bool | `True` | Inline CSS vs external |

**Custom Template**:
```python
reporter = get_reporter(
    "html",
    template_path="/path/to/custom_template.html",
)
```

Template context variables:
- `title`: Report title
- `result`: ValidationResult object
- `statistics`: ResultStatistics object
- `issues`: List of failed ValidatorResults
- `generated_at`: Timestamp string
- `config`: Reporter configuration

---

## 4. Configuration Reference

### Common Reporter Interface

All reporters implement the same interface:

```python
class BaseReporter(Generic[C], ABC):
    name: str                   # Reporter name
    file_extension: str         # Default file extension
    content_type: str           # MIME type

    def render(self, data: ValidationResult) -> str:
        """Render result to string."""
        ...

    def write(self, data: ValidationResult, path: str | Path) -> None:
        """Write result to file."""
        ...

    def report(self, data: ValidationResult) -> None:
        """Print to stdout (for console reporter)."""
        ...
```

### Common Configuration Options

Shared across all reporters:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title` | str | `Validation Report` | Report title |
| `timestamp_format` | str | `%Y-%m-%d %H:%M:%S` | Timestamp format |
| `output_path` | str | None | Default output path |

---

## 5. Custom Reporters

Create custom reporters by implementing the `BaseReporter` interface:

```python
from dataclasses import dataclass
from truthound.reporters import register_reporter, BaseReporter, ReporterConfig
from truthound.stores.results import ValidationResult

@dataclass
class SlackReporterConfig(ReporterConfig):
    """Configuration for Slack reporter."""
    webhook_url: str = ""
    channel: str = "#data-quality"
    mention_on_failure: str | None = None  # e.g., "@channel" or "@user"

@register_reporter("slack")
class SlackReporter(BaseReporter[SlackReporterConfig]):
    """Slack notification reporter."""

    name = "slack"
    file_extension = ".json"
    content_type = "application/json"

    def __init__(
        self,
        webhook_url: str = "",
        channel: str = "#data-quality",
        **kwargs,
    ):
        config = SlackReporterConfig(
            webhook_url=webhook_url,
            channel=channel,
            **kwargs,
        )
        super().__init__()
        self._config = config

    @classmethod
    def _default_config(cls) -> SlackReporterConfig:
        return SlackReporterConfig()

    def render(self, data: ValidationResult) -> str:
        """Build Slack message payload."""
        status_emoji = "✅" if data.success else "❌"
        color = "good" if data.success else "danger"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} {self._config.title}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Data Asset:*\n{data.data_asset}"},
                    {"type": "mrkdwn", "text": f"*Status:*\n{data.status.value}"},
                    {"type": "mrkdwn", "text": f"*Pass Rate:*\n{data.statistics.pass_rate:.1%}"},
                    {"type": "mrkdwn", "text": f"*Issues:*\n{data.statistics.total_issues}"},
                ]
            }
        ]

        # Add mention on failure
        if not data.success and self._config.mention_on_failure:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": self._config.mention_on_failure}
            })

        payload = {
            "channel": self._config.channel,
            "attachments": [{"color": color, "blocks": blocks}]
        }

        return json.dumps(payload)

    def write(self, data: ValidationResult, path: str = None) -> None:
        """Send to Slack."""
        import urllib.request

        payload = self.render(data)
        req = urllib.request.Request(
            self._config.webhook_url,
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req)

# Usage
reporter = get_reporter(
    "slack",
    webhook_url="https://hooks.slack.com/services/XXX/YYY/ZZZ",
    channel="#data-quality",
    mention_on_failure="@channel",
)
reporter.write(result)
```

### Email Reporter Example

```python
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

@dataclass
class EmailReporterConfig(ReporterConfig):
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_addr: str = ""
    to_addrs: list[str] = None
    subject_prefix: str = "[Data Quality]"

@register_reporter("email")
class EmailReporter(BaseReporter[EmailReporterConfig]):
    """Email reporter using HTML format."""

    name = "email"
    file_extension = ".html"
    content_type = "text/html"

    def __init__(self, **kwargs):
        config = EmailReporterConfig(**kwargs)
        super().__init__()
        self._config = config
        # Use HTML reporter for content
        self._html_reporter = get_reporter("html", title=config.title)

    @classmethod
    def _default_config(cls) -> EmailReporterConfig:
        return EmailReporterConfig()

    def render(self, data: ValidationResult) -> str:
        return self._html_reporter.render(data)

    def write(self, data: ValidationResult, path: str = None) -> None:
        html_content = self.render(data)

        status = "✅ PASSED" if data.success else "❌ FAILED"
        subject = f"{self._config.subject_prefix} {data.data_asset} - {status}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._config.from_addr
        msg["To"] = ", ".join(self._config.to_addrs)
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(self._config.smtp_host, self._config.smtp_port) as server:
            server.starttls()
            server.login(self._config.username, self._config.password)
            server.sendmail(
                self._config.from_addr,
                self._config.to_addrs,
                msg.as_string(),
            )
```

---

## 6. Integration Examples

### 6.1 CI/CD Pipeline (GitHub Actions)

```python
import sys
from truthound.reporters import get_reporter
import truthound as th

# Run validation
report = th.check("data.csv")
result = ValidationResult.from_report(report, "data.csv")

# Generate Markdown for GitHub summary
reporter = get_reporter("markdown", include_badges=True)
md_content = reporter.render(result)

# Write to GitHub step summary
with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
    f.write(md_content)

# Exit with error if validation failed
if not result.success:
    sys.exit(1)
```

### 6.2 Airflow Integration

```python
from airflow.decorators import task
from truthound.reporters import get_reporter

@task
def validate_and_report(data_path: str, report_path: str):
    import truthound as th
    from truthound.stores import ValidationResult

    report = th.check(data_path)
    result = ValidationResult.from_report(report, data_path)

    # Generate HTML report
    reporter = get_reporter("html", title="Daily Data Quality")
    reporter.write(result, report_path)

    # Fail task if validation failed
    if not result.success:
        raise ValueError(f"Data quality check failed for {data_path}")

    return report_path
```

### 6.3 FastAPI Endpoint

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from truthound.reporters import get_reporter
import truthound as th

app = FastAPI()

@app.post("/validate")
async def validate_data(file_path: str, format: str = "json"):
    report = th.check(file_path)
    result = ValidationResult.from_report(report, file_path)

    reporter = get_reporter(format)
    content = reporter.render(result)

    if format == "html":
        return HTMLResponse(content=content)
    return JSONResponse(content=json.loads(content))
```

### 6.4 Scheduled Reports

```python
import schedule
import time
from datetime import datetime
from truthound.reporters import get_reporter

def daily_report():
    result = run_validation("production_data.csv")

    # HTML report
    html_reporter = get_reporter("html", title="Daily Quality Report")
    filename = f"reports/report_{datetime.now():%Y%m%d}.html"
    html_reporter.write(result, filename)

    # Slack notification
    if not result.success:
        slack_reporter = get_reporter(
            "slack",
            webhook_url="https://hooks.slack.com/...",
            mention_on_failure="@data-team",
        )
        slack_reporter.write(result)

# Run daily at 9 AM
schedule.every().day.at("09:00").do(daily_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Summary

Truthound reporters provide flexible output formatting:

- **4 built-in formats**: JSON, Console, Markdown, HTML
- **Unified interface**: Same API across all reporters
- **Customizable**: Titles, themes, templates
- **Extensible**: Register custom reporters with `@register_reporter`

For more information:
- [Architecture Overview](ARCHITECTURE.md)
- [Stores Documentation](STORES.md)
- [Main README](../README.md)
