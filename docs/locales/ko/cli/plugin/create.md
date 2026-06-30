# truthound plugins create

CLI 명령 실행에서 Create을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Synopsis

```bash
truthound plugins create <NAME> [OPTIONS]
```

## Arguments

| CLI 명령 실행에서 Argument을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-------------|
| CLI 명령 실행에서 `NAME`, NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Plugin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Options

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|---------|-------------|
| CLI 명령 실행에서 `--output`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-o`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `.`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Output을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--type`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-t`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `validator`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Plugin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--author`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Author을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Description

CLI 명령 실행에서 `plugins create`을(를) 다루는 항목입니다:

1. CLI 명령 실행에서 `pyproject.toml`, Creates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. **Generates** plugin implementation 파일
3. CLI 명령 실행에서 Sets을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. CLI 명령 실행에서 Creates, README을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Plugin Types

| CLI 명령 실행에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Generated, Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|----------------|
| CLI 명령 실행에서 `validator`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Custom 검증기 plugin | 검증기 class |
| CLI 명령 실행에서 `reporter`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Reporter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `hook`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Event을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Hook을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `custom`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 General-purpose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Base을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 예시

### Create 검증기 Plugin

```bash
truthound plugins create my-validator
```

CLI 명령 실행에서 Generated을(를) 다루는 항목입니다:
```
truthound-plugin-my-validator/
├── my_validator/
│   ├── __init__.py
│   └── plugin.py
├── pyproject.toml
└── README.md
```

### Create Reporter Plugin

```bash
truthound plugins create my-reporter --type reporter
```

### Create Hook Plugin

```bash
truthound plugins create audit-hook --type hook
```

### With Author Info

```bash
truthound plugins create my-validator --author "John Doe"
```

### Custom Output Directory

```bash
truthound plugins create my-validator --output ./plugins
```

CLI 명령 실행에서 `./plugins/truthound-plugin-my-validator/`, Creates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Generated 파일

### pyproject.toml

```toml
[project]
name = "truthound-plugin-my-validator"
version = "0.1.0"
description = "Custom validator plugin for Truthound"
requires-python = ">=3.10"
dependencies = [
    "truthound>=3.0.0",
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
]

[project.entry-points."truthound.plugins"]
my-validator = "my_validator.plugin:MyValidatorPlugin"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### 검증기 Plugin (plugin.py)

```python
from truthound.plugins import Plugin, PluginType
from truthound.validators.base import Validator
from truthound.validators.report import ValidationIssue

class MyValidatorPlugin(Plugin):
    """Custom validator plugin."""

    name = "my-validator"
    version = "0.1.0"
    type = PluginType.VALIDATOR

    def get_validators(self):
        return [MyValidator()]

class MyValidator(Validator):
    """Custom validator implementation."""

    name = "my_validator"
    severity = "MEDIUM"

    def validate(self, df, columns=None):
        issues = []
        # Add validation logic here
        return issues
```

### Reporter Plugin (plugin.py)

```python
from truthound.plugins import Plugin, PluginType
from truthound.reporters.base import Reporter

class MyReporterPlugin(Plugin):
    """Custom reporter plugin."""

    name = "my-reporter"
    version = "0.1.0"
    type = PluginType.REPORTER

    def get_reporters(self):
        return [MyReporter()]

class MyReporter(Reporter):
    """Custom reporter implementation."""

    name = "my_reporter"
    extension = ".txt"
    content_type = "text/plain"

    def render(self, report):
        # Add rendering logic here
        return str(report)
```

### Hook Plugin (plugin.py)

```python
from truthound.plugins import Plugin, PluginType
from truthound.plugins.hooks import HookType

class AuditHookPlugin(Plugin):
    """Event hook plugin."""

    name = "audit-hook"
    version = "0.1.0"
    type = PluginType.HOOK

    def get_hooks(self):
        return {
            HookType.BEFORE_VALIDATION.value: self.on_validation_start,
            HookType.AFTER_VALIDATION.value: self.on_validation_complete,
        }

    def on_validation_start(self, context):
        """Called before validation starts."""
        print(f"Starting validation on {context.file_path}")

    def on_validation_complete(self, context, report):
        """Called after validation completes."""
        print(f"Validation complete: {len(report.issues)} issues found")
```

## Development 워크플로우

```bash
# 1. Create plugin template
truthound plugins create my-validator --type validator

# 2. Navigate to plugin directory
cd truthound-plugin-my-validator

# 3. Edit plugin implementation
# Edit my_validator/plugin.py

# 4. Install in development mode
pip install -e .

# 5. Verify plugin is discovered
truthound plugins list

# 6. Load and test
truthound plugins load my-validator
truthound check data.csv --validators my-validator

# 7. Build for distribution
pip install build
python -m build

# 8. Publish (optional)
pip install twine
twine upload dist/*
```

## Use Cases

### 1. Company-Specific 검증기

```bash
truthound plugins create company-validators \
  --type validator \
  --author "Data Team"
```

### 2. Custom 리포트 Format

```bash
truthound plugins create slack-reporter \
  --type reporter \
  --author "DevOps Team"
```

### 3. 감사 로깅 Hook

```bash
truthound plugins create compliance-audit \
  --type hook \
  --author "Compliance Team"
```

### 4. Multi-Purpose Plugin

```bash
truthound plugins create enterprise-suite \
  --type custom \
  --author "Enterprise Team"
```

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Invalid을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Related Commands

- CLI 명령 실행에서 `plugins list`, List을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 `plugins load`, Load을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 `new plugin`, Alternative을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- [플러그인 명령 개요](index.md)
- CLI 명령 실행에서 Plugin, System을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [스캐폴딩 명령](../scaffolding/index.md)
