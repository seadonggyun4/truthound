# Phase 9: Plugin Architecture

Truthound의 플러그인 아키텍처는 확장성과 유지보수성을 위해 설계되었습니다. 외부 패키지를 통해 validators, reporters, datasources 등을 확장할 수 있습니다.

## 목차

- [개요](#개요)
- [빠른 시작](#빠른-시작)
- [플러그인 타입](#플러그인-타입)
- [플러그인 생성](#플러그인-생성)
- [Hook 시스템](#hook-시스템)
- [CLI 명령어](#cli-명령어)
- [고급 사용법](#고급-사용법)

## 개요

플러그인 아키텍처의 주요 구성 요소:

- **PluginManager**: 플러그인 생명주기 관리 (발견, 로드, 활성화, 언로드)
- **PluginRegistry**: 플러그인 등록 및 조회
- **HookManager**: 이벤트 기반 확장 시스템
- **PluginDiscovery**: 플러그인 자동 발견 (Entry points, 디렉토리 스캔)

## 빠른 시작

### 플러그인 사용

```python
from truthound.plugins import PluginManager, get_plugin_manager

# 글로벌 매니저 사용
manager = get_plugin_manager()

# 플러그인 발견
manager.discover_plugins()

# 특정 플러그인 로드
manager.load_plugin("my-validator-plugin")

# 모든 플러그인 로드
manager.load_all()

# 활성 플러그인 확인
for plugin in manager.get_active_plugins():
    print(f"{plugin.name} v{plugin.version}")
```

### CLI로 플러그인 관리

```bash
# 발견된 플러그인 목록
truthound plugin list

# 플러그인 상세 정보
truthound plugin info my-plugin

# 플러그인 로드
truthound plugin load my-plugin

# 플러그인 활성화/비활성화
truthound plugin enable my-plugin
truthound plugin disable my-plugin

# 새 플러그인 템플릿 생성
truthound plugin create my-new-plugin --type validator
```

## 플러그인 타입

### 1. ValidatorPlugin

커스텀 검증 규칙을 추가합니다.

```python
from truthound.plugins import ValidatorPlugin, PluginInfo, PluginType
from truthound.validators.base import Validator, ValidationIssue
from truthound.types import Severity
import polars as pl

class MyValidator(Validator):
    name = "my_validator"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues = []
        # 검증 로직 구현
        return issues

class MyValidatorPlugin(ValidatorPlugin):
    def _get_plugin_name(self) -> str:
        return "my-validator-plugin"

    def _get_plugin_version(self) -> str:
        return "1.0.0"

    def _get_description(self) -> str:
        return "Custom validators for my use case"

    def get_validators(self) -> list[type]:
        return [MyValidator]
```

### 2. ReporterPlugin

새로운 출력 형식을 추가합니다.

```python
from truthound.plugins import ReporterPlugin
from truthound.reporters.base import ValidationReporter, ReporterConfig
from truthound.core import ValidationResult

class XMLReporter(ValidationReporter[ReporterConfig]):
    name = "xml"
    file_extension = ".xml"

    def render(self, data: ValidationResult) -> str:
        # XML 렌더링 로직
        return "<report>...</report>"

class XMLReporterPlugin(ReporterPlugin):
    def _get_plugin_name(self) -> str:
        return "xml-reporter"

    def get_reporters(self) -> dict[str, type]:
        return {"xml": XMLReporter}
```

### 3. HookPlugin

이벤트에 반응하는 훅을 등록합니다.

```python
from truthound.plugins import HookPlugin, HookType
from typing import Any, Callable

class NotifierPlugin(HookPlugin):
    def _get_plugin_name(self) -> str:
        return "notifier"

    def get_hooks(self) -> dict[str, Callable]:
        return {
            HookType.AFTER_VALIDATION.value: self._on_validation_complete,
            HookType.ON_ERROR.value: self._on_error,
        }

    def _on_validation_complete(self, datasource, result, issues, **kwargs):
        if issues:
            print(f"Found {len(issues)} issues!")

    def _on_error(self, error, context, **kwargs):
        print(f"Error occurred: {error}")
```

### 4. DataSourcePlugin

새로운 데이터 소스 타입을 추가합니다.

```python
from truthound.plugins import DataSourcePlugin
from truthound.datasources.base import BaseDataSource

class MongoDataSource(BaseDataSource):
    source_type = "mongodb"
    # 구현...

class MongoPlugin(DataSourcePlugin):
    def _get_plugin_name(self) -> str:
        return "mongodb-source"

    def get_datasource_types(self) -> dict[str, type]:
        return {"mongodb": MongoDataSource}
```

## 플러그인 생성

### 디렉토리 구조

```
truthound-plugin-myfeature/
├── myfeature/
│   ├── __init__.py
│   └── plugin.py
├── pyproject.toml
└── README.md
```

### pyproject.toml 설정

```toml
[project]
name = "truthound-plugin-myfeature"
version = "0.1.0"
dependencies = ["truthound>=0.1.0"]

[project.entry-points."truthound.plugins"]
myfeature = "myfeature:MyFeaturePlugin"
```

### CLI로 템플릿 생성

```bash
# Validator 플러그인 템플릿
truthound plugin create my-validator --type validator --author "Your Name"

# Reporter 플러그인 템플릿
truthound plugin create my-reporter --type reporter

# Hook 플러그인 템플릿
truthound plugin create my-notifier --type hook
```

## Hook 시스템

### 사용 가능한 Hook 타입

| Hook | 설명 | Handler 시그니처 |
|------|------|-----------------|
| `before_validation` | 검증 시작 전 | `(datasource, validators, **kwargs)` |
| `after_validation` | 검증 완료 후 | `(datasource, result, issues, **kwargs)` |
| `on_issue_found` | 이슈 발견 시 | `(issue, validator, **kwargs)` |
| `before_profile` | 프로파일링 시작 전 | `(datasource, config, **kwargs)` |
| `after_profile` | 프로파일링 완료 후 | `(datasource, profile, **kwargs)` |
| `on_report_generate` | 리포트 생성 시 | `(report, format, **kwargs)` |
| `on_error` | 에러 발생 시 | `(error, context, **kwargs)` |
| `on_plugin_load` | 플러그인 로드 시 | `(plugin, manager)` |
| `on_plugin_unload` | 플러그인 언로드 시 | `(plugin, manager)` |

### 데코레이터 사용

```python
from truthound.plugins import before_validation, after_validation, on_error

@before_validation(priority=50)  # 낮은 우선순위가 먼저 실행
def log_start(datasource, validators, **kwargs):
    print(f"Validating {datasource} with {len(validators)} validators")

@after_validation()
def log_complete(datasource, result, issues, **kwargs):
    print(f"Found {len(issues)} issues")

@on_error()
def handle_error(error, context, **kwargs):
    print(f"Error: {error}")
```

### HookManager 직접 사용

```python
from truthound.plugins import HookManager, HookType

hooks = HookManager()

# 훅 등록
hooks.register(
    HookType.BEFORE_VALIDATION,
    my_handler,
    priority=100,
    source="my-plugin"
)

# 훅 트리거
results = hooks.trigger(
    HookType.BEFORE_VALIDATION,
    datasource=source,
    validators=["null", "range"]
)

# 특정 소스의 훅 비활성화
hooks.disable(source="my-plugin")
```

## CLI 명령어

```bash
# 플러그인 목록 (상세 정보 포함)
truthound plugin list --verbose

# JSON 출력
truthound plugin list --json

# 타입별 필터
truthound plugin list --type validator

# 상태별 필터
truthound plugin list --state active

# 플러그인 정보
truthound plugin info my-plugin --json

# 플러그인 로드
truthound plugin load my-plugin --activate

# 플러그인 언로드
truthound plugin unload my-plugin

# 플러그인 활성화/비활성화
truthound plugin enable my-plugin
truthound plugin disable my-plugin

# 새 플러그인 생성
truthound plugin create my-validator \
    --type validator \
    --author "Your Name" \
    --output ./my-plugins/
```

## 고급 사용법

### 플러그인 설정

```python
from truthound.plugins import PluginManager, PluginConfig

manager = PluginManager()

# 플러그인별 설정
config = PluginConfig(
    enabled=True,
    priority=50,  # 로드 순서 (낮을수록 먼저)
    settings={
        "api_key": "...",
        "timeout": 30,
    },
    auto_load=True,
)

manager.set_plugin_config("my-plugin", config)
manager.load_plugin("my-plugin")
```

### 의존성 관리

```python
from truthound.plugins import Plugin, PluginInfo, PluginType

class DependentPlugin(Plugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="dependent-plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            dependencies=("base-plugin",),  # 다른 플러그인 의존
            python_dependencies=("requests", "jinja2"),  # Python 패키지 의존
        )
```

### 버전 호환성

```python
PluginInfo(
    name="my-plugin",
    version="1.0.0",
    plugin_type=PluginType.VALIDATOR,
    min_truthound_version="0.5.0",
    max_truthound_version="2.0.0",
)
```

### Context Manager 사용

```python
from truthound.plugins import PluginManager

with PluginManager() as manager:
    manager.discover_plugins()
    manager.load_all()
    # 작업 수행...
# 자동으로 모든 플러그인 언로드
```

### 디렉토리에서 플러그인 로드

```python
from truthound.plugins import PluginManager
from pathlib import Path

manager = PluginManager()
manager.add_plugin_directory(Path("./my-plugins"))
manager.discover_plugins()
```

## 예제 플러그인

Truthound에는 참고용 예제 플러그인이 포함되어 있습니다:

```python
from truthound.plugins.examples import (
    CustomValidatorPlugin,  # 커스텀 비즈니스 규칙 validator
    SlackNotifierPlugin,    # Slack 알림 hook
    XMLReporterPlugin,      # XML 리포터
)
```

자세한 구현은 `truthound/plugins/examples/` 디렉토리를 참조하세요.

## API 참조

### Core Classes

- `Plugin[ConfigT]`: 플러그인 기본 클래스
- `PluginConfig`: 플러그인 설정
- `PluginInfo`: 플러그인 메타데이터
- `PluginType`: 플러그인 타입 열거형
- `PluginState`: 플러그인 상태 열거형

### Specialized Base Classes

- `ValidatorPlugin`: Validator 플러그인 기본 클래스
- `ReporterPlugin`: Reporter 플러그인 기본 클래스
- `DataSourcePlugin`: DataSource 플러그인 기본 클래스
- `HookPlugin`: Hook 플러그인 기본 클래스

### Management

- `PluginManager`: 플러그인 생명주기 관리
- `PluginRegistry`: 플러그인 등록/조회
- `PluginDiscovery`: 플러그인 발견
- `HookManager`: 훅 등록/실행

### Exceptions

- `PluginError`: 기본 플러그인 에러
- `PluginLoadError`: 로드 실패
- `PluginNotFoundError`: 플러그인 미발견
- `PluginDependencyError`: 의존성 미충족
- `PluginCompatibilityError`: 버전 비호환

## Enterprise Features

엔터프라이즈 환경을 위한 고급 플러그인 기능이 포함되어 있습니다:

### Enterprise Plugin Manager

```python
from truthound.plugins import create_enterprise_manager

# 보안 수준과 함께 엔터프라이즈 매니저 생성
manager = create_enterprise_manager(
    security_level="enterprise",  # "development", "standard", "enterprise", "strict"
    require_signature=True,       # 플러그인 서명 요구
    enable_hot_reload=True,       # 핫 리로드 활성화
)

# 플러그인 로드
plugin = await manager.load("my-plugin")

# 샌드박스에서 실행
result = await manager.execute_in_sandbox("my-plugin", my_function, arg1, arg2)
```

### Security Sandbox

플러그인을 격리된 환경에서 실행하여 시스템 보안을 강화합니다:

```python
from truthound.plugins import (
    SandboxFactory,
    IsolationLevel,
    SecurityPolicyPresets,
)

# 격리 수준별 샌드박스 생성
sandbox = SandboxFactory().create(IsolationLevel.PROCESS)

# 보안 정책 프리셋 사용
policy = SecurityPolicyPresets.ENTERPRISE.to_policy()
```

### Code Signing

플러그인 무결성과 출처를 검증합니다:

```python
from pathlib import Path
from truthound.plugins import (
    SigningServiceImpl,
    SignatureAlgorithm,
    TrustStoreImpl,
    TrustLevel,
    create_verification_chain,
)

# 플러그인 서명
service = SigningServiceImpl(
    algorithm=SignatureAlgorithm.HMAC_SHA256,
    signer_id="my-org",
)
signature = service.sign(
    plugin_path=Path("my_plugin/"),
    private_key=b"secret_key",
)

# 신뢰 저장소 설정
trust_store = TrustStoreImpl()
trust_store.set_signer_trust("my-org", TrustLevel.TRUSTED)

# 서명 검증
chain = create_verification_chain(trust_store=trust_store)
result = chain.verify(plugin_path, signature, context={})
```

### Hot Reload

애플리케이션 재시작 없이 플러그인을 리로드합니다:

```python
from truthound.plugins import HotReloadManager, ReloadStrategy, LifecycleManager

lifecycle = LifecycleManager()
reload_manager = HotReloadManager(
    lifecycle,
    default_strategy=ReloadStrategy.GRACEFUL,
)

# 플러그인 감시 시작
await reload_manager.watch(
    plugin_id="my-plugin",
    plugin_path=Path("plugins/my-plugin/"),
    auto_reload=True,
)

# 수동 리로드
result = await reload_manager.reload("my-plugin")
```

### Version Constraints

시맨틱 버전 제약을 지원합니다:

```python
from truthound.plugins import parse_constraint

# 다양한 버전 제약 표현
constraint = parse_constraint("^1.2.3")  # >=1.2.3 && <2.0.0
constraint = parse_constraint("~1.2.3")  # >=1.2.3 && <1.3.0
constraint = parse_constraint(">=1.0.0,<2.0.0")  # 범위 지정

# 버전 호환성 확인
is_compatible = constraint.is_satisfied_by("1.5.0")
```

### Dependency Graph

플러그인 의존성을 자동으로 관리합니다:

```python
from truthound.plugins import DependencyGraph, DependencyType

graph = DependencyGraph()
graph.add_node("plugin-c", "1.0.0")
graph.add_node("plugin-b", "1.0.0",
    dependencies={"plugin-c": DependencyType.REQUIRED})
graph.add_node("plugin-a", "1.0.0",
    dependencies={"plugin-b": DependencyType.REQUIRED})

# 로드 순서 결정
load_order = graph.get_load_order()
# -> ['plugin-c', 'plugin-b', 'plugin-a']

# 순환 의존성 감지
cycles = graph.detect_cycles()
```

자세한 Enterprise 기능은 `.claude/docs/phase-09-plugins.md`를 참조하세요.
