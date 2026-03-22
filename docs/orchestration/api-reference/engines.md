---
title: Engines API Reference
---

# Engines API Reference

## engines/base.py

### DataQualityEngine Protocol

```python
@runtime_checkable
class DataQualityEngine(Protocol):
    @property
    def engine_name(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CheckResult: ...

    def profile(self, data: Any, **kwargs: Any) -> ProfileResult: ...

    def learn(self, data: Any, **kwargs: Any) -> LearnResult: ...
```

### EngineCapabilities

```python
@dataclass(frozen=True)
class EngineCapabilities:
    supports_check: bool = True
    supports_profile: bool = True
    supports_learn: bool = True
    supports_async: bool = False
    supports_streaming: bool = False
    supported_data_types: tuple[str, ...] = ()
    supported_rule_types: tuple[str, ...] = ()
```

### EngineInfo

```python
@dataclass(frozen=True)
class EngineInfo:
    name: str
    version: str
    description: str | None = None
    capabilities: EngineCapabilities | None = None
```

## engines/lifecycle.py

### EngineState

```python
class EngineState(Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
```

### ManagedEngine Protocol

```python
@runtime_checkable
class ManagedEngine(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def health_check(self) -> HealthCheckResult: ...
    def get_state(self) -> EngineState: ...
```

### EngineStateSnapshot

```python
@dataclass(frozen=True)
class EngineStateSnapshot:
    state: EngineState
    uptime_seconds: float
    error_count: int
    last_error: Exception | None
    health_status: HealthStatus | None
```

## engines/batch.py

### BatchConfig

```python
@dataclass(frozen=True)
class BatchConfig:
    batch_size: int = 1000
    max_workers: int = 4
    execution_strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MERGE
    fail_fast: bool = False
    timeout_seconds: float | None = None
```

### ExecutionStrategy

```python
class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
```

### BatchResult

```python
@dataclass(frozen=True)
class BatchResult:
    status: CheckStatus
    total_chunks: int
    passed_chunks: int
    failed_chunks: int
    duration_seconds: float
    results: tuple[CheckResult, ...] = ()
```

## engines/chain.py

### FallbackConfig

```python
@dataclass(frozen=True)
class FallbackConfig:
    strategy: FallbackStrategy = FallbackStrategy.SEQUENTIAL
    retry_count: int = 1
    retry_delay_seconds: float = 0.0
    check_health: bool = False
    skip_unhealthy: bool = False
    timeout_seconds: float | None = None
    weights: Mapping[str, float] = field(default_factory=dict)
```

### FallbackStrategy

```python
class FallbackStrategy(Enum):
    SEQUENTIAL = "sequential"
    FIRST_AVAILABLE = "first_available"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    PRIORITY = "priority"
    WEIGHTED = "weighted"
```

### ChainExecutionResult

```python
@dataclass(frozen=True)
class ChainExecutionResult:
    success: bool
    final_engine: str | None
    result: CheckResult | None
    attempts: tuple[ChainExecutionAttempt, ...]
    total_duration_ms: float
```

## engines/context.py

### ContextConfig

```python
@dataclass(frozen=True)
class ContextConfig:
    auto_start_engine: bool = True
    auto_stop_engine: bool = True
    cleanup_strategy: CleanupStrategy = CleanupStrategy.ALWAYS
    track_resources: bool = True
    enable_savepoints: bool = False
    propagate_exceptions: bool = True
    cleanup_timeout_seconds: float = 30.0
```

### CleanupStrategy

```python
class CleanupStrategy(Enum):
    ALWAYS = "always"
    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    NEVER = "never"
```

## engines/aggregation.py

### AggregationConfig

```python
@dataclass(frozen=True)
class AggregationConfig:
    strategy: ResultAggregationStrategy = ResultAggregationStrategy.MERGE
    conflict_resolution: ConflictResolution = ConflictResolution.PREFER_FAILURE
    consensus_threshold: float = 0.5
    weights: Mapping[str, float] = field(default_factory=dict)
    preserve_individual_results: bool = False
```

### ResultAggregationStrategy

```python
class ResultAggregationStrategy(Enum):
    MERGE = "merge"
    WORST = "worst"
    BEST = "best"
    MAJORITY = "majority"
    FIRST_FAILURE = "first_failure"
    WEIGHTED = "weighted"
    CONSENSUS = "consensus"
    STRICT_ALL = "strict_all"
    LENIENT_ANY = "lenient_any"
```

## engines/version.py

### SemanticVersion

```python
@dataclass(frozen=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None

    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
```

### VersionConstraint

```python
class VersionConstraint:
    def __init__(self, operator: VersionOperator, version: SemanticVersion): ...
    def satisfies(self, version: SemanticVersion) -> bool: ...
```

## engines/plugin.py

### PluginSpec

```python
@dataclass(frozen=True)
class PluginSpec:
    name: str
    module_path: str
    class_name: str
    plugin_type: PluginType = PluginType.ENGINE
    priority: int = 100
    enabled: bool = True
    aliases: tuple[str, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
```

### PluginType

```python
class PluginType(Enum):
    ENGINE = "engine"
    VALIDATOR = "validator"
    HOOK = "hook"
    SOURCE = "source"
```

### PluginState

```python
class PluginState(Enum):
    DISCOVERED = "discovered"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
```
