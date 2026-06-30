# 프로파일러 설정

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
from truthound.profiler import profile, generate_suite

# Profile data
profile_result = profile(df)

# Generate validation suite
suite = generate_suite(df, strictness="medium")
```

## SuiteGeneratorConfig

실무 운영 가이드에서 Main을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.suite_config import SuiteGeneratorConfig

config = SuiteGeneratorConfig(
    name="my_suite",
    strictness="medium",                # loose, medium, strict
    categories=CategoryConfig(),
    confidence=ConfidenceConfig(),
    output=OutputConfig(),
    generators=GeneratorConfig(),
    custom_options={},
)
```

### CategoryConfig

실무 운영 가이드에서 Control을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.suite_config import CategoryConfig

config = CategoryConfig(
    include=[],                         # Categories to include (empty = all)
    exclude=[],                         # Categories to exclude
    priority_order=[],                  # Preferred ordering
)

# Check if category should be included
if config.should_include("completeness"):
    # Include completeness validators
    pass
```

실무 운영 가이드에서 Available, Categories을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

- `schema` - 스키마 검증
- 실무 운영 가이드에서 `completeness`, Null/missing을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `uniqueness`, Unique을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `distribution`, Value을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `format`, String을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `pattern`, Regex을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `range`, Value을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- `consistency` - Cross-컬럼 consistency

### ConfidenceConfig

실무 운영 가이드에서 Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.suite_config import ConfidenceConfig

config = ConfidenceConfig(
    min_level="low",                    # low, medium, high
    include_rationale=True,             # Include rationale in output
    show_in_output=True,                # Show confidence level
)
```

| 실무 운영 가이드에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `low`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `medium`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Moderate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `high`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 High을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### OutputConfig

실무 운영 가이드에서 Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.suite_config import OutputConfig

config = OutputConfig(
    format="yaml",                      # yaml, json, python, toml, checkpoint
    include_metadata=True,
    include_summary=True,
    include_description=True,
    group_by_category=False,
    sort_rules=True,
    indent=2,

    # Python-specific options
    code_style="functional",            # functional, class_based, declarative
    include_docstrings=True,
    include_type_hints=True,
    max_line_length=88,
)
```

실무 운영 가이드에서 Output, Formats을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 실무 운영 가이드에서 `yaml`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | YAML 설정 파일 |
| 실무 운영 가이드에서 `json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | JSON 설정 파일 |
| 실무 운영 가이드에서 `python`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `toml`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | TOML 설정 파일 |
| 실무 운영 가이드에서 `checkpoint`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트 설정 |

실무 운영 가이드에서 Python, Code, Styles을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Style을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `functional`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Function-based 검증기 |
| 실무 운영 가이드에서 `class_based`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Class-based 검증기 |
| 실무 운영 가이드에서 `declarative`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Declarative을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### GeneratorConfig

실무 운영 가이드에서 Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.suite_config import GeneratorConfig

config = GeneratorConfig(
    mode="full",                        # full, fast, custom
    enabled_generators=[],              # Generators to enable
    disabled_generators=[],             # Generators to disable
    generator_options={},               # Generator-specific options
)

# Check if generator should be used
if config.should_use_generator("schema"):
    # Use schema generator
    pass

# Get generator options
opts = config.get_generator_options("completeness")
```

실무 운영 가이드에서 Generator, Modes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Mode을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| 실무 운영 가이드에서 `full`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `fast`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `custom`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Manual을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 설정 Presets

```python
from truthound.profiler.suite_config import (
    SuiteGeneratorConfig,
    ConfigPreset,
)

# Use preset
config = SuiteGeneratorConfig.from_preset(ConfigPreset.DEFAULT)
config = SuiteGeneratorConfig.from_preset(ConfigPreset.STRICT)
config = SuiteGeneratorConfig.from_preset(ConfigPreset.PRODUCTION)
```

실무 운영 가이드에서 Available, Presets을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Preset을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Strictness을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Key, Settings을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-----------|--------------|
| 실무 운영 가이드에서 `DEFAULT`, DEFAULT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Standard을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `STRICT`, STRICT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Group을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `LOOSE`, LOOSE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Permissive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MINIMAL`, MINIMAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Schema을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COMPREHENSIVE`, COMPREHENSIVE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SCHEMA_ONLY`, SCHEMA_ONLY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 스키마 + completeness only |
| 실무 운영 가이드에서 `FORMAT_ONLY`, FORMAT_ONLY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CI_CD`, CI_CD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트 output format |
| 실무 운영 가이드에서 `DEVELOPMENT`, DEVELOPMENT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PRODUCTION`, PRODUCTION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Grouped을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Loading 설정

```python
from truthound.profiler.suite_config import (
    load_config,
    save_config,
    SuiteGeneratorConfig,
)

# Load from file
config = load_config("suite_config.yaml")
config = load_config("suite_config.json")

# Load from environment
config = SuiteGeneratorConfig.from_env(prefix="TRUTHOUND_SUITE")

# Load from dictionary
config = SuiteGeneratorConfig.from_dict({
    "strictness": "strict",
    "categories": {"include": ["schema", "completeness"]},
})

# Clone with overrides
new_config = config.with_overrides(strictness="loose")

# Save to file
save_config(config, "suite_config.yaml")
```

**환경 변수:**

```bash
export TRUTHOUND_SUITE_STRICTNESS=strict
export TRUTHOUND_SUITE_MIN_CONFIDENCE=medium
export TRUTHOUND_SUITE_FORMAT=yaml
export TRUTHOUND_SUITE_INCLUDE_CATEGORIES=schema,completeness
export TRUTHOUND_SUITE_EXCLUDE_CATEGORIES=pattern
```

## Scheduling 설정

### 프로파일 Triggers

```python
from truthound.profiler.scheduling.triggers import (
    CronTrigger,
    IntervalTrigger,
    DataChangeTrigger,
    EventTrigger,
    CompositeTrigger,
    AlwaysTrigger,
    ManualTrigger,
)

# Cron-based scheduling
trigger = CronTrigger("0 2 * * *")              # Daily at 2 AM
trigger = CronTrigger("0 */6 * * *")            # Every 6 hours
trigger = CronTrigger("0 0 * * 0", timezone="UTC")  # Weekly

# Interval-based
trigger = IntervalTrigger(hours=6)
trigger = IntervalTrigger(minutes=30)
trigger = IntervalTrigger(days=1, hours=2)

# Data change trigger
trigger = DataChangeTrigger(
    change_threshold=0.05,              # 5% change threshold
    change_type="row_count",            # row_count, schema, hash
    min_interval_seconds=60,            # Minimum between runs
)

# Event-based
trigger = EventTrigger(event_name="profile_requested")

# Composite (combine multiple)
trigger = CompositeTrigger(
    triggers=[
        CronTrigger("0 2 * * *"),
        DataChangeTrigger(change_threshold=0.1),
    ],
    mode="any",                         # any = OR, all = AND
)

# Always/manual
trigger = AlwaysTrigger()               # Always run
trigger = ManualTrigger()               # Only manual
```

### 프로파일 Storage

```python
from truthound.profiler.scheduling.storage import (
    InMemoryProfileStorage,
    FileProfileStorage,
)

# In-memory (development/testing)
storage = InMemoryProfileStorage(max_profiles=100)

# File-based (production)
storage = FileProfileStorage(
    base_path="./profiles",
    max_profiles=100,
    compress=False,                     # Enable gzip
)

# Storage operations
storage.save(profile, metadata={"source": "scheduled"})
last_profile = storage.get_last_profile()
last_run = storage.get_last_run_time()
profiles = storage.list_profiles(limit=10)
```

### SchedulerConfig

```python
from truthound.profiler.scheduling.scheduler import SchedulerConfig

config = SchedulerConfig(
    enable_incremental=True,            # Incremental profiling
    compute_data_hash=True,             # Track data changes
    save_history=True,                  # Save profile history
    on_profile_complete=None,           # Callback on completion
    on_profile_skip=None,               # Callback on skip
    max_history_age_days=30,            # History retention
    context_providers=[],               # Context provider functions
)
```

### IncrementalProfileScheduler

```python
from truthound.profiler.scheduling.scheduler import (
    IncrementalProfileScheduler,
    create_scheduler,
)

# Direct construction
scheduler = IncrementalProfileScheduler(
    trigger=CronTrigger("0 2 * * *"),
    storage=FileProfileStorage("./profiles"),
    config=SchedulerConfig(
        enable_incremental=True,
        compute_data_hash=True,
    ),
)

# Factory function
scheduler = create_scheduler(
    trigger_type="interval",            # interval, cron, manual, always
    storage_type="file",                # memory, file
    hours=1,
    storage_path="./profiles",
    enable_incremental=True,
)

# Run if trigger condition met
profile = scheduler.run_if_needed(data)

# Force run
profile = scheduler.run(data, incremental=True)

# Get next scheduled run
next_run = scheduler.get_next_run_time()

# Get run history
history = scheduler.get_run_history(limit=10)

# Get metrics
metrics = scheduler.get_metrics()
print(f"Total runs: {metrics.total_runs}")
print(f"Incremental: {metrics.incremental_runs}")
print(f"Skipped: {metrics.skipped_runs}")
```

## 스키마 Evolution 설정

### SchemaEvolutionDetector

```python
from truthound.profiler.evolution.detector import SchemaEvolutionDetector
from truthound.profiler.evolution.changes import (
    ChangeType,
    ChangeSeverity,
    CompatibilityLevel,
)

detector = SchemaEvolutionDetector(
    storage=storage,                    # Profile storage for baseline
    detect_renames=True,                # Detect column renames
    rename_similarity_threshold=0.8,    # 80% similarity for rename
)

# Detect changes
changes = detector.detect_changes(
    current_schema=profile.schema,
    baseline_schema=None,               # None = use stored baseline
)

# Get change summary
summary = detector.get_change_summary(changes)
print(f"Breaking changes: {summary.breaking_count}")
print(f"Total changes: {summary.total_count}")
```

실무 운영 가이드에서 Change, Types을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|----------|
| 실무 운영 가이드에서 `COLUMN_ADDED`, COLUMN_ADDED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | New 컬럼 added | 실무 운영 가이드에서 INFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COLUMN_REMOVED`, COLUMN_REMOVED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 removed | 실무 운영 가이드에서 CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COLUMN_RENAMED`, COLUMN_RENAMED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 renamed | 실무 운영 가이드에서 CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TYPE_CHANGED`, TYPE_CHANGED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Variable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NULLABLE_CHANGED`, NULLABLE_CHANGED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Nullability을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Variable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CONSTRAINT_ADDED`, CONSTRAINT_ADDED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 New을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 WARNING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CONSTRAINT_REMOVED`, CONSTRAINT_REMOVED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Constraint을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 WARNING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DEFAULT_CHANGED`, DEFAULT_CHANGED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 INFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ORDER_CHANGED`, ORDER_CHANGED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 order changed | 실무 운영 가이드에서 INFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 Change, Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `INFO`, INFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Non-breaking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `WARNING`, WARNING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Potentially을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CRITICAL`, CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Breaking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

**호환성 Levels:**

| 실무 운영 가이드에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `FULL`, FULL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Forward을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FORWARD`, FORWARD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 New을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `BACKWARD`, BACKWARD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Old을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NONE`, NONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Not을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Type 호환성

실무 운영 가이드에서 Compatible을(를) 다루는 항목입니다:

```
Int8 → Int16, Int32, Int64
Int16 → Int32, Int64
Int32 → Int64
UInt8 → UInt16, UInt32, UInt64
Float32 → Float64
```

## Suite Execution 설정

### SuiteExecutor

```python
from truthound.profiler.integration.executor import (
    SuiteExecutor,
    ExecutionContext,
)

executor = SuiteExecutor(
    parallel=False,                     # Parallel execution
    fail_fast=False,                    # Stop on first failure
    max_workers=None,                   # Max parallel workers
    timeout_seconds=None,               # Execution timeout
    registry=None,                      # Validator registry
    listeners=[],                       # Execution listeners
    progress_reporter=None,             # Progress reporter
)

# Execution context
context = ExecutionContext(
    parallel=False,
    fail_fast=False,
    max_workers=4,
    timeout_seconds=300.0,
    dry_run=False,                      # Dry run mode
)

# Execute suite
result = executor.execute(suite, data, context)

# Async execution
result = await executor.execute_async(suite, data, context)
```

## Complete Example

```python
from truthound.profiler import profile, generate_suite
from truthound.profiler.suite_config import (
    SuiteGeneratorConfig,
    CategoryConfig,
    ConfidenceConfig,
    OutputConfig,
)
from truthound.profiler.scheduling import (
    IncrementalProfileScheduler,
    CronTrigger,
    FileProfileStorage,
    SchedulerConfig,
)
from truthound.profiler.evolution import SchemaEvolutionDetector

# 1. Configure suite generation
suite_config = SuiteGeneratorConfig(
    name="production_validation",
    strictness="strict",
    categories=CategoryConfig(
        include=["schema", "completeness", "distribution"],
        exclude=["pattern"],
    ),
    confidence=ConfidenceConfig(
        min_level="medium",
        include_rationale=True,
    ),
    output=OutputConfig(
        format="yaml",
        group_by_category=True,
        include_metadata=True,
    ),
)

# 2. Configure scheduling
storage = FileProfileStorage("./profiles", max_profiles=100)

scheduler = IncrementalProfileScheduler(
    trigger=CronTrigger("0 2 * * *"),   # Daily at 2 AM
    storage=storage,
    config=SchedulerConfig(
        enable_incremental=True,
        compute_data_hash=True,
        save_history=True,
        max_history_age_days=30,
    ),
)

# 3. Configure schema evolution detection
detector = SchemaEvolutionDetector(
    storage=storage,
    detect_renames=True,
    rename_similarity_threshold=0.8,
)

# 4. Run profiling
profile_result = scheduler.run_if_needed(data)

if profile_result:
    # Check for schema changes
    changes = detector.detect_changes(profile_result.schema)

    if changes:
        summary = detector.get_change_summary(changes)
        print(f"Schema changes detected: {summary.total_count}")

        for change in changes:
            if change.breaking:
                print(f"  BREAKING: {change.description}")

    # Generate validation suite
    suite = generate_suite(data, config=suite_config)
    print(f"Generated {len(suite.validators)} validators")
```

## 환경 변수

```bash
# Suite generation
export TRUTHOUND_SUITE_STRICTNESS=medium
export TRUTHOUND_SUITE_MIN_CONFIDENCE=low
export TRUTHOUND_SUITE_FORMAT=yaml
export TRUTHOUND_SUITE_INCLUDE_CATEGORIES=schema,completeness

# Scheduling
export TRUTHOUND_PROFILE_INTERVAL_HOURS=6
export TRUTHOUND_PROFILE_STORAGE_PATH=./profiles
export TRUTHOUND_PROFILE_MAX_HISTORY_DAYS=30

# Schema evolution
export TRUTHOUND_DETECT_RENAMES=true
export TRUTHOUND_RENAME_THRESHOLD=0.8
```
