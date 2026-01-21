# Profiler Configuration

Truthound profiler analyzes data to generate validation rules automatically.

## Quick Start

```python
from truthound.profiler import profile, generate_suite

# Profile data
profile_result = profile(df)

# Generate validation suite
suite = generate_suite(df, strictness="medium")
```

## SuiteGeneratorConfig

Main configuration for suite generation.

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

Control which validator categories to include.

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

**Available Categories:**

- `schema` - Schema validation
- `completeness` - Null/missing values
- `uniqueness` - Unique constraints
- `distribution` - Value distributions
- `format` - String formats
- `pattern` - Regex patterns
- `range` - Value ranges
- `consistency` - Cross-column consistency

### ConfidenceConfig

Configure confidence level filtering.

```python
from truthound.profiler.suite_config import ConfidenceConfig

config = ConfidenceConfig(
    min_level="low",                    # low, medium, high
    include_rationale=True,             # Include rationale in output
    show_in_output=True,                # Show confidence level
)
```

| Level | Description |
|-------|-------------|
| `low` | All detected rules |
| `medium` | Moderate confidence |
| `high` | High confidence only |

### OutputConfig

Configure output format and style.

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

**Output Formats:**

| Format | Description |
|--------|-------------|
| `yaml` | YAML configuration file |
| `json` | JSON configuration file |
| `python` | Python code |
| `toml` | TOML configuration file |
| `checkpoint` | Checkpoint configuration |

**Python Code Styles:**

| Style | Description |
|-------|-------------|
| `functional` | Function-based validators |
| `class_based` | Class-based validators |
| `declarative` | Declarative definitions |

### GeneratorConfig

Configure rule generators.

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

**Generator Modes:**

| Mode | Description |
|------|-------------|
| `full` | All generators |
| `fast` | Quick analysis only |
| `custom` | Manual selection |

## Configuration Presets

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

**Available Presets:**

| Preset | Strictness | Key Settings |
|--------|-----------|--------------|
| `DEFAULT` | medium | Standard, low confidence |
| `STRICT` | strict | Group by category, medium confidence |
| `LOOSE` | loose | Permissive rules |
| `MINIMAL` | loose | Schema only, high confidence, fast |
| `COMPREHENSIVE` | strict | All categories, low confidence |
| `SCHEMA_ONLY` | medium | Schema + completeness only |
| `FORMAT_ONLY` | medium | Format + pattern only |
| `CI_CD` | medium | Checkpoint output format |
| `DEVELOPMENT` | loose | Python functional style |
| `PRODUCTION` | strict | Grouped, high confidence |

## Loading Configuration

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

**Environment Variables:**

```bash
export TRUTHOUND_SUITE_STRICTNESS=strict
export TRUTHOUND_SUITE_MIN_CONFIDENCE=medium
export TRUTHOUND_SUITE_FORMAT=yaml
export TRUTHOUND_SUITE_INCLUDE_CATEGORIES=schema,completeness
export TRUTHOUND_SUITE_EXCLUDE_CATEGORIES=pattern
```

## Scheduling Configuration

### Profile Triggers

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

### Profile Storage

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

## Schema Evolution Configuration

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

**Change Types:**

| Type | Description | Severity |
|------|-------------|----------|
| `COLUMN_ADDED` | New column added | INFO |
| `COLUMN_REMOVED` | Column removed | CRITICAL |
| `COLUMN_RENAMED` | Column renamed | CRITICAL |
| `TYPE_CHANGED` | Type changed | Variable |
| `NULLABLE_CHANGED` | Nullability changed | Variable |
| `CONSTRAINT_ADDED` | New constraint | WARNING |
| `CONSTRAINT_REMOVED` | Constraint removed | WARNING |
| `DEFAULT_CHANGED` | Default value changed | INFO |
| `ORDER_CHANGED` | Column order changed | INFO |

**Change Severity:**

| Severity | Description |
|----------|-------------|
| `INFO` | Non-breaking change |
| `WARNING` | Potentially breaking |
| `CRITICAL` | Breaking change |

**Compatibility Levels:**

| Level | Description |
|-------|-------------|
| `FULL` | Forward and backward compatible |
| `FORWARD` | New schema can read old data |
| `BACKWARD` | Old schema can read new data |
| `NONE` | Not compatible |

### Type Compatibility

Compatible type changes (non-breaking):

```
Int8 → Int16, Int32, Int64
Int16 → Int32, Int64
Int32 → Int64
UInt8 → UInt16, UInt32, UInt64
Float32 → Float64
```

## Suite Execution Configuration

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

## Environment Variables

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
