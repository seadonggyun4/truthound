<div align="center">
  <img width="100%" alt="Truthound Banner" src="https://raw.githubusercontent.com/seadonggyun4/Truthound/main/docs/assets/truthound_banner.png" />
</div>

<h1 align="center">Truthound</h1>

<p align="center">
  <strong>Zero-Configuration Data Quality Framework Powered by Polars</strong>
</p>

<p align="center">
  <em>Sniffs out bad data.</em>
</p>

<p align="center">
  <a href="https://truthound.netlify.app/"><img src="https://img.shields.io/badge/docs-truthound.netlify.app-blue" alt="Documentation"></a>
  <a href="https://pypi.org/project/truthound/"><img src="https://img.shields.io/pypi/v/truthound" alt="PyPI"></a>
  <a href="https://pypi.org/project/truthound/"><img src="https://img.shields.io/pypi/pyversions/truthound" alt="Python"></a>
</p>

---

## Abstract
<img width="300" height="300" alt="Truthound_icon" src="https://github.com/user-attachments/assets/90d9e806-8895-45ec-97dc-f8300da4d997" />

Truthound is a high-performance data quality validation framework designed for modern data engineering pipelines. The library leverages the computational efficiency of Polars—a Rust-based DataFrame library—to achieve order-of-magnitude performance improvements over traditional Python-based validation solutions.

**Keywords**: Data Quality, Data Validation, Statistical Drift Detection, Anomaly Detection, PII Detection, Polars, Schema Inference

---

## Key Features

| Feature | Description |
|---------|-------------|
| **275+ Validators** | Schema, completeness, uniqueness, distribution, string patterns, datetime, and more |
| **Zero Configuration** | Automatic schema inference with fingerprint-based caching |
| **High Performance** | Polars LazyFrame architecture achieving 1M+ rows/sec throughput |
| **100M+ Scale** | Enterprise sampling strategies with O(1) memory footprint |
| **Parallel Execution** | DAG-based validator orchestration with dependency-aware parallel execution |
| **Custom Validator SDK** | Decorators, fluent builder, testing utilities, and pre-built templates |
| **Statistical Analysis** | 13 drift detection methods, 15 anomaly detection algorithms |
| **Privacy Compliance** | GDPR, CCPA, LGPD, PIPEDA, APPI + plugin-based regulation system |
| **SQL Security** | Multi-level SQL injection protection with parameterized queries |
| **ReDoS Protection** | Regex safety analysis, ML-based prediction, complexity scoring, safe execution, trainable models |
| **Multi-Backend Support** | Polars, Pandas, SQL databases, Spark, and cloud data warehouses |
| **CI/CD Integration** | Native support for 12 CI platforms with checkpoint orchestration, 9 notification providers |
| **Streaming Sources** | Parquet, CSV, JSONL, Arrow IPC, Arrow Flight streaming |
| **Auto-Profiling** | Streaming profiler with schema versioning, distributed processing, and incremental scheduling |
| **Schema Evolution** | Automatic schema change detection, compatibility analysis, and breaking change alerts |
| **Unified Resilience** | Circuit breaker, retry with backoff, bulkhead, rate limiting with fluent builder |
| **Data Docs** | Pipeline engine, 15-language i18n, report versioning, custom templates, white-labeling |
| **Plugin Architecture** | Enterprise plugin system with security sandbox, code signing, version constraints, dependency management, hot reload |
| **Internationalization** | Error messages in 7 languages (EN, KO, JA, ZH, DE, FR, ES) |
| **ML Integration** | Anomaly detection, model monitoring, drift alerting, ReDoS ML training, approximate k-NN, online SVM |
| **Geospatial** | Shapely polygon support with point-in-polygon validation |
| **Data Lineage** | Graph-based lineage tracking, impact analysis, 4 visualization renderers (D3, Cytoscape, Graphviz, Mermaid), OpenLineage integration |
| **Realtime Validation** | Protocol-based streaming with Kafka (aiokafka), Kinesis (aiobotocore), Pub/Sub, Testcontainers integration testing |
| **Distributed Timeout** | Deadline propagation, cascading timeout, and graceful degradation |
| **Enterprise Encryption** | AES-256-GCM, ChaCha20-Poly1305 with key management and streaming |
| **Compression** | gzip, zstd, lz4, snappy, brotli with adaptive selection |
| **Cloud Storage** | S3, GCS, Azure Blob with versioning, retention policies, tiered storage |
| **Data Lifecycle** | TTL/retention policies, automatic tiering (Hot→Warm→Cold), schema migration |
| **Streaming Backpressure** | 6 strategies (Memory, Queue, Latency, TokenBucket, Adaptive, Composite), CircuitBreaker |
| **Batch Write Optimization** | Memory-aware buffering, async batch writer, automatic flush |
| **Result Caching** | LRU, LFU, TTL caches with 4 cache modes (read-through, write-through, write-behind, cache-aside) |
| **Cross-region Replication** | Sync/Async/Semi-Sync modes, conflict resolution, health monitoring, disaster recovery |
| **Job Queue Monitoring** | InMemory, Redis, Prometheus collectors with aggregators and views |
| **Historical Analytics** | Time-bucket aggregation, rollup, trend analysis, forecasting, gap filling |
| **Enterprise SDK** | Sandbox execution, code signing, versioning, licensing, fuzzing |
| **Enterprise i18n** | 15+ languages, CLDR plurals, RTL support, TMS integration |
| **Enterprise Infrastructure** | Structured logging, Prometheus metrics, audit logging, Cloud KMS |

---

## Quick Start

### Installation

```bash
# Basic installation
pip install truthound

# With all optional features
pip install truthound[all]

# With encryption support
pip install truthound[encryption]
```

### Python API

```python
import truthound as th

# Basic validation
report = th.check("data.csv")

# Parallel validation (DAG-based execution)
report = th.check("data.csv", parallel=True, max_workers=4)

# Schema-based validation
schema = th.learn("baseline.csv")
report = th.check("new_data.csv", schema=schema)

# Drift detection
drift = th.compare("train.csv", "production.csv")

# PII scanning and masking
pii_report = th.scan(df)
masked_df = th.mask(df, strategy="hash")

# Statistical profiling
profile = th.profile("data.csv")

# Generate and execute validation suite
from truthound.profiler import generate_suite
suite = generate_suite(profile)
result = suite.execute("new_data.csv", parallel=True)

# Internationalized error messages
from truthound.validators.i18n import set_validator_locale
set_validator_locale("ko")  # Korean messages
```

### CLI

```bash
truthound check data.csv                    # Validate
truthound check data.csv --strict           # CI/CD mode
truthound compare baseline.csv current.csv  # Drift detection
truthound scan data.csv                     # PII scanning
truthound auto-profile data.csv -o profile.json  # Profiling
truthound docs generate profile.json -o report.html  # HTML report

# Code scaffolding (new)
truthound new validator my_validator        # Create validator
truthound new reporter json_export          # Create reporter
truthound new plugin my_plugin              # Create plugin
truthound new list --verbose                # List available templates
```

---

## Performance

| Operation | 10M Rows | 100M Rows | Throughput |
|-----------|----------|-----------|------------|
| `th.check()` | 9.4s | ~94s | 1.06M rows/sec |
| `th.profile()` | 0.15s | - | 66.7M rows/sec |
| `th.learn()` | 0.27s | - | 37.0M rows/sec |

**Enterprise Scale Features:**
- **DAG-based Parallel Execution**: Dependency-aware validator orchestration with `parallel=True`
- **Enterprise Sampling (100M+ rows)**: Block, multi-stage, column-aware, and progressive sampling strategies
- **Validator Performance Profiling**: Per-validator timing, memory, and throughput metrics with Prometheus export
- **Memory-Aware Processing**: O(1) memory footprint with backpressure and time budgets
- Streaming validation with bounded memory (~300MB for 100GB files)
- Sampling-based validation for ML anomaly detection
- Approximate k-NN (Annoy, HNSW, Faiss) for memory-efficient LOF
- LRU cache for reference data statistics

---

## Documentation

### Getting Started
- **[Getting Started Guide](docs/GETTING_STARTED.md)** — Installation, quick start, and basic usage

### Core Concepts
- **[Architecture Overview](docs/ARCHITECTURE.md)** — System design and core principles
- **[Validators Reference](docs/VALIDATORS.md)** — Complete reference for all 265+ validators
- **[Statistical Methods](docs/STATISTICAL_METHODS.md)** — Mathematical foundations for drift and anomaly detection

### Features by Phase

| Phase | Documentation | Description |
|-------|---------------|-------------|
| **Phase 1-3** | [Core Validators](docs/VALIDATORS.md) | 275 validators across 22 categories |
| **Phase 4** | [Storage & Reporters](docs/STORES.md), [Reporters](docs/REPORTERS.md) | Persistence and output formats |
| **Phase 5** | [Data Sources](docs/DATASOURCES.md) | Multi-backend support (BigQuery, Snowflake, etc.) |
| **Phase 6** | [Checkpoint & CI/CD](docs/CHECKPOINT.md) | Saga pattern, 9 notification providers, job monitoring, analytics |
| **Phase 7** | [Auto-Profiling](docs/PROFILER.md) | Streaming profiler with distributed processing |
| **Phase 8** | [Data Docs](docs/DATADOCS.md) | Pipeline engine, i18n, versioning, themes, PDF |
| **Phase 9** | [Plugin Architecture](docs/PLUGINS.md) | Enterprise plugins with security, versioning, hot reload |
| **Phase 10** | [Advanced Features](docs/ADVANCED.md) | ML monitoring, Lineage visualization, OpenLineage, Protocol-based streaming |
| **Enterprise** | [Enterprise Features](#enterprise-features) | SDK, Security, i18n, Infrastructure |

### Reference
- **[API Reference](docs/API_REFERENCE.md)** — Complete API documentation
- **[Performance Guide](docs/PERFORMANCE.md)** — Benchmarks and optimization strategies
- **[Examples](docs/EXAMPLES.md)** — Usage examples and patterns
- **[Test Coverage](docs/TEST_COVERAGE.md)** — 4,200+ tests across all features

---

## Enterprise Features

Truthound provides comprehensive enterprise-grade features for production deployments. All features are fully tested with 550+ dedicated tests.

### CLI Code Scaffolding

Generate production-ready code templates for validators, reporters, and plugins:

```bash
# Create a new validator with different templates
truthound new validator my_validator                    # Basic validator
truthound new validator null_check --template column    # Column-level validator
truthound new validator email_format --template pattern --pattern "^[a-z@.]+$"
truthound new validator percentage --template range --min 0 --max 100
truthound new validator date_order --template comparison
truthound new validator full_check --template full --docs --author "John Doe"

# Create a new reporter
truthound new reporter json_export --extension .json --content-type application/json
truthound new reporter detailed_report --template full --docs

# Create a new plugin
truthound new plugin my_validators                       # Validator plugin
truthound new plugin custom_reports --type reporter      # Reporter plugin
truthound new plugin monitoring --type hook              # Hook plugin
truthound new plugin custom_db --type datasource         # DataSource plugin
truthound new plugin notify --type action                # Checkpoint action plugin
truthound new plugin enterprise --type full              # Full-featured plugin

# List available scaffolds and templates
truthound new list --verbose
truthound new templates validator
```

**Validator Templates:**

| Template | Description | Use Case |
|----------|-------------|----------|
| `basic` | Minimal validator structure | Quick prototypes |
| `column` | Column-level validation with target column support | Per-column checks |
| `pattern` | Regex pattern matching with safety checks | Format validation |
| `range` | Numeric range validation | Value bounds |
| `comparison` | Cross-column comparison | Column relationships |
| `composite` | Multi-validator composite | Complex validations |
| `full` | Full-featured with tests and docs | Production validators |

**Plugin Types:**

| Type | Description | Components |
|------|-------------|------------|
| `validator` | Custom validators | Validator classes, registration |
| `reporter` | Custom report formats | Reporter classes, config |
| `hook` | Event hooks | Before/after callbacks |
| `datasource` | Data connectors | Connection, read, schema |
| `action` | Checkpoint actions | Post-validation actions |
| `full` | All components | Validators + reporters + hooks |

**Generated Structure (Validator Example):**

```
my_validator/
├── __init__.py              # Package exports
├── validator.py             # Validator implementation
├── tests/
│   ├── __init__.py
│   └── test_validator.py    # Pytest test cases
├── docs/
│   └── README.md            # Documentation (with --docs)
└── examples/
    └── basic_usage.py       # Usage examples (with --template full)
```

**Generated Structure (Plugin Example):**

```
truthound-plugin-my_plugin/
├── pyproject.toml           # Package configuration with entry points
├── README.md                # Plugin documentation
├── my_plugin/
│   ├── __init__.py          # Package exports
│   └── plugin.py            # Plugin implementation
└── tests/
    ├── __init__.py
    └── test_plugin.py       # Plugin tests
```

**Extensibility:**

The scaffolding system uses a registry pattern for easy extension:

```python
from truthound.cli_modules.scaffolding import (
    BaseScaffold,
    register_scaffold,
    ScaffoldConfig,
    ScaffoldResult,
)

@register_scaffold(
    name="my_scaffold",
    description="My custom scaffold",
    aliases=("ms",),
)
class MyScaffold(BaseScaffold):
    TEMPLATE_VARIANTS = ("basic", "advanced")

    def _generate_files(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        result.add_file(f"{config.name}/main.py", self._generate_content(config))

# Now available via CLI:
# truthound new my_scaffold my_component
```

### Custom Validator SDK

Build custom validators with a powerful, intuitive SDK:

```python
from truthound.validators.sdk import custom_validator, ValidatorBuilder
from truthound.validators.sdk.testing import ValidatorTestCase

# Decorator-based approach
@custom_validator(
    name="check_positive",
    category="numeric",
    description="Validates all values are positive"
)
def check_positive(df, column: str, strict: bool = True):
    """Custom validator using decorator."""
    values = df[column]
    if strict:
        return (values > 0).all()
    return (values >= 0).all()

# Fluent builder approach
validator = (
    ValidatorBuilder("revenue_check")
    .with_category("business")
    .with_column("revenue")
    .with_condition(lambda df: df["revenue"] > 0)
    .with_error_message("Revenue must be positive")
    .build()
)

# Testing utilities
class TestMyValidator(ValidatorTestCase):
    def test_positive_values(self):
        df = self.create_df({"value": [1, 2, 3]})
        result = self.run_validator(check_positive, df, "value")
        self.assert_passed(result)
```

### Enterprise SDK

Production-ready features for enterprise deployments:

```python
from truthound.validators.sdk.enterprise import (
    SandboxExecutor, SandboxConfig,
    ResourceLimits, ResourceMonitor,
    SignatureManager, ValidatorSignature,
    VersionChecker, SemanticVersion,
    LicenseManager, LicenseInfo,
    FuzzRunner, FuzzConfig,
)

# 1. Sandbox Execution - Isolated validator execution
config = SandboxConfig(
    backend="subprocess",  # or "docker" for full isolation
    timeout=30.0,
    memory_limit_mb=512,
    cpu_limit=1.0,
    allowed_imports=["polars", "numpy"],
    blocked_imports=["os", "subprocess", "socket"],
)
executor = SandboxExecutor(config)
result = executor.execute(my_validator, df, column="value")

# 2. Resource Limits - Prevent runaway validators
limits = ResourceLimits(
    max_memory_mb=1024,
    max_cpu_percent=80.0,
    max_wall_time_seconds=60.0,
    max_file_descriptors=100,
)
monitor = ResourceMonitor(limits)
with monitor.track():
    result = expensive_validator(large_df)

# 3. Code Signing - Verify validator integrity
signer = SignatureManager(algorithm="ed25519")
signature = signer.sign(
    validator_code,
    private_key=my_private_key,
    metadata={"author": "data-team", "version": "1.0.0"}
)
is_valid = signer.verify(validator_code, signature, public_key)

# 4. Version Compatibility - Semver-based checking
checker = VersionChecker()
checker.add_constraint("truthound", ">=0.2.0,<1.0.0")
checker.add_constraint("python", ">=3.11")
is_compatible = checker.check_compatibility()

# 5. License Management - Track commercial validators
license_mgr = LicenseManager(secret_key="your-secret")
license_key = license_mgr.generate_key(
    license_type="enterprise",
    features=["ml-validators", "streaming"],
    max_rows=100_000_000,
    expires_at=datetime(2025, 12, 31),
)
is_valid = license_mgr.validate_key(license_key)

# 6. Fuzz Testing - Find edge cases automatically
fuzzer = FuzzRunner(FuzzConfig(
    max_iterations=1000,
    strategies=["random", "boundary", "mutation"],
    timeout_per_test=1.0,
))
results = fuzzer.run(my_validator, column_type="string")
print(f"Found {len(results.failures)} edge cases")
```

### ReDoS Protection

Comprehensive protection against Regular Expression Denial of Service:

```python
from truthound.validators.security.redos import (
    # Core protection
    ReDoSChecker, SafeRegexExecutor,
    # ML-based analysis
    MLPatternAnalyzer, FeatureExtractor,
    # Pattern optimization
    PatternOptimizer, OptimizationResult,
    # CVE database
    CVEDatabase, CVEMatch,
    # Runtime monitoring
    CPUMonitor, PatternProfiler,
    # Alternative engine
    RE2Engine,
)

# Basic safety check
checker = ReDoSChecker()
result = checker.analyze(r"(a+)+$")
print(f"Safe: {result.is_safe}, Score: {result.complexity_score}")

# ML-based prediction (32 features)
analyzer = MLPatternAnalyzer()
prediction = analyzer.predict(r"(.*a){10}")
print(f"Risk: {prediction.risk_probability:.2%}")
print(f"Confidence: {prediction.confidence:.2%}")

# Auto-optimize dangerous patterns
optimizer = PatternOptimizer()
result = optimizer.optimize(r"(a+)+$")
print(f"Original: {result.original}")
print(f"Optimized: {result.optimized}")  # -> "a+"
print(f"Transformations: {result.transformations}")

# Check against CVE database
cve_db = CVEDatabase()
matches = cve_db.check(r"(a+)+$")
for match in matches:
    print(f"CVE: {match.cve_id}, Severity: {match.severity}")

# Safe execution with CPU monitoring
executor = SafeRegexExecutor(
    timeout=1.0,
    max_cpu_percent=50.0,
)
match = executor.match(pattern, text)

# Use RE2 for guaranteed linear time
re2 = RE2Engine()
if re2.is_available():
    match = re2.match(pattern, text)  # O(n) guaranteed
```

### ReDoS ML Training Framework

Train and deploy ML models for ReDoS vulnerability prediction:

```python
from truthound.validators.security.redos.ml import (
    # High-level API
    ReDoSMLPredictor,
    train_redos_model,
    load_trained_model,
    # Training pipeline
    TrainingPipeline, TrainingConfig, TrainingResult,
    # Models
    RandomForestReDoSModel, GradientBoostingReDoSModel, EnsembleReDoSModel,
    # Feature extraction
    PatternFeatureExtractor, PatternFeatures,
    # Dataset generation
    ReDoSDatasetGenerator, generate_training_dataset,
    # Model storage
    ModelStorage, save_model, load_model,
)

# Quick start - predict with default model
predictor = ReDoSMLPredictor()
result = predictor.predict(r"(a+)+b")
print(f"Risk: {result.risk_level.name}")  # CRITICAL
print(f"Probability: {result.risk_probability:.2%}")
print(f"Confidence: {result.confidence:.2%}")

# Train custom model with your data
patterns = [r"(a+)+", r"^[a-z]+$", r"(x+x+)+y", r"\\d{3}-\\d{4}"]
labels = [1, 0, 1, 0]  # 1=vulnerable, 0=safe

predictor = train_redos_model(patterns, labels)
predictor.save("my_model.pkl")

# Auto-train using generated dataset
predictor = ReDoSMLPredictor()
result = predictor.auto_train(n_samples=1000)
print(f"Accuracy: {result.metrics.accuracy:.2%}")
print(f"F1 Score: {result.metrics.f1_score:.2%}")

# Advanced: Training pipeline with cross-validation
config = TrainingConfig(
    model_type="random_forest",
    cv_folds=5,
    test_split=0.2,
)
pipeline = TrainingPipeline(config=config)

dataset = generate_training_dataset(n_samples=500)
result = pipeline.train(dataset)

print(f"Cross-validation scores: {result.cv_result.fold_metrics}")
print(f"Feature importance: {result.model.get_feature_importance_dict()}")

# Load and use trained model
predictor = load_trained_model("my_model.pkl")
is_safe = predictor.is_safe(r"^\\w+@\\w+\\.\\w+$")
```

**ML Framework Features:**

| Component | Description |
|-----------|-------------|
| **PatternFeatureExtractor** | Extracts 32 features from regex patterns |
| **RuleBasedReDoSModel** | Fast baseline, always available |
| **RandomForestReDoSModel** | sklearn Random Forest with rule-based fallback |
| **GradientBoostingReDoSModel** | sklearn Gradient Boosting with rule-based fallback |
| **EnsembleReDoSModel** | Combines rule-based + ML + signature matching |
| **TrainingPipeline** | Cross-validation, metrics, model comparison |
| **ModelStorage** | Versioned storage with metadata tracking |
| **ReDoSDatasetGenerator** | Built-in vulnerable/safe pattern collections |

### Distributed Timeout

Enterprise-grade timeout management for distributed systems:

```python
from truthound.validators.timeout import (
    # Core components
    DeadlineContext, TimeoutBudget, with_deadline,
    CascadeTimeoutHandler, CascadeLevel,
    GracefulDegradation, DegradationPolicy,
    DistributedTimeoutManager,
)
from truthound.validators.timeout.advanced import (
    # Advanced features
    TelemetryIntegration,  # OpenTelemetry
    PerformancePredictor,  # ML-based prediction
    AdaptiveSampler,       # Data-aware sampling
    PriorityExecutor,      # Priority queues
    RetryPolicy,           # Exponential backoff
    SLAMonitor,            # SLA tracking
    RedisBackend,          # Distributed coordination
    CircuitBreaker,        # Fault tolerance
)

# Basic deadline propagation
with DeadlineContext(timeout=30.0) as ctx:
    result = validate_large_dataset(df)
    print(f"Remaining: {ctx.remaining_time():.1f}s")

# Cascading timeout levels
handler = CascadeTimeoutHandler(
    levels=[
        CascadeLevel("full", timeout=60.0, validators="all"),
        CascadeLevel("critical", timeout=30.0, validators="critical_only"),
        CascadeLevel("minimal", timeout=10.0, validators="schema_only"),
    ]
)
result = handler.execute(validators, df)

# Graceful degradation
degradation = GracefulDegradation(
    policy=DegradationPolicy.PROGRESSIVE,
    fallback_actions={
        "timeout": lambda: {"status": "partial", "validated": False},
        "memory": lambda: sample_and_validate(df, ratio=0.1),
    }
)
result = degradation.execute(validate, df)

# Priority-based execution
executor = PriorityExecutor()
executor.submit(schema_validator, priority=1)   # Highest
executor.submit(business_validator, priority=5) # Lower
results = executor.run_all(timeout=30.0)

# SLA monitoring with alerts
sla = SLAMonitor(
    targets={
        "p50_latency_ms": 100,
        "p99_latency_ms": 500,
        "error_rate": 0.01,
    },
    alert_callback=send_pagerduty_alert,
)
with sla.track("validation_job"):
    result = validate(df)

# Circuit breaker for external dependencies
breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_requests=3,
)
with breaker.protect():
    result = call_external_validator(df)
```

### Enterprise i18n

Comprehensive internationalization with 15+ languages:

```python
from truthound.validators.i18n import (
    set_validator_locale, get_message, MessageCatalog,
)
from truthound.validators.i18n.enterprise import (
    # CLDR pluralization
    PluralRules, PluralCategory,
    # RTL support
    BiDiHandler, TextDirection,
    # Locale-aware formatting
    LocaleFormatter, NumberFormat, DateFormat,
    # Regional dialects
    DialectManager, LocaleChain,
    # TMS integration
    TMSClient, CrowdinProvider, LokaliseProvider,
    # Dynamic loading
    CatalogLoader, CatalogCache,
)

# Basic usage - 7 core languages
set_validator_locale("ko")  # Korean
error = get_message("null_values_found", count=5)
# -> "5개의 null 값이 발견되었습니다"

# CLDR plural rules (handles complex cases like Russian, Arabic)
rules = PluralRules("ru")
category = rules.select(5)  # -> PluralCategory.FEW
message = get_message("rows_found", count=5, locale="ru")
# -> "Найдено 5 строк" (correct Russian plural)

# RTL language support
bidi = BiDiHandler()
text = bidi.format("Error in column: {name}", name="عمود_البيانات", locale="ar")
# Correctly handles RTL text with LTR embeddings

# Locale-aware number/date formatting
formatter = LocaleFormatter("de-DE")
print(formatter.format_number(1234567.89))  # -> "1.234.567,89"
print(formatter.format_currency(99.99, "EUR"))  # -> "99,99 €"
print(formatter.format_date(datetime.now()))  # -> "28.12.2025"

# Regional dialect support
dialects = DialectManager()
dialects.register("en-US", "en-GB")  # American -> British fallback
dialects.register("zh-TW", "zh-CN")  # Traditional -> Simplified
message = get_message("color_mismatch", locale="en-GB")
# -> "Colour values do not match" (British spelling)

# TMS integration for translation management
tms = TMSClient(CrowdinProvider(api_key="..."))
await tms.push_source_strings(catalog)
await tms.pull_translations(languages=["ja", "ko", "zh"])

# Dynamic catalog loading (lazy load by language)
loader = CatalogLoader(cache_size=10)
catalog = loader.load("ko", context="validation_errors")
```

**Supported Languages:**

| Category | Languages |
|----------|-----------|
| **Core (7)** | English, Korean, Japanese, Chinese, German, French, Spanish |
| **Extended (8)** | Portuguese (BR/PT), Italian, Russian, Arabic, Hebrew, Persian, Turkish, Polish |

### Enterprise Infrastructure

Production-ready infrastructure components:

```python
from truthound.infrastructure import (
    # Structured logging
    EnterpriseLogger, LogConfig, CorrelationContext,
    # Prometheus metrics
    MetricsManager, ValidatorMetrics, MetricsServer,
    # Environment configuration
    ConfigManager, ConfigProfile, Environment,
    # Audit logging
    EnterpriseAuditLogger, ComplianceReporter,
    # Data encryption
    AtRestEncryption, FieldLevelEncryption,
    AWSKMSProvider, GCPKMSProvider, AzureKeyVaultProvider,
)

# 1. Structured Logging with correlation IDs
logger = EnterpriseLogger(LogConfig(
    format="json",
    level="INFO",
    sinks=["console", "elasticsearch"],
    elasticsearch_url="http://elk:9200",
))
with CorrelationContext(request_id="req-123"):
    logger.info("Validation started", dataset="sales_data", rows=1000000)
    # -> {"timestamp": "...", "correlation_id": "req-123", "dataset": "sales_data", ...}

# 2. Prometheus Metrics
metrics = MetricsManager()
metrics.validator_executions.labels(name="null_check", status="success").inc()
metrics.validation_duration.labels(dataset="sales").observe(1.5)

# Start metrics HTTP server
server = MetricsServer(port=9090)
server.start()  # -> http://localhost:9090/metrics

# 3. Environment Configuration
config = ConfigManager()
config.add_source(Environment())  # ENV vars
config.add_source(ConfigProfile("config/production.yaml"))
config.add_source(VaultProvider("secret/truthound"))  # HashiCorp Vault

db_url = config.get("database.url")
api_key = config.get_secret("api_key")

# Hot reload on config changes
config.watch(callback=on_config_change)

# 4. Audit Logging for Compliance
audit = EnterpriseAuditLogger(
    storage=["elasticsearch", "s3"],
    retention_days=365,
)
audit.log_operation(
    action="VALIDATION_RUN",
    user="data-engineer@company.com",
    resource="customer_pii_table",
    result="PASSED",
    metadata={"rows": 1000000, "duration_ms": 5432},
)

# Generate compliance reports
reporter = ComplianceReporter(audit)
soc2_report = reporter.generate("SOC2", date_range=("2025-01-01", "2025-12-31"))
gdpr_report = reporter.generate("GDPR", data_subject="user@example.com")

# 5. Data Encryption with Cloud KMS
# AWS KMS
encryption = AtRestEncryption(
    provider=AWSKMSProvider(
        key_id="arn:aws:kms:us-east-1:123456:key/abc-123",
        region="us-east-1",
    )
)
encrypted_report = encryption.encrypt(validation_report)

# Field-level encryption for sensitive columns
field_encryption = FieldLevelEncryption(
    provider=AzureKeyVaultProvider(vault_url="https://myvault.vault.azure.net"),
    fields={
        "ssn": "key-ssn",
        "credit_card": "key-cc",
    }
)
encrypted_df = field_encryption.encrypt_dataframe(df)
```

### Infrastructure Summary

| Component | Features | Integrations |
|-----------|----------|--------------|
| **Logging** | JSON format, correlation IDs, async logging | ELK, Loki, Fluentd |
| **Metrics** | Counter, Gauge, Histogram, Summary | Prometheus, Grafana |
| **Config** | Profiles, hot reload, validation | Vault, AWS Secrets, Azure |
| **Audit** | Full operation tracking, retention | Elasticsearch, S3, Kafka |
| **Encryption** | AES-256-GCM, field-level, streaming | AWS/GCP/Azure KMS, Vault |

### Enterprise Storage Features

Production-ready storage capabilities for high-throughput and disaster recovery:

```python
from truthound.stores.backpressure import (
    BackpressureConfig, MemoryBasedBackpressure,
    CompositeBackpressure, CircuitBreaker,
)
from truthound.stores.batching import BatchConfig, AsyncBatchWriter, BatchedStore
from truthound.stores.caching import CacheConfig, LRUCache, CachedStore, CacheMode
from truthound.stores.replication import (
    ReplicationConfig, ReplicationMode, ConflictResolution,
    ReplicaTarget, ReplicatedStore,
)

# 1. Streaming Backpressure - Prevent overwhelming downstream systems
backpressure = CompositeBackpressure([
    MemoryBasedBackpressure(BackpressureConfig(memory_threshold_percent=80.0)),
    QueueDepthBackpressure(BackpressureConfig(queue_depth_threshold=10000)),
])
if backpressure.should_pause():
    backpressure.apply_backoff()

# Circuit Breaker for fault tolerance
breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_requests=3,
))
result = await breaker.call(external_service.save, data)

# 2. Batch Write Optimization - Maximize throughput
writer = AsyncBatchWriter(
    store,
    BatchConfig(
        batch_size=1000,
        flush_interval_seconds=5.0,
        max_buffer_memory_mb=100,
        parallelism=4,
    ),
)
await writer.start_auto_flush()
for result in results:
    await writer.add(result)  # Buffered, batched writes

# 3. Result Caching - Reduce read latency
cached_store = CachedStore(
    store=underlying_store,
    cache=LRUCache(CacheConfig(max_size=10000, ttl_seconds=300)),
    mode=CacheMode.READ_THROUGH,
)
# Automatic cache population on reads
result = cached_store.get(item_id)  # Cached on first access

# 4. Cross-region Replication - Disaster recovery
replica_eu = ReplicaTarget(name="eu", store=eu_store, region="eu-west-1")
replica_asia = ReplicaTarget(name="asia", store=asia_store, region="ap-northeast-1")

replicated_store = ReplicatedStore(
    primary=us_store,
    config=ReplicationConfig(
        mode=ReplicationMode.SEMI_SYNC,  # Wait for at least 1 replica
        min_sync_replicas=1,
        targets=[replica_eu, replica_asia],
        conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
        read_preference=ReadPreference.NEAREST,  # Read from lowest latency
    ),
)
async with replicated_store:
    await replicated_store.save_async(result)  # Auto-replicated
    stats = replicated_store.get_stats()  # Health and metrics
```

**Backpressure Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **MemoryBased** | Monitors system memory usage | Memory-constrained environments |
| **QueueDepth** | Tracks pending write queue size | High-throughput ingestion |
| **LatencyBased** | Measures operation latency | Latency-sensitive workloads |
| **TokenBucket** | Rate limiting with burst support | API rate limiting |
| **Adaptive** | Auto-adjusts based on history | Variable workloads |
| **Composite** | Combines multiple strategies | Production deployments |

**Replication Modes:**

| Mode | Behavior | Use Case |
|------|----------|----------|
| **SYNC** | Wait for all replicas | Strong consistency |
| **ASYNC** | Fire and forget | High throughput |
| **SEMI_SYNC** | Wait for N replicas | Balance consistency/performance |

### Data Docs - Enterprise Reporting

Generate beautiful, internationalized reports with versioning and custom branding:

```python
from truthound.datadocs import (
    # Pipeline engine
    ReportPipeline, ReportContext, ComponentRegistry,
    # I18n support
    ReportCatalog, PluralRules, NumberFormatter, DateFormatter,
    # Versioning
    InMemoryVersionStorage, FileVersionStorage,
    IncrementalStrategy, SemanticStrategy, TimestampStrategy, GitLikeStrategy,
    ReportDiffer, diff_versions,
    # Themes & Renderers
    HTMLReportBuilder, generate_html_report,
)

# 1. Multi-language Reports with CLDR plurals
catalog = ReportCatalog(locale="ko")
message = catalog.get("validation_passed", count=5)
# -> "5개 검증 통과"

# 2. Locale-aware formatting
formatter = NumberFormatter(locale="de-DE")
print(formatter.format(1234567.89))  # -> "1.234.567,89"

date_formatter = DateFormatter(locale="ja-JP")
print(date_formatter.format(datetime.now()))  # -> "2025年12月28日"

# 3. Report versioning with diff
storage = FileVersionStorage("/reports")
v1 = storage.save("my_report", report_html, message="Initial version")
v2 = storage.save("my_report", updated_html, message="Updated charts")

# Compare versions
differ = ReportDiffer()
diff_result = differ.compare(v1, v2)
print(f"Changes: {diff_result.added_count} added, {diff_result.removed_count} removed")
print(differ.format_diff(diff_result, format="unified"))

# Rollback to previous version
old_version = storage.load("my_report", version=1)

# 4. Custom branded reports
builder = HTMLReportBuilder(theme=ReportTheme.PROFESSIONAL)
builder.set_branding(
    logo_url="https://company.com/logo.png",
    primary_color="#1a365d",
    company_name="Acme Corp",
)
html = builder.build(profile)
```

**Supported Languages (15):**

| Core (7) | Extended (8) |
|----------|--------------|
| English, Korean, Japanese | Portuguese (BR/PT), Italian, Russian |
| Chinese, German, French, Spanish | Arabic, Hebrew, Turkish, Polish |

**Versioning Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `IncrementalStrategy` | Simple 1, 2, 3... numbering | Most common |
| `SemanticStrategy` | 0.0.1, 0.1.0, 1.0.0 format | API/schema changes |
| `TimestampStrategy` | Millisecond timestamps | Audit trails |
| `GitLikeStrategy` | Content-based hashing | Deduplication |

### Schema Evolution Detection

Automatically detect and alert on schema changes in your data pipelines:

```python
from truthound.profiler.evolution import (
    SchemaEvolutionDetector,
    SchemaCompatibilityAnalyzer,
    SchemaChangeAlertManager,
    ConsoleAlertHandler,
)

# Detect schema changes with rename detection
detector = SchemaEvolutionDetector(detect_renames=True)
changes = detector.detect_changes(current_schema, baseline_schema)

for change in changes:
    print(f"{change.change_type}: {change.column}")
    if change.breaking:
        print(f"  ⚠️ BREAKING CHANGE: {change.description}")

# Analyze compatibility level
analyzer = SchemaCompatibilityAnalyzer()
result = analyzer.analyze(old_schema, new_schema)
print(f"Compatible: {result.is_compatible}")
print(f"Level: {result.level}")  # FULL, BACKWARD, FORWARD, NONE

# Alert on breaking changes
manager = SchemaChangeAlertManager(handlers=[
    ConsoleAlertHandler(use_colors=True),
])
manager.alert_if_breaking(changes)
```

**Change Types Detected:**

| Type | Description | Breaking |
|------|-------------|----------|
| `COLUMN_ADDED` | New column added | No |
| `COLUMN_REMOVED` | Column deleted | Yes |
| `TYPE_CHANGED` | Column type changed | Depends |
| `COLUMN_RENAMED` | Column renamed (detected via similarity) | Yes |
| `NULLABLE_CHANGED` | Nullability changed | Depends |

### Unified Resilience Patterns

Enterprise-grade resilience patterns consolidated in `common/resilience/`:

```python
from truthound.common.resilience import (
    # Circuit Breaker
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState,
    CircuitBreakerRegistry,
    # Retry
    RetryPolicy, RetryConfig, RetryExhaustedError,
    ExponentialBackoff, LinearBackoff, ConstantBackoff,
    # Bulkhead
    SemaphoreBulkhead, BulkheadConfig, BulkheadFullError,
    # Rate Limiting
    TokenBucketRateLimiter, FixedWindowRateLimiter, RateLimiterConfig,
    # Composite Builder
    ResilienceBuilder, ResilientWrapper,
)

# 1. Circuit Breaker - Prevent cascading failures
cb = CircuitBreaker("my-service", CircuitBreakerConfig.aggressive())
with cb.protect():
    external_service.call()

# 2. Retry with exponential backoff
retry = RetryPolicy(RetryConfig.exponential())
result = retry.execute(flaky_function)

# 3. Bulkhead - Limit concurrent operations
bulkhead = SemaphoreBulkhead("db-pool", BulkheadConfig(max_concurrent=10))
with bulkhead.limit():
    database.query()

# 4. Rate Limiting - Control throughput
limiter = TokenBucketRateLimiter("api", RateLimiterConfig.per_second(100))
with limiter.limit():
    api.call()

# 5. Combined patterns with fluent builder
wrapper = (
    ResilienceBuilder("database")
    .with_rate_limit(RateLimiterConfig.per_second(100))
    .with_bulkhead(BulkheadConfig(max_concurrent=10))
    .with_circuit_breaker(CircuitBreakerConfig.for_database())
    .with_retry(RetryConfig.quick())
    .build()
)

@wrapper
def protected_operation():
    return database.execute_query()

# Presets for common use cases
db_wrapper = ResilienceBuilder.for_database("postgres")
api_wrapper = ResilienceBuilder.for_external_api("payment-gateway")
simple_wrapper = ResilienceBuilder.simple("my-service")
```

**Resilience Pattern Summary:**

| Pattern | Purpose | Use Case |
|---------|---------|----------|
| **Circuit Breaker** | Fail fast when downstream is unhealthy | External APIs, databases |
| **Retry** | Handle transient failures | Network timeouts, rate limits |
| **Bulkhead** | Isolate resources | Connection pools, thread pools |
| **Rate Limiter** | Control throughput | API rate limiting, quota management |
| **Composite** | Combine multiple patterns | Production services |

### Enterprise Plugin Architecture

Secure, extensible plugin system with enterprise-grade features:

```python
from truthound.plugins import (
    # Core Plugin System
    PluginManager, PluginRegistry, PluginDiscovery,
    Plugin, PluginConfig, PluginType,
    # Security
    SecurityPolicyPresets, create_policy, IsolationLevel,
    SandboxFactory, SigningServiceImpl, SignatureAlgorithm,
    TrustStoreImpl, TrustLevel, create_verification_chain,
    # Versioning & Dependencies
    parse_constraint, VersionConstraint,
    DependencyGraph, DependencyResolver, DependencyType,
    # Lifecycle
    LifecycleManager, HotReloadManager, ReloadStrategy,
    # Documentation
    DocumentationExtractor, render_documentation,
    # Enterprise Manager
    create_enterprise_manager,
)

# 1. Create enterprise plugin manager with security preset
manager = create_enterprise_manager(
    security_preset=SecurityPolicyPresets.ENTERPRISE,
)

# 2. Load plugin with signature verification
await manager.load("my-plugin", verify_signature=True)

# 3. Execute plugin in sandbox
result = await manager.execute_in_sandbox(
    "my-plugin",
    my_function,
    arg1, arg2,
)

# 4. Hot reload without restart
await manager.reload("my-plugin")

# 5. Generate plugin documentation
docs = manager.generate_docs("my-plugin")

# Custom security policy
policy = create_policy(
    preset="standard",
    isolation_level="container",
    max_memory_mb=512,
    max_execution_time_sec=30,
    allow_network=False,
    blocked_modules=["os", "subprocess", "socket"],
)

# Plugin signing with Ed25519
service = SigningServiceImpl(
    algorithm=SignatureAlgorithm.ED25519,
    signer_id="my-org",
)
signature = service.sign(
    plugin_path=Path("my_plugin/"),
    private_key=private_key,
)

# Verify with trust chain
trust_store = TrustStoreImpl()
trust_store.set_signer_trust("my-org", TrustLevel.TRUSTED)
chain = create_verification_chain(trust_store=trust_store)
result = chain.verify(plugin_path, signature, context={})

# Version constraints (semver)
constraint = parse_constraint("^1.2.3")  # >=1.2.3 && <2.0.0
constraint.is_satisfied_by("1.5.0")  # True
constraint.is_satisfied_by("2.0.0")  # False

# Dependency management
graph = DependencyGraph()
graph.add_node("plugin-c", "1.0.0")
graph.add_node("plugin-b", "1.0.0",
    dependencies={"plugin-c": DependencyType.REQUIRED})
graph.add_node("plugin-a", "1.0.0",
    dependencies={"plugin-b": DependencyType.REQUIRED})

order = graph.get_load_order()  # ['plugin-c', 'plugin-b', 'plugin-a']
cycles = graph.detect_cycles()  # Circular dependency detection

# Hot reload with rollback
reload_manager = HotReloadManager(
    lifecycle_manager,
    default_strategy=ReloadStrategy.GRACEFUL,
)
await reload_manager.watch("my-plugin", Path("plugins/my-plugin/"), auto_reload=True)
result = await reload_manager.reload("my-plugin")
if not result.success and result.rolled_back:
    print("Rolled back to previous version")
```

**Security Policy Presets:**

| Preset | Isolation | Network | Subprocess | Signatures | Use Case |
|--------|-----------|---------|------------|------------|----------|
| **DEVELOPMENT** | None | Yes | Yes | 0 | Local dev |
| **TESTING** | None | Yes | Yes | 0 | Test environment |
| **STANDARD** | Process | No | No | 1 | General production |
| **ENTERPRISE** | Process | No | No | 1 + trusted | Enterprise |
| **STRICT** | Container | No | No | 2 + trusted | High-security |
| **AIRGAPPED** | Container | No | No | 2 + trusted + blocked syscalls | Air-gapped |

**Sandbox Engines:**

| Engine | Isolation Level | Features |
|--------|----------------|----------|
| **NoopSandboxEngine** | NONE | Timeout only, development use |
| **ProcessSandboxEngine** | PROCESS | Subprocess isolation, resource limits, module blocking |
| **ContainerSandboxEngine** | CONTAINER | Docker/Podman, network isolation, cgroups |

**Signature Algorithms:**

| Algorithm | Description | Recommended |
|-----------|-------------|-------------|
| SHA256/SHA512 | Hash only (integrity) | No |
| HMAC_SHA256/512 | Symmetric key | Yes (internal) |
| RSA_SHA256 | Asymmetric key | Yes (public distribution) |
| ED25519 | Modern asymmetric | Yes (recommended) |

**Plugin Lifecycle States:**

```
DISCOVERED → LOADING → LOADED → ACTIVATING → ACTIVE
     ↓          ↓         ↓          ↓          ↓
  FAILED    FAILED    FAILED    FAILED    DEACTIVATING
                                              ↓
                                          UNLOADING
                                              ↓
                                          UNLOADED
```

### ML Model Monitoring

Enterprise-grade ML model observability and alerting:

```python
from truthound.ml.monitoring import (
    ModelMonitor, InMemoryMetricStore,
    ThresholdRule, ThresholdConfig, AlertSeverity,
    SlackAlertHandler, SlackConfig,
    PerformanceCollector, QualityCollector,
)

# Create monitor with collectors
store = InMemoryMetricStore()
monitor = ModelMonitor(store=store)
monitor.add_collector(PerformanceCollector())
monitor.add_collector(QualityCollector())

# Add alert rules
monitor.add_rule(ThresholdRule(
    name="high-latency",
    config=ThresholdConfig(
        metric_name="latency_ms",
        threshold=100.0,
        comparison="gt",
    ),
    severity=AlertSeverity.WARNING,
))

# Add handlers (Slack, Webhook)
monitor.add_handler(SlackAlertHandler(SlackConfig(
    webhook_url="https://hooks.slack.com/...",
)))

# Collect metrics and evaluate alerts
await monitor.collect({
    "model_id": "fraud-detection-v2",
    "latency_ms": 45.2,
    "predictions": 1000,
})
alerts = await monitor.evaluate()
```

### Lineage Visualization

Multi-format lineage graph rendering:

```python
from truthound.lineage import LineageTracker
from truthound.lineage.visualization.renderers import (
    D3Renderer, CytoscapeRenderer, GraphvizRenderer, MermaidRenderer,
)
from truthound.lineage.visualization.protocols import RenderConfig

# Build lineage
tracker = LineageTracker()
tracker.track_source("raw_data", source_type="csv")
tracker.track_transformation("cleaned", sources=["raw_data"])
tracker.track_output("warehouse", sources=["cleaned"])

graph = tracker.graph
config = RenderConfig(width=1200, height=800, orientation="LR")

# Render to different formats
d3_json = D3Renderer().render(graph, config)       # D3.js JSON
cyto_json = CytoscapeRenderer().render(graph, config)  # Cytoscape.js
dot = GraphvizRenderer().render(graph, config)     # DOT format
mermaid = MermaidRenderer().render(graph, config)  # Mermaid syntax

# Generate interactive HTML pages
d3_html = D3Renderer().render_html(graph, config)
mermaid_md = MermaidRenderer().render_markdown(graph, config)
```

**Visualization Renderers:**

| Renderer | Output | Use Case |
|----------|--------|----------|
| **D3Renderer** | JSON + HTML | Interactive web dashboards |
| **CytoscapeRenderer** | JSON + HTML | Complex graph analysis |
| **GraphvizRenderer** | DOT, SVG, PNG | Static documentation |
| **MermaidRenderer** | Markdown | GitHub/GitLab README |

### OpenLineage Integration

Industry-standard lineage event emission:

```python
from truthound.lineage.integrations import OpenLineageEmitter

emitter = OpenLineageEmitter(
    namespace="my-company",
    producer="truthound",
)

# Start a run with inputs
run = emitter.start_run(
    job_name="data-validation",
    inputs=[
        emitter.build_dataset("raw_sales"),
        emitter.build_dataset("raw_customers"),
    ],
)

# Complete with outputs
emitter.emit_complete(run, outputs=[
    emitter.build_dataset("validated_sales"),
])

# Convert Truthound graph to OpenLineage events
runs = emitter.emit_from_graph(graph, job_name="pipeline")
```

### Protocol-based Streaming

Unified streaming adapter abstraction for Kafka and Kinesis:

```python
from truthound.realtime import create_adapter, register_adapter

# Create Kafka adapter
kafka = create_adapter("kafka", {
    "bootstrap_servers": "localhost:9092",
    "topic": "validation-events",
    "consumer_group": "truthound-validators",
})

# Or Kinesis adapter
kinesis = create_adapter("kinesis", {
    "stream_name": "validation-stream",
    "region": "us-east-1",
})

# Unified interface
async with kafka:
    await kafka.produce({"event": "validated", "result": True})

    async for message in kafka.consume():
        result = validate(message.value)
        await kafka.produce(result)
```

**Streaming Adapters:**

| Adapter | Library | Features |
|---------|---------|----------|
| **KafkaAdapter** | aiokafka | Auto-reconnect, batch produce, consumer groups, partition info |
| **KinesisAdapter** | aiobotocore | Stream management, shard iterators, batch put |

### Testcontainers Integration Testing

Docker-based integration testing for streaming:

```python
from truthound.realtime.testing import (
    KafkaTestContainer, RedisTestContainer, LocalStackTestContainer,
    StreamTestEnvironment,
)

# Single container
async with KafkaTestContainer() as kafka:
    adapter = create_adapter("kafka", {
        "bootstrap_servers": kafka.get_bootstrap_servers(),
    })
    # Run tests against real Kafka

# Multi-container environment
env = StreamTestEnvironment()
env.add_container("kafka", KafkaTestContainer())
env.add_container("redis", RedisTestContainer())

async with env:
    # Kafka + Redis integration tests
    pass
```

### Incremental Profiling Scheduling

Schedule automatic profiling with various trigger strategies:

```python
from truthound.profiler.scheduling import (
    create_scheduler,
    CronTrigger, IntervalTrigger, DataChangeTrigger,
    CompositeTrigger, ManualTrigger,
    InMemoryProfileStorage, FileProfileStorage,
)

# Cron-based scheduling (e.g., every day at 2am)
scheduler = create_scheduler(
    profiler=profiler,
    trigger_type="cron",
    expression="0 2 * * *",
)

# Interval-based scheduling (e.g., every hour)
scheduler = create_scheduler(
    profiler=profiler,
    trigger_type="interval",
    interval_seconds=3600,
)

# Data change-based scheduling (profile when data changes by 5%)
scheduler = create_scheduler(
    profiler=profiler,
    trigger_type="data_change",
    change_threshold=0.05,
)

# Run profiling if trigger conditions are met
profile = scheduler.run_if_needed(data)
if profile:
    print(f"New profile: {profile.profiled_at}")
else:
    print("Profile not needed yet")

# Get scheduler metrics
metrics = scheduler.get_metrics()
print(f"Total runs: {metrics['total_runs']}")
print(f"Skipped: {metrics['skipped_runs']}")
```

**Trigger Types:**

| Trigger | Description |
|---------|-------------|
| `CronTrigger` | Cron expression based scheduling |
| `IntervalTrigger` | Fixed time interval |
| `DataChangeTrigger` | Trigger on data changes (row count, hash) |
| `EventTrigger` | Event-driven triggering |
| `ManualTrigger` | On-demand triggering |
| `CompositeTrigger` | Combine multiple triggers (ANY/ALL mode) |

---

## Validator Categories

| Category | Count | Description |
|----------|-------|-------------|
| Schema | 14 | Column structure, types, relationships |
| Completeness | 7 | Null detection, required fields |
| Uniqueness | 13 | Duplicates, primary keys, composite keys |
| Distribution | 15 | Range, outliers, statistical tests |
| String | 18 | Regex, email, URL, JSON validation |
| Datetime | 10 | Format, range, sequence validation |
| Aggregate | 8 | Mean, median, sum constraints |
| Cross-table | 4 | Multi-table relationships |
| Multi-column | 21 | Column comparisons, conditional logic |
| Query | 20 | SQL/Polars expression validation |
| Table | 18 | Row count, freshness, metadata |
| Geospatial | 12 | Coordinates, bounding boxes, Shapely polygons |
| Drift | 13 | KS, PSI, Chi-square, Wasserstein |
| Anomaly | 15 | IQR, Z-score, Isolation Forest, LOF |
| Business | 8 | Luhn, IBAN, VAT, ISBN validation |
| Localization | 9 | Korean, Japanese, Chinese identifiers |
| ML Feature | 5 | Leakage detection, correlation |
| Profiling | 7 | Cardinality, entropy, frequency |
| Referential | 13 | Foreign keys, orphan records |
| Time Series | 14 | Gaps, seasonality, trend detection |
| Privacy | 21 | GDPR, CCPA, LGPD + 6 plugin regulations |
| Security | 8 | SQL injection, ReDoS protection (ML training), regex safety |
| SDK | 27 | Decorators, builder, testing, templates |
| Enterprise SDK | 58 | Sandbox, signing, versioning, licensing, fuzzing |
| Timeout | 90 | Deadline, cascade, degradation, priority, SLA |
| i18n | 118 | 15+ languages, CLDR plurals, RTL, TMS |
| Infrastructure | 162 | Logging, metrics, config, audit, encryption |
| Resilience | 41 | Circuit breaker, retry, bulkhead, rate limiting |
| Schema Evolution | 33 | Change detection, compatibility, alerts |
| Scheduling | 30 | Cron, interval, data change triggers |
| Data Docs | 190 | Pipeline, i18n (15 langs), versioning, themes, PDF |
| Plugin Architecture | 190 | Security sandbox, signing, versioning, dependencies, lifecycle, hot reload |
| ML Monitoring | 11 | Model metrics, alerting, drift detection, handlers |
| Lineage Visualization | 5 | D3, Cytoscape, Graphviz, Mermaid renderers |
| OpenLineage | 6 | Industry-standard lineage events, run lifecycle |
| Streaming Adapters | 21 | Kafka (aiokafka), Kinesis (aiobotocore), Testcontainers |

---

## Data Sources

| Category | Sources |
|----------|---------|
| **DataFrame** | Polars, Pandas, PySpark |
| **Core SQL** | PostgreSQL, MySQL, SQLite |
| **Cloud DW** | BigQuery, Snowflake, Redshift, Databricks |
| **Enterprise** | Oracle, SQL Server |
| **File** | CSV, Parquet, JSON, NDJSON |

---

## Installation Options

```bash
# Core installation
pip install truthound

# Feature-specific extras
pip install truthound[drift]      # Drift detection (scipy)
pip install truthound[anomaly]    # Anomaly detection (scikit-learn)
pip install truthound[pdf]        # PDF export (weasyprint)
pip install truthound[dashboard]  # Interactive dashboard (reflex)

# Data source extras
pip install truthound[bigquery]   # Google BigQuery
pip install truthound[snowflake]  # Snowflake
pip install truthound[redshift]   # Amazon Redshift
pip install truthound[databricks] # Databricks
pip install truthound[oracle]     # Oracle Database
pip install truthound[sqlserver]  # SQL Server
pip install truthound[enterprise] # All enterprise sources

# Security & storage extras
pip install truthound[encryption] # Encryption (cryptography)
pip install truthound[compression] # Advanced compression (zstd, lz4)

# Full installation
pip install truthound[all]
```

---

## Security Features

### Encryption

Truthound provides enterprise-grade encryption for sensitive validation results:

```python
from truthound.stores.encryption import (
    get_encryptor,
    generate_key,
    EncryptionAlgorithm,
    create_secure_pipeline,
    derive_key,
    KeyDerivation,
)

# Generate encryption key
key = generate_key(EncryptionAlgorithm.AES_256_GCM)

# Simple encryption
encryptor = get_encryptor("aes-256-gcm")
encrypted = encryptor.encrypt(sensitive_data, key)
decrypted = encryptor.decrypt(encrypted, key)

# Password-based encryption
key, salt = derive_key("my_password", kdf=KeyDerivation.ARGON2ID)

# Compress-then-encrypt pipeline (recommended)
pipeline = create_secure_pipeline(key, compression="gzip")
result = pipeline.process(data)
original = pipeline.reverse(result.data, result.header)
```

**Supported Algorithms:**
| Algorithm | Key Size | Use Case |
|-----------|----------|----------|
| AES-256-GCM | 256-bit | Default, widely supported |
| ChaCha20-Poly1305 | 256-bit | Fast on CPUs without AES-NI |
| XChaCha20-Poly1305 | 256-bit | Extended nonce for high-volume |
| Fernet | 128-bit | Simple symmetric encryption |

**Key Derivation Functions:**
- Argon2id (recommended for passwords)
- PBKDF2-SHA256/SHA512
- scrypt
- HKDF-SHA256

---

## Comparative Analysis

| Feature | Truthound | Great Expectations | Pandera |
|---------|-----------|-------------------|---------|
| Zero Configuration | Yes | No | No |
| Polars Native | Yes | No | No |
| LazyFrame Support | Yes | No | No |
| Drift Detection | 13 methods | Plugin | No |
| Anomaly Detection | 15 methods | No | No |
| PII Detection | Yes | No | No |
| Cross-table Validation | Yes | Yes | No |
| Geospatial Validation | Yes | No | No |
| Time Series Validation | 14 validators | No | No |
| Privacy Compliance | GDPR/CCPA/LGPD | No | No |
| Validator Count | 265+ | 300+ | 50+ |

---

## Requirements

- Python 3.11+
- Polars 1.x
- PyYAML
- Rich (console output)
- Typer (CLI)

---

## Development

```bash
git clone https://github.com/seadonggyun4/Truthound.git
cd Truthound
pip install hatch
hatch env create
hatch run test
```

---

## References

1. Polars Documentation. https://pola.rs/
2. Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione"
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
4. Breunig, M. M., et al. (2000). "LOF: Identifying Density-Based Local Outliers"

---

## License

MIT License — Copyright (c) 2024-2025 Truthound Contributors

---

## Acknowledgments

Built with [Polars](https://pola.rs/), [Rich](https://rich.readthedocs.io/), [Typer](https://typer.tiangolo.com/), [scikit-learn](https://scikit-learn.org/), and [SciPy](https://scipy.org/).

---

<p align="center">
  <strong>Truthound — Your data's loyal guardian.</strong>
</p>
