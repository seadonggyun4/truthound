"""Auto-Profiling and Rule Generation module (Phase 7).

This module provides automatic data profiling and validation rule generation:
- Profile data to understand structure, patterns, and statistics
- Generate validation rules based on profile results
- Export rules as YAML, Python code, or JSON

NOTE: This module also provides backward compatibility with the legacy
`profile_data` function used by api.py.

Example:
    # Profile data
    from truthound.profiler import DataProfiler, profile_file

    profile = profile_file("data.parquet")
    print(f"Rows: {profile.row_count}, Columns: {profile.column_count}")

    for col in profile:
        print(f"{col.name}: {col.inferred_type}, null_ratio={col.null_ratio}")

    # Generate validation rules
    from truthound.profiler import generate_suite

    suite = generate_suite(
        profile,
        strictness="medium",
        include_categories=["schema", "completeness", "format"]
    )

    # Export as YAML
    print(suite.to_yaml())

    # Export as Python code
    print(suite.to_python_code())
"""

import polars as pl

from truthound.profiler.base import (
    # Enums
    DataType,
    Strictness,
    ProfileCategory,
    # Data structures
    PatternMatch,
    DistributionStats,
    ValueFrequency,
    ColumnProfile,
    TableProfile,
    # Base classes
    Profiler,
    ProfilerProtocol,
    TypeInferrer,
    # Configuration
    ProfilerConfig,
    # Registry
    ProfilerRegistry,
    profiler_registry,
    register_profiler,
    register_type_inferrer,
)

from truthound.profiler.column_profiler import (
    # Analyzers
    ColumnAnalyzer,
    BasicStatsAnalyzer,
    NumericAnalyzer,
    StringAnalyzer,
    DatetimeAnalyzer,
    ValueFrequencyAnalyzer,
    PatternAnalyzer,
    # Type inferrers
    PhysicalTypeInferrer,
    PatternBasedTypeInferrer,
    CardinalityTypeInferrer,
    # Main profiler
    ColumnProfiler,
    # Patterns
    PatternDefinition,
    BUILTIN_PATTERNS,
)

from truthound.profiler.table_profiler import (
    # Analyzers
    TableAnalyzer,
    DuplicateRowAnalyzer,
    MemoryEstimator,
    CorrelationAnalyzer,
    # Main profiler
    DataProfiler,
    # Convenience functions
    profile_dataframe,
    profile_file,
    save_profile,
    load_profile,
)

from truthound.profiler.generators import (
    # Base
    RuleGenerator,
    GeneratedRule,
    RuleGeneratorRegistry,
    rule_generator_registry,
    register_generator,
    # Generators
    SchemaRuleGenerator,
    StatsRuleGenerator,
    PatternRuleGenerator,
    MLRuleGenerator,
    # Suite
    ValidationSuiteGenerator,
    generate_suite,
)

from truthound.profiler.generators.base import (
    RuleCategory,
    RuleConfidence,
    RuleBuilder,
    StrictnessThresholds,
    DEFAULT_THRESHOLDS,
)

from truthound.profiler.generators.suite_generator import (
    ValidationSuite,
    save_suite,
    load_suite,
)

# P0 Improvements: Error handling, native patterns, streaming, schema versioning
from truthound.profiler.errors import (
    # Severity and categories
    ErrorSeverity,
    ErrorCategory,
    # Exception hierarchy
    ProfilerError,
    AnalysisError,
    PatternError,
    TypeInferenceError,
    ValidationError,
    # Error collection
    ErrorRecord,
    ErrorCollector,
    ErrorCatcher,
    # Decorator
    with_error_handling,
)

from truthound.profiler.native_patterns import (
    # Pattern system
    PatternSpec,
    PatternBuilder,
    PatternPriority,
    PatternRegistry,
    BUILTIN_PATTERNS as NATIVE_PATTERNS,
    # Pattern matcher
    NativePatternMatcher,
    NativePatternAnalyzer,
    PatternMatchResult,
    # Convenience functions
    match_patterns,
    infer_column_type,
)

# P0 Critical: Memory-safe sampling for pattern matching
from truthound.profiler.sampling import (
    # Enums
    SamplingMethod,
    ConfidenceLevel,
    # Configuration
    SamplingConfig,
    DEFAULT_SAMPLING_CONFIG,
    # Metrics
    SamplingMetrics,
    SamplingResult,
    # Strategies
    SamplingStrategy,
    NoSamplingStrategy,
    HeadSamplingStrategy,
    RandomSamplingStrategy,
    SystematicSamplingStrategy,
    HashSamplingStrategy,
    StratifiedSamplingStrategy,
    ReservoirSamplingStrategy,
    AdaptiveSamplingStrategy,
    # Registry
    SamplingStrategyRegistry,
    sampling_strategy_registry,
    # Data size estimation
    DataSizeEstimator,
    # Main interface
    Sampler,
    # Convenience functions
    create_sampler,
    sample_data,
    calculate_sample_size,
)

# Enterprise-scale sampling for 100M+ datasets
from truthound.profiler.enterprise_sampling import (
    # Scale classification
    ScaleCategory,
    SamplingQuality,
    # Configuration
    MemoryBudgetConfig,
    EnterpriseScaleConfig,
    # Extended metrics
    BlockSamplingMetrics,
    ProgressiveResult,
    # Monitoring
    MemoryMonitor,
    TimeBudgetManager,
    # Strategies
    BlockSamplingStrategy,
    MultiStageSamplingStrategy,
    ColumnAwareSamplingStrategy,
    ProgressiveSamplingStrategy,
    # Main interface
    EnterpriseScaleSampler,
    # Convenience functions
    sample_large_dataset,
    estimate_optimal_sample_size,
    classify_dataset_scale,
)

# Parallel block processing for maximum throughput
from truthound.profiler.parallel_sampling import (
    # Configuration
    ParallelSamplingConfig,
    ExecutionMode,
    SchedulingPolicy,
    # Strategies
    ParallelBlockSampler,
    SketchBasedSampler,
    # Work stealing
    WorkStealingQueue,
    BlockTask,
    BlockResult,
    # Metrics
    ParallelSamplingMetrics,
    # Convenience functions
    sample_parallel,
)

# Probabilistic data structures for O(1) memory aggregations
from truthound.profiler.sketches import (
    # Core implementations
    HyperLogLog,
    CountMinSketch,
    BloomFilter,
    # Factory
    create_sketch,
    SketchType,
    SketchFactory,
    # Protocols
    Sketch,
    MergeableSketch,
    CardinalityEstimator,
    FrequencyEstimator,
    MembershipTester,
)

from truthound.profiler.sampled_matcher import (
    # Results
    SampledPatternMatchResult,
    SampledColumnMatchResult,
    # Configuration
    SampledMatcherConfig,
    # Main interface
    SampledPatternMatcher,
    SafeNativePatternMatcher,
    # Factory functions
    create_sampled_matcher,
    match_patterns_safe,
    infer_column_type_safe,
)

from truthound.profiler.streaming import (
    # Incremental stats
    IncrementalStats,
    # Chunk iterators
    FileChunkIterator,
    DataFrameChunkIterator,
    # Progress tracking
    StreamingProgress,
    ProgressCallback,
    # Streaming profiler
    StreamingProfiler,
    # Convenience functions
    stream_profile_file,
    stream_profile_dataframe,
)

from truthound.profiler.schema import (
    # Schema versioning
    SchemaVersion,
    CURRENT_SCHEMA_VERSION,
    # Migration
    MigrationStep,
    SchemaMigrator,
    schema_migrator,
    # Serialization
    ProfileSerializer,
    # Validation
    SchemaValidationStatus,
    SchemaValidationResult,
    SchemaValidator,
    # Convenience (these override table_profiler versions with versioning support)
    save_profile as save_profile_versioned,
    load_profile as load_profile_versioned,
    validate_profile,
)

# P1 Improvements: Progress, comparison, incremental, timeout
from truthound.profiler.progress import (
    # Events
    ProgressStage,
    ProgressEvent,
    ProgressCallback,
    # Tracker
    ProgressTracker,
    ProgressAggregator,
    # Reporters
    ConsoleProgressReporter,
    # Convenience
    create_progress_callback,
)

from truthound.profiler.comparison import (
    # Drift types
    DriftType,
    DriftSeverity,
    ChangeDirection,
    # Results
    DriftResult,
    ColumnComparison,
    ProfileComparison,
    # Configuration
    DriftThresholds,
    # Detectors
    DriftDetector,
    CompletenessDriftDetector,
    UniquenessDriftDetector,
    DistributionDriftDetector,
    RangeDriftDetector,
    CardinalityDriftDetector,
    # Comparator
    ProfileComparator,
    # Convenience
    compare_profiles,
    detect_drift,
)

from truthound.profiler.incremental import (
    # Change detection
    ChangeReason,
    ColumnFingerprint,
    ChangeDetectionResult,
    FingerprintCalculator,
    # Configuration
    IncrementalConfig,
    # Profiler
    IncrementalProfiler,
    ProfileMerger,
    # Convenience
    profile_incrementally,
)

# P1 Improvements: Incremental Profiling Validation
from truthound.profiler.incremental_validation import (
    # Types
    ValidationSeverity,
    ValidationCategory,
    ValidationType,
    # Results
    ValidationIssue,
    ValidationMetrics,
    ValidationResult,
    # Context
    ValidationContext,
    # Base classes
    BaseValidator,
    ValidatorProtocol,
    # Change detection validators
    ChangeDetectionAccuracyValidator,
    SchemaChangeValidator,
    StalenessValidator,
    # Fingerprint validators
    FingerprintConsistencyValidator,
    FingerprintSensitivityValidator,
    # Merge validators
    ProfileMergeValidator,
    # Data integrity validators
    DataIntegrityValidator,
    # Performance validators
    PerformanceValidator,
    # Registry
    ValidatorRegistry,
    validator_registry,
    register_validator,
    # Configuration
    ValidationConfig,
    # Runner
    ValidationRunner,
    # Main validator
    IncrementalValidator,
    # Convenience functions
    validate_incremental,
    validate_merge,
    validate_fingerprints,
)

from truthound.profiler.timeout import (
    # Configuration
    TimeoutAction,
    TimeoutConfig,
    # Results
    TimeoutResult,
    # Executor
    TimeoutExecutor,
    # Utilities
    timeout_context,
    TimeoutAwareMixin,
    DeadlineTracker,
    # Convenience
    with_timeout,
    create_timeout_config,
)

# P0 Critical: Process-isolated timeout for reliable termination
from truthound.profiler.process_timeout import (
    # Enums
    ExecutionBackend,
    TimeoutAction as ProcessTimeoutAction,
    TerminationMethod,
    CircuitState,
    # Metrics and Results
    ExecutionMetrics,
    ExecutionResult,
    # Complexity Estimation
    ComplexityEstimate,
    ComplexityEstimator,
    DefaultComplexityEstimator,
    default_complexity_estimator,
    # Circuit Breaker
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerRegistry,
    circuit_breaker_registry,
    # Execution Strategies
    ExecutionStrategy,
    ThreadExecutionStrategy,
    ProcessExecutionStrategy,
    AdaptiveExecutionStrategy,
    InlineExecutionStrategy,
    ExecutionStrategyRegistry,
    execution_strategy_registry,
    # Resource Monitoring
    ResourceLimits,
    ResourceUsage,
    ResourceMonitor,
    resource_monitor,
    # Main Interface
    ProcessTimeoutConfig,
    ProcessTimeoutExecutor,
    # Convenience Functions
    with_process_timeout,
    estimate_execution_time,
    create_timeout_executor,
    # Context Manager
    process_timeout_context,
    # Decorator
    timeout_protected,
)

# P2 Improvements: Caching, Observability, Quality Scoring, Custom Patterns
from truthound.profiler.caching import (
    # Cache keys
    CacheKey,
    CacheKeyProtocol,
    FileHashCacheKey,
    DataFrameHashCacheKey,
    # Cache entry
    CacheEntry,
    # Cache backends
    CacheBackend,
    MemoryCacheBackend,
    FileCacheBackend,
    RedisCacheBackend,
    CacheBackendRegistry,
    cache_backend_registry,
    # Profile cache
    CacheConfig,
    ProfileCache,
    # Decorator
    cached_profile,
    # Convenience
    create_cache,
    hash_file,
    # Redis error
    RedisConnectionError,
)

# P2 Improvements: Resilience and fallback patterns
from truthound.profiler.resilience import (
    # Circuit breaker states
    CircuitState,
    BackendHealth,
    FailureType,
    # Configuration
    CircuitBreakerConfig,
    RetryConfig,
    HealthCheckConfig,
    ResilienceConfig,
    # Circuit breaker
    CircuitBreaker,
    CircuitOpenError,
    # Retry logic
    RetryPolicy,
    # Health monitoring
    HealthMonitor,
    # Resilient backends
    ResilientCacheBackend,
    FallbackChain,
    # Factory functions
    create_resilient_redis_backend,
    create_high_availability_cache,
)

from truthound.profiler.observability import (
    # Types
    MetricType,
    SpanStatus,
    # Span
    Span,
    SpanEvent,
    SpanProtocol,
    # Metrics
    Metric,
    Counter,
    Gauge,
    Histogram,
    MetricValue,
    MetricsCollector,
    # Span exporters
    SpanExporter,
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OTLPSpanExporter,
    SpanExporterRegistry,
    span_exporter_registry,
    # Telemetry
    TelemetryConfig,
    ProfilerTelemetry,
    # Decorators
    traced,
    timed,
    # OpenTelemetry
    OpenTelemetryIntegration,
    # Global access
    get_telemetry,
    set_telemetry,
    get_metrics,
    create_telemetry,
)

from truthound.profiler.quality import (
    # Quality levels
    QualityLevel,
    RuleType,
    # Confusion matrix
    ConfusionMatrix,
    # Metrics
    QualityMetrics,
    # Rules
    RuleProtocol,
    ValidationRule,
    # Estimators
    QualityEstimator,
    SamplingQualityEstimator,
    HeuristicQualityEstimator,
    CrossValidationEstimator,
    QualityEstimatorRegistry,
    quality_estimator_registry,
    # Scorer
    ScoringConfig,
    RuleQualityScore,
    RuleQualityScorer,
    # Trend analysis
    QualityTrendPoint,
    QualityTrendAnalyzer,
    # Convenience
    estimate_quality,
    score_rule,
    compare_rules,
)

from truthound.profiler.custom_patterns import (
    # Pattern configuration
    PatternPriority as CustomPatternPriority,
    PatternExample,
    PatternConfig,
    PatternGroup,
    PatternConfigSchema,
    # Loader
    PatternConfigLoader,
    # Registry
    PatternRegistry as CustomPatternRegistry,
    pattern_registry,
    # Default patterns
    DEFAULT_PATTERNS_YAML,
    load_default_patterns,
    # Convenience
    load_patterns,
    load_patterns_directory,
    register_pattern,
    match_patterns as match_custom_patterns,
    infer_type_from_patterns,
    export_patterns,
)

# P3 Improvements: Distributed Processing, ML Inference, Auto Threshold, Visualization
from truthound.profiler.distributed import (
    # Backend types
    BackendType,
    PartitionStrategy,
    # Data structures
    PartitionInfo,
    WorkerResult,
    # Configurations
    BackendConfig,
    SparkConfig,
    DaskConfig,
    RayConfig,
    DistributedProfileConfig,
    # Abstract base
    DistributedBackend,
    # Implementations
    LocalBackend,
    SparkBackend,
    DaskBackend,
    RayBackend,
    # Registry
    BackendRegistry,
    # Main interface
    DistributedProfiler,
    # Convenience
    create_distributed_profiler,
    profile_distributed,
    get_available_backends,
)

from truthound.profiler.ml_inference import (
    # Feature types
    FeatureType,
    Feature,
    FeatureVector,
    # Abstract base
    FeatureExtractor,
    # Feature extractors
    NameFeatureExtractor,
    ValueFeatureExtractor,
    StatisticalFeatureExtractor,
    ContextFeatureExtractor,
    # Registry
    FeatureExtractorRegistry,
    # Inference results
    InferenceResult,
    # Abstract model
    InferenceModel,
    # Model implementations
    RuleBasedModel,
    NaiveBayesModel,
    EnsembleModel,
    # Model registry
    ModelRegistry as InferenceModelRegistry,
    # Configuration
    InferrerConfig as MLInferenceConfig,
    # Main interface
    MLTypeInferrer,
    # Convenience
    create_inference_model,
    infer_column_type_ml,
    infer_table_types_ml,
)

from truthound.profiler.auto_threshold import (
    # Strategy types
    TuningStrategy,
    ThresholdType,
    # Data structures
    ColumnThresholds,
    TableThresholds,
    # Presets
    StrictnessPreset,
    # Abstract base
    TuningStrategyImpl,
    # Strategy implementations
    ConservativeStrategy,
    BalancedStrategy,
    PermissiveStrategy,
    AdaptiveStrategy,
    StatisticalStrategy,
    DomainAwareStrategy,
    # Registry
    StrategyRegistry as TuningStrategyRegistry,
    # Configuration
    TunerConfig as TuningConfig,
    # Main interface
    ThresholdTuner,
    # A/B Testing
    ThresholdTestResult,
    ThresholdTester,
    # Convenience
    tune_thresholds,
    get_available_strategies,
    create_tuner,
)

from truthound.profiler.visualization import (
    # Chart types
    ChartType,
    ColorScheme,
    ReportTheme,
    SectionType,
    # Color palettes
    COLOR_PALETTES,
    # Data structures
    ChartData,
    ChartConfig,
    ThemeConfig,
    SectionContent,
    ReportConfig,
    ProfileData,
    # Theme configs
    THEME_CONFIGS,
    # Abstract base
    ChartRenderer,
    SectionRenderer,
    # Registries
    ChartRendererRegistry,
    chart_renderer_registry,
    SectionRegistry,
    section_registry,
    ThemeRegistry,
    theme_registry,
    # Renderers
    SVGChartRenderer,
    BaseSectionRenderer,
    OverviewSectionRenderer,
    DataQualitySectionRenderer,
    ColumnDetailsSectionRenderer,
    PatternsSectionRenderer,
    RecommendationsSectionRenderer,
    CustomSectionRenderer,
    # Template
    ReportTemplate,
    # Converter
    ProfileDataConverter,
    # Main interface
    HTMLReportGenerator,
    ReportExporter,
    # Convenience
    generate_report,
    compare_profiles as compare_profile_reports,
)

# Phase 7 P0: Suite Export System
from truthound.profiler.suite_export import (
    # Enums
    ExportFormat,
    CodeStyle,
    OutputMode,
    # Configuration
    ExportConfig,
    DEFAULT_CONFIG,
    MINIMAL_CONFIG,
    VERBOSE_CONFIG,
    # Protocol and base
    SuiteFormatterProtocol,
    SuiteFormatter,
    # Built-in formatters
    YAMLFormatter,
    JSONFormatter,
    PythonFormatter,
    TOMLFormatter,
    CheckpointFormatter,
    # Registry
    FormatterRegistry,
    formatter_registry,
    register_formatter,
    # Post-processors
    ExportPostProcessor,
    AddHeaderPostProcessor,
    AddFooterPostProcessor,
    TemplatePostProcessor,
    # Exporter
    ExportResult,
    SuiteExporter,
    # Convenience
    create_exporter,
    export_suite,
    format_suite,
    get_available_formats,
)

# Phase 7 P0: Suite Configuration System
from truthound.profiler.suite_config import (
    # Enums
    ConfigPreset,
    OutputFormat as SuiteOutputFormat,
    GeneratorMode,
    # Sub-configurations
    CategoryConfig,
    ConfidenceConfig,
    OutputConfig as SuiteOutputConfig,
    GeneratorConfig as SuiteGeneratorOptionsConfig,
    # Main configuration
    SuiteGeneratorConfig,
    # Presets
    PRESETS,
    # File I/O
    load_config as load_suite_config,
    save_config as save_suite_config,
    # CLI support
    CLIArguments,
    build_config_from_cli,
)

# Phase 7 P0: Suite CLI Handlers
from truthound.profiler.suite_cli import (
    # Result types
    GenerateSuiteResult,
    QuickSuiteResult,
    # Progress tracking
    SuiteGenerationProgress,
    # Handlers
    SuiteGenerationHandler,
    # Helpers
    get_available_formats as get_suite_formats,
    get_available_presets,
    get_available_categories,
    format_category_help,
    format_preset_help,
    validate_format,
    validate_preset,
    validate_strictness,
    validate_confidence,
    # CLI integration
    create_cli_handler,
    run_generate_suite,
    run_quick_suite,
)

# Phase 7 P0: Streaming Pattern Matching
# Phase 7 P0: Standardized Progress Callbacks
from truthound.profiler.progress_callbacks import (
    # Event types and levels
    EventLevel,
    EventType,
    # Context and metrics
    ProgressContext,
    ProgressMetrics,
    StandardProgressEvent,
    # Protocols
    ProgressCallback as StandardProgressCallback,
    AsyncProgressCallback,
    LifecycleCallback,
    # Base adapter
    CallbackAdapter,
    # Console adapters
    ConsoleStyle,
    ConsoleAdapter,
    MinimalConsoleAdapter,
    # Logging adapter
    LoggingAdapter,
    # File adapter
    FileOutputConfig,
    FileAdapter,
    # Callback chain
    CallbackChain,
    # Filtering and throttling
    FilterConfig,
    FilteringAdapter,
    ThrottleConfig,
    ThrottlingAdapter,
    # Buffering
    BufferConfig,
    BufferingAdapter,
    # Async
    AsyncAdapter,
    # Registry
    CallbackRegistry,
    # Emitter
    ProgressEmitter,
    # Presets
    CallbackPresets,
    # Convenience functions
    create_callback_chain,
    create_console_callback,
    create_logging_callback,
    create_file_callback,
    with_throttling,
    with_filtering,
    with_buffering,
)

# Phase 7 P0: Internationalization (i18n) System
from truthound.profiler.i18n import (
    # Message codes
    MessageCode,
    # Locale management
    LocaleInfo,
    LocaleManager,
    BUILTIN_LOCALES,
    set_locale,
    get_locale,
    # Message catalog
    MessageEntry,
    MessageCatalog,
    # Message loaders
    MessageLoader,
    DictMessageLoader,
    FileMessageLoader,
    # Formatter
    PlaceholderFormatter,
    # Main interface
    I18n,
    # I18n exceptions
    I18nError,
    I18nAnalysisError,
    I18nPatternError,
    I18nTypeError,
    I18nIOError,
    I18nTimeoutError,
    I18nValidationError,
    # Registry
    MessageCatalogRegistry,
    # Convenience functions
    get_message,
    t as translate,
    register_messages,
    load_messages_from_file,
    create_message_loader,
    # Context manager
    locale_context,
    # Presets
    I18nPresets,
)

from truthound.profiler.streaming_patterns import (
    # Enums
    AggregationMethod,
    ChunkProcessingStatus,
    # Pattern state management
    PatternChunkStats,
    PatternState,
    ColumnPatternState,
    # Aggregation strategies
    AggregationStrategy,
    IncrementalAggregation,
    WeightedAggregation,
    SlidingWindowAggregation,
    ExponentialAggregation,
    ConsensusAggregation,
    AdaptiveAggregation,
    # Aggregation registry
    AggregationStrategyRegistry,
    aggregation_strategy_registry,
    # Result types
    StreamingPatternResult,
    # Events
    PatternEvent,
    PatternEventCallback,
    # Configuration
    StreamingPatternConfig,
    # Main interface
    StreamingPatternMatcher,
    # Integration
    StreamingPatternIntegration,
    # Convenience functions
    create_streaming_matcher,
    stream_match_patterns,
    get_available_aggregation_methods,
)


# =============================================================================
# Legacy Compatibility: profile_data function (moved to legacy.py)
# =============================================================================
from truthound.profiler.legacy import profile_data


__all__ = [
    # === Legacy compatibility ===
    "profile_data",
    # === Base module ===
    # Enums
    "DataType",
    "Strictness",
    "ProfileCategory",
    # Data structures
    "PatternMatch",
    "DistributionStats",
    "ValueFrequency",
    "ColumnProfile",
    "TableProfile",
    # Base classes
    "Profiler",
    "ProfilerProtocol",
    "TypeInferrer",
    # Configuration
    "ProfilerConfig",
    # Registry
    "ProfilerRegistry",
    "profiler_registry",
    "register_profiler",
    "register_type_inferrer",
    # === Column profiler ===
    # Analyzers
    "ColumnAnalyzer",
    "BasicStatsAnalyzer",
    "NumericAnalyzer",
    "StringAnalyzer",
    "DatetimeAnalyzer",
    "ValueFrequencyAnalyzer",
    "PatternAnalyzer",
    # Type inferrers
    "PhysicalTypeInferrer",
    "PatternBasedTypeInferrer",
    "CardinalityTypeInferrer",
    # Main profiler
    "ColumnProfiler",
    # Patterns
    "PatternDefinition",
    "BUILTIN_PATTERNS",
    # === Table profiler ===
    # Analyzers
    "TableAnalyzer",
    "DuplicateRowAnalyzer",
    "MemoryEstimator",
    "CorrelationAnalyzer",
    # Main profiler
    "DataProfiler",
    # Convenience functions
    "profile_dataframe",
    "profile_file",
    "save_profile",
    "load_profile",
    # === Generators ===
    # Base
    "RuleGenerator",
    "GeneratedRule",
    "RuleCategory",
    "RuleConfidence",
    "RuleBuilder",
    "StrictnessThresholds",
    "DEFAULT_THRESHOLDS",
    "RuleGeneratorRegistry",
    "rule_generator_registry",
    "register_generator",
    # Generators
    "SchemaRuleGenerator",
    "StatsRuleGenerator",
    "PatternRuleGenerator",
    "MLRuleGenerator",
    # Suite
    "ValidationSuite",
    "ValidationSuiteGenerator",
    "generate_suite",
    "save_suite",
    "load_suite",
    # === P0: Error Handling ===
    "ErrorSeverity",
    "ErrorCategory",
    "ProfilerError",
    "AnalysisError",
    "PatternError",
    "TypeInferenceError",
    "ValidationError",
    "ErrorRecord",
    "ErrorCollector",
    "ErrorCatcher",
    "with_error_handling",
    # === P0: Native Pattern Matching ===
    "PatternSpec",
    "PatternBuilder",
    "PatternPriority",
    "PatternRegistry",
    "NATIVE_PATTERNS",
    "NativePatternMatcher",
    "NativePatternAnalyzer",
    "PatternMatchResult",
    "match_patterns",
    "infer_column_type",
    # === P0 Critical: Memory-Safe Sampling ===
    # Enums
    "SamplingMethod",
    "ConfidenceLevel",
    # Configuration
    "SamplingConfig",
    "DEFAULT_SAMPLING_CONFIG",
    # Metrics
    "SamplingMetrics",
    "SamplingResult",
    # Strategies
    "SamplingStrategy",
    "NoSamplingStrategy",
    "HeadSamplingStrategy",
    "RandomSamplingStrategy",
    "SystematicSamplingStrategy",
    "HashSamplingStrategy",
    "StratifiedSamplingStrategy",
    "ReservoirSamplingStrategy",
    "AdaptiveSamplingStrategy",
    # Registry
    "SamplingStrategyRegistry",
    "sampling_strategy_registry",
    # Data size estimation
    "DataSizeEstimator",
    # Main interface
    "Sampler",
    # Convenience functions
    "create_sampler",
    "sample_data",
    "calculate_sample_size",
    # === Enterprise-Scale Sampling (100M+ rows) ===
    # Scale classification
    "ScaleCategory",
    "SamplingQuality",
    # Configuration
    "MemoryBudgetConfig",
    "EnterpriseScaleConfig",
    # Extended metrics
    "BlockSamplingMetrics",
    "ProgressiveResult",
    # Monitoring
    "MemoryMonitor",
    "TimeBudgetManager",
    # Strategies
    "BlockSamplingStrategy",
    "MultiStageSamplingStrategy",
    "ColumnAwareSamplingStrategy",
    "ProgressiveSamplingStrategy",
    # Main interface
    "EnterpriseScaleSampler",
    # Convenience functions
    "sample_large_dataset",
    "estimate_optimal_sample_size",
    "classify_dataset_scale",
    # === P0 Critical: Sampled Pattern Matcher ===
    # Results
    "SampledPatternMatchResult",
    "SampledColumnMatchResult",
    # Configuration
    "SampledMatcherConfig",
    # Main interface
    "SampledPatternMatcher",
    "SafeNativePatternMatcher",
    # Factory functions
    "create_sampled_matcher",
    "match_patterns_safe",
    "infer_column_type_safe",
    # === P0: Streaming Profiler ===
    "IncrementalStats",
    "FileChunkIterator",
    "DataFrameChunkIterator",
    "StreamingProgress",
    "ProgressCallback",
    "StreamingProfiler",
    "stream_profile_file",
    "stream_profile_dataframe",
    # === P0: Schema Versioning ===
    "SchemaVersion",
    "CURRENT_SCHEMA_VERSION",
    "MigrationStep",
    "SchemaMigrator",
    "schema_migrator",
    "ProfileSerializer",
    "SchemaValidationStatus",
    "SchemaValidationResult",
    "SchemaValidator",
    "save_profile_versioned",
    "load_profile_versioned",
    "validate_profile",
    # === P1: Progress Tracking ===
    "ProgressStage",
    "ProgressEvent",
    "ProgressTracker",
    "ProgressAggregator",
    "ConsoleProgressReporter",
    "create_progress_callback",
    # === P1: Profile Comparison ===
    "DriftType",
    "DriftSeverity",
    "ChangeDirection",
    "DriftResult",
    "ColumnComparison",
    "ProfileComparison",
    "DriftThresholds",
    "DriftDetector",
    "CompletenessDriftDetector",
    "UniquenessDriftDetector",
    "DistributionDriftDetector",
    "RangeDriftDetector",
    "CardinalityDriftDetector",
    "ProfileComparator",
    "compare_profiles",
    "detect_drift",
    # === P1: Incremental Profiling ===
    "ChangeReason",
    "ColumnFingerprint",
    "ChangeDetectionResult",
    "FingerprintCalculator",
    "IncrementalConfig",
    "IncrementalProfiler",
    "ProfileMerger",
    "profile_incrementally",
    # === P1: Incremental Profiling Validation ===
    # Types
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationType",
    # Results
    "ValidationIssue",
    "ValidationMetrics",
    "ValidationResult",
    # Context
    "ValidationContext",
    # Base classes
    "BaseValidator",
    "ValidatorProtocol",
    # Change detection validators
    "ChangeDetectionAccuracyValidator",
    "SchemaChangeValidator",
    "StalenessValidator",
    # Fingerprint validators
    "FingerprintConsistencyValidator",
    "FingerprintSensitivityValidator",
    # Merge validators
    "ProfileMergeValidator",
    # Data integrity validators
    "DataIntegrityValidator",
    # Performance validators
    "PerformanceValidator",
    # Registry
    "ValidatorRegistry",
    "validator_registry",
    "register_validator",
    # Configuration
    "ValidationConfig",
    # Runner
    "ValidationRunner",
    # Main validator
    "IncrementalValidator",
    # Convenience functions
    "validate_incremental",
    "validate_merge",
    "validate_fingerprints",
    # === P1: Timeout Settings ===
    "TimeoutAction",
    "TimeoutConfig",
    "TimeoutResult",
    "TimeoutExecutor",
    "timeout_context",
    "TimeoutAwareMixin",
    "DeadlineTracker",
    "with_timeout",
    "create_timeout_config",
    # === P0 Critical: Process-Isolated Timeout ===
    # Enums
    "ExecutionBackend",
    "ProcessTimeoutAction",
    "TerminationMethod",
    "CircuitState",
    # Metrics and Results
    "ExecutionMetrics",
    "ExecutionResult",
    # Complexity Estimation
    "ComplexityEstimate",
    "ComplexityEstimator",
    "DefaultComplexityEstimator",
    "default_complexity_estimator",
    # Circuit Breaker
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "circuit_breaker_registry",
    # Execution Strategies
    "ExecutionStrategy",
    "ThreadExecutionStrategy",
    "ProcessExecutionStrategy",
    "AdaptiveExecutionStrategy",
    "InlineExecutionStrategy",
    "ExecutionStrategyRegistry",
    "execution_strategy_registry",
    # Resource Monitoring
    "ResourceLimits",
    "ResourceUsage",
    "ResourceMonitor",
    "resource_monitor",
    # Main Interface
    "ProcessTimeoutConfig",
    "ProcessTimeoutExecutor",
    # Convenience Functions
    "with_process_timeout",
    "estimate_execution_time",
    "create_timeout_executor",
    # Context Manager
    "process_timeout_context",
    # Decorator
    "timeout_protected",
    # === P2: Caching Layer ===
    "CacheKey",
    "CacheKeyProtocol",
    "FileHashCacheKey",
    "DataFrameHashCacheKey",
    "CacheEntry",
    "CacheBackend",
    "MemoryCacheBackend",
    "FileCacheBackend",
    "RedisCacheBackend",
    "CacheBackendRegistry",
    "cache_backend_registry",
    "CacheConfig",
    "ProfileCache",
    "cached_profile",
    "create_cache",
    "hash_file",
    "RedisConnectionError",
    # === P2: Resilience and Fallback ===
    # States and types
    "CircuitState",
    "BackendHealth",
    "FailureType",
    # Configuration
    "CircuitBreakerConfig",
    "RetryConfig",
    "HealthCheckConfig",
    "ResilienceConfig",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitOpenError",
    # Retry logic
    "RetryPolicy",
    # Health monitoring
    "HealthMonitor",
    # Resilient backends
    "ResilientCacheBackend",
    "FallbackChain",
    # Factory functions
    "create_resilient_redis_backend",
    "create_high_availability_cache",
    # === P2: Observability/Metrics ===
    "MetricType",
    "SpanStatus",
    "Span",
    "SpanEvent",
    "SpanProtocol",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricValue",
    "MetricsCollector",
    "SpanExporter",
    "ConsoleSpanExporter",
    "InMemorySpanExporter",
    "OTLPSpanExporter",
    "SpanExporterRegistry",
    "span_exporter_registry",
    "TelemetryConfig",
    "ProfilerTelemetry",
    "traced",
    "timed",
    "OpenTelemetryIntegration",
    "get_telemetry",
    "set_telemetry",
    "get_metrics",
    "create_telemetry",
    # === P2: Rule Quality Scoring ===
    "QualityLevel",
    "RuleType",
    "ConfusionMatrix",
    "QualityMetrics",
    "RuleProtocol",
    "ValidationRule",
    "QualityEstimator",
    "SamplingQualityEstimator",
    "HeuristicQualityEstimator",
    "CrossValidationEstimator",
    "QualityEstimatorRegistry",
    "quality_estimator_registry",
    "ScoringConfig",
    "RuleQualityScore",
    "RuleQualityScorer",
    "QualityTrendPoint",
    "QualityTrendAnalyzer",
    "estimate_quality",
    "score_rule",
    "compare_rules",
    # === P2: Custom Patterns YAML ===
    "CustomPatternPriority",
    "PatternExample",
    "PatternConfig",
    "PatternGroup",
    "PatternConfigSchema",
    "PatternConfigLoader",
    "CustomPatternRegistry",
    "pattern_registry",
    "DEFAULT_PATTERNS_YAML",
    "load_default_patterns",
    "load_patterns",
    "load_patterns_directory",
    "register_pattern",
    "match_custom_patterns",
    "infer_type_from_patterns",
    "export_patterns",
    # === P3: Distributed Processing ===
    "BackendType",
    "PartitionStrategy",
    "PartitionInfo",
    "WorkerResult",
    "BackendConfig",
    "SparkConfig",
    "DaskConfig",
    "RayConfig",
    "DistributedProfileConfig",
    "DistributedBackend",
    "LocalBackend",
    "SparkBackend",
    "DaskBackend",
    "RayBackend",
    "BackendRegistry",
    "DistributedProfiler",
    "create_distributed_profiler",
    "profile_distributed",
    "get_available_backends",
    # === P3: ML-based Type Inference ===
    "FeatureType",
    "Feature",
    "FeatureVector",
    "FeatureExtractor",
    "NameFeatureExtractor",
    "ValueFeatureExtractor",
    "StatisticalFeatureExtractor",
    "ContextFeatureExtractor",
    "FeatureExtractorRegistry",
    "InferenceResult",
    "InferenceModel",
    "RuleBasedModel",
    "NaiveBayesModel",
    "EnsembleModel",
    "InferenceModelRegistry",
    "MLInferenceConfig",
    "MLTypeInferrer",
    "create_inference_model",
    "infer_column_type_ml",
    "infer_table_types_ml",
    # === P3: Automatic Threshold Tuning ===
    "TuningStrategy",
    "ThresholdType",
    "ColumnThresholds",
    "TableThresholds",
    "StrictnessPreset",
    "TuningStrategyImpl",
    "ConservativeStrategy",
    "BalancedStrategy",
    "PermissiveStrategy",
    "AdaptiveStrategy",
    "StatisticalStrategy",
    "DomainAwareStrategy",
    "TuningStrategyRegistry",
    "TuningConfig",
    "ThresholdTuner",
    "ThresholdTestResult",
    "ThresholdTester",
    "create_tuner",
    "tune_thresholds",
    "get_available_strategies",
    # === P3: Profile Visualization ===
    "ChartType",
    "ColorScheme",
    "ReportTheme",
    "SectionType",
    "COLOR_PALETTES",
    "ChartData",
    "ChartConfig",
    "ThemeConfig",
    "SectionContent",
    "ReportConfig",
    "ProfileData",
    "THEME_CONFIGS",
    "ChartRenderer",
    "SectionRenderer",
    "ChartRendererRegistry",
    "chart_renderer_registry",
    "SectionRegistry",
    "section_registry",
    "ThemeRegistry",
    "theme_registry",
    "SVGChartRenderer",
    "BaseSectionRenderer",
    "OverviewSectionRenderer",
    "DataQualitySectionRenderer",
    "ColumnDetailsSectionRenderer",
    "PatternsSectionRenderer",
    "RecommendationsSectionRenderer",
    "CustomSectionRenderer",
    "ReportTemplate",
    "ProfileDataConverter",
    "HTMLReportGenerator",
    "ReportExporter",
    "generate_report",
    "compare_profile_reports",
    # === Phase 7 P0: Suite Export System ===
    "ExportFormat",
    "CodeStyle",
    "OutputMode",
    "ExportConfig",
    "DEFAULT_CONFIG",
    "MINIMAL_CONFIG",
    "VERBOSE_CONFIG",
    "SuiteFormatterProtocol",
    "SuiteFormatter",
    "YAMLFormatter",
    "JSONFormatter",
    "PythonFormatter",
    "TOMLFormatter",
    "CheckpointFormatter",
    "FormatterRegistry",
    "formatter_registry",
    "register_formatter",
    "ExportPostProcessor",
    "AddHeaderPostProcessor",
    "AddFooterPostProcessor",
    "TemplatePostProcessor",
    "ExportResult",
    "SuiteExporter",
    "create_exporter",
    "export_suite",
    "format_suite",
    "get_available_formats",
    # === Phase 7 P0: Suite Configuration System ===
    "ConfigPreset",
    "SuiteOutputFormat",
    "GeneratorMode",
    "CategoryConfig",
    "ConfidenceConfig",
    "SuiteOutputConfig",
    "SuiteGeneratorOptionsConfig",
    "SuiteGeneratorConfig",
    "PRESETS",
    "load_suite_config",
    "save_suite_config",
    "CLIArguments",
    "build_config_from_cli",
    # === Phase 7 P0: Suite CLI Handlers ===
    "GenerateSuiteResult",
    "QuickSuiteResult",
    "SuiteGenerationProgress",
    "SuiteGenerationHandler",
    "get_suite_formats",
    "get_available_presets",
    "get_available_categories",
    "format_category_help",
    "format_preset_help",
    "validate_format",
    "validate_preset",
    "validate_strictness",
    "validate_confidence",
    "create_cli_handler",
    "run_generate_suite",
    "run_quick_suite",
    # === Phase 7 P0: Streaming Pattern Matching ===
    "AggregationMethod",
    # === Phase 7 P0: Progress Callback System ===
    # Event types and levels
    "EventLevel",
    "EventType",
    # Context and metrics
    "ProgressContext",
    "ProgressMetrics",
    "StandardProgressEvent",
    # Protocols
    "StandardProgressCallback",
    "AsyncProgressCallback",
    "LifecycleCallback",
    # Base adapter
    "CallbackAdapter",
    # Console adapters
    "ConsoleStyle",
    "ConsoleAdapter",
    "MinimalConsoleAdapter",
    # Logging adapter
    "LoggingAdapter",
    # File adapter
    "FileOutputConfig",
    "FileAdapter",
    # Callback chain
    "CallbackChain",
    # Filtering and throttling
    "FilterConfig",
    "FilteringAdapter",
    "ThrottleConfig",
    "ThrottlingAdapter",
    # Buffering
    "BufferConfig",
    "BufferingAdapter",
    # Async
    "AsyncAdapter",
    # Registry
    "CallbackRegistry",
    # Emitter
    "ProgressEmitter",
    # Presets
    "CallbackPresets",
    # Convenience functions
    "create_callback_chain",
    "create_console_callback",
    "create_logging_callback",
    "create_file_callback",
    "with_throttling",
    "with_filtering",
    "with_buffering",
    # === Phase 7 P0: Internationalization (i18n) ===
    # Message codes
    "MessageCode",
    # Locale management
    "LocaleInfo",
    "LocaleManager",
    "BUILTIN_LOCALES",
    "set_locale",
    "get_locale",
    # Message catalog
    "MessageEntry",
    "MessageCatalog",
    # Message loaders
    "MessageLoader",
    "DictMessageLoader",
    "FileMessageLoader",
    # Formatter
    "PlaceholderFormatter",
    # Main interface
    "I18n",
    # I18n exceptions
    "I18nError",
    "I18nAnalysisError",
    "I18nPatternError",
    "I18nTypeError",
    "I18nIOError",
    "I18nTimeoutError",
    "I18nValidationError",
    # Registry
    "MessageCatalogRegistry",
    # Convenience functions
    "get_message",
    "translate",
    "register_messages",
    "load_messages_from_file",
    "create_message_loader",
    # Context manager
    "locale_context",
    # Presets
    "I18nPresets",
    "ChunkProcessingStatus",
    "PatternChunkStats",
    "PatternState",
    "ColumnPatternState",
    "AggregationStrategy",
    "IncrementalAggregation",
    "WeightedAggregation",
    "SlidingWindowAggregation",
    "ExponentialAggregation",
    "ConsensusAggregation",
    "AdaptiveAggregation",
    "AggregationStrategyRegistry",
    "aggregation_strategy_registry",
    "StreamingPatternResult",
    "PatternEvent",
    "PatternEventCallback",
    "StreamingPatternConfig",
    "StreamingPatternMatcher",
    "StreamingPatternIntegration",
    "create_streaming_matcher",
    "stream_match_patterns",
    "get_available_aggregation_methods",
]
