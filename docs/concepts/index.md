# Concepts

Concepts and Architecture explains why Truthound 3.0 works the way it does. Read this section when you want the system model behind zero-config behavior, the result model, planner/runtime boundaries, or extension architecture.

## Start Here If You Need To Understand

- why `ValidationRunResult` is the canonical output model
- how zero-config local state is managed through `TruthoundContext`
- how planners, validators, datasources, and reporters fit together
- where advanced modules such as lineage, realtime, and orchestration attach to the core platform

## Architecture

### [Architecture Overview](architecture.md)

Understand Truthound's internal design:

- Core design principles
- Module structure
- Data flow architecture
- Extension points
- Performance architecture

### [Data Sources Architecture](datasources-architecture.md)

Technical deep dive into multi-backend support:

- DataSource protocol
- Execution engine abstraction
- Query pushdown optimization
- Connection pooling
- Error handling strategies

## Advanced Features

### [Advanced Features (ML, Lineage, Realtime)](advanced.md)

Enterprise-grade capabilities:

- ML-based anomaly detection
- Data drift detection
- Data lineage tracking
- Real-time streaming validation
- Model monitoring

### [Plugin Architecture](plugins.md)

Extend Truthound with custom functionality:

- Plugin system design
- Security sandbox
- Code signing
- Version constraints
- Hot reload support

## Technical Reference

### [Statistical Methods](statistical-methods.md)

Mathematical foundations for data quality:

- Kolmogorov-Smirnov test
- Chi-squared test
- Population Stability Index (PSI)
- Jensen-Shannon divergence
- Anomaly detection algorithms

### [Test Coverage](test-coverage.md)

Quality assurance practices:

- Test suite overview
- Coverage metrics
- Testing strategies
- CI/CD test integration

## Concept Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        Truthound Core                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Validators  │    │  Profiler    │    │  Reporters   │       │
│  │  (289 types) │    │  (Auto-gen)  │    │  (5 formats) │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │   Core Engine   │                          │
│                    │  (Polars-based) │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │                │
│  ┌──────▼───────┐    ┌──────▼───────┐    ┌──────▼───────┐       │
│  │ Data Sources │    │    Stores    │    │   Plugins    │       │
│  │ (12 backends)│    │ (5 backends) │    │  (Extensible)│       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     Advanced Modules                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  ML Module   │    │   Lineage    │    │   Realtime   │       │
│  │  (Anomaly,   │    │  (Tracking,  │    │  (Kafka,     │       │
│  │   Drift)     │    │   OpenLineage│    │   Kinesis)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Links

| Concept | Document |
|---------|----------|
| Concepts overview | [Concepts](index.md) |
| System design | [Architecture](architecture.md) |
| Data backends | [DataSources Architecture](datasources-architecture.md) |
| ML & Lineage | [Advanced Features](advanced.md) |
| Extensibility | [Plugin Architecture](plugins.md) |
| Statistics | [Statistical Methods](statistical-methods.md) |

## Recommended Reading Path

1. [Architecture](architecture.md)
2. [Zero-Config Context](zero-config.md)
3. [Plugin Architecture](plugins.md)
4. [Advanced Features](advanced.md)
5. [OpenLineage](openlineage.md) or [Statistical Methods](statistical-methods.md), depending on your use case
