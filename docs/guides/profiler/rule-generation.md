# Rule Generation

This document describes the system for automatically generating validation rules from profile results.

## Overview

The rule generation system defined in `src/truthound/profiler/generators/base.py` automatically generates validation rules based on profile analysis results.

## RuleCategory

```python
class RuleCategory(str, Enum):
    """Generatable rule categories"""

    SCHEMA = "schema"             # Schema validation (column existence, types)
    COMPLETENESS = "completeness" # Completeness (null ratio)
    UNIQUENESS = "uniqueness"     # Uniqueness (duplicate checking)
    FORMAT = "format"             # Format validation (regex)
    DISTRIBUTION = "distribution" # Distribution validation (range, cardinality)
    PATTERN = "pattern"           # Pattern validation
    TEMPORAL = "temporal"         # Temporal validation
    RELATIONSHIP = "relationship" # Relationship validation
    ANOMALY = "anomaly"           # Anomaly validation
```

## Strictness Levels

```python
class Strictness(str, Enum):
    """Rule generation strictness"""

    LOOSE = "loose"    # Loose thresholds, fewer rules
    MEDIUM = "medium"  # Balanced default
    STRICT = "strict"  # Strict thresholds, comprehensive rules
```

## GeneratedRule

```python
@dataclass
class GeneratedRule:
    """Generated validation rule"""

    name: str                          # Rule name
    category: RuleCategory             # Rule category
    column: str | None                 # Target column (None = table level)
    validator_type: str                # Validator class name
    parameters: dict[str, Any]         # Validator parameters
    confidence: RuleConfidence         # Confidence level
    description: str                   # Rule description
    severity: str = "high"             # Default severity

    # Metadata
    source: str = "profiler"           # Generation source
    generated_at: datetime = field(default_factory=datetime.now)
```

## RuleConfidence

```python
class RuleConfidence(str, Enum):
    """Rule confidence level"""

    LOW = "low"        # Low confidence (review needed)
    MEDIUM = "medium"  # Medium confidence
    HIGH = "high"      # High confidence
```

## RuleGenerator ABC

Base class for all rule generators.

```python
from abc import ABC, abstractmethod

class RuleGenerator(ABC):
    """Rule generator abstract class"""

    @abstractmethod
    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness = Strictness.MEDIUM,
    ) -> list[GeneratedRule]:
        """Generate rules from table profile"""
        ...

    @abstractmethod
    def generate_for_column(
        self,
        column_profile: ColumnProfile,
        strictness: Strictness = Strictness.MEDIUM,
    ) -> list[GeneratedRule]:
        """Generate rules from column profile"""
        ...
```

## Built-in Rule Generators

### SchemaRuleGenerator

Generates schema validation rules.

```python
from truthound.profiler.generators import SchemaRuleGenerator

generator = SchemaRuleGenerator()
rules = generator.generate(profile, Strictness.STRICT)

# Generated rules:
# - Column existence validation
# - Data type validation
# - Nullable validation
```

### CompletenessRuleGenerator

Generates completeness validation rules.

```python
from truthound.profiler.generators import CompletenessRuleGenerator

generator = CompletenessRuleGenerator()
rules = generator.generate(profile)

# Generated rules:
# - Maximum null ratio
# - NotNull constraint (for columns with no nulls)
```

### UniquenessRuleGenerator

Generates uniqueness validation rules.

```python
from truthound.profiler.generators import UniquenessRuleGenerator

generator = UniquenessRuleGenerator()
rules = generator.generate(profile)

# Generated rules:
# - Unique constraint (100% unique ratio)
# - Primary Key candidates
```

### PatternRuleGenerator

Generates pattern validation rules.

```python
from truthound.profiler.generators import PatternRuleGenerator

generator = PatternRuleGenerator()
rules = generator.generate(profile)

# Generated rules:
# - Email format validation
# - Phone number format validation
# - Custom pattern validation
```

### DistributionRuleGenerator

Generates distribution validation rules.

```python
from truthound.profiler.generators import DistributionRuleGenerator

generator = DistributionRuleGenerator()
rules = generator.generate(profile)

# Generated rules:
# - Range validation (min/max)
# - Allowed values validation (categorical)
# - Cardinality validation
```

## Unified Rule Generation

```python
from truthound.profiler import generate_suite, Strictness

# Generate rule suite from profile
suite = generate_suite(
    profile,
    strictness=Strictness.STRICT,
    include=["schema", "completeness"],  # Categories to include
    exclude=["anomaly"],                  # Categories to exclude
)

# Review rules
for rule in suite.rules:
    print(f"{rule.name}: {rule.validator_type}")
    print(f"  Column: {rule.column}")
    print(f"  Confidence: {rule.confidence}")
```

## Presets

| Preset | Description |
|--------|-------------|
| `default` | General use (balanced rules) |
| `strict` | Production data (strict rules) |
| `loose` | Development/testing (loose rules) |
| `minimal` | Essential rules only |
| `comprehensive` | All possible rules |
| `ci_cd` | CI/CD pipeline optimized |
| `schema_only` | Schema validation only |
| `format_only` | Format/pattern validation only |

```python
from truthound.profiler import generate_suite

suite = generate_suite(profile, preset="ci_cd")
```

## Rule Export

### YAML Format

```python
from truthound.profiler.generators import save_suite

save_suite(suite, "rules.yaml", format="yaml")
```

```yaml
# rules.yaml
version: "1.0"
rules:
  - name: email_not_null
    category: completeness
    column: email
    validator: NotNullValidator
    parameters: {}
    severity: high

  - name: age_range
    category: distribution
    column: age
    validator: BetweenValidator
    parameters:
      min_value: 0
      max_value: 120
    severity: medium
```

### JSON Format

```python
save_suite(suite, "rules.json", format="json")
```

### Python Format

```python
save_suite(suite, "rules.py", format="python")
```

```python
# rules.py
from truthound import Suite, NotNullValidator, BetweenValidator

suite = Suite(
    validators=[
        NotNullValidator(columns=["email"]),
        BetweenValidator(columns=["age"], min_value=0, max_value=120),
    ]
)
```

## Custom Rule Generators

```python
from truthound.profiler.generators import RuleGenerator, GeneratedRule

class MyCustomGenerator(RuleGenerator):
    """Custom rule generator"""

    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness = Strictness.MEDIUM,
    ) -> list[GeneratedRule]:
        rules = []

        for col in profile.columns:
            if col.name.endswith("_id"):
                rules.append(
                    GeneratedRule(
                        name=f"{col.name}_unique",
                        category=RuleCategory.UNIQUENESS,
                        column=col.name,
                        validator_type="UniqueValidator",
                        parameters={},
                        confidence=RuleConfidence.HIGH,
                        description=f"{col.name} must be unique",
                    )
                )

        return rules

    def generate_for_column(self, column_profile, strictness):
        return []  # Generate only at table level
```

## Rule Generator Registry

```python
from truthound.profiler.generators import GeneratorRegistry

# Register custom generator
GeneratorRegistry.register("custom", MyCustomGenerator)

# Retrieve registered generator
generator = GeneratorRegistry.get("schema")

# Generate rules with all generators
all_rules = GeneratorRegistry.generate_all(profile, Strictness.MEDIUM)
```

## CLI Usage

```bash
# Generate rules from profile
th generate-suite profile.json -o rules.yaml

# Profile and generate rules at once
th quick-suite data.csv -o rules.yaml

# Specify strictness
th quick-suite data.csv -o rules.yaml --strictness strict

# Use preset
th quick-suite data.csv -o rules.yaml --preset ci_cd

# Filter categories
th quick-suite data.csv -o rules.yaml --include schema,completeness
```

## Next Steps

- [Quality Scoring](quality-scoring.md) - Evaluate quality of generated rules
- [Threshold Tuning](threshold-tuning.md) - Automatic threshold adjustment
