# Rule Generation

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 `src/truthound/profiler/generators/base.py`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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
    validator_class: str                # Validator class name
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

실무 운영 가이드에서 Base을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

Generates 스키마 검증 rules.

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

Generates completeness 검증 rules.

```python
from truthound.profiler.generators import CompletenessRuleGenerator

generator = CompletenessRuleGenerator()
rules = generator.generate(profile)

# Generated rules:
# - Maximum null ratio
# - NotNull constraint (for columns with no nulls)
```

### UniquenessRuleGenerator

Generates uniqueness 검증 rules.

```python
from truthound.profiler.generators import UniquenessRuleGenerator

generator = UniquenessRuleGenerator()
rules = generator.generate(profile)

# Generated rules:
# - Unique constraint (100% unique ratio)
# - Primary Key candidates
```

### PatternRuleGenerator

Generates pattern 검증 rules.

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

Generates distribution 검증 rules.

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

네이티브 `generate_suite`는 복원한 사전을 포함해 불완전한 프로파일에
`IncompleteProfileError`를 발생시킵니다. 선택한 생성기가 실패하면 원래 예외 본문을
포함하지 않는 `ProfilerError`를 발생시키며 해당 생성기를 조용히 누락하지 않습니다.
생성 전에 `profile.complete`와 테이블·컬럼의 `execution_issues`를 확인합니다.

```python
from truthound.profiler import generate_suite, Strictness

# Generate rule suite from profile
suite = generate_suite(
    profile,
    strictness=Strictness.STRICT,
    include_categories=["schema", "completeness"],  # Categories to include
    exclude_categories=["anomaly"],                  # Categories to exclude
)

# Review rules
for rule in suite.rules:
    print(f"{rule.name}: {rule.validator_class}")
    print(f"  Column: {rule.column}")
    print(f"  Confidence: {rule.confidence}")
```

## Presets

| 실무 운영 가이드에서 Preset을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 실무 운영 가이드에서 `default`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 General을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `strict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Production을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `loose`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Development/testing을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `minimal`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Essential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `comprehensive`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ci_cd`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CI/CD 파이프라인 optimized |
| 실무 운영 가이드에서 `schema_only`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 스키마 검증 only |
| 실무 운영 가이드에서 `format_only`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Format/pattern 검증 only |

```python
from truthound.profiler import generate_suite

from truthound.profiler.suite_config import SuiteGeneratorConfig

config = SuiteGeneratorConfig.from_preset("ci_cd")
suite = generate_suite(
    profile, strictness=config.strictness,
    include_categories=config.categories.include or None,
    exclude_categories=config.categories.exclude or None,
    min_confidence=config.confidence.min_level,
    name=config.name,
)
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
                        validator_class="UniqueValidator",
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

## 다음 단계

- 실무 운영 가이드에서 Quality, Scoring, Evaluate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Threshold, Tuning, Automatic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
