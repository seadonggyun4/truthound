# Rule Generation

이 문서는 프로파일 결과에서 검증 규칙을 자동 생성하는 시스템을 설명합니다.

## 개요

`src/truthound/profiler/generators/base.py`에 정의된 규칙 생성 시스템은 프로파일 분석 결과를 기반으로 검증 규칙을 자동으로 생성합니다.

## RuleCategory

```python
class RuleCategory(str, Enum):
    """생성 가능한 규칙 카테고리"""

    SCHEMA = "schema"             # 스키마 검증 (컬럼 존재, 타입)
    COMPLETENESS = "completeness" # 완전성 (null 비율)
    UNIQUENESS = "uniqueness"     # 고유성 (중복 검사)
    FORMAT = "format"             # 형식 검증 (정규식)
    DISTRIBUTION = "distribution" # 분포 검증 (범위, 카디널리티)
    PATTERN = "pattern"           # 패턴 검증
    TEMPORAL = "temporal"         # 시간 검증
    RELATIONSHIP = "relationship" # 관계 검증
    ANOMALY = "anomaly"           # 이상치 검증
```

## Strictness 수준

```python
class Strictness(str, Enum):
    """규칙 생성 엄격도"""

    LOOSE = "loose"    # 느슨한 임계값, 적은 규칙
    MEDIUM = "medium"  # 균형잡힌 기본값
    STRICT = "strict"  # 엄격한 임계값, 포괄적 규칙
```

## GeneratedRule

```python
@dataclass
class GeneratedRule:
    """생성된 검증 규칙"""

    name: str                          # 규칙 이름
    category: RuleCategory             # 규칙 카테고리
    column: str | None                 # 대상 컬럼 (None = 테이블 수준)
    validator_type: str                # Validator 클래스 이름
    parameters: dict[str, Any]         # Validator 파라미터
    confidence: RuleConfidence        # 신뢰 수준
    description: str                   # 규칙 설명
    severity: str = "high"             # 기본 심각도

    # 메타데이터
    source: str = "profiler"           # 생성 소스
    generated_at: datetime = field(default_factory=datetime.now)
```

## RuleConfidence

```python
class RuleConfidence(str, Enum):
    """규칙 신뢰 수준"""

    LOW = "low"        # 낮은 신뢰도 (검토 필요)
    MEDIUM = "medium"  # 중간 신뢰도
    HIGH = "high"      # 높은 신뢰도
```

## RuleGenerator ABC

모든 규칙 생성기의 기본 클래스입니다.

```python
from abc import ABC, abstractmethod

class RuleGenerator(ABC):
    """규칙 생성기 추상 클래스"""

    @abstractmethod
    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness = Strictness.MEDIUM,
    ) -> list[GeneratedRule]:
        """테이블 프로파일에서 규칙 생성"""
        ...

    @abstractmethod
    def generate_for_column(
        self,
        column_profile: ColumnProfile,
        strictness: Strictness = Strictness.MEDIUM,
    ) -> list[GeneratedRule]:
        """컬럼 프로파일에서 규칙 생성"""
        ...
```

## 내장 규칙 생성기

### SchemaRuleGenerator

스키마 검증 규칙을 생성합니다.

```python
from truthound.profiler.generators import SchemaRuleGenerator

generator = SchemaRuleGenerator()
rules = generator.generate(profile, Strictness.STRICT)

# 생성되는 규칙:
# - 컬럼 존재 검증
# - 데이터 타입 검증
# - Nullable 검증
```

### CompletenessRuleGenerator

완전성 검증 규칙을 생성합니다.

```python
from truthound.profiler.generators import CompletenessRuleGenerator

generator = CompletenessRuleGenerator()
rules = generator.generate(profile)

# 생성되는 규칙:
# - 최대 null 비율
# - NotNull 제약 (null이 없는 컬럼)
```

### UniquenessRuleGenerator

고유성 검증 규칙을 생성합니다.

```python
from truthound.profiler.generators import UniquenessRuleGenerator

generator = UniquenessRuleGenerator()
rules = generator.generate(profile)

# 생성되는 규칙:
# - Unique 제약 (고유 비율 100%)
# - Primary Key 후보
```

### PatternRuleGenerator

패턴 검증 규칙을 생성합니다.

```python
from truthound.profiler.generators import PatternRuleGenerator

generator = PatternRuleGenerator()
rules = generator.generate(profile)

# 생성되는 규칙:
# - 이메일 형식 검증
# - 전화번호 형식 검증
# - 커스텀 패턴 검증
```

### DistributionRuleGenerator

분포 검증 규칙을 생성합니다.

```python
from truthound.profiler.generators import DistributionRuleGenerator

generator = DistributionRuleGenerator()
rules = generator.generate(profile)

# 생성되는 규칙:
# - 범위 검증 (min/max)
# - 허용 값 검증 (카테고리)
# - 카디널리티 검증
```

## 통합 규칙 생성

```python
from truthound.profiler import generate_suite, Strictness

# 프로파일에서 규칙 스위트 생성
suite = generate_suite(
    profile,
    strictness=Strictness.STRICT,
    include=["schema", "completeness"],  # 포함할 카테고리
    exclude=["anomaly"],                  # 제외할 카테고리
)

# 규칙 확인
for rule in suite.rules:
    print(f"{rule.name}: {rule.validator_type}")
    print(f"  Column: {rule.column}")
    print(f"  Confidence: {rule.confidence}")
```

## 프리셋

| 프리셋 | 설명 |
|--------|------|
| `default` | 일반적인 사용 (균형잡힌 규칙) |
| `strict` | 프로덕션 데이터 (엄격한 규칙) |
| `loose` | 개발/테스트 (느슨한 규칙) |
| `minimal` | 필수 규칙만 |
| `comprehensive` | 모든 가능한 규칙 |
| `ci_cd` | CI/CD 파이프라인 최적화 |
| `schema_only` | 스키마 검증만 |
| `format_only` | 형식/패턴 검증만 |

```python
from truthound.profiler import generate_suite

suite = generate_suite(profile, preset="ci_cd")
```

## 규칙 내보내기

### YAML 형식

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

### JSON 형식

```python
save_suite(suite, "rules.json", format="json")
```

### Python 형식

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

## 커스텀 규칙 생성기

```python
from truthound.profiler.generators import RuleGenerator, GeneratedRule

class MyCustomGenerator(RuleGenerator):
    """커스텀 규칙 생성기"""

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
        return []  # 테이블 수준에서만 생성
```

## 규칙 생성기 레지스트리

```python
from truthound.profiler.generators import GeneratorRegistry

# 커스텀 생성기 등록
GeneratorRegistry.register("custom", MyCustomGenerator)

# 등록된 생성기 조회
generator = GeneratorRegistry.get("schema")

# 모든 생성기로 규칙 생성
all_rules = GeneratorRegistry.generate_all(profile, Strictness.MEDIUM)
```

## CLI 사용법

```bash
# 프로파일에서 규칙 생성
th generate-suite profile.json -o rules.yaml

# 한 번에 프로파일링 + 규칙 생성
th quick-suite data.csv -o rules.yaml

# 엄격도 지정
th quick-suite data.csv -o rules.yaml --strictness strict

# 프리셋 사용
th quick-suite data.csv -o rules.yaml --preset ci_cd

# 카테고리 필터링
th quick-suite data.csv -o rules.yaml --include schema,completeness
```

## 다음 단계

- [품질 스코어링](quality-scoring.md) - 생성된 규칙의 품질 평가
- [임계값 튜닝](threshold-tuning.md) - 자동 임계값 조정
