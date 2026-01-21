# 보안 가이드

Truthound는 검증기 실행 시 발생할 수 있는 보안 위협을 방지하기 위한 포괄적인 보안 기능을 제공합니다.

## 개요

보안 모듈 아키텍처:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Security Module                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│      SQL Injection 방지        │   │          ReDoS 보호                    │
├───────────────────────────────┤   ├───────────────────────────────────────┤
│ • Query Validator             │   │ • Static Analyzer                     │
│ • Parameterized Query         │   │ • ML Pattern Analyzer                 │
│ • Whitelist Validator         │   │ • Pattern Optimizer                   │
│ • Security Policy             │   │ • CVE Database                        │
│ • Audit Logger                │   │ • CPU Monitor                         │
└───────────────────────────────┘   │ • Profiler                            │
                                    │ • RE2 Engine                          │
                                    └───────────────────────────────────────┘
```

---

## 1. SQL 인젝션 방지

SQL 데이터소스 검증 시 인젝션 공격을 방지합니다.

### SecurityLevel

| 레벨 | 설명 |
|------|------|
| `STRICT` | 최대 보안, 최소 허용 연산 |
| `STANDARD` | 균형 잡힌 보안 (기본값) |
| `PERMISSIVE` | 신뢰 환경용 완화된 보안 |

### SecurityPolicy

```python
from truthound.validators.security import (
    SecurityPolicy,
    SecurityLevel,
    SQLQueryValidator,
)

# 프리셋 정책
strict_policy = SecurityPolicy.strict()
standard_policy = SecurityPolicy.standard()
permissive_policy = SecurityPolicy.permissive()

# 커스텀 정책
policy = SecurityPolicy(
    level=SecurityLevel.STANDARD,
    max_query_length=10000,           # 최대 쿼리 길이
    max_identifier_length=128,        # 최대 식별자 길이

    # 구조적 권한
    allow_joins=True,                 # JOIN 허용
    allow_subqueries=True,            # 서브쿼리 허용
    allow_aggregations=True,          # 집계 함수 허용
    allow_window_functions=True,      # 윈도우 함수 허용
    allow_cte=True,                   # WITH 절 허용
    allow_union=False,                # UNION 차단 (인젝션 벡터)

    # 허용 문장 타입
    allowed_statements={"SELECT", "WITH"},

    # 차단 패턴 (정규식)
    blocked_patterns=[r"xp_cmdshell", r"sp_executesql"],

    # 차단 함수
    blocked_functions=[
        "SLEEP",
        "BENCHMARK",
        "LOAD_FILE",
        "INTO OUTFILE",
        "INTO DUMPFILE",
    ],

    # 화이트리스트 (빈 경우 모두 허용)
    allowed_tables={"orders", "customers"},
    allowed_columns={"id", "name", "amount"},

    # 위반 콜백
    on_violation=lambda name, matched: print(f"Violation: {name}"),
)
```

### SQLQueryValidator

```python
from truthound.validators.security import (
    SQLQueryValidator,
    validate_sql_query,
    SQLInjectionError,
    QueryValidationError,
)

# 검증기 생성
validator = SQLQueryValidator(policy=policy)

# 쿼리 검증
try:
    validator.validate("SELECT * FROM orders WHERE amount > 100")
    print("Query is safe")
except SQLInjectionError as e:
    print(f"Injection detected: {e.pattern}")
except QueryValidationError as e:
    print(f"Validation failed: {e}")

# 편의 함수
validate_sql_query(
    "SELECT id, amount FROM orders",
    allowed_tables=["orders", "customers"],
)
```

### 위험 패턴 감지

내장된 위험 패턴 레지스트리:

| 카테고리 | 패턴 | 심각도 |
|----------|------|--------|
| DDL | `CREATE`, `ALTER`, `DROP`, `TRUNCATE` | HIGH |
| DCL | `GRANT`, `REVOKE`, `DENY` | HIGH |
| DML 수정 | `INSERT`, `UPDATE`, `DELETE` | HIGH |
| 실행 | `EXEC`, `EXECUTE`, `CALL` | HIGH |
| 파일 | `LOAD_FILE`, `INTO OUTFILE` | HIGH |
| 스택 쿼리 | `; SELECT`, `; DROP` | HIGH |
| UNION 인젝션 | `UNION SELECT` | MEDIUM |
| 시간 기반 | `SLEEP`, `WAITFOR DELAY`, `BENCHMARK` | HIGH |
| 에러 기반 | `EXTRACTVALUE`, `UPDATEXML` | MEDIUM |
| 불리언 기반 | `OR 1=1`, `AND '1'='1'` | HIGH |
| 주석 | `--`, `/* */` | LOW-MEDIUM |

### SecureSQLBuilder

플루언트 인터페이스로 안전한 쿼리 빌드:

```python
from truthound.validators.security import (
    SecureSQLBuilder,
    ParameterizedQuery,
)

builder = SecureSQLBuilder(
    allowed_tables=["orders", "customers"],
    policy=SecurityPolicy.standard(),
)

# 쿼리 빌드
query = (
    builder
    .select("orders", ["id", "amount", "status"])
    .join("customers", "orders.customer_id = customers.id")
    .where("amount > :min_amount")
    .where("status = :status")
    .group_by("status")
    .having("COUNT(*) > :min_count")
    .order_by("amount", desc=True)
    .limit(100)
    .offset(0)
    .build({
        "min_amount": 100,
        "status": "pending",
        "min_count": 5,
    })
)

# Polars SQL 컨텍스트로 실행
import polars as pl
ctx = pl.SQLContext()
ctx.register("orders", orders_lf)
ctx.register("customers", customers_lf)

result = builder.execute(ctx, query)
```

### ParameterizedQuery

```python
from truthound.validators.security import ParameterizedQuery

query = ParameterizedQuery(
    template="SELECT * FROM orders WHERE amount > :min_amount AND status = :status",
    parameters={"min_amount": 100, "status": "pending"},
)

# 렌더링 (값 이스케이프)
rendered = query.render()
# SELECT * FROM orders WHERE amount > 100 AND status = 'pending'
```

지원 타입과 이스케이프:

| 타입 | 이스케이프 |
|------|------------|
| `None` | `NULL` |
| `bool` | `TRUE`/`FALSE` |
| `int`, `float` | 그대로 |
| `str` | 작은따옴표, `'` → `''` |
| `list`, `tuple` | `(val1, val2, ...)` |

### SchemaWhitelist

```python
from truthound.validators.security import (
    SchemaWhitelist,
    WhitelistValidator,
)

# 스키마 화이트리스트 정의
whitelist = SchemaWhitelist()
whitelist.add_table("orders", ["id", "customer_id", "amount", "status"])
whitelist.add_table("customers", ["id", "name", "email"])

# 테이블/컬럼 검증
whitelist.validate_table("orders")  # OK
whitelist.validate_column("orders", "amount")  # OK
whitelist.validate_column("orders", "password")  # QueryValidationError

# 쿼리 검증
validator = WhitelistValidator(whitelist)
validator.validate_query("SELECT id, amount FROM orders")  # OK
validator.validate_query("SELECT password FROM users")  # Error
```

### SecureQueryMixin

검증기에서 안전한 쿼리 실행:

```python
from truthound.validators.security import SecureQueryMixin
from truthound.validators.base import Validator

class MyValidator(Validator, SecureQueryMixin):
    def __init__(self):
        super().__init__()
        self.set_security_policy(SecurityPolicy.strict())

    def validate(self, lf):
        # 안전한 쿼리 빌드
        query = self.build_secure_query(
            table="data",
            columns=["id", "value"],
            where="value > :threshold",
            parameters={"threshold": 100},
            allowed_tables=["data"],
        )

        # 안전한 쿼리 실행
        result = self.execute_secure_query(lf, query, table_name="data")
        return self.process_result(result)
```

### QueryAuditLogger

쿼리 실행 감사 로깅:

```python
from truthound.validators.security import QueryAuditLogger

logger = QueryAuditLogger(
    max_entries=10000,
    log_full_queries=False,  # 값 마스킹
    python_logger=logging.getLogger("sql_audit"),
)

# 쿼리 로깅
logger.log_query(
    query="SELECT * FROM users WHERE email = 'test@example.com'",
    success=True,
    user="admin",
    context={"source": "api"},
)

# 감사 조회
recent = logger.get_recent(100)
failures = logger.get_failures(50)
by_hash = logger.get_by_hash("abc123...")

# 통계
stats = logger.get_stats()
# {
#   "total_queries": 1000,
#   "successful": 950,
#   "failed": 50,
#   "success_rate": 0.95,
#   "unique_queries": 120,
# }

# 파일 내보내기
logger.export_to_file("audit.log")
```

---

## 2. ReDoS 보호

정규식 서비스 거부(ReDoS) 공격을 방지합니다.

### ReDoSRisk

| 레벨 | 설명 |
|------|------|
| `NONE` | 알려진 취약점 없음 |
| `LOW` | 미미한 우려, 대부분 안전 |
| `MEDIUM` | 일부 위험 패턴, 주의 필요 |
| `HIGH` | 위험 패턴 감지, 사용 자제 |
| `CRITICAL` | 알려진 ReDoS 패턴, 거부 |

### SafeRegexConfig

```python
from truthound.validators.security import (
    SafeRegexConfig,
    RegexSafetyChecker,
    check_regex_safety,
)

# 프리셋
strict_config = SafeRegexConfig.strict()    # 신뢰할 수 없는 패턴용
lenient_config = SafeRegexConfig.lenient()  # 신뢰 패턴용

# 커스텀 설정
config = SafeRegexConfig(
    max_pattern_length=1000,      # 최대 패턴 길이
    max_groups=20,                # 최대 캡처 그룹
    max_quantifier_range=100,     # 최대 {n,m} 범위
    max_alternations=50,          # 최대 대안 분기
    max_nested_depth=10,          # 최대 중첩 깊이
    allow_backreferences=False,   # 역참조 허용
    allow_lookaround=True,        # lookahead/lookbehind 허용
    timeout_seconds=1.0,          # 매칭 제한 시간
    max_input_length=100_000,     # 최대 입력 길이
)
```

### RegexComplexityAnalyzer

정적 분석으로 위험 패턴 감지:

```python
from truthound.validators.security import (
    RegexComplexityAnalyzer,
    analyze_regex_complexity,
)

analyzer = RegexComplexityAnalyzer(config)
result = analyzer.analyze(r"(a+)+b")

print(result.risk_level)          # ReDoSRisk.CRITICAL
print(result.complexity_score)    # 높은 점수
print(result.dangerous_constructs)  # ["nested_quantifiers"]
print(result.is_safe)             # False
print(result.recommendation)      # 안전한 대안 제안

# 편의 함수
result = analyze_regex_complexity(r"(a+)+b")
```

### 감지되는 위험 패턴

| 패턴 | 이름 | 위험 | 설명 |
|------|------|------|------|
| `(a+)+` | nested_quantifiers | CRITICAL | 지수적 백트래킹 |
| `(a+){2,}` | nested_quantifiers_bounded | CRITICAL | 제한 중첩 양화사 |
| `((a)+)+` | deeply_nested_quantifiers | CRITICAL | 깊은 중첩 |
| `(a\|b)+` | alternation_with_quantifier | HIGH | 양화사 대안 |
| `\1+` | quantified_backreference | HIGH | 양화사 역참조 |
| `.*.*` | adjacent_quantifiers | MEDIUM | 인접 양화사 |
| `(a\|b\|c\|...)+` | long_alternation_chain | MEDIUM | 긴 대안 체인 |
| `.+.` | greedy_dot_conflict | MEDIUM | 탐욕적 충돌 |

### RegexSafetyChecker

```python
from truthound.validators.security import (
    RegexSafetyChecker,
    check_regex_safety,
)

checker = RegexSafetyChecker(config)

# 안전성 검사
is_safe, result = checker.check(r"^[a-z]+$")
if not is_safe:
    print(f"Unsafe: {result.dangerous_constructs}")

# 편의 함수
is_safe, result = check_regex_safety(r"(a+)+b")
```

### SafeRegexExecutor

타임아웃과 함께 안전한 정규식 실행:

```python
from truthound.validators.security import (
    SafeRegexExecutor,
    create_safe_regex,
    safe_match,
    safe_search,
)

# 안전한 정규식 생성
executor = create_safe_regex(r"^[a-z]+$", config)

# 안전한 매칭 (타임아웃 적용)
match = executor.match("hello")
match = executor.search("test string")
matches = executor.findall("hello world")

# 편의 함수
match = safe_match(r"^[a-z]+$", "hello")
match = safe_search(r"[0-9]+", "test123")
```

### ML 기반 위험 예측

머신러닝을 사용한 ReDoS 위험 예측:

```python
from truthound.validators.security import (
    MLPatternAnalyzer,
    predict_redos_risk,
    FeatureExtractor,
)

# ML 분석기
analyzer = MLPatternAnalyzer()
result = analyzer.analyze(r"(a+)+b")

print(result.risk_probability)  # 0.95
print(result.confidence)        # 0.87
print(result.features)          # 추출된 피처

# 편의 함수
risk_level = predict_redos_risk(r"(a+)+b")
```

### PatternOptimizer

위험한 패턴을 안전하게 최적화:

```python
from truthound.validators.security import (
    PatternOptimizer,
    optimize_pattern,
    OptimizationRule,
)

optimizer = PatternOptimizer()
result = optimizer.optimize(r"(a+)+b")

print(result.original_pattern)    # (a+)+b
print(result.optimized_pattern)   # a+b
print(result.rules_applied)       # 적용된 규칙
print(result.is_equivalent)       # 동등성 여부

# 편의 함수
optimized = optimize_pattern(r"(a+)+b")
```

### CVE 데이터베이스

알려진 취약 패턴 데이터베이스:

```python
from truthound.validators.security import (
    CVEDatabase,
    check_cve_vulnerability,
    CVEEntry,
)

db = CVEDatabase()

# CVE 검사
result = db.check(r"(a+)+b")
if result.is_vulnerable:
    print(f"CVE: {result.cve_id}")
    print(f"Severity: {result.severity}")
    print(f"Description: {result.description}")

# 편의 함수
result = check_cve_vulnerability(r"pattern")
```

### CPU 모니터링

실행 중 리소스 모니터링:

```python
from truthound.validators.security import (
    CPUMonitor,
    execute_with_monitoring,
    ResourceLimits,
)

limits = ResourceLimits(
    max_cpu_percent=50.0,
    max_memory_mb=100,
    max_time_seconds=1.0,
)

monitor = CPUMonitor(limits)

# 모니터링 실행
result = execute_with_monitoring(
    lambda: re.match(pattern, input_text),
    monitor=monitor,
)

if result.timed_out:
    print(f"Timeout after {result.elapsed_seconds}s")
print(f"CPU: {result.cpu_percent}%")
print(f"Memory: {result.memory_mb}MB")
```

### 패턴 프로파일링

정규식 성능 프로파일링:

```python
from truthound.validators.security import (
    PatternProfiler,
    profile_pattern,
    BenchmarkConfig,
)

config = BenchmarkConfig(
    iterations=1000,
    input_sizes=[100, 1000, 10000],
    timeout_per_iteration=0.1,
)

profiler = PatternProfiler(config)
result = profiler.profile(r"^[a-z]+$")

print(result.mean_time_ms)
print(result.std_time_ms)
print(result.complexity_class)  # O(n), O(n^2), O(2^n)
print(result.backtrack_count)

# 편의 함수
result = profile_pattern(r"pattern")
```

### RE2 엔진

선형 시간 보장 엔진 (google-re2 필요):

```python
from truthound.validators.security import (
    RE2Engine,
    safe_match_re2,
    safe_search_re2,
    is_re2_available,
    check_re2_compatibility,
)

# RE2 사용 가능 확인
if is_re2_available():
    # 호환성 검사
    compatible, reason = check_re2_compatibility(r"pattern")
    if not compatible:
        print(f"Not compatible: {reason}")

    # RE2 엔진 사용
    engine = RE2Engine()
    match = engine.match(r"^[a-z]+$", "hello")

    # 편의 함수
    match = safe_match_re2(r"^[a-z]+$", "hello")
    match = safe_search_re2(r"[0-9]+", "test123")
```

RE2 지원하지 않는 기능:
- 역참조 (`\1`, `\2`, ...)
- Lookahead (`(?=...)`, `(?!...)`)
- Lookbehind (`(?<=...)`, `(?<!...)`)
- 조건부 패턴
- 원자 그룹

---

## 3. 통합 사용

### 검증기에서 보안 적용

```python
from truthound.validators.base import Validator
from truthound.validators.security import (
    SecureQueryMixin,
    SecurityPolicy,
    RegexSafetyChecker,
    SafeRegexConfig,
)

class SecurePatternValidator(Validator, SecureQueryMixin):
    def __init__(self, pattern: str):
        super().__init__()
        self.set_security_policy(SecurityPolicy.strict())

        # 패턴 안전성 검사
        checker = RegexSafetyChecker(SafeRegexConfig.strict())
        is_safe, result = checker.check(pattern)
        if not is_safe:
            raise ValueError(
                f"Unsafe pattern: {result.dangerous_constructs}"
            )

        self.pattern = pattern

    def validate(self, lf):
        # 안전한 쿼리 및 패턴 사용
        ...
```

### 엔터프라이즈 SDK와 통합

```python
from truthound.validators.sdk.enterprise import EnterpriseSDKManager

manager = EnterpriseSDKManager()

# 보안 기능 포함 실행
result = await manager.execute_validator(
    validator_class=SecurePatternValidator,
    data=my_dataframe,
)
```

---

## 다음 단계

- [엔터프라이즈 SDK](enterprise-sdk.md) - 샌드박스, 서명, 라이선스
- [커스텀 검증기](custom-validators.md) - SDK 기본 사용법
- [내장 검증기](built-in.md) - 289개 내장 검증기 참조
