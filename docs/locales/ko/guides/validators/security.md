# 보안 Guide

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

보안 Module 아키텍처:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Security Module                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│      SQL Injection Prevention  │   │          ReDoS Protection              │
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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 1. SQL Injection Prevention

실무 운영 가이드에서 SQL, Prevents을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 보안Level

| 실무 운영 가이드에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `STRICT`, STRICT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `STANDARD`, STANDARD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Balanced 보안 (default) |
| 실무 운영 가이드에서 `PERMISSIVE`, PERMISSIVE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Relaxed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 보안Policy

```python
from truthound.validators.security import (
    SecurityPolicy,
    SecurityLevel,
    SQLQueryValidator,
)

# Preset policies
strict_policy = SecurityPolicy.strict()
standard_policy = SecurityPolicy.standard()
permissive_policy = SecurityPolicy.permissive()

# Custom policy
policy = SecurityPolicy(
    level=SecurityLevel.STANDARD,
    max_query_length=10000,           # Maximum query length
    max_identifier_length=128,        # Maximum identifier length

    # Structural permissions
    allow_joins=True,                 # Allow JOIN
    allow_subqueries=True,            # Allow subqueries
    allow_aggregations=True,          # Allow aggregate functions
    allow_window_functions=True,      # Allow window functions
    allow_cte=True,                   # Allow WITH clause
    allow_union=False,                # Block UNION (injection vector)

    # Allowed statement types
    allowed_statements={"SELECT", "WITH"},

    # Blocked patterns (regex)
    blocked_patterns=[r"xp_cmdshell", r"sp_executesql"],

    # Blocked functions
    blocked_functions=[
        "SLEEP",
        "BENCHMARK",
        "LOAD_FILE",
        "INTO OUTFILE",
        "INTO DUMPFILE",
    ],

    # Whitelist (empty allows all)
    allowed_tables={"orders", "customers"},
    allowed_columns={"id", "name", "amount"},

    # Violation callback
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

# Create validator
validator = SQLQueryValidator(policy=policy)

# Validate query
try:
    validator.validate("SELECT * FROM orders WHERE amount > 100")
    print("Query is safe")
except SQLInjectionError as e:
    print(f"Injection detected: {e.pattern}")
except QueryValidationError as e:
    print(f"Validation failed: {e}")

# Convenience function
validate_sql_query(
    "SELECT id, amount FROM orders",
    allowed_tables=["orders", "customers"],
)
```

### Dangerous Pattern Detection

실무 운영 가이드에서 Built-in을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|---------|----------|
| 실무 운영 가이드에서 DDL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, CREATE, ALTER, DROP, TRUNCATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 DCL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `GRANT`, `REVOKE`, `DENY`, GRANT, REVOKE, DENY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 DML, Modification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `INSERT`, `UPDATE`, `DELETE`, INSERT, UPDATE, DELETE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `EXEC`, `EXECUTE`, `CALL`, EXEC, EXECUTE, CALL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 파일 | 실무 운영 가이드에서 `LOAD_FILE`, `INTO OUTFILE`, LOAD_FILE, INTO, OUTFILE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Stacked, Query을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `; SELECT`, `; DROP`, SELECT, DROP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 UNION, Injection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `UNION SELECT`, UNION, SELECT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Time-Based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `SLEEP`, `WAITFOR DELAY`, `BENCHMARK`, SLEEP, WAITFOR, DELAY, BENCHMARK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Error-Based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `EXTRACTVALUE`, `UPDATEXML`, EXTRACTVALUE, UPDATEXML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Boolean-Based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `OR 1=1`, `AND '1'='1'`, AND을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Comment을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `--`, `/* */`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 LOW-MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### SecureSQLBuilder

실무 운영 가이드에서 Fluent을(를) 다루는 항목입니다:

```python
from truthound.validators.security import (
    SecureSQLBuilder,
    ParameterizedQuery,
)

builder = SecureSQLBuilder(
    allowed_tables=["orders", "customers"],
    policy=SecurityPolicy.standard(),
)

# Build query
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

# Execute with Polars SQL context
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

# Render (escape values)
rendered = query.render()
# SELECT * FROM orders WHERE amount > 100 AND status = 'pending'
```

실무 운영 가이드에서 Supported을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Escaping을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|----------|
| 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `NULL`, NULL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `TRUE`, `FALSE`, TRUE, FALSE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `int`, `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 As-is을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `'`, `''`, Single을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `list`, `tuple`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `(val1, val2, ...)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### SchemaWhitelist

```python
from truthound.validators.security import (
    SchemaWhitelist,
    WhitelistValidator,
)

# Define schema whitelist
whitelist = SchemaWhitelist()
whitelist.add_table("orders", ["id", "customer_id", "amount", "status"])
whitelist.add_table("customers", ["id", "name", "email"])

# Validate table/column
whitelist.validate_table("orders")  # OK
whitelist.validate_column("orders", "amount")  # OK
whitelist.validate_column("orders", "password")  # QueryValidationError

# Validate query
validator = WhitelistValidator(whitelist)
validator.validate_query("SELECT id, amount FROM orders")  # OK
validator.validate_query("SELECT password FROM users")  # Error
```

### SecureQueryMixin

Secure query execution in 검증기:

```python
from truthound.validators.security import SecureQueryMixin
from truthound.validators.base import Validator

class MyValidator(Validator, SecureQueryMixin):
    def __init__(self):
        super().__init__()
        self.set_security_policy(SecurityPolicy.strict())

    def validate(self, lf):
        # Build secure query
        query = self.build_secure_query(
            table="data",
            columns=["id", "value"],
            where="value > :threshold",
            parameters={"threshold": 100},
            allowed_tables=["data"],
        )

        # Execute secure query
        result = self.execute_secure_query(lf, query, table_name="data")
        return self.process_result(result)
```

### Query감사Logger

Query execution 감사 logging:

```python
from truthound.validators.security import QueryAuditLogger

logger = QueryAuditLogger(
    max_entries=10000,
    log_full_queries=False,  # Mask values
    python_logger=logging.getLogger("sql_audit"),
)

# Log query
logger.log_query(
    query="SELECT * FROM users WHERE email = 'test@example.com'",
    success=True,
    user="admin",
    context={"source": "api"},
)

# Query audit
recent = logger.get_recent(100)
failures = logger.get_failures(50)
by_hash = logger.get_by_hash("abc123...")

# Statistics
stats = logger.get_stats()
# {
#   "total_queries": 1000,
#   "successful": 950,
#   "failed": 50,
#   "success_rate": 0.95,
#   "unique_queries": 120,
# }

# Export to file
logger.export_to_file("audit.log")
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 2. ReDoS Protection

실무 운영 가이드에서 Prevents, Regular, Expression, Denial, Service, ReDoS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### ReDoSRisk

| 실무 운영 가이드에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `NONE`, NONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `LOW`, LOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Minimal을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MEDIUM`, MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Some을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HIGH`, HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dangerous을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CRITICAL`, CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Known, ReDoS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### SafeRegexConfig

```python
from truthound.validators.security import (
    SafeRegexConfig,
    RegexSafetyChecker,
    check_regex_safety,
)

# Presets
strict_config = SafeRegexConfig.strict()    # For untrusted patterns
lenient_config = SafeRegexConfig.lenient()  # For trusted patterns

# Custom configuration
config = SafeRegexConfig(
    max_pattern_length=1000,      # Maximum pattern length
    max_groups=20,                # Maximum capture groups
    max_quantifier_range=100,     # Maximum {n,m} range
    max_alternations=50,          # Maximum alternation branches
    max_nested_depth=10,          # Maximum nesting depth
    allow_backreferences=False,   # Allow backreferences
    allow_lookaround=True,        # Allow lookahead/lookbehind
    timeout_seconds=1.0,          # Matching timeout
    max_input_length=100_000,     # Maximum input length
)
```

### RegexComplexityAnalyzer

실무 운영 가이드에서 Static을(를) 다루는 항목입니다:

```python
from truthound.validators.security import (
    RegexComplexityAnalyzer,
    analyze_regex_complexity,
)

analyzer = RegexComplexityAnalyzer(config)
result = analyzer.analyze(r"(a+)+b")

print(result.risk_level)          # ReDoSRisk.CRITICAL
print(result.complexity_score)    # High score
print(result.dangerous_constructs)  # ["nested_quantifiers"]
print(result.is_safe)             # False
print(result.recommendation)      # Safe alternative suggestion

# Convenience function
result = analyze_regex_complexity(r"(a+)+b")
```

### Detected Dangerous Patterns

| 실무 운영 가이드에서 Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Name을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Risk을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|------|------|-------------|
| 실무 운영 가이드에서 `(a+)+`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Exponential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `(a+){2,}`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Bounded을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `((a)+)+`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Deeply을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Alternation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `\1+`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Quantified을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `.*.*`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Adjacent을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Long을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `.+.`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Greedy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### RegexSafetyChecker

```python
from truthound.validators.security import (
    RegexSafetyChecker,
    check_regex_safety,
)

checker = RegexSafetyChecker(config)

# Safety check
is_safe, result = checker.check(r"^[a-z]+$")
if not is_safe:
    print(f"Unsafe: {result.dangerous_constructs}")

# Convenience function
is_safe, result = check_regex_safety(r"(a+)+b")
```

### SafeRegexExecutor

실무 운영 가이드에서 Safe을(를) 다루는 항목입니다:

```python
from truthound.validators.security import (
    SafeRegexExecutor,
    create_safe_regex,
    safe_match,
    safe_search,
)

# Create safe regex
executor = create_safe_regex(r"^[a-z]+$", config)

# Safe matching (with timeout)
match = executor.match("hello")
match = executor.search("test string")
matches = executor.findall("hello world")

# Convenience functions
match = safe_match(r"^[a-z]+$", "hello")
match = safe_search(r"[0-9]+", "test123")
```

### ML-Based Risk Prediction

실무 운영 가이드에서 ReDoS을(를) 다루는 항목입니다:

```python
from truthound.validators.security import (
    MLPatternAnalyzer,
    predict_redos_risk,
    FeatureExtractor,
)

# ML analyzer
analyzer = MLPatternAnalyzer()
result = analyzer.analyze(r"(a+)+b")

print(result.risk_probability)  # 0.95
print(result.confidence)        # 0.87
print(result.features)          # Extracted features

# Convenience function
risk_level = predict_redos_risk(r"(a+)+b")
```

### PatternOptimizer

실무 운영 가이드에서 Safely을(를) 다루는 항목입니다:

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
print(result.rules_applied)       # Applied rules
print(result.is_equivalent)       # Equivalence status

# Convenience function
optimized = optimize_pattern(r"(a+)+b")
```

### CVE 데이터베이스

Known vulnerable pattern 데이터베이스:

```python
from truthound.validators.security import (
    CVEDatabase,
    check_cve_vulnerability,
    CVEEntry,
)

db = CVEDatabase()

# CVE check
result = db.check(r"(a+)+b")
if result.is_vulnerable:
    print(f"CVE: {result.cve_id}")
    print(f"Severity: {result.severity}")
    print(f"Description: {result.description}")

# Convenience function
result = check_cve_vulnerability(r"pattern")
```

### CPU 모니터링

런타임 resource 모니터링:

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

# Execute with monitoring
result = execute_with_monitoring(
    lambda: re.match(pattern, input_text),
    monitor=monitor,
)

if result.timed_out:
    print(f"Timeout after {result.elapsed_seconds}s")
print(f"CPU: {result.cpu_percent}%")
print(f"Memory: {result.memory_mb}MB")
```

### Pattern 프로파일링

Regex 성능 프로파일링:

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

# Convenience function
result = profile_pattern(r"pattern")
```

### RE2 엔진

실무 운영 가이드에서 Linear-time을(를) 다루는 항목입니다:

```python
from truthound.validators.security import (
    RE2Engine,
    safe_match_re2,
    safe_search_re2,
    is_re2_available,
    check_re2_compatibility,
)

# Check RE2 availability
if is_re2_available():
    # Compatibility check
    compatible, reason = check_re2_compatibility(r"pattern")
    if not compatible:
        print(f"Not compatible: {reason}")

    # Use RE2 engine
    engine = RE2Engine()
    match = engine.match(r"^[a-z]+$", "hello")

    # Convenience functions
    match = safe_match_re2(r"^[a-z]+$", "hello")
    match = safe_search_re2(r"[0-9]+", "test123")
```

실무 운영 가이드에서 Features, RE2을(를) 다루는 항목입니다:
- 실무 운영 가이드에서 `\1`, `\2`, Backreferences을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `(?=...)`, `(?!...)`, Lookahead을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `(?<=...)`, `(?<!...)`, Lookbehind을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Conditional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Atomic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 3. Integrated Usage

### Applying 보안 in 검증기

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

        # Pattern safety check
        checker = RegexSafetyChecker(SafeRegexConfig.strict())
        is_safe, result = checker.check(pattern)
        if not is_safe:
            raise ValueError(
                f"Unsafe pattern: {result.dangerous_constructs}"
            )

        self.pattern = pattern

    def validate(self, lf):
        # Use secure queries and patterns
        ...
```

### 통합 with Enterprise SDK

```python
from truthound.validators.sdk.enterprise import EnterpriseSDKManager

manager = EnterpriseSDKManager()

# Execute with security features included
result = await manager.execute_validator(
    validator_class=SecurePatternValidator,
    data=my_dataframe,
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 다음 단계

- 실무 운영 가이드에서 Enterprise, SDK, Sandbox을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Custom, Validators, SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Built-in, Validators을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
