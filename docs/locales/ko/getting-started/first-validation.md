# 첫 검증 Tutorial

초기 설정과 첫 실행에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Objectives

초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

1. Understand Truthound's 검증 워크플로우
2. 초기 설정과 첫 실행에서 Create을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. Interpret 검증 결과
4. Fix 데이터 품질 issues
5. Set up continuous 검증

## Step 1: Prepare Your Data

초기 설정과 첫 실행에서 Let을(를) 다루는 항목입니다:

```python
import polars as pl
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
random.seed(42)

# Generate sample customer data
n_rows = 1000

data = {
    "customer_id": list(range(1, n_rows + 1)),
    "name": [
        f"Customer {i}" if random.random() > 0.05 else None
        for i in range(n_rows)
    ],
    "email": [
        f"customer{i}@example.com" if random.random() > 0.1 else "invalid-email"
        for i in range(n_rows)
    ],
    "age": [
        random.randint(18, 80) if random.random() > 0.02 else -1
        for _ in range(n_rows)
    ],
    "signup_date": [
        (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        if random.random() > 0.03 else "invalid-date"
        for _ in range(n_rows)
    ],
    "country": random.choices(
        ["US", "UK", "CA", "AU", None],
        weights=[40, 20, 15, 15, 10],
        k=n_rows
    ),
}

# Add some duplicates
data["customer_id"][500:510] = list(range(1, 11))  # Duplicate IDs

df = pl.DataFrame(data)
df.write_csv("customers.csv")

print(f"Created customers.csv with {n_rows} rows")
```

## Step 2: Learn the 스키마

초기 설정과 첫 실행에서 Truthound을(를) 다루는 항목입니다:

초기 설정과 첫 실행에서 CLI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

    ```bash
    truthound learn customers.csv -o customer_schema.yaml
    ```

초기 설정과 첫 실행에서 Output을(를) 다루는 항목입니다:
    ```
    Schema saved to customer_schema.yaml
      Columns: 6
      Rows: 1000
    ```

초기 설정과 첫 실행에서 Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

    ```python
    from truthound.schema import learn

    schema = learn("customers.csv")
    schema.save("customer_schema.yaml")

    print(f"Learned schema with {len(schema.columns)} columns")
    ```

The 스키마 captures:

- 초기 설정과 첫 실행에서 Column을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Unique을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Min/max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Allowed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Step 3: Validate Your Data

초기 설정과 첫 실행에서 Truthound, Run을(를) 다루는 항목입니다:

초기 설정과 첫 실행에서 CLI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

    ```bash
    truthound check customers.csv --schema customer_schema.yaml
    ```

초기 설정과 첫 실행에서 Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

    ```python
    import truthound as th

    run = th.check(
        "customers.csv",
        schema="customer_schema.yaml"
    )

    # Print formatted summary (uses Rich for pretty output)
    run.print()

    # Or access detailed issues programmatically
    for issue in run.issues:
        detail = f" - {issue.details}" if issue.details else ""
        print(f"[{issue.severity.value}] {issue.column}: {issue.issue_type}{detail}")
    ```

## Step 4: Understand the 결과

The 검증 리포트 shows:

```
Truthound Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Column       ┃ Issue              ┃ Count ┃ Severity ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ customer_id  │ unique_violation   │    10 │ critical │
│ age          │ out_of_range       │    20 │   high   │
│ name         │ null               │    50 │   high   │
│ email        │ invalid_format     │   100 │   high   │
│ signup_date  │ invalid_format     │    30 │  medium  │
│ country      │ null               │   100 │  medium  │
└──────────────┴────────────────────┴───────┴──────────┘

Summary: 6 issues found
```

### Severity Levels

| 초기 설정과 첫 실행에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 초기 설정과 첫 실행에서 `critical`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 초기 설정과 첫 실행에서 `high`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Business을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 초기 설정과 첫 실행에서 `medium`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 초기 설정과 첫 실행에서 `low`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Minor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Issue Types

| 초기 설정과 첫 실행에서 Issue, Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------------|-------------|
| 초기 설정과 첫 실행에서 `unique_violation`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Duplicate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 초기 설정과 첫 실행에서 `out_of_range`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Values을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 초기 설정과 첫 실행에서 `null`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Null values in 컬럼 |
| 초기 설정과 첫 실행에서 `invalid_format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Values을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Step 5: Generate Detailed 리포트

초기 설정과 첫 실행에서 HTML, Create을(를) 다루는 항목입니다:

초기 설정과 첫 실행에서 CLI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

    ```bash
    truthound check customers.csv \
        --schema customer_schema.yaml \
        --format html \
        --output report.html
    ```

    !!! note "참고"
초기 설정과 첫 실행에서 `pip install truthound[reports]`, `pip install jinja2`, Install을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

초기 설정과 첫 실행에서 Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

    ```python
    import truthound as th
    from truthound.html_reporter import write_html_report, HTMLReportConfig

    # Run validation
    run = th.check("customers.csv", schema="customer_schema.yaml")

    # Write HTML report to file (option 1: using config)
    config = HTMLReportConfig(title="Customer Data Quality Report")
    write_html_report(run, "report.html", config=config)

    # Or simply (option 2: using kwargs)
    write_html_report(run, "report.html", title="Customer Data Quality Report")

    # Generate HTML string without writing to file
    from truthound.html_reporter import generate_html_report
    html_content = generate_html_report(run, title="Customer Data Quality Report")
    ```

    !!! note "참고"
초기 설정과 첫 실행에서 `pip install truthound[reports]`, `pip install jinja2`, Install을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Step 6: Fix Issues Programmatically

초기 설정과 첫 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
import polars as pl
import truthound as th

# Load data
df = pl.read_csv("customers.csv")

# Fix duplicates - keep first occurrence
df_cleaned = df.unique(subset=["customer_id"], keep="first")

# Fix invalid ages (negative values to null)
df_cleaned = df_cleaned.with_columns(
    pl.when(pl.col("age") < 0)
    .then(None)
    .otherwise(pl.col("age"))
    .alias("age")
)

# Fix invalid emails - mark as null
df_cleaned = df_cleaned.with_columns(
    pl.when(~pl.col("email").str.contains("@"))
    .then(None)
    .otherwise(pl.col("email"))
    .alias("email")
)

# Fix invalid dates - mark as null
df_cleaned = df_cleaned.with_columns(
    pl.when(~pl.col("signup_date").str.contains(r"^\d{4}-\d{2}-\d{2}$"))
    .then(None)
    .otherwise(pl.col("signup_date"))
    .alias("signup_date")
)

# Save cleaned data
df_cleaned.write_csv("customers_cleaned.csv")

# Re-validate
run = th.check("customers_cleaned.csv", schema="customer_schema.yaml")
print(f"Issues remaining: {len(run.issues)}")
```

## Step 7: Set Up Continuous 검증

Create a 체크포인트 for ongoing 검증:

```yaml
# truthound.yaml
checkpoints:
  - name: customer_data_check
    data_source: customers.csv
    validators:
      - "null"
      - "duplicate"
      - "range"
      - "format"
    min_severity: medium
    fail_on_critical: true
    fail_on_high: false
    timeout_seconds: 3600
    tags:
      dataset: customers
      environment: production
    actions:
      - type: store_result
        store_path: ./validation_results
      - type: slack
        webhook_url: ${SLACK_WEBHOOK}
```

!!! info "참고"
    | 초기 설정과 첫 실행에서 Field을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
    |-------|-------------|---------|
    | 초기 설정과 첫 실행에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Unique 체크포인트 identifier | 초기 설정과 첫 실행에서 `"default_checkpoint"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
    | 초기 설정과 첫 실행에서 `data_source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 파일 path or connection string | 초기 설정과 첫 실행에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
    | 초기 설정과 첫 실행에서 `validators`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | List of 검증기 names | All 검증기 |
    | 초기 설정과 첫 실행에서 `min_severity`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Minimum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
    | 초기 설정과 첫 실행에서 `fail_on_critical`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Fail을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 `true`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
    | 초기 설정과 첫 실행에서 `fail_on_high`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Fail을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 `false`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
    | 초기 설정과 첫 실행에서 `timeout_seconds`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 `3600`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
    | 초기 설정과 첫 실행에서 `tags`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 Key-value을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 초기 설정과 첫 실행에서 `{}`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

Run 검증:

```bash
truthound checkpoint run customer_data_check --config truthound.yaml --strict
```

Or run ad-hoc 검증 without a 설정 파일:

```bash
truthound checkpoint run quick_check \
    --data customers.csv \
    --validators null,duplicate,range \
    --strict \
    --store ./validation_results
```

### Integrating with GitHub Actions

```yaml
# .github/workflows/data-quality.yml
name: Customer Data Quality

on:
  push:
    paths:
      - 'data/customers.csv'
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Truthound
        run: pip install truthound

      - name: Validate Customer Data
        run: |
          truthound checkpoint run customer_data_check \
            --config truthound.yaml \
            --strict \
            --github-summary
```

## Summary

초기 설정과 첫 실행에서 You을(를) 다루는 항목입니다:

1. 초기 설정과 첫 실행에서 Automatically을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 초기 설정과 첫 실행에서 Run, Check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 초기 설정과 첫 실행에서 Interpret, Understand을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 초기 설정과 첫 실행에서 HTML, Generate, Create을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 초기 설정과 첫 실행에서 Fix, Programmatically을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 초기 설정과 첫 실행에서 Automate, Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 다음 단계

- 초기 설정과 첫 실행에서 Validators, Guide, Explore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 CI/CD, Integration, Advanced을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Custom, Validators, Create을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 초기 설정과 첫 실행에서 Data, Sources, Connect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
