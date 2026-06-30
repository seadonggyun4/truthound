# truthound ml 드리프트

CLI 명령 실행에서 Detect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Synopsis

```bash
truthound ml drift <baseline> <current> [OPTIONS]
```

## Arguments

| CLI 명령 실행에서 Argument을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-------------|
| CLI 명령 실행에서 `baseline`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 JSON, Path, CSV, Parquet, NDJSON, JSONL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `current`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 JSON, Path, CSV, Parquet, NDJSON, JSONL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Options

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|---------|-------------|
| CLI 명령 실행에서 `--method`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-m`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `feature`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Detection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-t`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `0.1`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 드리프트 threshold (0.0-1.0) |
| CLI 명령 실행에서 `--columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 to compare (comma-separated) |
| CLI 명령 실행에서 `--output`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-o`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Output 파일 path |

## Description

CLI 명령 실행에서 `ml drift`을(를) 다루는 항목입니다:

1. CLI 명령 실행에서 Compares을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. CLI 명령 실행에서 Detects을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. **리포트** 드리프트 scores per 컬럼
4. CLI 명령 실행에서 Identifies을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Detection Methods

### Distribution (`distribution`)

Compares individual 컬럼 distributions.

- CLI 명령 실행에서 Tests, KS-test, Chi-squared을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Best, Feature-level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Speed, Fast을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Interpretability, High을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```bash
truthound ml drift baseline.csv current.csv --method distribution
```

### Feature (`feature`)

CLI 명령 실행에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

- CLI 명령 실행에서 Tests, Mean/std을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- **Best for**: ML feature 모니터링
- CLI 명령 실행에서 Speed, Fast을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Interpretability, High을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```bash
truthound ml drift baseline.csv current.csv --method feature
```

### Multivariate (`multivariate`)

CLI 명령 실행에서 Detects을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

- CLI 명령 실행에서 Algorithm, Compares을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Best, Correlated을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Speed, Slower을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Advantage, Catches을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```bash
truthound ml drift baseline.csv current.csv --method multivariate
```

## 예시

### Basic 드리프트 Detection

```bash
truthound ml drift baseline.csv current.csv
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
ML Drift Detection Report
=========================
Baseline: baseline.csv (10,000 rows)
Current: current.csv (12,000 rows)
Method: distribution
Threshold: 0.1

Overall Drift Score: 0.23 (DRIFT DETECTED)

Column Results
──────────────────────────────────────────────────────────────────
Column          Score     Threshold   Drift Detected    Details
──────────────────────────────────────────────────────────────────
age             0.05      0.10        No                -
income          0.34      0.10        Yes ⚠️            Mean shift: +15%
category        0.08      0.10        No                -
region          0.42      0.10        Yes ⚠️            New values: 3
score           0.12      0.10        Yes ⚠️            Std shift: +22%
──────────────────────────────────────────────────────────────────

Drift Summary:
  Total Columns: 5
  Drifted Columns: 3 (60%)
  Status: DRIFT DETECTED

Recommendations:
  - Review 'income' for mean shift (+15%)
  - Check 'region' for new category values
  - Investigate 'score' distribution change
```

### Specific 컬럼

Compare only selected 컬럼:

```bash
truthound ml drift baseline.csv current.csv --columns age,income,score
```

### Custom Threshold

Adjust 드리프트 sensitivity:

```bash
# More sensitive (lower threshold)
truthound ml drift baseline.csv current.csv --threshold 0.05

# Less sensitive (higher threshold)
truthound ml drift baseline.csv current.csv --threshold 0.2
```

### Multivariate Detection

Detect complex 드리프트 patterns:

```bash
truthound ml drift train_data.csv production_data.csv --method multivariate
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
ML Drift Detection Report
=========================
Method: multivariate

Multivariate Drift Analysis
──────────────────────────────────────────────────────────────────
Metric                    Value       Threshold   Status
──────────────────────────────────────────────────────────────────
Overall Drift Score       0.28        0.10        DRIFT DETECTED
Correlation Change        0.15        0.10        Changed
Covariance Shift          0.22        0.10        Shifted
Principal Component Drift 0.18        0.10        Drifted
──────────────────────────────────────────────────────────────────

Most Affected Feature Combinations:
  1. income × tenure: correlation changed from 0.72 to 0.45
  2. age × salary: distribution shift detected
  3. region × category: joint distribution changed
```

### JSON Output

```bash
truthound ml drift baseline.csv current.csv --output drift_report.json
```

Output 파일 (`drift_report.json`):
```json
{
  "baseline": {
    "file": "baseline.csv",
    "rows": 10000
  },
  "current": {
    "file": "current.csv",
    "rows": 12000
  },
  "method": "distribution",
  "threshold": 0.1,
  "overall_drift_score": 0.23,
  "has_drift": true,
  "column_results": [
    {
      "column": "age",
      "score": 0.05,
      "has_drift": false,
      "details": null
    },
    {
      "column": "income",
      "score": 0.34,
      "has_drift": true,
      "details": {
        "type": "mean_shift",
        "baseline_mean": 50000,
        "current_mean": 57500,
        "percent_change": 15.0
      }
    },
    {
      "column": "region",
      "score": 0.42,
      "has_drift": true,
      "details": {
        "type": "new_categories",
        "new_values": ["west", "southwest", "northwest"],
        "count": 3
      }
    }
  ],
  "summary": {
    "total_columns": 5,
    "drifted_columns": 3,
    "drift_ratio": 0.6
  }
}
```

## Method Comparison

| CLI 명령 실행에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Speed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Complexity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Correlated, Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|------------|---------------------|----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Fast을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Low을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Fast을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Medium을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | ML 모니터링 |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Slow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 High을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Complex 드리프트 |

## Comparison: `ml drift` vs `compare`

| CLI 명령 실행에서 Feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `ml drift`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `compare`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|------------|-----------|
| CLI 명령 실행에서 Focus을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | ML-oriented 드리프트 detection | CLI 명령 실행에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 Methods을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 Multivariate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 Speed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Varies을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Fast을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | ML 파이프라인 | CLI 명령 실행에서 General을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Use Cases

### 1. ML Model 모니터링

```bash
# Check if production data has drifted from training
truthound ml drift training_data.csv production_data.csv --method multivariate --threshold 0.1
```

### 2. Feature Store 모니터링

```bash
# Monitor feature drift daily
truthound ml drift features_yesterday.parquet features_today.parquet --method feature
```

### 3. A/B Test 검증

```bash
# Ensure test groups are comparable
truthound ml drift control_group.csv treatment_group.csv --columns demographics
```

### 4. CI/CD 파이프라인

```yaml
# GitHub Actions
- name: Check Feature Drift
  run: |
    truthound ml drift baseline.csv current.csv --method multivariate --threshold 0.15 --output drift.json
    # Parse result and fail if drift detected
    python -c "
    import json
    with open('drift.json') as f:
        result = json.load(f)
    if result['has_drift']:
        print(f'Drift detected! Score: {result[\"overall_drift_score\"]}')
        for col in result['column_results']:
            if col['has_drift']:
                print(f'  - {col[\"column\"]}: {col[\"score\"]}')
        exit(1)
    "
```

### 5. Retraining Trigger

```bash
# Check drift and trigger retraining if needed
if truthound ml drift train.csv prod.csv --method multivariate --threshold 0.2; then
  echo "No significant drift"
else
  echo "Drift detected, triggering retraining"
  python retrain_model.py
fi
```

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

> CLI 명령 실행에서 JSON, `--output result.json`, `has_drift`, Note, Drift, CI/CD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Commands

- CLI 명령 실행에서 `compare`, Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [`ml anomaly`](anomaly.md) - 이상치 detection
- [`ml learn-rules`](learn-rules.md) - Learn 검증 rules

## 함께 보기

- CLI 명령 실행에서 Statistical, Methods을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Advanced, Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [ML Model 모니터링](../../guides/ml-monitoring.md)
